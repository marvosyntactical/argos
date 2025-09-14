import os
from functools import lru_cache

import dash
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from argos_viz import generate_steps

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "distilgpt2")
DEVICE = "cpu" if os.environ.get("DEVICE", "cpu") == "cpu" else "cuda"

# ------------------------- Model loading (cached) -------------------------

@lru_cache(maxsize=2)
def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(DEVICE)
    model.eval()
    return tok, model

# ------------------------------- Dash App --------------------------------

external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/milligram/1.4.1/milligram.min.css",
]

app = Dash(__name__, external_stylesheets=external_stylesheets, title="Argos 🌌")
server = app.server

app.layout = html.Div(id="root-div", children=[
    # Inline CSS (works without assets/ folder)
    dcc.Markdown(
        """
        <style>
        :root {
          --bg-light:#ffffff; --fg-light:#000000;
          --bg-dark:#0e0e0f;  --fg-dark:#e4e4e7;
        }
        #root-div.light { background: var(--bg-light); color: var(--fg-light); }
        #root-div.dark  { background: var(--bg-dark);  color: var(--fg-dark); }
        /* make Loading overlay sit on top */
        .dash-spinner { z-index: 9999 !important; }
        </style>
        """,
        dangerously_allow_html=True
    ),

    html.Div(style={"padding":"1.2rem"}, children=[
        html.Div([
            html.H2("Argos — token flow on the hypersphere"),
            html.P("Type a prompt, hit RUN, then step through generation with NEXT/PREV. "
                   "Use the slider in the plot to sweep LayerNorm indices. "
                   "Tokens are colour-coded in both text and the sphere."),
        ], style={"maxWidth":"980px"}),

        # Global loading overlay wrapping the whole grid (shows while model loads)
        dcc.Loading(
            id="global-loading",
            type="default",
            children=html.Div(
                style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"1rem","alignItems":"start"},
                children=[
                    # LEFT PANEL (controls)
                    html.Div(children=[
                        html.Label("Hugging Face model ID"),
                        dcc.Input(id="model-id", type="text", value=DEFAULT_MODEL_ID, debounce=True, style={"width":"100%"}),

                        html.Label("Prompt"),
                        dcc.Textarea(id="prompt", value="Once upon a time", style={"width":"100%", "height":"8rem"}),

                        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"0.5rem"}, children=[
                            html.Div([html.Label("Max new tokens"),
                                      dcc.Input(id="max-new", type="number", value=32, min=1, max=128, step=1, style={"width":"100%"})]),
                            html.Div([html.Label("Temperature"),
                                      dcc.Input(id="temp", type="number", value=0.9, min=0.05, max=2.0, step=0.05, style={"width":"100%"})]),
                            html.Div([html.Label("Trajectories"),
                                      dcc.Checklist(id="traj", options=[{"label":"show", "value":"yes"}], value=["yes"]) ]),
                            html.Div([html.Label("Freeze PCA"),
                                      dcc.Checklist(id="freeze-pca", options=[{"label":"yes", "value":"freeze"}], value=[]) ]),
                            html.Div([html.Label("Dark theme"),
                                      dcc.Checklist(id="theme-toggle", options=[{"label":"on", "value":"dark"}], value=["dark"]) ]),
                        ]),

                        html.Div(style={"marginTop":"0.5rem","display":"flex","gap":"0.5rem"}, children=[
                            html.Button("RUN", id="run", className="button-primary"),
                            html.Button("PREV", id="prev"),
                            html.Button("NEXT", id="next"),
                        ]),

                        # Put status inside its own Loading too (so you see a spinner near the button)
                        dcc.Loading(html.Div(id="status",
                                             style={"marginTop":"0.25rem","fontSize":"0.9rem","opacity":0.85}), type="default"),
                        dcc.Store(id="steps-store"),
                        dcc.Store(id="step-index", data=0),
                    ]),

                    # RIGHT PANEL (graph + text)
                    html.Div(children=[
                        dcc.Loading(dcc.Graph(id="sphere", style={"height":"720px"}), type="default"),
                        html.Div(id="texts", style={"marginTop":"0.5rem"}),
                    ]),
                ]
            )
        ),

        # Hidden target that also triggers the global spinner
        html.Div(id="busy", style={"display":"none"})
    ])
])

# ------------------------------ Callbacks --------------------------------

# Theme class on the root (default = dark)
@app.callback(
    Output("root-div", "className"),
    Input("theme-toggle", "value")
)
def set_theme(value):
    return "dark" if "dark" in (value or []) else "light"

# RUN -> generate + store steps
@app.callback(
    Output("steps-store", "data"),
    Output("step-index", "data"),
    Output("status", "children"),
    Output("busy", "children"),  # dummy output to keep the global spinner honest
    Input("run", "n_clicks"),
    State("model-id", "value"),
    State("prompt", "value"),
    State("max-new", "value"),
    State("temp", "value"),
    State("traj", "value"),
    State("freeze-pca", "value"),
    prevent_initial_call=True
)
def on_run(n_clicks, model_id, prompt, max_new, temp, traj, freeze_pca):
    # The global spinner shows during this whole callback
    model_id = model_id or DEFAULT_MODEL_ID
    show_traj = ("yes" in (traj or []))
    pca_freeze = ("freeze" in (freeze_pca or []))

    # load (can be slow first time) – spinner is visible because this is an Output-producing callback
    tok, model = load_model(model_id)

    steps, _ = generate_steps(
        model, tok, (prompt or "").strip(),
        max_new_tokens=int(max_new or 32),
        temperature=float(temp or 0.9),
        pca_freeze_after_first=pca_freeze,
        show_trajectories=show_traj
    )
    prompt_len = len(tok.encode((prompt or "").strip()))
    payload = [dict(tokens=s.tokens, colors=s.colors, X=s.X.tolist(), fig=s.figure_json, prompt_len=prompt_len)
               for s in steps]

    status = f"Generated {len(payload)} steps with {model_id}."
    return payload, max(0, len(payload)-1), status, "done"

# NEXT/PREV
@app.callback(
    Output("step-index", "data", allow_duplicate=True),
    Input("prev", "n_clicks"),
    Input("next", "n_clicks"),
    State("steps-store", "data"),
    State("step-index", "data"),
    prevent_initial_call=True
)
def on_prev_next(prev_clicks, next_clicks, steps, idx):
    if not steps:
        return no_update
    trig = ctx.triggered_id
    idx = int(idx or 0)
    if trig == "prev":
        idx = max(0, idx-1)
    elif trig == "next":
        idx = min(len(steps)-1, idx+1)
    return idx

# Update figure + text; also apply dark template if needed
@app.callback(
    Output("sphere", "figure"),
    Output("texts", "children"),
    Input("steps-store", "data"),
    Input("step-index", "data"),
    Input("theme-toggle", "value"),
    prevent_initial_call=True
)
def on_step_change(steps, idx, theme_value):
    if not steps:
        return go.Figure(), ""
    idx = int(idx or 0)
    step = steps[idx]

    fig = go.Figure(step["fig"])

    # Theme the figure
    dark = ("dark" in (theme_value or []))
    if dark:
        fig.update_layout(template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e4e4e7"))
        if "scene" in fig.layout:
            fig.update_layout(scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ))
    else:
        fig.update_layout(template="plotly",
                          paper_bgcolor="rgba(0,0,0,0)")

    # Colourised token strip
    p_len = int(step.get("prompt_len", 0))
    spans = []
    for i, (tok, col) in enumerate(zip(step["tokens"], step["colors"])):
        style = {"color": col, "paddingRight": "2px"}
        if i < p_len:
            style.update({"fontWeight": "600", "opacity": 0.85})
        spans.append(html.Span(tok.replace("Ġ"," ").replace("▁"," "), style=style))
    token_row = html.Div(spans, style={"fontFamily":"monospace","fontSize":"1.05rem","lineHeight":"1.8"})
    caption = html.Div(f"Step {idx+1}/{len(steps)} — tokens so far: {len(step['tokens'])}")

    return fig, html.Div([caption, token_row])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run_server(host="0.0.0.0", port=port, debug=False)

