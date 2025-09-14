import os
import unicodedata
from functools import lru_cache

import dash
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from argos_viz import generate_steps  # unchanged API

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "distilgpt2")
DEVICE = "cpu" if os.environ.get("DEVICE", "cpu") == "cpu" else "cuda"

# ------------------------- Utilities -------------------------

def strip_diacritics(s: str) -> str:
    """
    Remove diacritics and non-ASCII oddities for display.
    """
    if not isinstance(s, str):
        return s
    # NFKD -> drop combining marks -> keep ascii only
    norm = unicodedata.normalize("NFKD", s)
    without_marks = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")
    return without_marks.encode("ascii", "ignore").decode("ascii")


def sanitize_tokens(tokens):
    return [strip_diacritics(t).replace("Ġ", " ").replace("▁", " ") for t in tokens]


def sanitize_figure_text(fig: go.Figure):
    """
    In-place: strip diacritics in text labels across base data and frames.
    """
    def fix_trace_text(tr):
        if hasattr(tr, "text") and tr.text is not None:
            if isinstance(tr.text, (list, tuple)):
                tr.text = [strip_diacritics(x) for x in tr.text]
            else:
                tr.text = strip_diacritics(tr.text)

    for tr in fig.data:
        fix_trace_text(tr)
    if fig.frames:
        for fr in fig.frames:
            for tr in fr.data:
                fix_trace_text(tr)
    return fig

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

app = Dash(__name__, external_stylesheets=external_stylesheets, title="Argos — Latent Flow")
server = app.server

# minimal CSS; theme set via root-div style (not class) so it always wins
app.layout = html.Div(id="root-div", style={"backgroundColor":"#0f1620","color":"#e4e7ee"}, children=[
    dcc.Markdown(
        """
        <style>
        .status-pill {
          display:flex; gap:.35rem; align-items:center;
          background: rgba(255,255,255,.07); border-radius: 999px; padding:.25rem .6rem;
          backdrop-filter: blur(3px);
        }
        .spinner {
          display:inline-block; width:14px; height:14px; border-radius:50%;
          border: 2px solid rgba(255,255,255,.22); border-top-color: rgba(255,255,255,.9);
          animation: spin 0.9s linear infinite;
        }
        @keyframes spin { from { transform: rotate(0deg);} to { transform: rotate(360deg);} }
        .floating-status { position: fixed; top: 12px; right: 12px; z-index: 50; }
        .infobox {
          position: fixed; left: 12px; bottom: 12px; max-width: 540px;
          background: rgba(255,255,255,.07); color: inherit;
          border-radius: 12px; padding: .75rem .9rem; font-size: .92rem; line-height: 1.4;
          box-shadow: 0 4px 16px rgba(0,0,0,.25);
          backdrop-filter: blur(6px);
        }
        a { color: #80caff; font-weight: 600; }
        </style>
        """,
        dangerously_allow_html=True
    ),

    # floating, non-blocking loader + status text
    html.Div(className="floating-status", children=[
        dcc.Loading(id="status-loading", type="default",
                    children=html.Div(id="status-area", className="status-pill"))
    ]),

    html.Div(style={"padding":"1.2rem"}, children=[
        html.Div([
            html.H2("Argos — token flow on the hypersphere"),
            html.P("Click (>|) to generate one token and scrub the slider across LayerNorms with the Play button in the plot. "
                   "(|>|>) generates many tokens in one go. Tokens are colour-coded along a smooth spectrum by position."),
        ], style={"maxWidth":"980px"}),

        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"1rem","alignItems":"start"}, children=[
            # LEFT PANEL (controls)
            html.Div(children=[
                html.Label("Hugging Face model ID"),
                dcc.Input(id="model-id", type="text", value=DEFAULT_MODEL_ID, debounce=True, style={"width":"100%"}),

                html.Label("Prompt"),
                dcc.Textarea(id="prompt", value="Once upon a time", style={"width":"100%", "height":"8rem"}),

                html.Div(style={"display":"grid","gridTemplateColumns":"repeat(6, minmax(0,1fr))","gap":"0.5rem"}, children=[
                    html.Div([html.Label("Max new tokens"),
                              dcc.Input(id="max-new", type="number", value=32, min=1, max=256, step=1, style={"width":"100%"})], style={"gridColumn":"span 2"}),
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
                    html.Button(">|", id="run_one", className="button-primary", title="Generate 1 token"),
                    html.Button("|>|>", id="run_many", title="Generate Max new tokens"),
                    html.Button("PREV", id="prev"),
                    html.Button("NEXT", id="next"),
                ]),

                # stores
                dcc.Store(id="busy"),
                dcc.Store(id="steps-store"),
                dcc.Store(id="step-index", data=0),
            ]),

            # RIGHT PANEL (graph + text)
            html.Div(children=[
                dcc.Graph(id="sphere", style={"height":"720px"}),
                html.Div(id="texts", style={"marginTop":"0.5rem"}),
            ]),
        ]),
    ]),

    # Info box (markdown)
    html.Div(className="infobox", children=[
        dcc.Markdown(
            """
**What you’re seeing**

- Each dot is a **token** (prompt + generated so far).
- The 3D plot shows tokens embedded on a **unit sphere** after applying PCA to *all LayerNorm activations* across the network, so sliding the bar moves you through the **norms** (pre/post-attention etc.).
- Lines (optional) trace how a token **moves** across norms.
- Colors follow a smooth gradient by token **position** (earliest → latest).

**Why it looks like a parachute**

- After attention, many tokens borrow from shared components (often the **first token**, the *attention sink*), so they drift toward a common direction; MLP layers re-differentiate them.

**Links**
- More experiments & contact: [marvosyntactical.github.io](https://marvosyntactical.github.io)
            """
        )
    ])
])

# ------------------------------ Callbacks --------------------------------

# Theme: set explicit style so it overrides framework CSS
@app.callback(
    Output("root-div", "style"),
    Input("theme-toggle", "value")
)
def theme_style(value):
    dark = ("dark" in (value or []))
    if dark:
        return {"backgroundColor":"#0f1620", "color":"#e4e7ee"}
    else:
        return {"backgroundColor":"#ffffff", "color":"#111111"}

# Small non-blocking status indicator
@app.callback(
    Output("status-area", "children"),
    Input("busy", "data")
)
def status_view(msg):
    if not msg:
        return ""
    return html.Div([html.Div(className="spinner"), html.Span(msg)])

def _generate(model_id, prompt, temp, traj_flag, freeze_flag, k_tokens):
    """Shared worker: returns steps payload + status text."""
    status = "Loading model…"
    tok, model = load_model(model_id or DEFAULT_MODEL_ID)

    show_traj = ("yes" in (traj_flag or []))
    pca_freeze = ("freeze" in (freeze_flag or []))
    max_new = int(k_tokens)

    status = f"Generating ({max_new})…"
    steps, _ = generate_steps(
        model, tok, (prompt or "").strip(),
        max_new_tokens=max_new,
        temperature=float(temp or 0.9),
        pca_freeze_after_first=pca_freeze,
        show_trajectories=show_traj
    )
    # include prompt length for styling; sanitize after
    prompt_len = len(tok.encode((prompt or '').strip()))
    payload = [dict(tokens=s.tokens, colors=s.colors, X=s.X.tolist(), fig=s.figure_json, prompt_len=prompt_len)
               for s in steps]
    status = f"Ready. Steps: {len(payload)}."
    return payload, status

# (>|) one token
@app.callback(
    Output("steps-store", "data"),
    Output("step-index", "data"),
    Output("busy", "data"),
    Input("run_one", "n_clicks"),
    State("model-id", "value"),
    State("prompt", "value"),
    State("temp", "value"),
    State("traj", "value"),
    State("freeze-pca", "value"),
    prevent_initial_call=True
)
def on_run_one(n_clicks, model_id, prompt, temp, traj, freeze_pca):
    payload, status = _generate(model_id, prompt, temp, traj, freeze_pca, k_tokens=1)
    return payload, max(0, len(payload)-1), status

# (|>|>) many tokens
@app.callback(
    Output("steps-store", "data", allow_duplicate=True),
    Output("step-index", "data", allow_duplicate=True),
    Output("busy", "data", allow_duplicate=True),
    Input("run_many", "n_clicks"),
    State("model-id", "value"),
    State("prompt", "value"),
    State("max-new", "value"),
    State("temp", "value"),
    State("traj", "value"),
    State("freeze-pca", "value"),
    prevent_initial_call=True
)
def on_run_many(n_clicks, model_id, prompt, max_new, temp, traj, freeze_pca):
    payload, status = _generate(model_id, prompt, temp, traj, freeze_pca, k_tokens=int(max_new or 32))
    return payload, max(0, len(payload)-1), status

# NEXT/PREV within already generated steps
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

# Update figure + text; apply dark template; strip diacritics everywhere
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
    # sanitize all text labels used in markers/hover
    fig = sanitize_figure_text(fig)

    # theme
    dark = ("dark" in (theme_value or []))
    if dark:
        # deep slate bg (matches root-div), transparent paper so page shows through
        fig.update_layout(template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e4e7ee"))
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

    # ensure Play button exists (to animate slider)
    if not fig.layout.updatemenus:
        fig.update_layout(updatemenus=[dict(
            type="buttons", showactive=False, buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame":{"duration":400, "redraw": True}, "transition":{"duration":0}}])
            ]
        )])

    # colorized prompt+output strip, sanitized
    p_len = int(step.get("prompt_len", 0))
    safe_tokens = sanitize_tokens(step["tokens"])
    spans = []
    for i, (tok, col) in enumerate(zip(safe_tokens, step["colors"])):
        style = {"color": col, "paddingRight": "2px"}
        if i < p_len:
            style.update({"fontWeight": "600", "opacity": 0.85})
        spans.append(html.Span(tok, style=style))
    token_row = html.Div(spans, style={"fontFamily":"monospace","fontSize":"1.05rem","lineHeight":"1.8"})
    caption = html.Div(f"Step {idx+1}/{len(steps)} — tokens so far: {len(step['tokens'])}")

    return fig, html.Div([caption, token_row])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run_server(host="0.0.0.0", port=port, debug=False)

