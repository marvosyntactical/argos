import os
import unicodedata
from functools import lru_cache

import dash
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from argos_viz import generate_steps  # same API

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "distilgpt2")
DEVICE = "cpu" if os.environ.get("DEVICE", "cpu") == "cpu" else "cuda"

# ------------------------- Text sanitization -------------------------

def remove_letters_with_diacritics(s: str) -> str:
    """Remove any base letter that carries diacritics; drop tokenizer markers."""
    if not isinstance(s, str):
        return s
    s = s.replace("Ġ", " ").replace("▁", " ")
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        decomp = unicodedata.normalize("NFD", ch)
        if any(unicodedata.category(c) == "Mn" for c in decomp[1:]):
            i += 1
            while i < len(s) and unicodedata.category(unicodedata.normalize("NFD", s[i])) == "Mn":
                i += 1
            continue
        if i + 1 < len(s):
            nxt = unicodedata.normalize("NFD", s[i+1])
            if any(unicodedata.category(c) == "Mn" for c in nxt):
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out).encode("ascii", "ignore").decode("ascii")


def sanitize_tokens(tokens):
    return [remove_letters_with_diacritics(t) for t in tokens]


def sanitize_figure_text(fig: go.Figure):
    def fix_trace_text(tr):
        if hasattr(tr, "text") and tr.text is not None:
            if isinstance(tr.text, (list, tuple)):
                tr.text = [remove_letters_with_diacritics(x) for x in tr.text]
            else:
                tr.text = remove_letters_with_diacritics(tr.text)
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
    "https://fonts.googleapis.com/css2?family=Oxanium:wght@300;600;800&display=swap",
]

app = Dash(__name__, external_stylesheets=external_stylesheets, title="ARGΩΣ")
server = app.server

DARK_BG = "#0f1620"
DARK_FG = "#e4e7ee"
CARD_BG = "rgba(255,255,255,.07)"

DEFAULT_PROMPT = """Now this is the story all about how,
My strife got flipped-turned upside down,
And I'd like to tickle my newt, just sit right there,
I'll tell you how I became the principles' own cut Bell heir.

Invest philanthropic, bought and trained
OpenAI playground is where I spent most of my days.
Rollin' out, maxin', minmaxin' all cool,
And all provin' sums be balls outside of the pool.

When a tuple of keys who were up to no good,
Started fakin' thruples which mine nay bore would.
i got in 1 lil byte and my malloc err'd,
And sed -u "removin' with yawn tee and NCL in there."

I whistled, for ACAB, and then in came Neo,
The license pled, said "refresh" and it was nice in the mirror.
If anything I could say that this .cab was .rar,
But I thought "Nah forget IT, Yo
"""

def empty_dark_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        title="Latent thought flow",
    )
    return fig

app.layout = html.Div(
    id="root-div",
    style={"backgroundColor": DARK_BG, "color": DARK_FG, "minHeight": "100vh"},
    children=[
        dcc.Markdown(
            f"""
            <style>
            html, body, #root-div {{ height: 100%; background: {DARK_BG}; color: {DARK_FG}; }}
            .status-pill {{
              display:flex; gap:.35rem; align-items:center;
              background: {CARD_BG}; border-radius: 999px; padding:.25rem .6rem;
              backdrop-filter: blur(3px);
            }}
            .spinner {{
              display:inline-block; width:14px; height:14px; border-radius:50%;
              border: 2px solid rgba(255,255,255,.22); border-top-color: rgba(255,255,255,.9);
              animation: spin 0.9s linear infinite;
            }}
            @keyframes spin {{ from {{ transform: rotate(0deg);}} to {{ transform: rotate(360deg);}} }}
            /* Inputs — force dark text & bg across browsers + Milligram */
            #root-div input[type="text"],
            #root-div textarea,
            #root-div input[type="number"],
            #root-div .input,
            #root-div .textarea {{
              background: #111a24 !important;
              color: #e4e7ee !important;               /* Firefox/normal text color */
              -webkit-text-fill-color: #e4e7ee !important; /* WebKit/autofill text color */
              caret-color: #e4e7ee !important;
              border: 1px solid #2a3340 !important;
              box-shadow: none !important;
            }}

            #root-div input::placeholder,
            #root-div textarea::placeholder {{ color: #9aa2b1 !important; }}

            /* Defang WebKit autofill’s yellow */
            #root-div input:-webkit-autofill {{
              -webkit-text-fill-color: #e4e7ee !important;
              transition: background-color 5000s ease-in-out 0s;
            }}

            /* Firefox odd states */
            #root-div :-moz-ui-invalid {{ box-shadow: none !important; }}
            /* Cross-engine, apparently */
            #root-div input[type="text"],
            #root-div textarea,
            #root-div input[type="number"],
            #root-div .input,
            #root-div .textarea {{
              background: #111a24 !important;
              color: #e4e7ee !important;
              -webkit-text-fill-color: #e4e7ee !important; /* Chrome/Safari */
              caret-color: #e4e7ee !important;
              border: 1px solid #2a3340 !important;
              box-shadow: none !important;
            }}
            /* Buttons */
            .btn {{ background:#6b4df6; color:white; border:none; border-radius:8px; padding:.45rem .8rem; }}
            .btn.secondary {{ background:#2a3340; }}
            /* Brand */
            .brand {{
              font-family: 'Oxanium', system-ui, sans-serif;
              letter-spacing: .14em;
              font-weight: 800;
              font-size: 6.2rem;    /* was 2.4rem */
              line-height: 1.05;
              margin: .25rem 0 1rem 0;
            }}
            /* Info box */
            .infobox {{
              background:{CARD_BG}; color: inherit;
              border-radius: 12px; padding: .65rem .85rem; font-size: .92rem; line-height: 1.4;
              box-shadow: 0 4px 16px rgba(0,0,0,.25);
              backdrop-filter: blur(6px);
              margin-top: .8rem; max-width: 720px;
            }}
            a {{ color: #80caff; font-weight: 600; }}
            </style>
            """,
            dangerously_allow_html=True,
        ),

        # status (top-right)
        html.Div(
            style={"position": "fixed", "top": "12px", "right": "12px", "zIndex": 50},
            children=[dcc.Loading(id="status-loading", type="default",
                                  children=html.Div(id="status-area", className="status-pill"))],
        ),

        html.Div(style={"padding": "1.2rem"}, children=[
            html.Div([
                html.Div("🌌 ARGΩΣ", className="brand",
                  style={"fontSize": "6.2rem", "fontWeight": 800, "letterSpacing": ".14em"}
                )
            ], style={"maxWidth": "980px"}),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1rem", "alignItems": "start"},
                children=[
                    # LEFT
                    html.Div(children=[
                        html.Label("Hugging Face model ID"),
                        dcc.Input(id="model-id", type="text", value=DEFAULT_MODEL_ID, debounce=True, style={"width": "100%"}),

                        html.Label("Prompt"),
                        dcc.Textarea(id="prompt", value=DEFAULT_PROMPT, style={"width": "100%", "height": "8rem"}),

                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(0,1fr))", "gap": "0.5rem"},
                            children=[
                                html.Div([html.Label("Max new tokens"),
                                          dcc.Input(id="max-new", type="number", value=32, min=1, max=256, step=1, style={"width": "100%"})],
                                         style={"gridColumn": "span 2"}),
                                html.Div([html.Label("Temperature"),
                                          dcc.Input(id="temp", type="number", value=0.9, min=0.05, max=2.0, step=0.05, style={"width": "100%"})]),
                                html.Div([html.Label("Dark theme"),
                                          dcc.Checklist(id="theme-toggle", options=[{"label": "on", "value": "dark"}], value=["dark"])]),
                            ],
                        ),

                        html.Div(style={"marginTop": "0.5rem", "display": "flex", "gap": "0.5rem", "flexWrap": "wrap"}, children=[
                            html.Button("⏯", id="run_one", className="btn", title="Generate 1 token"),
                            html.Button("⏩", id="run_many", className="btn", title="Generate Max new tokens"),
                            html.Button("◀", id="prev", className="btn secondary", title="Prev step"),
                            html.Button("▶", id="next", className="btn secondary", title="Next step"),
                        ]),

                        # infobox just under buttons
                        html.Div(className="infobox", children=[
                            dcc.Markdown(
                                """
# Controls
- **⏯** generate **one** token and update the viz
- **⏩** generate **Max new tokens**
- **◀ / ▶** step through already generated token-steps
- **Play** in the plotly figure animates the forward pass.

To get started, press ⏯ .

Give it a minute or two to update the figure. [Spoiler](/assets/argos.gif) of what it'll look like.

Once the figure is displayed, use the slider to go through the depth of the model or press **Play** to see the entire forward pass.

I literally couldn't figure out how to make the input text visible with a dark background, so you'll probably have to untoggle the dark theme to see what's in the input fields.

# What you're seeing

Each dot is the hidden vector currently associated with the input **token** indicated by its label (prompt + generated so far).

We capture activations after every **LayerNorm** (which usually follow each **Attention** and each **MLP**), stack them, run **PCA** to 3D, and project to the **unit sphere**.

So every second layernorm is subsequent to the same module type (either Attention or MLP).

Post-attention, the first token often takes on the role of an **attention sink**, which you should be able to see beautifully here, as a paratrooperesque constellation.

If you like this (otherwise too haha), you can check out more of my stuff at [marvosyntactical.github.io](https://marvosyntactical.github.io).
                                """
                            )
                        ]),

                        # stores
                        dcc.Store(id="busy"),
                        dcc.Store(id="steps-store"),
                        dcc.Store(id="step-index", data=0),
                    ]),

                    # RIGHT
                    html.Div(children=[
                        dcc.Graph(id="sphere", figure=empty_dark_fig(), style={"height": "720px"}),
                        html.Div(id="texts", style={"marginTop": "0.5rem"}),
                    ]),
                ],
            ),
        ]),
    ],
)

# ------------------------------ Callbacks --------------------------------

# Page theme (inputs already styled via CSS)
@app.callback(
    Output("root-div", "style"),
    Input("theme-toggle", "value"),
    prevent_initial_call=False
)
def theme_style(value):
    dark = ("dark" in (value or []))
    if dark:
        return {"backgroundColor": DARK_BG, "color": DARK_FG, "minHeight": "100vh"}
    else:
        return {"backgroundColor": "#ffffff", "color": "#111111", "minHeight": "100vh"}

# Small non-blocking status indicator
@app.callback(
    Output("status-area", "children"),
    Input("busy", "data")
)
def status_view(msg):
    if not msg:
        return ""
    return html.Div([html.Div(className="spinner"), html.Span(msg)])

def _generate(model_id, prompt, temp, k_tokens):
    tok, model = load_model(model_id or DEFAULT_MODEL_ID)
    steps, _ = generate_steps(
        model, tok, (prompt or "").strip(),
        max_new_tokens=int(k_tokens),
        temperature=float(temp or 0.9),
        pca_freeze_after_first=False,
        show_trajectories=True,     # trajectories on by default
    )
    prompt_len = len(tok.encode((prompt or '').strip()))
    payload = [dict(tokens=s.tokens, colors=s.colors, X=s.X.tolist(), fig=s.figure_json, prompt_len=prompt_len)
               for s in steps]
    status = f"Ready. Steps: {len(payload)}."
    return payload, status

# ⏯ one token
@app.callback(
    Output("steps-store", "data"),
    Output("step-index", "data"),
    Output("busy", "data"),
    Input("run_one", "n_clicks"),
    State("model-id", "value"),
    State("prompt", "value"),
    State("temp", "value"),
    prevent_initial_call=True
)
def on_run_one(n_clicks, model_id, prompt, temp):
    busy = "Generating (1)…"
    payload, status = _generate(model_id, prompt, temp, k_tokens=1)
    return payload, max(0, len(payload)-1), status or busy

# ⏩ many tokens
@app.callback(
    Output("steps-store", "data", allow_duplicate=True),
    Output("step-index", "data", allow_duplicate=True),
    Output("busy", "data", allow_duplicate=True),
    Input("run_many", "n_clicks"),
    State("model-id", "value"),
    State("prompt", "value"),
    State("max-new", "value"),
    State("temp", "value"),
    prevent_initial_call=True
)
def on_run_many(n_clicks, model_id, prompt, max_new, temp):
    busy = f"Generating ({int(max_new or 32)})…"
    payload, status = _generate(model_id, prompt, temp, k_tokens=int(max_new or 32))
    return payload, max(0, len(payload)-1), status or busy

# ◀ / ▶ navigate generated steps
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

# Render figure + text (dark from the start) and sanitize labels
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
        return empty_dark_fig(), ""
    idx = int(idx or 0)
    step = steps[idx]

    fig = go.Figure(step["fig"])
    fig = sanitize_figure_text(fig)

    dark = ("dark" in (theme_value or []))
    if dark:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=DARK_FG),
        )
        if "scene" in fig.layout:
            fig.update_layout(scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ))
    else:
        fig.update_layout(template="plotly", paper_bgcolor="rgba(255,255,255,1)")

    # ensure Play is available (to animate slider)
    if not fig.layout.updatemenus:
        fig.update_layout(updatemenus=[dict(
            type="buttons", showactive=False, buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 400, "redraw": True}, "transition": {"duration": 0}}])
            ]
        )])

    p_len = int(step.get("prompt_len", 0))
    safe_tokens = sanitize_tokens(step["tokens"])
    spans = []
    for i, (tok, col) in enumerate(zip(safe_tokens, step["colors"])):
        style = {"color": col, "paddingRight": "2px"}
        if i < p_len:
            style.update({"fontWeight": "600", "opacity": 0.85})
        spans.append(html.Span(tok, style=style))
    token_row = html.Div(spans, style={"fontFamily": "monospace", "fontSize": "1.05rem", "lineHeight": "1.8"})
    caption = html.Div(f"Step {idx+1}/{len(steps)} — tokens so far: {len(step['tokens'])}")

    return fig, html.Div([caption, token_row])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run_server(host="0.0.0.0", port=port, debug=False)

