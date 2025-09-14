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
    """
    Remove any base letter that carries diacritics in NFD/NFKD.
    Also drop tokenization markers 'Ġ' and '▁' entirely.
    """
    if not isinstance(s, str):
        return s
    # strip tokenization markers up front
    s = s.replace("Ġ", " ").replace("▁", " ")
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        # decompose at this position
        decomp = unicodedata.normalize("NFD", ch)
        # if this single char expands to base+marks, drop all
        if any(unicodedata.category(c) == "Mn" for c in decomp[1:]):
            # skip this char (and any subsequent combining marks)
            i += 1
            while i < len(s) and unicodedata.category(unicodedata.normalize("NFD", s[i])) == "Mn":
                i += 1
            continue
        # also handle sequences like 'a' + combining mark spread across string
        if i + 1 < len(s):
            nxt = unicodedata.normalize("NFD", s[i+1])
            if any(unicodedata.category(c) == "Mn" for c in nxt):
                # drop base char and its following combining mark char
                i += 2
                continue
        out.append(ch)
        i += 1
    # finally, force ASCII-safe (drop any stragglers)
    return "".join(out).encode("ascii", "ignore").decode("ascii")


def sanitize_tokens(tokens):
    return [remove_letters_with_diacritics(t) for t in tokens]


def sanitize_figure_text(fig: go.Figure):
    """
    In-place: strip problematic diacritics/markers from text labels in traces + frames.
    """
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
    # lightweight geometric sci-fi font
    "https://fonts.googleapis.com/css2?family=Oxanium:wght@300;600;800&display=swap",
]

app = Dash(__name__, external_stylesheets=external_stylesheets, title="ARGOS")
server = app.server

DARK_BG = "#0f1620"     # deep slate gray
DARK_FG = "#e4e7ee"
CARD_BG = "rgba(255,255,255,.07)"

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
        title="Latent flow across layer norms",
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
            /* Inputs dark */
            input[type="text"], textarea, input[type="number"] {{
              background:#111a24 !important; color:{DARK_FG} !important;
              border: 1px solid #2a3340 !important; box-shadow:none !important;
            }}
            input::placeholder, textarea::placeholder {{ color: #7f8693; }}
            /* Buttons */
            .btn {{ background:#6b4df6; color:white; border:none; border-radius:8px; padding:.45rem .8rem; }}
            .btn.secondary {{ background:#2a3340; }}
            /* Headline */
            h1, h2, .brand {{ font-family: 'Oxanium', system-ui, sans-serif; letter-spacing: .08em; }}
            .brand {{ font-weight: 800; font-size: 1.6rem; }}
            /* Info box near buttons (not fixed) */
            .infobox {{
              background:{CARD_BG}; color: inherit;
              border-radius: 12px; padding: .65rem .85rem; font-size: .92rem; line-height: 1.4;
              box-shadow: 0 4px 16px rgba(0,0,0,.25);
              backdrop-filter: blur(6px);
              margin-top: .8rem;
              max-width: 720px;
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
                html.Div("🌌 ARGOS", className="brand"),
                html.P("Generate a token, then scrub through LayerNorms with ⏯ (or the Play button in the plot)."),
            ], style={"maxWidth": "980px"}),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "1rem",
                    "alignItems": "start",
                },
                children=[
                    # LEFT
                    html.Div(children=[
                        html.Label("Hugging Face model ID"),
                        dcc.Input(id="model-id", type="text", value=DEFAULT_MODEL_ID, debounce=True, style={"width": "100%"}),

                        html.Label("Prompt"),
                        dcc.Textarea(id="prompt", value="Once upon a time", style={"width": "100%", "height": "8rem"}),

                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(0,1fr))", "gap": "0.5rem"},
                            children=[
                                html.Div([html.Label("Max new tokens"),
                                          dcc.Input(id="max-new", type="number", value=32, min=1, max=256, step=1, style={"width": "100%"})],
                                         style={"gridColumn": "span 2"}),
                                html.Div([html.Label("Temperature"),
                                          dcc.Input(id="temp", type="number", value=0.9, min=0.05, max=2.0, step=0.05, style={"width": "100%"})]),
                                html.Div([html.Label("Trajectories"),
                                          dcc.Checklist(id="traj", options=[{"label": "show", "value": "yes"}], value=["yes"])]),
                                html.Div([html.Label("Freeze PCA"),
                                          dcc.Checklist(id="freeze-pca", options=[{"label": "yes", "value": "freeze"}], value=[])]),
                                html.Div([html.Label("Dark theme"),
                                          dcc.Checklist(id="theme-toggle", options=[{"label": "on", "value": "dark"}], value=["dark"])]),
                            ],
                        ),

                        html.Div(style={"marginTop": "0.5rem", "display": "flex", "gap": "0.5rem", "flexWrap": "wrap"}, children=[
                            html.Button("▶", id="run_one", className="btn", title="Generate 1 token"),
                            html.Button("⏩", id="run_many", className="btn", title="Generate Max new tokens"),
                            html.Button("◀", id="prev", className="btn secondary", title="Prev step"),
                            html.Button("▶", id="next", className="btn secondary", title="Next step"),
                            html.Button("⏯", id="play_anim", className="btn secondary", title="Animate LN slider"),
                        ]),

                        # infobox just under buttons
                        html.Div(className="infobox", children=[
                            dcc.Markdown(
                                """
**Controls**
- **▶** generate **one** token and update the viz  
- **⏩** generate **Max new tokens**  
- **◀ / ▶** step through already generated token-steps  
- **⏯** animate the layer slider (same as the plot’s Play)

**What you’re seeing**
- Each dot = a **token** (prompt + generated so far).
- We capture activations after every **LayerNorm**, stack them, run **PCA**, and project to a **unit sphere**. The slider sweeps norms (pre/post-attention).
- Optional thin lines show each token’s **trajectory** across norms.
- Colors follow a smooth gradient by **token position** (earliest → latest).

**Parachute effect (attention sink)**
- Post-attention, tokens absorb a shared component (often the **first token**). They drift toward a common direction; MLP layers re-differentiate.

**Links**
- Experiments & contact: [marvosyntactical.github.io](https://marvosyntactical.github.io)
                                """
                            )
                        ]),

                        # stores
                        dcc.Store(id="busy"),
                        dcc.Store(id="steps-store"),
                        dcc.Store(id="step-index", data=0),
                        dcc.Store(id="animate-trigger"),  # gets poked by ⏯
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

# Apply dark/light page colors (inputs already styled; this is the container bg/fg)
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

def _generate(model_id, prompt, temp, traj_flag, freeze_flag, k_tokens):
    """Shared worker: returns steps payload + status text."""
    # load (cached)
    tok, model = load_model(model_id or DEFAULT_MODEL_ID)

    show_traj = ("yes" in (traj_flag or []))
    pca_freeze = ("freeze" in (freeze_flag or []))
    max_new = int(k_tokens)

    steps, _ = generate_steps(
        model, tok, (prompt or "").strip(),
        max_new_tokens=max_new,
        temperature=float(temp or 0.9),
        pca_freeze_after_first=pca_freeze,
        show_trajectories=show_traj
    )
    # include prompt length; sanitize on render
    prompt_len = len(tok.encode((prompt or '').strip()))
    payload = [dict(tokens=s.tokens, colors=s.colors, X=s.X.tolist(), fig=s.figure_json, prompt_len=prompt_len)
               for s in steps]
    status = f"Ready. Steps: {len(payload)}."
    return payload, status

# ▶ one token
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
    busy = "Generating (1)…"
    payload, status = _generate(model_id, prompt, temp, traj, freeze_pca, k_tokens=1)
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
    State("traj", "value"),
    State("freeze-pca", "value"),
    prevent_initial_call=True
)
def on_run_many(n_clicks, model_id, prompt, max_new, temp, traj, freeze_pca):
    busy = f"Generating ({int(max_new or 32)})…"
    payload, status = _generate(model_id, prompt, temp, traj, freeze_pca, k_tokens=int(max_new or 32))
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

# Render figure + text (dark from the start) and sanitize all labels
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

    # theme
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

    # ensure Play is available
    if not fig.layout.updatemenus:
        fig.update_layout(updatemenus=[dict(
            type="buttons", showactive=False, buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 400, "redraw": True}, "transition": {"duration": 0}}])
            ]
        )])

    # token strip (sanitized)
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

# ⏯ trigger: client-side animate current figure (same as Plotly Play)
app.clientside_callback(
    """
    function(n, fig) {
      if (!n) { return window.dash_clientside.no_update; }
      try {
        var el = document.getElementById('sphere');
        if (el && el.children && el.children[0]) {
          var gd = el.children[0];
          Plotly.animate(gd, null, {frame:{duration:400, redraw:true}, transition:{duration:0}});
        }
      } catch(e) {}
      return 0;
    }
    """,
    Output("animate-trigger", "data"),
    Input("play_anim", "n_clicks"),
    State("sphere", "figure"),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run_server(host="0.0.0.0", port=port, debug=False)

