import os
import unicodedata
from functools import lru_cache

import dash
from dash import Dash, dcc, html, ctx, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from argos_viz import generate_steps  # server-side capture

# -------------------- Constants --------------------
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "distilgpt2")
DEVICE = "cpu" if os.environ.get("DEVICE", "cpu") == "cpu" else "cuda"

DARK_BG = "#0f1620"    # deep slate
DARK_FG = "#e4e7ee"
CARD_BG = "rgba(255,255,255,.07)"

# Amber controls panel
AMBER_BG = "#F0B45A"
AMBER_TEXT = "#111"
AMBER_INPUT_BG = "#FFF4DE"
AMBER_INPUT_BORDER = "#B88A37"

# -------------------- Sanitization --------------------
def remove_letters_with_diacritics(s: str) -> str:
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
    def fix(tr):
        if hasattr(tr, "text") and tr.text is not None:
            if isinstance(tr.text, (list, tuple)):
                tr.text = [remove_letters_with_diacritics(x) for x in tr.text]
            else:
                tr.text = remove_letters_with_diacritics(tr.text)
    for tr in fig.data:
        fix(tr)
    if fig.frames:
        for fr in fig.frames:
            for tr in fr.data:
                fix(tr)
    return fig

# -------------------- Model cache --------------------
@lru_cache(maxsize=2)
def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(DEVICE)
    model.eval()
    return tok, model

# -------------------- Defaults --------------------
DEFAULT_PROMPT = """Now this is the story all about how,
My life got flipped-turned upside down,
And I'd like to take a minute, just sit right there,
I'll tell you how I became the prince of a town called Bel Air.

In West Philadelphia, born and raised
On the playground is where I spent most of my days.
Chillin' out, maxin', relaxin all cool,
And all shootin' some B-ball outside of the school.

When a couple of guys who were up to no good,
Started makin' trouble in my neighborhood.
I got in one little fight and my mom got scared,
And said "You're movin' with your auntie and uncle in Bel Air."

I whistled for a cab, and when it came near,
The license plate said "fresh" and it had dice in the mirror.
If anything I could say that this cab was rare,
But I thought "Nah forget it, Yo home to Bel"
"""

def empty_dark_fig(title="Latent flow across layer norms") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        title=title,
    )
    return fig

# -------------------- App --------------------
external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/milligram/1.4.1/milligram.min.css",
    "https://fonts.googleapis.com/css2?family=Oxanium:wght@300;600;800&display=swap",
]

app = Dash(__name__, external_stylesheets=external_stylesheets, title="ARGΩΣ")
server = app.server

app.layout = html.Div(
    id="root-div",
    style={"backgroundColor": DARK_BG, "color": DARK_FG, "minHeight": "100vh"},
    children=[
        dcc.Markdown(
            f"""
            <style>
            html, body, #root-div {{ height: 100%; background: {DARK_BG}; color: {DARK_FG}; }}
            #root-div .js-plotly-plot, #root-div .plot-container {{ background: transparent !important; }}

            /* BIG brand */
            .brand {{
              font-family: 'Oxanium', system-ui, sans-serif;
              letter-spacing: .14em; font-weight: 800; font-size: 6.2rem;
              line-height: 1.05; margin: .25rem 0 1rem 0;
            }}

            /* Status pill */
            .status-pill {{ display:flex; gap:.35rem; align-items:center;
              background: {CARD_BG}; border-radius: 999px; padding:.25rem .6rem; backdrop-filter: blur(3px); }}
            .spinner {{ display:inline-block; width:14px; height:14px; border-radius:50%;
              border: 2px solid rgba(255,255,255,.22); border-top-color: rgba(255,255,255,.9);
              animation: spin 0.9s linear infinite; }}
            @keyframes spin {{ from {{ transform: rotate(0deg);}} to {{ transform: rotate(360deg);}} }}

            /* Amber controls panel */
            #controls {{
              background: {AMBER_BG}; color: {AMBER_TEXT};
              border: 1px solid {AMBER_INPUT_BORDER};
              border-radius: 12px; padding: .8rem;
            }}
            #controls input[type="text"], #controls textarea, #controls input[type="number"] {{
              background:{AMBER_INPUT_BG} !important; color:{AMBER_TEXT} !important;
              -webkit-text-fill-color:{AMBER_TEXT} !important; caret-color:{AMBER_TEXT} !important;
              border:1px solid {AMBER_INPUT_BORDER} !important; border-radius:6px; padding:.5rem .6rem;
              box-shadow:none !important;
            }}
            #controls .btn {{ background:#6b4df6; color:white; border:none; border-radius:8px; padding:.45rem .8rem; }}
            #controls .btn.secondary {{ background:#344054; color:#e8eef7; }}

            /* Switch look for dcc.Checklist toggles inside #controls */
            #controls .dash-checklist label {{ display:inline-block; }}
            #controls .dash-checklist input[type="checkbox"] {{
              appearance: none; -webkit-appearance:none; -moz-appearance:none;
              width: 46px; height: 24px; border-radius: 24px; border: 0;
              background: #8f6e2f; position: relative; cursor: pointer; outline: none; vertical-align: middle; margin: 0;
            }}
            #controls .dash-checklist input[type="checkbox"]::after {{
              content:""; position:absolute; left:3px; top:3px; width:18px; height:18px; border-radius:50%;
              background:#fff; transition: transform .2s ease;
            }}
            #controls .dash-checklist input[type="checkbox"]:checked {{ background:#6b4df6; }}
            #controls .dash-checklist input[type="checkbox"]:checked::after {{ transform: translateX(22px); }}

            /* Info box */
            .infobox {{
              background:{CARD_BG}; color: inherit; border-radius: 12px; padding: .65rem .85rem;
              font-size:.92rem; line-height:1.4; box-shadow: 0 4px 16px rgba(0,0,0,.25); backdrop-filter: blur(6px);
              margin-top:.8rem; max-width:720px;
            }}
            a {{ color:#80caff; font-weight:600; }}
            </style>
            """,
            dangerously_allow_html=True,
        ),

        # status
        html.Div(
            style={"position": "fixed", "top": "12px", "right": "12px", "zIndex": 50},
            children=[dcc.Loading(id="status-loading", type="default",
                                  children=html.Div(id="status-area", className="status-pill"))],
        ),

        html.Div(style={"padding": "1.2rem"}, children=[
            html.Div([html.Div("🌌 ARGΩΣ", className="brand",
                               style={"fontSize":"6.2rem","fontWeight":800,"letterSpacing":".14em"})],
                     style={"maxWidth": "980px"}),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1rem", "alignItems": "start"},
                children=[
                    # LEFT — amber controls panel
                    html.Div(id="controls", children=[
                        html.Label("Hugging Face model ID"),
                        dcc.Input(id="model-id", type="text", value=DEFAULT_MODEL_ID, style={"width":"100%"}),

                        html.Label("Prompt"),
                        dcc.Textarea(id="prompt", value=DEFAULT_PROMPT, style={"width":"100%","height":"8rem"}),

                        html.Div(
                            style={"display":"grid","gridTemplateColumns":"repeat(6, minmax(0,1fr))","gap":"0.5rem"},
                            children=[
                                html.Div([html.Label("Max new tokens"),
                                          dcc.Input(id="max-new", type="number", value=32, min=1, max=256, step=1, style={"width":"100%"})],
                                         style={"gridColumn":"span 2"}),
                                html.Div([html.Label("Temperature"),
                                          dcc.Input(id="temp", type="number", value=0.9, min=0.05, max=2.0, step=0.05, style={"width":"100%"})]),
                                html.Div([html.Label("Dark theme"),
                                          dcc.Checklist(id="theme-toggle",
                                                        options=[{"label":"","value":"on"}], value=["on"])]),
                                html.Div([html.Label("Show axes (PC1–PC3)"),
                                          dcc.Checklist(id="axes-toggle",
                                                        options=[{"label":"","value":"on"}], value=[])]),
                                html.Div([html.Label("Track token₀ trajectory"),
                                          dcc.Checklist(id="track-first-toggle",
                                                        options=[{"label":"","value":"on"}], value=[])]),
                                html.Div([html.Label("Project to unit sphere"),
                                          dcc.Checklist(id="sphere-proj-toggle",
                                                        options=[{"label":"","value":"on"}], value=["on"])]),
                            ],
                        ),

                        html.Div(style={"marginTop":"0.5rem","display":"flex","gap":"0.5rem","flexWrap":"wrap"}, children=[
                            html.Button("⏯", id="run_one", className="btn", title="Generate 1 token"),
                            html.Button("⏩", id="run_many", className="btn", title="Generate Max new tokens"),
                            html.Button("◀", id="prev", className="btn secondary", title="Prev step"),
                            html.Button("▶", id="next", className="btn secondary", title="Next step"),
                        ]),

                        html.Div(className="infobox", children=[
                            dcc.Markdown(
                                """
**Controls**
- **⏯** generate **one** token • **⏩** generate **Max new tokens** • **◀/▶** step through generated steps  
- **Show axes** overlays PC1–PC3; **Track token₀** draws a red path of the first token across layer norms  
- **Project to unit sphere** toggles radial normalization for a sanity check  
- Bored while it loads? **[Watch the animation (opens GIF)](/assets/argos.gif){:target="_blank"}**

**What you’re seeing**
- Tokens’ post-norm hidden states, PCA→3D, optionally normalized to a unit sphere.  
- Slider scrubs LayerNorms; you’ll see the **attention sink** canopy after attention blocks.  
- Lines (optional) trace per-token paths across norms.

More: [marvosyntactical.github.io](https://marvosyntactical.github.io)
                                """,
                                dangerously_allow_html=True
                            )
                        ]),

                        # stores
                        dcc.Store(id="busy"),
                        dcc.Store(id="steps-store"),
                        dcc.Store(id="step-index", data=0),
                    ]),

                    # RIGHT — always show graph (no GIF placeholder)
                    html.Div(id="viz-panel", children=[
                        dcc.Graph(id="sphere", figure=empty_dark_fig(), style={"height":"720px"}),
                        html.Div(id="texts", style={"marginTop":"0.5rem"}),
                    ]),
                ],
            ),
        ]),
    ],
)

# -------------------- Callbacks --------------------

# Theme container
@app.callback(Output("root-div","style"), Input("theme-toggle","value"), prevent_initial_call=False)
def on_theme(value):
    if "on" in (value or []):
        return {"backgroundColor": DARK_BG, "color": DARK_FG, "minHeight": "100vh"}
    return {"backgroundColor": "#ffffff", "color": "#111111", "minHeight": "100vh"}

# Small status pill view
@app.callback(Output("status-area","children"), Input("busy","data"))
def status_view(msg):
    if not msg: return ""
    return html.Div([html.Div(className="spinner"), html.Span(msg)])

def _generate(model_id, prompt, temp, k_tokens):
    tok, model = load_model(model_id or DEFAULT_MODEL_ID)
    steps, _ = generate_steps(
        model, tok, (prompt or "").strip(),
        max_new_tokens=int(k_tokens),
        temperature=float(temp or 0.9),
        pca_freeze_after_first=False,
        show_trajectories=True,
    )
    prompt_len = len(tok.encode((prompt or '').strip()))
    payload = [dict(tokens=s.tokens, colors=s.colors, X=s.X.tolist(), fig=s.figure_json, prompt_len=prompt_len)
               for s in steps]
    return payload

# Generate 1
@app.callback(
    Output("steps-store","data"),
    Output("step-index","data"),
    Output("busy","data"),
    Input("run_one","n_clicks"),
    State("model-id","value"),
    State("prompt","value"),
    State("temp","value"),
    prevent_initial_call=True
)
def on_run_one(_, model_id, prompt, temp):
    payload = _generate(model_id, prompt, temp, k_tokens=1)
    return payload, max(0, len(payload)-1), "Ready."

# Generate many
@app.callback(
    Output("steps-store","data", allow_duplicate=True),
    Output("step-index","data", allow_duplicate=True),
    Output("busy","data", allow_duplicate=True),
    Input("run_many","n_clicks"),
    State("model-id","value"),
    State("prompt","value"),
    State("max-new","value"),
    State("temp","value"),
    prevent_initial_call=True
)
def on_run_many(_, model_id, prompt, max_new, temp):
    payload = _generate(model_id, prompt, temp, k_tokens=int(max_new or 32))
    return payload, max(0, len(payload)-1), "Ready."

# Prev/Next
@app.callback(
    Output("step-index","data", allow_duplicate=True),
    Input("prev","n_clicks"),
    Input("next","n_clicks"),
    State("steps-store","data"),
    State("step-index","data"),
    prevent_initial_call=True
)
def on_prev_next(p, n, steps, idx):
    if not steps: return no_update
    idx = int(idx or 0)
    trig = ctx.triggered_id
    if trig == "prev": idx = max(0, idx-1)
    elif trig == "next": idx = min(len(steps)-1, idx+1)
    return idx

# Render with toggles (axes/track/sphere) and dark backgrounds
@app.callback(
    Output("sphere","figure"),
    Output("texts","children"),
    Input("steps-store","data"),
    Input("step-index","data"),
    Input("theme-toggle","value"),
    Input("axes-toggle","value"),
    Input("track-first-toggle","value"),
    Input("sphere-proj-toggle","value"),
    prevent_initial_call=True
)
def on_step_change(steps, idx, theme_val, axes_val, track_val, sphere_val):
    if not steps:
        return empty_dark_fig(), ""
    idx = int(idx or 0)
    step = steps[idx]

    fig = go.Figure(step["fig"])
    fig = sanitize_figure_text(fig)

    dark_on = ("on" in (theme_val or []))
    show_axes = ("on" in (axes_val or []))
    track_first = ("on" in (track_val or []))
    project_sphere = ("on" in (sphere_val or []))

    if "X" in step and step["X"] is not None:
        X = np.array(step["X"])  # (T,L,3)
        if project_sphere:
            R = np.linalg.norm(X, axis=2, keepdims=True) + 1e-8
            X = X / R
            frames = []
            T, L, _ = X.shape
            for ell in range(L):
                frames.append(go.Frame(data=[go.Scatter3d(
                    x=X[:, ell, 0], y=X[:, ell, 1], z=X[:, ell, 2]
                )], name=f"LN{ell}"))
            fig.frames = tuple(frames) if frames else None
            if L > 0 and len(fig.data) > 0:
                fig.data[0].x = X[:, -1, 0]; fig.data[0].y = X[:, -1, 1]; fig.data[0].z = X[:, -1, 2]

        if show_axes:
            c0 = step["colors"][0] if step.get("colors") else "#9bdcff"
            axes_len = 1.05
            for vx,vy,vz in [(axes_len,0,0),(0,axes_len,0),(0,0,axes_len)]:
                fig.add_trace(go.Scatter3d(x=[-vx, vx], y=[-vy, vy], z=[-vz, vz],
                                           mode="lines", line=dict(width=3, color=c0),
                                           hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Cone(x=[axes_len,0,0], y=[0,axes_len,0], z=[0,0,axes_len],
                                  u=[0.12,0,0], v=[0,0.12,0], w=[0,0,0.12],
                                  colorscale=[[0, c0],[1, c0]], showscale=False, sizemode="absolute",
                                  sizeref=0.2, anchor="tail", hoverinfo="skip"))

        if track_first:
            x0 = X[0, :, 0]; y0 = X[0, :, 1]; z0 = X[0, :, 2]
            fig.add_trace(go.Scatter3d(x=x0, y=y0, z=z0, mode="lines+markers",
                                       line=dict(width=2, color="#ff5252"),
                                       marker=dict(size=2, color="#ff5252"),
                                       name="token₀", showlegend=False))

    if dark_on:
        fig.update_layout(template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=DARK_FG))
        if "scene" in fig.layout:
            fig.update_layout(scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ))
    else:
        fig.update_layout(template="plotly", paper_bgcolor="rgba(255,255,255,1)")

    if not fig.layout.updatemenus:
        fig.update_layout(updatemenus=[dict(
            type="buttons", showactive=False, buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame":{"duration":400,"redraw":True},"transition":{"duration":0}}])
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
    token_row = html.Div(spans, style={"fontFamily":"monospace","fontSize":"1.05rem","lineHeight":"1.8"})
    caption = html.Div(f"Step {idx+1}/{len(steps)} — tokens so far: {len(step['tokens'])}")

    return fig, html.Div([caption, token_row])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run_server(host="0.0.0.0", port=port, debug=False)

