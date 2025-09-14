import os
import json
import unicodedata
from functools import lru_cache

import dash
from dash import Dash, dcc, html, ctx, no_update
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from argos_viz import generate_steps  # unchanged server-side capture

# ------------------------- Config / Env -------------------------
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "distilgpt2")
DEVICE = "cpu" if os.environ.get("DEVICE", "cpu") == "cpu" else "cuda"

# If set to "1", ⏯ / ⏩ are disabled (safe for HN spikes)
DISABLE_GENERATION = os.environ.get("DISABLE_GENERATION", "0") == "1"

# Hard cap to avoid abuse (applies to ⏩)
MAX_NEW_TOKENS_CAP = int(os.environ.get("MAX_NEW_TOKENS_CAP", "64"))

# Optional JSON cache path for precomputed steps; e.g. "/data/belair.json"
PRECOMPUTE_JSON = os.environ.get("PRECOMPUTE_JSON", "").strip()

# If "1", compute once on boot and optionally write PRECOMPUTE_JSON
PRECOMPUTE_ON_BOOT = os.environ.get("PRECOMPUTE_ON_BOOT", "1") == "1"

DARK_BG = "#0f1620"
DARK_FG = "#e4e7ee"
CARD_BG = "rgba(255,255,255,.07)"

# ------------------------- Text sanitization -------------------------

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

# ------------------------- Your prompt (unchanged) -------------------------
DEFAULT_PROMPT = """Now this is the story all about how,
My strife got flipped-turned upside down,
And I'd like to tickle my newt, just sit right there,
I'll tell you how I became the principles' own caught Bell heir.

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
        title="\t\tLatent Thought Flow",
    )
    return fig

# ------------------------- Precompute on boot -------------------------
def _generate_steps(model_id: str, prompt: str, temperature: float, max_new_tokens: int):
    tok, model = load_model(model_id or DEFAULT_MODEL_ID)
    steps, _ = generate_steps(
        model, tok, (prompt or "").strip(),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        pca_freeze_after_first=False,
        show_trajectories=True,
    )
    prompt_len = len(tok.encode((prompt or '').strip()))
    payload = [dict(tokens=s.tokens, colors=s.colors, X=s.X.tolist(), fig=s.figure_json, prompt_len=prompt_len)
               for s in steps]
    return payload

def _load_precomputed():
    if PRECOMPUTE_JSON and os.path.exists(PRECOMPUTE_JSON):
        with open(PRECOMPUTE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_precomputed(data):
    if PRECOMPUTE_JSON:
        os.makedirs(os.path.dirname(PRECOMPUTE_JSON), exist_ok=True)
        with open(PRECOMPUTE_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f)

PRECOMP_STEPS = _load_precomputed()
if PRECOMP_STEPS is None and PRECOMPUTE_ON_BOOT:
    try:
        # Modest default so it warms fast; you can change via env if you like
        pre_temp = float(os.environ.get("PRECOMPUTE_TEMPERATURE", "0.9"))
        pre_max = int(os.environ.get("PRECOMPUTE_MAX_NEW", "32"))
        PRECOMP_STEPS = _generate_steps(DEFAULT_MODEL_ID, DEFAULT_PROMPT, pre_temp, pre_max)
        _save_precomputed(PRECOMP_STEPS)
    except Exception as e:
        # Precompute failed; fall back to empty (page still loads)
        print(f"[argos] Precompute failed: {e}")
        PRECOMP_STEPS = None

# ------------------------- Layout -------------------------
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

/* Inputs */
#root-div input[type="text"],
#root-div textarea,
#root-div input[type="number"],
#root-div .input,
#root-div .textarea {{
  background: #111a24 !important;
  color: #e4e7ee !important;
  -webkit-text-fill-color: #e4e7ee !important;
  caret-color: #e4e7ee !important;
  border: 1px solid #2a3340 !important;
  box-shadow: none !important;
}}
.btn {{ background:#6b4df6; color:white; border:none; border-radius:8px; padding:.45rem .8rem; }}
.btn.secondary {{ background:#2a3340; }}

/* BIG brand */
.brand {{
  font-family: 'Oxanium', system-ui, sans-serif;
  letter-spacing: .14em;
  font-weight: 800;
  font-size: 6.2rem;
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

/* Token chips */
.tok {{
  font-family: monospace;
  padding: 0 .18rem;
  border-radius: .35rem;
  transition: background .12s ease, color .12s ease;
  cursor: pointer;
}}
.tok:hover {{ background: var(--tokcol); color: {DARK_BG}; }}
.tok.current {{ background: var(--tokcol); color: {DARK_BG}; }}
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
                    ]),

                        html.Div(style={"marginTop": "0.5rem", "display": "flex", "gap": "0.5rem", "flexWrap": "wrap"}, children=[
                            html.Button("⏯", id="run_one", className="btn", title="Generate 1 token"),
                            html.Button("⏩", id="run_many", className="btn", title="Generate Max new tokens"),
                            html.Button("◀", id="prev", className="btn secondary", title="Prev token"),
                            html.Button("▶", id="next", className="btn secondary", title="Next token"),
                        ]),

                        # ---------------------- YOUR INFOBOX (unchanged) ----------------------
                        html.Div(className="infobox", children=[
                            dcc.Markdown(
                                """
# Controls
- **⏯** generate **one** token and update the viz
- **⏩** generate **Max new tokens**
- **◀ / ▶** step through already generated token-steps
- **Play** in the plotly figure animates the forward pass.

**To get started, press ⏯ !**

**Give it a minute or two** to update the figure. [Spoiler](/assets/argos.gif) of what it'll look like.

Once the figure is displayed, use the slider to go through the depth of the model or press **Play** to see the entire forward pass.

On mobile, landscape mode should work best.

I literally couldn't figure out how to make the input text visible with a dark background, so you'll probably have to untoggle the dark theme to see what's in the input fields.

You can click on individual generated tokens and the figure will update to that token's forward pass.


# What is this?

Each dot is the hidden vector currently associated with the input **token** indicated by its label (prompt + generated so far).

I capture activations after every **LayerNorm** (which usually follow each **Attention** and each **MLP**), stack them and run **PCA** to 3D on this.

I visualise after each LayerNorm because they project to a **hyperellipsoid**, which most of the time even happens to be spherical, which is not only compact and much nicer to look at, but spherical geometry is what is of interest to understanding the topology induced by attention (log of attention is proportional to cosine similarity).

So every second layernorm is subsequent to the same module type (either Attention or MLP).

The PCA base is indeed the same across LayerNorms, the jumps come from the MLPs: MLPs are basically a disontinuous jump. If you try to conceptualise Transformers as smooth token flows, MLPs are difficult to treat, as they amount to a context-independent conceptual key-value dictionary. If you replace MLPs by learned rotation matrices, you get a smooth flow like [this](https://marvosyntactical.github.io/hydra_latent_labeled.html). For that model, see [here](https://github.com/marvosyntactical/hydra). It **halves** parameter count with no loss in performance at small model scales, I don't have the resources to see if it scales, though.


## Post-Attention Geometry

Post-attention, you can often see a beautiful paratrooperesque constellation, with the very first token being the trooper spanning the rest as a chute.

My good friend [Michi Staniek](https://www.cl.uni-heidelberg.de/statnlpgroup/members/staniek/) was quick to point out why you can expect this to happen: Attention attracts new tokens to all previous ones, using them as a sort of **reference frame**. He sent me [this article](https://arxiv.org/abs/2508.02546v1), which corroborates this view.

The literature also discusses the phenomenon of **attention sinks**, where the first token takes on a significant portion (think, a third) of most subsequent tokens' attention. It stands to reason that that phenomenon can be explained in terms of the induction of such a reference frame iteratively constructed from the first few tokens. It definitely seems like there's room for theoretical headway there. Ultimately, I am interested in the contractive dynamics on the token distribution induced by attention, as demonstrated empirically and theoretically by e.g. [this work](https://www.ams.org/journals/bull/2025-62-03/S0273-0979-2025-01863-7/). This model is fairly small, but with deeper models, you should see the tokens gradually contract to a point, at least according to the highly idealised model of that paper.


# Other stuff

Namesake of this tool is [Argos Panoptes](https://en.wikipedia.org/wiki/Argus_Panoptes).

It goes without saying that this is vibecoded.

If you like this (otherwise too haha), you can check out more of my stuff at [marvosyntactical.github.io](https://marvosyntactical.github.io). If you wanna chat about this or even want to buy me a coffee so I can deploy a better model on this page, just send me an email.

                                """
                            )
                        ]),
                        # ----------------------------------------------------------------------

                        # state stores (init with precomputed)
                        dcc.Store(id="busy"),
                        dcc.Store(id="steps-store", data=PRECOMP_STEPS if PRECOMP_STEPS else None),
                        dcc.Store(id="step-index", data=(len(PRECOMP_STEPS)-1 if PRECOMP_STEPS else 0)),
                        dcc.Store(id="focus-token", data=None),
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

# ------------------------------ Helpers --------------------------------

def build_fig_from_step(step, tokens_shown: int) -> go.Figure:
    X = np.array(step["X"])            # (T, L, 3)
    colors = step["colors"][:tokens_shown]
    texts = sanitize_tokens(step["tokens"][:tokens_shown])

    T, L, _ = X.shape
    T = min(T, tokens_shown)

    def scatter_at_ln(ell: int) -> go.Scatter3d:
        pts = X[:T, ell, :]
        return go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+text",
            text=texts,
            textposition="top center",
            marker=dict(size=4, color=colors, opacity=0.95),
            hoverinfo="text",
            showlegend=False
        )

    data = [scatter_at_ln(L - 1)]
    frames = [go.Frame(data=[scatter_at_ln(ell)], name=f"LN{ell}") for ell in range(L)]

    fig = go.Figure(data=data, frames=frames)
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
        updatemenus=[dict(
            type="buttons", showactive=False, buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 400, "redraw": True}, "transition": {"duration": 0}}])
            ],
            x=0.0, y=1.1
        )],
        sliders=[dict(
            active=L-1,
            pad={"l": 0, "b": 0, "t": 0, "r": 0},
            steps=[
                dict(args=[[f"LN{ell}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                     label=f"LN {ell}", method="animate")
                for ell in range(L)
            ]
        )]
    )
    return sanitize_figure_text(fig)

def _safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return default

# ------------------------------ Callbacks --------------------------------

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

@app.callback(Output("status-area", "children"), Input("busy", "data"))
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
        show_trajectories=True,
    )
    prompt_len = len(tok.encode((prompt or '').strip()))
    payload = [dict(tokens=s.tokens, colors=s.colors, X=s.X.tolist(), fig=s.figure_json, prompt_len=prompt_len)
               for s in steps]
    return payload

# ⏯ one token
@app.callback(
    Output("steps-store", "data"),
    Output("step-index", "data"),
    Output("focus-token", "data"),
    Output("busy", "data"),
    Input("run_one", "n_clicks"),
    State("model-id", "value"),
    State("prompt", "value"),
    State("temp", "value"),
    prevent_initial_call=True
)
def on_run_one(n_clicks, model_id, prompt, temp):
    if DISABLE_GENERATION:
        return no_update, no_update, None, "Generation disabled."
    payload = _generate(model_id, prompt, temp, k_tokens=1)
    return payload, max(0, len(payload) - 1), None, f"Ready. Steps: {len(payload)}."

# ⏩ many tokens (clamped)
@app.callback(
    Output("steps-store", "data", allow_duplicate=True),
    Output("step-index", "data", allow_duplicate=True),
    Output("focus-token", "data", allow_duplicate=True),
    Output("busy", "data", allow_duplicate=True),
    Input("run_many", "n_clicks"),
    State("model-id", "value"),
    State("prompt", "value"),
    State("max-new", "value"),
    State("temp", "value"),
    prevent_initial_call=True
)
def on_run_many(n_clicks, model_id, prompt, max_new, temp):
    if DISABLE_GENERATION:
        return no_update, no_update, None, "Generation disabled."
    cap = MAX_NEW_TOKENS_CAP
    want = _safe_int(max_new, 32)
    k = min(max(1, want), cap)
    payload = _generate(model_id, prompt, temp, k_tokens=k)
    return payload, max(0, len(payload) - 1), None, f"Ready. Steps: {len(payload)}."

# ◀ / ▶ navigate step index (token-by-token)
@app.callback(
    Output("step-index", "data", allow_duplicate=True),
    Output("focus-token", "data", allow_duplicate=True),
    Input("prev", "n_clicks"),
    Input("next", "n_clicks"),
    State("steps-store", "data"),
    State("step-index", "data"),
    prevent_initial_call=True
)
def on_prev_next(prev_clicks, next_clicks, steps, idx):
    if not steps:
        return no_update, no_update
    trig = ctx.triggered_id
    idx = int(idx or 0)
    if trig == "prev":
        idx = max(0, idx - 1)
    elif trig == "next":
        idx = min(len(steps) - 1, idx + 1)
    return idx, None

# Click a token chip → focus that token (figure shows tokens up to it) and jump to its earliest step
@app.callback(
    Output("focus-token", "data", allow_duplicate=True),
    Output("step-index", "data", allow_duplicate=True),
    Input({"type": "tok", "index": ALL}, "n_clicks"),
    State("steps-store", "data"),
    prevent_initial_call=True
)
def on_click_token(nclicks, steps):
    if not steps or not nclicks:
        return no_update, no_update
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or "index" not in trig:
        return no_update, no_update
    tok_i = int(trig["index"])
    new_idx = len(steps) - 1
    for s_idx, s in enumerate(steps):
        if len(s["tokens"]) >= (tok_i + 1):
            new_idx = s_idx
            break
    return tok_i, new_idx

# Render figure + interactive token strip
@app.callback(
    Output("sphere", "figure"),
    Output("texts", "children"),
    Input("steps-store", "data"),
    Input("step-index", "data"),
    Input("focus-token", "data"),
    Input("theme-toggle", "value"),
    prevent_initial_call=False
)
def on_step_change(steps, idx, focus_tok, theme_value):
    if not steps:
        return empty_dark_fig(), ""
    idx = int(idx or 0)
    step = steps[idx]
    tokens_shown = (focus_tok + 1) if isinstance(focus_tok, int) else len(step["tokens"])
    fig = build_fig_from_step(step, tokens_shown=tokens_shown)

    # Token strip: use the latest step to list all tokens
    full = steps[-1]
    all_tokens = sanitize_tokens(full["tokens"])
    all_colors = full["colors"]
    current_i = focus_tok if isinstance(focus_tok, int) else (len(step["tokens"]) - 1)

    chips = []
    for i, (tok, col) in enumerate(zip(all_tokens, all_colors)):
        style = {"--tokcol": col, "color": col, "marginRight": "2px", "paddingRight": "2px"}
        cls = "tok" + (" current" if i == current_i else "")
        chips.append(
            html.Span(
                tok,
                id={"type": "tok", "index": i},
                className=cls,
                style=style,
                title=f"Jump to token {i}",
                n_clicks=0
            )
        )

    token_row = html.Div(chips, style={"fontFamily": "monospace", "fontSize": "1.05rem", "lineHeight": "1.8"})
    caption = html.Div(f"Step {idx+1}/{len(steps)} — tokens so far: {len(step['tokens'])}")

    return fig, html.Div([caption, token_row])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run_server(host="0.0.0.0", port=port, debug=False)

