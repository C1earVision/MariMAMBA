import os
import gradio as gr
from PIL import Image

from map_fixer_system.map_fixer import (
    generate_level,
    bfs_reachability,
    run_fix_pipeline,
    render_level,
    count_tile_changes,
    TILE_PX,
)

# ── Helpers ────────────────────────────────────────────────────────────────

def _blank(msg="", w=640, h=200):
    img = Image.new("RGB", (w, h), (22, 26, 36))
    return img


def _render(grid, visited=None, path=None, stuck_col=None, title=""):
    return render_level(grid, visited=visited, path=path,
                        stuck_col=stuck_col, title=title, px=TILE_PX)


# ── CSS theme ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap');

body, .gradio-container {
    background: #0d0f14 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Header */
.studio-header {
    background: #161a23;
    border-bottom: 1px solid #2a3045;
    padding: 18px 28px;
    margin-bottom: 24px;
    border-radius: 10px;
}
.studio-title {
    font-family: 'Press Start 2P', monospace;
    font-size: 13px;
    color: #ff6b2b;
    text-shadow: 2px 2px 0 #7a3010;
    letter-spacing: 1px;
    margin: 0 0 6px 0;
}
.studio-sub {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #525d75;
    letter-spacing: 1px;
}

/* Section labels */
.section-label {
    font-family: 'Press Start 2P', monospace;
    font-size: 7px;
    color: #ff6b2b;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* Panels */
.gr-panel, .gr-box {
    background: #161a23 !important;
    border: 1px solid #2a3045 !important;
    border-radius: 10px !important;
}

/* Sliders */
.gr-slider input[type=range] {
    accent-color: #ff6b2b;
}

/* Buttons */
.gr-button-primary {
    background: #ff6b2b !important;
    border: none !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 0 #7a3010 !important;
    color: white !important;
    padding: 14px !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
}
.gr-button-primary:hover {
    background: #ff9a5c !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 0 #7a3010 !important;
}
.gr-button-primary:active {
    transform: translateY(2px) !important;
    box-shadow: 0 2px 0 #7a3010 !important;
}
.gr-button-secondary {
    background: #1e2330 !important;
    border: 1px solid #3a4560 !important;
    color: #8892a8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    border-radius: 8px !important;
}
.gr-button-secondary:hover {
    border-color: #4a8fff !important;
    color: #4a8fff !important;
}

/* Log textbox */
.log-textbox textarea {
    background: #0d0f14 !important;
    border: 1px solid #2a3045 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: #8892a8 !important;
    line-height: 1.8 !important;
}

/* API key input */
.api-key-input input {
    background: #0d0f14 !important;
    border: 1px solid #2a3045 !important;
    color: #e8ecf4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 6px !important;
}

/* Image panels */
.gr-image {
    border: 1px solid #2a3045 !important;
    border-radius: 8px !important;
    background: #0d0f14 !important;
}

/* Tab styling */
.gr-tab-item {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 7px !important;
    color: #525d75 !important;
}
.gr-tab-item.selected {
    color: #ff6b2b !important;
    border-bottom-color: #ff6b2b !important;
}

/* Status label */
.status-label {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 10px 14px;
    background: #0d0f14;
    border: 1px solid #2a3045;
    border-radius: 6px;
    color: #8892a8;
}

/* Legend */
.legend-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #525d75;
    padding: 8px 0;
}

/* Stats row */
.stat-box {
    background: #1e2330;
    border: 1px solid #2a3045;
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
}
.stat-val {
    font-family: 'Press Start 2P', monospace;
    font-size: 10px;
    color: #ff6b2b;
    display: block;
    margin-bottom: 4px;
}
.stat-key {
    font-size: 10px;
    color: #525d75;
}
"""

HEADER_HTML = """
<div class="studio-header">
  <div class="studio-title">🍄 MARIO LEVEL STUDIO</div>
  <div class="studio-sub">PHYSICS-GUIDED AI LEVEL GENERATOR &amp; FIXER</div>
</div>
"""

LEGEND_HTML = """
<div class="legend-grid">
  <span>🟫 X — Ground</span>      <span>🟨 ? — Question</span>
  <span>🟤 S — Break block</span>  <span>🔴 E — Enemy</span>
  <span>🔵 - — Air</span>          <span>🟩 &lt;&gt; — Pipe</span>
  <span>🟡 o — Coin</span>         <span>⚪ ● — Solution path</span>
  <span>🟠 col — Stuck point</span><span>💚 tint — Reachable</span>
</div>
"""


# ── Core handler ───────────────────────────────────────────────────────────

def run_pipeline(enemies, gaps, pipes, temperature, top_k, top_p, num_columns, cfg_scale, seed_str, api_key):
    """
    Generator yielding (img_generated, img_fixed, img_before, img_after, log_text, stats_html).
    Drives the Gradio UI live as the fixer runs.
    """
    blank = _blank()

    # ── Parse seed ──
    try:
        seed = int(seed_str) if seed_str.strip() else None
    except ValueError:
        seed = None

    log_lines = []
    def log(msg):
        log_lines.append(msg)
        return "\n".join(log_lines)

    # Stats placeholders
    def stats_html(gen_sol=None, fix_sol=None, changes=None, rounds=None):
        def cell(val, label, color="#ff6b2b"):
            v = "—" if val is None else str(val)
            return (
                f'<div class="stat-box">'
                f'<span class="stat-val" style="color:{color}">{v}</span>'
                f'<span class="stat-key">{label}</span>'
                f'</div>'
            )
        g_color = "#3ddc84" if gen_sol else ("#ff4757" if gen_sol is False else "#ff6b2b")
        f_color = "#3ddc84" if fix_sol else ("#ff4757" if fix_sol is False else "#ff6b2b")
        g_text  = ("YES" if gen_sol else "NO") if gen_sol is not None else "—"
        f_text  = ("YES" if fix_sol else "NO") if fix_sol is not None else "—"
        return (
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px">'
            + cell(g_text, "Gen Solvable", g_color)
            + cell(f_text, "Fixed Solvable", f_color)
            + cell(changes, "Tiles Changed")
            + cell(rounds, "Fix Rounds")
            + "</div>"
        )

    yield blank, blank, blank, blank, log("Starting…"), stats_html()

    # ── Step 1: Generate ──
    txt = log(f"[1/4] Generating level (enemies={enemies}, gaps={gaps}, pipes={pipes}, columns={num_columns}, seed={seed})…")
    yield blank, blank, blank, blank, txt, stats_html()

    grid_orig = generate_level(
        [enemies, gaps, pipes], patches=num_columns/16.0, seed=seed,
        temperature=temperature, top_k=int(top_k), top_p=top_p,
        cfg_scale=cfg_scale
    )
    h, w = len(grid_orig), len(grid_orig[0])
    txt = log(f"      Level size: {h} rows × {w} cols")
    yield blank, blank, blank, blank, txt, stats_html()

    # ── Step 2: BFS on generated ──
    txt = log("[2/4] Running BFS physics check on generated level…")
    yield blank, blank, blank, blank, txt, stats_html()

    vis0, path0, sol0, stuck0 = bfs_reachability(grid_orig)
    status0 = "SOLVABLE ✓" if sol0 else f"NOT SOLVABLE — stuck at col {stuck0}/{w-1}"
    txt = log(f"      Result: {status0}")

    img_gen = _render(
        grid_orig,
        visited=vis0,
        path=path0 if sol0 else None,
        stuck_col=None if sol0 else stuck0,
        title=f"GENERATED — {status0}",
    )
    yield img_gen, blank, img_gen, blank, txt, stats_html(gen_sol=sol0)

    if not api_key.strip():
        txt = log("⚠  No GROQ API key provided — skipping fix phase.")
        yield img_gen, img_gen, img_gen, img_gen, txt, stats_html(gen_sol=sol0, fix_sol=sol0, changes=0, rounds=0)
        return

    if sol0:
        txt = log("      Level already solvable — no fixes needed.")
        yield img_gen, img_gen, img_gen, img_gen, txt, stats_html(gen_sol=True, fix_sol=True, changes=0, rounds=0)
        return

    # ── Step 3: Fix pipeline ──
    txt = log(f"[3/4] Running AI fix engine (max {4} attempts per stuck column)…")
    yield img_gen, blank, img_gen, blank, txt, stats_html(gen_sol=sol0)

    img_fixed_latest = blank
    final_grid = grid_orig
    final_sol  = False
    final_stuck = stuck0
    total_rounds = 0

    for event in run_fix_pipeline(grid_orig, api_key.strip()):
        if "log" in event:
            txt = log("      " + event["log"])
            yield img_gen, img_fixed_latest, img_gen, img_fixed_latest, txt, stats_html(gen_sol=sol0)

        if "grid" in event:
            final_grid   = event["grid"]
            final_sol    = event["solvable"]
            final_stuck  = event["stuck_col"]
            total_rounds = event["round"]

            img_fixed_latest = _render(
                final_grid,
                visited=event.get("visited"),
                path=event.get("path") if final_sol else None,
                stuck_col=None if final_sol else final_stuck,
                title=f"Round {total_rounds} — {'SOLVABLE ✓' if final_sol else f'stuck col {final_stuck}'}",
            )
            changes = count_tile_changes(grid_orig, final_grid)
            txt = log(f"      [grid update round {total_rounds}]")
            yield (
                img_gen, img_fixed_latest,
                img_gen, img_fixed_latest,
                txt,
                stats_html(gen_sol=sol0, fix_sol=final_sol, changes=changes, rounds=total_rounds),
            )

    # ── Step 4: Final render ──
    txt = log("[4/4] Rendering final results…")
    changes = count_tile_changes(grid_orig, final_grid)

    vis_f, path_f, sol_f, stuck_f = bfs_reachability(final_grid)
    img_final = _render(
        final_grid,
        visited=vis_f,
        path=path_f if sol_f else None,
        stuck_col=None if sol_f else stuck_f,
        title=f"FIXED — {'SOLVABLE ✓' if sol_f else 'NOT SOLVABLE ✗'}",
    )

    txt = log(f"Done! Solvable: {sol_f} | Tiles changed: {changes} | Rounds: {total_rounds}")

    yield (
        img_gen, img_final,
        img_gen, img_final,
        txt,
        stats_html(gen_sol=sol0, fix_sol=sol_f, changes=changes, rounds=total_rounds),
    )


# ── Build Gradio UI ────────────────────────────────────────────────────────

with gr.Blocks(title="Mario Level Studio") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        # ── LEFT: controls ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            gr.HTML('<div class="section-label">⚙ Generation Parameters</div>')

            enemies = gr.Slider(
                minimum=0, maximum=15, step=1, value=3,
                label="Enemies (Target Count)",
            )
            gaps = gr.Slider(
                minimum=0, maximum=10, step=1, value=0,
                label="Gaps (Target Count)",
            )
            pipes = gr.Slider(
                minimum=0, maximum=10, step=1, value=0,
                label="Pipes (Target Count)",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=2.0, step=0.1, value=0.8,
                label="Temperature  (higher=more random)",
            )
            top_k = gr.Slider(
                minimum=0, maximum=100, step=1, value=20,
                label="Top-K  (0=disabled)",
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, step=0.05, value=0.9,
                label="Top-P  (1.0=disabled)",
            )
            cfg_scale = gr.Slider(
                minimum=1.0, maximum=10.0, step=0.5, value=3.0,
                label="CFG Scale  (1.0=no guidance)",
            )
            num_columns = gr.Slider(
                minimum=16, maximum=320, step=1, value=120,
                label="Level Columns",
            )
            seed_input = gr.Textbox(
                label="Random seed  (leave blank for random)",
                placeholder="e.g. 42",
                max_lines=1,
            )

            gr.HTML('<div style="margin:16px 0 8px"><div class="section-label">🔑 API Key</div></div>')
            api_key = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk-…",
                type="password",
                max_lines=1,
                elem_classes=["api-key-input"],
            )

            run_btn = gr.Button("▶  GENERATE + FIX", variant="primary")

            gr.HTML('<div style="margin:16px 0 8px"><div class="section-label">📖 Tile Legend</div></div>')
            gr.HTML(LEGEND_HTML)

        # ── RIGHT: results ─────────────────────────────────────────────
        with gr.Column(scale=3):

            stats_out = gr.HTML(
                '<div style="color:#525d75;font-family:\'DM Mono\',monospace;font-size:11px;padding:8px 0">'
                'Stats appear after generation.</div>'
            )

            with gr.Tabs():
                with gr.TabItem("Generated"):
                    img_generated = gr.Image(
                        label="Generated Level", type="pil",
                    )

                with gr.TabItem("Fixed"):
                    img_fixed = gr.Image(
                        label="Fixed Level", type="pil",
                    )

                with gr.TabItem("Before / After"):
                    with gr.Row():
                        img_before = gr.Image(
                            label="Before  (Generated)", type="pil",
                        )
                        img_after = gr.Image(
                            label="After  (Fixed)", type="pil",
                        )

                with gr.TabItem("Fix Log"):
                    log_out = gr.Textbox(
                        label="",
                        lines=22,
                        interactive=False,
                        elem_classes=["log-textbox"],
                    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[enemies, gaps, pipes, temperature, top_k, top_p, num_columns, cfg_scale, seed_input, api_key],
        outputs=[img_generated, img_fixed, img_before, img_after, log_out, stats_out],
    )


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, css=CUSTOM_CSS)