import copy
import os
import random
import textwrap
import time
import requests
import torch
from collections import deque
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont

from models.mamba import Mamba
from config.model_config import MambaConfig
from data.parser import LevelParser

SOLID_TILES = {"X", "S", "?", "Q", "<", ">", "[", "]", "B", "b", "E"}
VALID_TILES  = {"X", "S", "-", "?", "Q", "E", "<", ">", "[", "]", "o", "B", "b"}
EMPTY        = "-"

TILE_COLORS = {
    "X": (101,  67,  33),
    "S": (180, 120,  60),
    "-": ( 92, 148, 252),
    "?": (255, 200,   0),
    "Q": (200, 160,   0),
    "E": (220,  50,  50),
    "<": ( 50, 180,  50),
    ">": ( 50, 180,  50),
    "[": ( 40, 150,  40),
    "]": ( 40, 150,  40),
    "o": (255, 230,  80),
    "B": (160,  80,  40),
    "b": (130,  60,  20),
}
DEFAULT_COLOR = (160, 160, 160)

MAX_JUMP_H   = 5
MAX_JUMP_W   = 5
MAX_FIX_ROUNDS = 4
MAX_TOTAL_ROUNDS = 20

ANTHROPIC_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ANTHROPIC_MODEL   = "llama-3.1-8b-instant" 

TILE_PX = 16



def solid(g, r, c):
    h, w = len(g), len(g[0])
    if r < 0 or r >= h:
        return False
    if c < 0 or c >= w:
        return True
    return g[r][c] in SOLID_TILES


def passable(g, r, c):
    h, w = len(g), len(g[0])
    if r < 0 or r >= h:
        return True
    if c < 0 or c >= w:
        return False
    return g[r][c] not in SOLID_TILES


def grounded(g, r, c):
    return passable(g, r, c) and solid(g, r + 1, c)


def mario_moves(g, r, c):
    moves = []
    h, w = len(g), len(g[0])
    on_ground = grounded(g, r, c)

    if on_ground:
        for dc in (1, -1):
            nc = c + dc
            if 0 <= nc < w and grounded(g, r, nc):
                moves.append((r, nc))

    if not solid(g, r + 1, c) and passable(g, r + 1, c):
        moves.append((r + 1, c))
        for dc in (1, -1):
            if 0 <= c + dc < w and passable(g, r + 1, c + dc):
                moves.append((r + 1, c + dc))

    if on_ground:
        for di in (1, -1):
            for jh in range(1, MAX_JUMP_H + 1):
                for jd in range(1, MAX_JUMP_W + 1):
                    lc = c + di * jd
                    if lc < 0 or lc >= w:
                        continue
                    apex_r = r - jh
                    if apex_r < 0:
                        continue
                    clear = True
                    for s in range(1, jd + 1):
                        cc = c + di * s
                        if cc < 0 or cc >= w:
                            clear = False
                            break
                        cr = r - int(s * (jh / jd))
                        if cr < 0 or cr >= h or solid(g, cr, cc):
                            clear = False
                            break
                    if not clear:
                        continue
                    for lr in range(apex_r, h):
                        if grounded(g, lr, lc):
                            moves.append((lr, lc))
                            break
    return moves


def bfs_reachability(grid):

    h, w = len(grid), len(grid[0])
    starts = sorted(
        [(r, 0) for r in range(h) if grounded(grid, r, 0)],
        key=lambda p: -p[0],
    )
    end_cols = {w - 1, w - 2, w - 3}

    if not starts:
        return set(), [], False, 0

    visited = {}
    queue   = deque()
    for s in starts:
        if s not in visited:
            visited[s] = [s]
            queue.append(s)

    found_end = None
    while queue:
        cr, cc = queue.popleft()
        if cc in end_cols and passable(grid, cr, cc):
            found_end = (cr, cc)
            break
        for nb in mario_moves(grid, cr, cc):
            if nb not in visited:
                visited[nb] = visited[(cr, cc)] + [nb]
                queue.append(nb)

    visited_set = set(visited.keys())
    if found_end:
        return visited_set, visited[found_end], True, None

    stuck_col = max((c for _, c in visited_set), default=0)
    rightmost = [p for p in visited_set if p[1] == stuck_col]
    best_path = visited.get(rightmost[0], []) if rightmost else []
    return visited_set, best_path, False, stuck_col


def analyze_stuck(grid, visited, stuck_col):
    h, w = len(grid), len(grid[0])
    problems = []
    look_ahead = min(stuck_col + 12, w - 1)

    for c in range(stuck_col, look_ahead + 1):
        col_solid = [r for r in range(h) if solid(grid, r, c)]
        if len(col_solid) >= h - 1:
            problems.append(f"BLOCKAGE at col {c}: A solid vertical wall is preventing Mario from moving right.")
            break
        if passable(grid, h - 1, c):
            problems.append(f"DANGEROUS PIT at col {c}: There is no ground at the bottom of this column.")

    if not problems:
        problems.append(f"REACHABILITY: Mario is stuck at col {stuck_col} and cannot find a path forward.")

    return problems



SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert Super Mario Bros level designer.
Your task is to make a small LEVEL WINDOW (a text grid) passable so Mario can traverse it left to right.

TILE LEGEND:
X=solid ground, S=breakable block, -=empty air, ?=question block full, Q=question block empty,
E=enemy, <=pipe-top-left, >=pipe-top-right, [=pipe-left, ]=pipe-right, o=coin, B=cannon-top, b=cannon-bottom

MARIO'S CAPABILITIES:
- Max jump height: 5 tiles vertically
- Max jump width:  5 tiles horizontally
- Solid obstacles: X S ? Q < > [ ] B b E

STRICT FIXING RULES:
1. MINIMALISM – Change as few tiles as possible. Only remove tiles that are blocking Mario's path.
2. REMOVAL ONLY – Your primary tool is changing solid tiles to '-' (empty air) to clear a path for Mario.
3. DO NOT touch pipes, enemies, coins, or any tiles that are NOT directly blocking Mario's path.
4. DO NOT add new tiles. DO NOT add platforms, enemies, blocks, or any structures.
5. DO NOT fix cosmetic issues, broken structures, or anything unrelated to making the path passable.
6. Your ONLY goal is to let Mario walk or jump from the left side to the right side of this window.

CRITICAL OUTPUT RULES:
1. Output ONLY the exact text grid rows. NO markdown, NO backticks, NO explanations.
2. EXACT same number of rows and columns as the input window.
3. Only use characters from the TILE LEGEND.
""")


def call_llm(prompt, api_key, retries=3):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":    ANTHROPIC_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.2,
    }
    for attempt in range(retries):
        try:
            r    = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=120)
            data = r.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            code = data.get("error", {}).get("code")
            if code == 429:
                time.sleep((attempt + 1) * 15)
            else:
                time.sleep(5)
        except Exception as e:
            print(f"[LLM] attempt {attempt+1} exception: {e}")
            time.sleep(5)
    return ""


def build_window_prompt(grid, stuck_col, visited, problems, attempt):
    h, w = len(grid), len(grid[0])
    rows_at_stuck = [r for (r, c) in visited if c == stuck_col]
    stuck_row = min(rows_at_stuck) if rows_at_stuck else h - 1

    r0 = max(0, stuck_row - 2)
    r1 = min(h - 1, stuck_row + 2)
    c0 = max(0, stuck_col - 2)
    c1 = min(w - 1, stuck_col + 2)

    win_rows = [
        "".join(grid[r][c0:c1 + 1])
        for r in range(r0, r1 + 1)
    ]
    window_str = "\n".join(win_rows)
    win_h, win_w = r1 - r0 + 1, c1 - c0 + 1
    prob_text = "\n".join(f"  [{i+1}] {p}" for i, p in enumerate(problems))

    prompt = (
        f"=== LEVEL WINDOW ({win_h}r × {win_w}c) — attempt {attempt+1} ===\n"
        f"Mario is STUCK near the centre of this window and cannot move right.\n\n"
        f"PROBLEMS:\n{prob_text}\n\n"
        f"WINDOW:\n{window_str}\n\n"
        f"Remove the minimum number of blocking tiles (change them to '-') so Mario can pass through.\n"
        f"Return ONLY the {win_h} rows, each exactly {win_w} characters.\n"
    )
    return r0, r1, c0, c1, win_h, win_w, prompt


def parse_window_response(text, win_h, win_w):
    rows = []
    for line in text.splitlines():
        line = line.strip().replace("`", "")
        if not line:
            continue
        clean = line.replace(" ", "")
        if all(ch in VALID_TILES for ch in clean):
            rows.append(clean)
    if not rows:
        return None
    result = []
    for row in rows[:win_h]:
        row = (row + EMPTY * win_w)[:win_w]
        result.append(row)
    while len(result) < win_h:
        result.append(EMPTY * win_w)
    return result


def apply_window(grid, fixed_rows, r0, r1, c0, c1):
    new_grid = copy.deepcopy(grid)
    win_w = c1 - c0 + 1
    for i, r in enumerate(range(r0, r1 + 1)):
        if i < len(fixed_rows):
            row = list((fixed_rows[i] + EMPTY * win_w)[:win_w])
            for dc in range(win_w):
                if row[dc] in VALID_TILES:
                    new_grid[r][c0 + dc] = row[dc]
    return new_grid


def run_fix_pipeline(grid, api_key):

    h, w = len(grid), len(grid[0])

    yield {"log": f"Level size: {h} rows × {w} cols"}


    yield {"log": "Running initial BFS reachability check…"}
    visited0, path0, sol0, stuck0 = bfs_reachability(grid)
    status = "SOLVABLE" if sol0 else f"NOT SOLVABLE — stuck at col {stuck0}/{w-1}"
    yield {"log": f"Initial result: {status}"}
    yield {
        "grid": grid, "visited": visited0, "path": path0,
        "solvable": sol0, "stuck_col": stuck0, "round": 0,
    }


    if sol0:
        yield {"log": "Level is already solvable. No fixes needed."}
        return

    current = copy.deepcopy(grid)
    last_stuck_col      = -1
    attempts_at_col     = 0
    fix_round           = 0

    for total_round in range(MAX_TOTAL_ROUNDS):
        visited, path, solvable, stuck_col = bfs_reachability(current)
        fix_round = total_round

        if solvable:
            yield {"log": f"SUCCESS: Level is now solvable after {total_round} fix rounds."}
            yield {
                "grid": current, "visited": visited, "path": path,
                "solvable": True, "stuck_col": None, "round": total_round,
            }
            return

        if stuck_col == last_stuck_col:
            attempts_at_col += 1
        else:
            last_stuck_col  = stuck_col
            attempts_at_col = 1

        if attempts_at_col > MAX_FIX_ROUNDS:
            yield {"log": f"✗ Gave up on stuck col {stuck_col} after {MAX_FIX_ROUNDS} attempts."}
            break

        yield {"log": f"Round {total_round+1}: stuck at col {stuck_col}/{w-1} "
                      f"(attempt {attempts_at_col}/{MAX_FIX_ROUNDS} for this col)"}

        problems = analyze_stuck(current, visited, stuck_col)
        for p in problems:
            yield {"log": f"  Problem: {p[:100]}"}

        r0, r1, c0, c1, win_h, win_w, prompt = build_window_prompt(
            current, stuck_col, visited, problems, total_round
        )
        yield {"log": f"  Window rows {r0}–{r1}, cols {c0}–{c1} ({win_h}×{win_w}) → calling LLM…"}

        response = call_llm(prompt, api_key)
        if not response:
            yield {"log": "  LLM returned empty — skipping round."}
            continue

        fixed_rows = parse_window_response(response, win_h, win_w)
        if fixed_rows is None:
            yield {"log": f"  Could not parse LLM response — skipping."}
            continue

        current = apply_window(current, fixed_rows, r0, r1, c0, c1)
        yield {"log": "  Window patch applied."}


        v2, p2, s2, sc2 = bfs_reachability(current)
        yield {
            "grid": current, "visited": v2, "path": p2,
            "solvable": s2, "stuck_col": sc2, "round": total_round + 1,
        }


    visited_f, path_f, sol_f, stuck_f = bfs_reachability(current)
    changes = sum(
        1 for r in range(h) for c in range(w)
        if grid[r][c] != current[r][c]
    )
    yield {"log": f"Final: {'SOLVABLE ✓' if sol_f else 'NOT SOLVABLE ✗'} | "
                  f"Tiles changed: {changes} | Fix rounds: {fix_round+1}"}
    yield {
        "grid": current, "visited": visited_f, "path": path_f,
        "solvable": sol_f, "stuck_col": stuck_f, "round": fix_round + 1,
    }


_mamba_model = None
_mamba_parser = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_mamba_model():
    global _mamba_model, _mamba_parser
    if _mamba_model is None:
        from config.training_config import MambaTrainingConfig
        t_cfg = MambaTrainingConfig()
        m_cfg = MambaConfig()
        
        print("[Mamba] Initializing Mamba...")
        _mamba_parser = LevelParser()
        _mamba_model = Mamba(
            num_tile_types=m_cfg.num_tile_types,
            column_height=m_cfg.column_height,
            tile_embed_dim=m_cfg.tile_embed_dim,
            d_model=m_cfg.d_model,
            n_layers=m_cfg.n_layers,
            d_state=m_cfg.d_state,
            d_conv=m_cfg.d_conv,
            expand=m_cfg.expand,
            dropout=0.0,
            max_seq_len=m_cfg.max_seq_len,
            num_attributes=m_cfg.num_attributes,
            columns_per_token=m_cfg.columns_per_token,
        ).to(_device)
        

        checkpoint_path = t_cfg.save_path.replace('.pth', '_best.pth')

        ema_path = checkpoint_path.replace('.pth', '_ema.pth')
        if t_cfg.use_ema and os.path.exists(ema_path):
            checkpoint_path = ema_path
            
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=_device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            _mamba_model.load_state_dict(state_dict)
            _mamba_model.eval()
            print(f"[Mamba] Loaded weights from: {checkpoint_path}")
        else:
            print(f"[Mamba] ⚠ Weights not found at {checkpoint_path}")
            
    return _mamba_model, _mamba_parser



def create_difficulty_schedule(num_columns: int, peak_difficulty: float) -> torch.Tensor:
    """
    Creates a triangular difficulty schedule: 0 -> peak -> 0.
    """
    num_columns = int(num_columns)
    mid = num_columns // 2

    ramp_up = torch.linspace(0.0, float(peak_difficulty), int(mid))

    ramp_down = torch.linspace(float(peak_difficulty), 0.0, int(num_columns - mid))
    return torch.cat([ramp_up, ramp_down])


def generate_level(attributes: List[float], patches: int, seed: int | None = None, 
                   temperature: float = 0.8, top_k: int = 20, top_p: float = 0.9,
                   cfg_scale: float = 3.0) -> list[list[str]]:
    """
    Returns a 2-D list[list[str]] representing a Mario level generated by Mamba.
    """
    model, parser = get_mamba_model()
    if model is None:
        raise RuntimeError("Could not load Mamba model for generation.")

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)


    num_columns = int(patches * 16)
    
    attr_tensor = torch.tensor(attributes).float().to(_device)

    with torch.no_grad():
        generated_indices = model.generate(
            num_columns=num_columns,
            attributes=attr_tensor,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale,
            device=_device
        )
    

    level_array = generated_indices.cpu().numpy().T
    
    grid = []
    for r in range(level_array.shape[0]):
        row = [parser.idx_to_tile[int(tile_idx)] for tile_idx in level_array[r]]
        grid.append(row)
        
    return grid








_TITLE_H = 22          # pixels for title bar
_mario_lm_cache = None


def _get_mario_lm():
    """Lazy-load MarioLM once and cache it."""
    global _mario_lm_cache
    if _mario_lm_cache is None:
        from mario_gpt import MarioLM          # noqa: PLC0415
        _mario_lm_cache = MarioLM()
    return _mario_lm_cache


def render_level(grid, visited=None, path=None, stuck_col=None, title="", px=16, **kwargs):
    """
    Hybrid renderer:
      1. MarioGPT's convert_level_to_png for authentic tile sprites.
      2. PIL overlays for BFS visited (green tint), solution path (white dots),
         stuck column (orange stripe), and a title bar.
    Falls back to pure-PIL on any MarioGPT import error.
    """
    h = len(grid)
    w = len(grid[0])

    # ── Try MarioGPT base render ────────────────────────────────────────────
    base_img = None
    tile_px   = px          # actual tile size in the base image
    try:
        from mario_gpt.utils import convert_level_to_png   # noqa: PLC0415
        mario_lm  = _get_mario_lm()
        row_list  = ["".join(row) for row in grid]
        base_img, _, _ = convert_level_to_png(row_list, mario_lm.tokenizer)
        # convert_level_to_png always produces 16-px tiles
        tile_px = base_img.width // w if w > 0 else 16
    except Exception:
        base_img = None   # fall through to PIL fallback

    # ── Build canvas (with optional title bar) ──────────────────────────────
    title_h = _TITLE_H if title else 0
    canvas_w = (base_img.width  if base_img else w * tile_px)
    canvas_h = (base_img.height if base_img else h * tile_px) + title_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (22, 26, 36))

    if title:
        from PIL import ImageDraw as _ID, ImageFont as _IF  # noqa: PLC0415
        _d = _ID.Draw(canvas)
        _d.rectangle([0, 0, canvas_w, title_h - 1], fill=(30, 35, 50))
        try:
            _font = _IF.truetype("DejaVuSansMono.ttf", 11)
        except Exception:
            _font = _IF.load_default()
        _d.text((6, 4), title, fill=(255, 200, 80), font=_font)

    # Paste base image below the title bar
    if base_img:
        canvas.paste(base_img, (0, title_h))
    else:
        # Pure-PIL fallback: draw solid tile colours
        _d2 = ImageDraw.Draw(canvas)
        for r in range(h):
            for c in range(w):
                color = TILE_COLORS.get(grid[r][c], DEFAULT_COLOR)
                x0, y0 = c * tile_px, r * tile_px + title_h
                _d2.rectangle([x0, y0, x0 + tile_px - 1, y0 + tile_px - 1], fill=color)

    # ── PIL overlays ────────────────────────────────────────────────────────
    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    path_set = set(map(tuple, path))    if path    else set()
    vis_set  = set(map(tuple, visited)) if visited else set()

    for r in range(h):
        for c in range(w):
            x0 = c * tile_px
            y0 = r * tile_px + title_h

            # Green tint — BFS-reachable cells
            if (r, c) in vis_set:
                draw.rectangle([x0, y0, x0 + tile_px - 1, y0 + tile_px - 1],
                               fill=(0, 200, 60, 60))

            # Orange stripe — stuck column
            if stuck_col is not None and c == stuck_col:
                draw.rectangle([x0, y0, x0 + tile_px - 1, y0 + tile_px - 1],
                               fill=(255, 120, 0, 100))

            # White dot — solution path
            if (r, c) in path_set:
                cr, cy = x0 + tile_px // 2, y0 + tile_px // 2
                r2 = max(2, tile_px // 4)
                draw.ellipse([cr - r2, cy - r2, cr + r2, cy + r2],
                             fill=(255, 255, 255, 220))

    canvas = canvas.convert("RGBA")
    canvas.alpha_composite(overlay)
    return canvas.convert("RGB")



def count_tile_changes(original, fixed):
    h, w = len(original), len(original[0])
    return sum(
        1 for r in range(h) for c in range(w)
        if original[r][c] != fixed[r][c]
    )