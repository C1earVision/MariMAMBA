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
MAX_FIX_ROUNDS = 4     # attempts per stuck column
MAX_TOTAL_ROUNDS = 20  # hard cap on total LLM calls

ANTHROPIC_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ANTHROPIC_MODEL   = "llama-3.1-8b-instant" 

TILE_PX = 16   # pixels per tile in rendered images



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


def check_structural_integrity(grid, start_col=0, end_col=None):
    """
    Checks for mismatched tiles (incomplete pipes, floating cannons).
    Returns a list of (column, problem_description) tuples.
    """
    h, w = len(grid), len(grid[0])
    if end_col is None:
        end_col = w
    problems = []
    
    for c in range(start_col, min(end_col, w)):
        for r in range(h):
            tile = grid[r][c]
            
            # Pipe tops: < must have >
            if tile == "<":
                if c + 1 >= w or grid[r][c+1] != ">":
                    problems.append((c, f"INCOMPLETE PIPE TOP: '<' at ({r},{c}) is missing its right half '>'."))
            elif tile == ">":
                if c - 1 < 0 or grid[r][c-1] != "<":
                    problems.append((c, f"INCOMPLETE PIPE TOP: '>' at ({r},{c}) is missing its left half '<'."))
            
            # Pipe bodies: [ must have ]
            elif tile == "[":
                if c + 1 >= w or grid[r][c+1] != "]":
                    problems.append((c, f"INCOMPLETE PIPE BODY: '[' at ({r},{c}) is missing its right half ']'."))
                if r > 0 and grid[r-1][c] not in ("<", "["):
                    problems.append((c, f"FLOATING PIPE BODY: '[' at ({r},{c}) should have '<' or '[' above it."))
            elif tile == "]":
                if c - 1 < 0 or grid[r][c-1] != "[":
                    problems.append((c, f"INCOMPLETE PIPE BODY: ']' at ({r},{c}) is missing its left half '['."))
                if r > 0 and grid[r-1][c] not in (">", "]"):
                    problems.append((c, f"FLOATING PIPE BODY: ']' at ({r},{c}) should have '>' or ']' above it."))
            
            # Cannons
            elif tile == "B":
                if r + 1 >= h or grid[r+1][c] != "b":
                    problems.append((c, f"FLOATING CANNON: 'B' at ({r},{c}) is missing its base 'b' below it."))
            elif tile == "b":
                if r - 1 < 0 or grid[r-1][c] not in ("B", "b"):
                    problems.append((c, f"MISPLACED CANNON BASE: 'b' at ({r},{c}) has no cannon 'B' above it."))

    return problems


def analyze_stuck(grid, visited, stuck_col):
    h, w = len(grid), len(grid[0])
    problems = []
    look_ahead = min(stuck_col + 12, w - 1)

    # 1. Physics/Reachability checks
    for c in range(stuck_col, look_ahead + 1):
        col_solid = [r for r in range(h) if solid(grid, r, c)]
        if len(col_solid) >= h - 1:
            problems.append(f"SOLID WALL at col {c}: Too high to jump.")
            break
        if passable(grid, h - 1, c):
            problems.append(f"PIT at col {c}: Row {h-1} must be 'X'.")

    # 2. Structural checks
    struct_probs = check_structural_integrity(grid, max(0, stuck_col - 4), look_ahead + 4)
    for _, desc in struct_probs:
        problems.append(f"STRUCTURAL ERROR: {desc}")

    if not problems:
        problems.append(f"Mario is stuck at col {stuck_col}. Add platforms or clear paths to the right.")

    return problems



SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert Super Mario Bros level designer and bug-fixer.
Your task is to fix a small LEVEL WINDOW (a text grid) so Mario can traverse it left to right.

TILE LEGEND:
X=solid ground, S=breakable block, -=empty air, ?=question block full, Q=question block empty,
E=enemy, <=pipe-top-left, >=pipe-top-right, [=pipe-left, ]=pipe-right, o=coin, B=cannon-top, b=cannon-bottom

MARIO'S CAPABILITIES:
- Max jump height: 5 tiles vertically
- Max jump width:  5 tiles horizontally
- Solid obstacles: X S ? Q < > [ ] B b E

FIXING STRATEGIES:
1. TALL WALLS  – lower wall or add 'S' step blocks so height diff ≤ 5 tiles.
2. WIDE GAPS   – place 'X' stepping stones so no gap exceeds 5 tiles.
3. BOTTOMLESS  – fill empty bottom rows with 'X'.
4. MISMATCHED TILES – Fix broken pipes (ensure < is with >, [ with ]) and cannons (B must have b below).
5. MINIMALISM  – change as few tiles as possible.
6. CONSISTENCY – Ensure the structure makes visual and logical sense.

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
        f"Mario is STUCK near the centre of this window.\n\n"
        f"PROBLEMS:\n{prob_text}\n\n"
        f"WINDOW:\n{window_str}\n\n"
        f"Fix the window so Mario can pass.\n"
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

    # Initial BFS
    yield {"log": "Running initial BFS reachability check…"}
    visited0, path0, sol0, stuck0 = bfs_reachability(grid)
    status = "SOLVABLE" if sol0 else f"NOT SOLVABLE — stuck at col {stuck0}/{w-1}"
    yield {"log": f"Initial result: {status}"}
    yield {
        "grid": grid, "visited": visited0, "path": path0,
        "solvable": sol0, "stuck_col": stuck0, "round": 0,
    }

    if sol0:
        yield {"log": "Level is already solvable — no fixes needed."}
        return

    current = copy.deepcopy(grid)
    last_stuck_col      = -1
    attempts_at_col     = 0
    fix_round           = 0

    for total_round in range(MAX_TOTAL_ROUNDS):
        visited, path, solvable, stuck_col = bfs_reachability(current)
        fix_round = total_round

        if solvable:
            # Even if solvable, check for structural errors
            integrity_probs = check_structural_integrity(current)
            if not integrity_probs:
                yield {"log": f"SOLVED and Structural Integrity Verified after {total_round} fix round(s)!"}
                yield {
                    "grid": current, "visited": visited, "path": path,
                    "solvable": True, "stuck_col": None, "round": total_round,
                }
                return
            else:
                # Pick the first structural problem to fix
                first_prob_col, _ = integrity_probs[0]
                stuck_col = first_prob_col
                yield {"log": f"Level is solvable, but found {len(integrity_probs)} structural issues. Fixing..."}


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

        # Show interim grid
        v2, p2, s2, sc2 = bfs_reachability(current)
        yield {
            "grid": current, "visited": v2, "path": p2,
            "solvable": s2, "stuck_col": sc2, "round": total_round + 1,
        }

    # Final state
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
        print("[Mamba] Initializing Mamba...")
        config = MambaConfig()
        _mamba_parser = LevelParser()
        _mamba_model = Mamba(
            num_tile_types=config.num_tile_types,
            column_height=config.column_height,
            tile_embed_dim=config.tile_embed_dim,
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dropout=0.0,
            max_seq_len=config.max_seq_len,
            columns_per_token=config.columns_per_token,
        ).to(_device)
        
        checkpoint_path = "checkpoints/mamba_best_ema.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=_device)
            # Handle both full state and state_dict only
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            _mamba_model.load_state_dict(state_dict)
            _mamba_model.eval()
            print(f"[Mamba] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[Mamba] ⚠ Checkpoint not found at {checkpoint_path}")
            
    return _mamba_model, _mamba_parser



def create_difficulty_schedule(num_columns: int, peak_difficulty: float) -> torch.Tensor:
    """
    Creates a triangular difficulty schedule: 0 -> peak -> 0.
    """
    num_columns = int(num_columns)
    mid = num_columns // 2
    # Linear ramp up to mid
    ramp_up = torch.linspace(0.0, float(peak_difficulty), int(mid))
    # Linear ramp down to end
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

    # Convert patches to columns (each patch is 16 columns)
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
    
    # Transpose to [H, W] and convert to list of lists of characters
    level_array = generated_indices.cpu().numpy().T  # [14, W]
    
    grid = []
    for r in range(level_array.shape[0]):
        row = [parser.idx_to_tile[int(tile_idx)] for tile_idx in level_array[r]]
        grid.append(row)
        
    return grid


# ══════════════════════════════════════════════════════════════════════════
#  RENDERER
# ══════════════════════════════════════════════════════════════════════════

from mario_gpt import MarioLM
from mario_gpt.utils import view_level, convert_level_to_png
mario_lm = MarioLM()



def render_level(grid, **kwargs):
    row_list = ["".join(row) for row in grid]
    img, _, _ = convert_level_to_png(row_list, mario_lm.tokenizer)
    return img


def count_tile_changes(original, fixed):
    h, w = len(original), len(original[0])
    return sum(
        1 for r in range(h) for c in range(w)
        if original[r][c] != fixed[r][c]
    )