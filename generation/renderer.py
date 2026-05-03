import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional


TILE_COLORS = {
    0:  (139,  90,  43),   # X  - solid ground (brown)
    1:  (185, 122,  55),   # S  - breakable brick (light brown)
    2:  (107, 170, 228),   # -  - empty / sky (sky blue)
    3:  (255, 200,  37),   # ?  - question block full (yellow)
    4:  (180, 140,  30),   # Q  - question block empty (dark yellow)
    5:  (210,  50,  50),   # E  - enemy (red)
    6:  ( 50, 180,  50),   # <  - top-left pipe (green)
    7:  ( 50, 180,  50),   # >  - top-right pipe (green)
    8:  ( 34, 130,  34),   # [  - left pipe body (dark green)
    9:  ( 34, 130,  34),   # ]  - right pipe body (dark green)
    10: (255, 215,   0),   # o  - coin (gold)
    11: ( 80,  80,  80),   # B  - cannon top (dark gray)
    12: (120, 120, 120),   # b  - cannon bottom (gray)
}

DEFAULT_COLOR = (200, 200, 200)  # fallback for unknown tiles


def render_level_to_image(
    level: np.ndarray,
    tile_size: int = 16,
    grid_lines: bool = False,
    grid_color: tuple = (60, 60, 60),
) -> Image.Image:
    height, width = level.shape
    img_width = width * tile_size
    img_height = height * tile_size

    img = Image.new('RGB', (img_width, img_height), color=(107, 170, 228))
    draw = ImageDraw.Draw(img)

    for row in range(height):
        for col in range(width):
            tile_id = int(level[row, col])
            color = TILE_COLORS.get(tile_id, DEFAULT_COLOR)

            x0 = col * tile_size
            y0 = row * tile_size
            x1 = x0 + tile_size - 1
            y1 = y0 + tile_size - 1

            draw.rectangle([x0, y0, x1, y1], fill=color)

            # Add subtle inner details for certain tiles
            if tile_id == 3:  # Question block - draw "?" mark
                cx, cy = x0 + tile_size // 2, y0 + tile_size // 2
                r = max(tile_size // 6, 2)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 160, 20))
            elif tile_id == 5:  # Enemy - draw eyes
                ey = y0 + tile_size // 3
                draw.rectangle([x0 + 3, ey, x0 + 5, ey + 2], fill=(255, 255, 255))
                draw.rectangle([x1 - 5, ey, x1 - 3, ey + 2], fill=(255, 255, 255))
            elif tile_id == 10:  # Coin - draw circle
                cx, cy = x0 + tile_size // 2, y0 + tile_size // 2
                r = max(tile_size // 3, 3)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 235, 60))

    if grid_lines:
        for col in range(1, width):
            x = col * tile_size
            draw.line([(x, 0), (x, img_height)], fill=grid_color, width=1)
        for row in range(1, height):
            y = row * tile_size
            draw.line([(0, y), (img_width, y)], fill=grid_color, width=1)

    return img


def save_level_image(
    level: np.ndarray,
    path: str,
    tile_size: int = 16,
    grid_lines: bool = False,
) -> None:
    img = render_level_to_image(level, tile_size=tile_size, grid_lines=grid_lines)
    img.save(path)
