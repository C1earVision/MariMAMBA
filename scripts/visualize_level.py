"""Visualize a generated Mario level using MarioGPT's sprite renderer."""
import sys
from mario_gpt import MarioLM
from mario_gpt.utils import convert_level_to_png

level_path = sys.argv[1] if len(sys.argv) > 1 else "output/generated_levels/column_level_1.txt"
output_path = sys.argv[2] if len(sys.argv) > 2 else "output/generated_levels/column_level_1_rendered.png"

# Read level text
with open(level_path, "r") as f:
    level_lines = [line.rstrip("\r\n") for line in f.readlines()]

print(f"Loaded level: {len(level_lines)} rows x {len(level_lines[0])} cols")

# MarioLM is needed just for its tokenizer
mario_lm = MarioLM()

# Convert to sprite-based PNG
img, _, _ = convert_level_to_png(level_lines, mario_lm.tokenizer)
img.save(output_path)
print(f"Rendered image saved to: {output_path}")
