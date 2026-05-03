import numpy as np
from typing import List, Dict


class LevelParser:
    def __init__(self):
        self.tile_to_idx = {
            "X": 0,   #solid ground
            "S": 1,   # solid breakable
            "-": 2,   # passable empty
            "?": 3,   # question block full
            "Q": 4,   # question block empty
            "E": 5,   # enemy
            "<": 6,   # top-left pipe
            ">": 7,   # top-right pipe
            "[": 8,   # left pipe
            "]": 9,   # right pipe
            "o": 10,  # coin
            "B": 11,  # cannon top
            "b": 12   # cannon bottom
        }

        self.idx_to_tile = {v: k for k, v in self.tile_to_idx.items()}

        self.num_tile_types = len(self.tile_to_idx)


    def parse_level_list(self, level_lines: List[str]) -> np.ndarray:
        parsed_rows = []
        for row in level_lines:
            parsed_row = [self.tile_to_idx[char] for char in row.strip()]
            parsed_rows.append(parsed_row)
        level_array = np.array(parsed_rows, dtype=np.int32)
        return level_array


    def parse_dataset(self, levels: List[str], level_names: List[str] = None,
                      start_x: int = 0) -> Dict:
        if level_names is None:
            level_names = [f"map_{i+1}" for i in range(len(levels))]

        parsed_data = {}

        for idx, (level, name) in enumerate(zip(levels, level_names)):
            level_array = self.parse_level_list(level)
            entry = {
                'data': level_array,
                'height': level_array.shape[0],
                'width': level_array.shape[1],
                'original': level,
                'index': idx
            }


            parsed_data[name] = entry

        return parsed_data


    def decode_level(self, level_array: np.ndarray) -> str:
        rows = []
        for row in level_array:
            row_str = ''.join([self.idx_to_tile[idx] for idx in row])
            rows.append(row_str)
        return '\n'.join(rows)

    def get_statistics(self, parsed_data: Dict) -> Dict:
        heights = [data['height'] for data in parsed_data.values()]
        widths = [data['width'] for data in parsed_data.values()]

        tile_counts = np.zeros(self.num_tile_types, dtype=np.int32)
        for data in parsed_data.values():
            unique, counts = np.unique(data['data'], return_counts=True)
            for tile_idx, count in zip(unique, counts):
                tile_counts[tile_idx] += count

        stats = {
            'num_levels': len(parsed_data),
            'height_range': (min(heights), max(heights)),
            'width_range': (min(widths), max(widths)),
            'avg_height': np.mean(heights),
            'avg_width': np.mean(widths),
            'tile_frequencies': {
                self.idx_to_tile[i]: int(count)
                for i, count in enumerate(tile_counts)
            },
            'total_tiles': int(tile_counts.sum())
        }

        return stats

    def visualize_level(self, level_array: np.ndarray, use_colors: bool = False):
            print(self.decode_level(level_array))