import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import heapq


@dataclass
class MarioState:
    """Represents Mario's state in the level."""
    x: int
    y: int
    on_ground: bool = True
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)


class AStarAgent:
    SOLID_TILES = {0, 1, 3, 4, 6, 7, 8, 9, 11, 12}  
    PASSABLE_TILES = {2, 5, 10}  
    ENEMY_TILES = {5}  
    HAZARD_TILES = set()  
    
    MAX_JUMP_HEIGHT = 4  
    MAX_JUMP_DISTANCE = 5  
    FALL_SPEED = 1  
    
    def __init__(self, level: np.ndarray):
        self.level = level
        self.height, self.width = level.shape
        self.ground_row = self.height - 1  
        
    def is_solid(self, x: int, y: int) -> bool:
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return y >= self.height
        return int(self.level[y, x]) in self.SOLID_TILES
    
    def is_passable(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width:
            return False
        if y < 0: 
            return True
        if y >= self.height:
            return False  
        return int(self.level[y, x]) not in self.SOLID_TILES
    
    def is_enemy(self, x: int, y: int) -> bool:
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return False
        return int(self.level[y, x]) in self.ENEMY_TILES
    
    def find_ground(self, x: int, start_y: int) -> int:
        y = start_y
        while y < self.height and self.is_passable(x, y):
            y += 1
        return y - 1
    
    def can_stand(self, x: int, y: int) -> bool:
        """Check if Mario can stand at this position."""
        if not self.is_passable(x, y):
            return False
        # Must have solid ground below or be at bottom
        return self.is_solid(x, y + 1) or y == self.height - 1
    
    def get_valid_moves(self, state: MarioState) -> List[Tuple[MarioState, int]]:
        """
        Get all valid moves from current state.
        Returns list of (new_state, cost) tuples.
        """
        moves = []
        x, y = state.x, state.y
        
        # Walk right
        if self.is_passable(x + 1, y):
            new_y = self.find_ground(x + 1, y)
            if new_y < self.height:
                moves.append((MarioState(x + 1, new_y), 1))
        
        # Walk left (for backtracking if needed)
        if self.is_passable(x - 1, y):
            new_y = self.find_ground(x - 1, y)
            if new_y < self.height:
                moves.append((MarioState(x - 1, new_y), 2))  # Higher cost for going back
        
        # Jump moves (various heights and distances)
        for jump_height in range(1, self.MAX_JUMP_HEIGHT + 1):
            for jump_dist in range(1, self.MAX_JUMP_DISTANCE + 1):
                # Jump right
                new_x = x + jump_dist
                # Check if jump arc is clear
                if self._can_jump_to(x, y, new_x, jump_height):
                    new_y = self.find_ground(new_x, y - jump_height)
                    if 0 <= new_y < self.height and self.is_passable(new_x, new_y):
                        moves.append((MarioState(new_x, new_y), 1 + jump_dist))
                
                # Jump left (for platforming puzzles)
                new_x = x - jump_dist
                if self._can_jump_to(x, y, new_x, jump_height):
                    new_y = self.find_ground(new_x, y - jump_height)
                    if 0 <= new_y < self.height and self.is_passable(new_x, new_y):
                        moves.append((MarioState(new_x, new_y), 2 + jump_dist))
        
        # Jump straight up (for tight spaces)
        for jump_height in range(1, self.MAX_JUMP_HEIGHT + 1):
            if self._can_jump_up(x, y, jump_height):
                # Check if there's a platform to land on at this height
                if self.can_stand(x, y - jump_height):
                    moves.append((MarioState(x, y - jump_height), 2))
        
        return moves
    
    def _can_jump_to(self, from_x: int, from_y: int, to_x: int, jump_height: int) -> bool:
        """Check if the jump arc is clear of obstacles."""
        # Simplified: check if the peak and landing are clear
        peak_y = from_y - jump_height
        
        # Check peak position
        if not self.is_passable(from_x, peak_y):
            return False
        
        # Check landing position
        if not self.is_passable(to_x, peak_y) and not self.is_passable(to_x, peak_y + 1):
            return False
        
        # Check horizontal path at peak
        step = 1 if to_x > from_x else -1
        for check_x in range(from_x, to_x + step, step):
            if not self.is_passable(check_x, peak_y):
                return False
        
        return True
    
    def _can_jump_up(self, x: int, y: int, height: int) -> bool:
        """Check if Mario can jump straight up."""
        for h in range(1, height + 1):
            if not self.is_passable(x, y - h):
                return False
        return True
    
    def heuristic(self, state: MarioState, goal_x: int) -> float:
        """A* heuristic: horizontal distance to goal."""
        return abs(goal_x - state.x)
    
    def find_path(self, start_x: int = 0, goal_x: int = None) -> dict:
        """
        Find a path from start to goal using A*.
        
        Args:
            start_x: Starting x position (default: 0, left edge)
            goal_x: Goal x position (default: width-1, right edge)
            
        Returns:
            dict with 'playable', 'path', 'path_length', 'enemies_encountered'
        """
        if goal_x is None:
            goal_x = self.width - 1
        
        # Find starting position (ground level at start_x)
        start_y = self.find_ground(start_x, 0)
        if start_y >= self.height:
            return {'playable': False, 'reason': 'No valid start position', 
                    'path': [], 'path_length': 0, 'enemies_encountered': 0}
        
        start_state = MarioState(start_x, start_y)
        
        # A* algorithm
        open_set = [(0, start_state)]
        came_from = {}
        g_score = {start_state: 0}
        f_score = {start_state: self.heuristic(start_state, goal_x)}
        visited = set()
        enemies_hit = set()
        
        max_iterations = self.width * self.height * 10
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            # Goal reached?
            if current.x >= goal_x:
                path = self._reconstruct_path(came_from, current)
                return {
                    'playable': True,
                    'path': path,
                    'path_length': len(path),
                    'enemies_encountered': len(enemies_hit),
                    'iterations': iterations
                }
            
            if current in visited:
                continue
            visited.add(current)
            
            # Check for enemies at current position
            if self.is_enemy(current.x, current.y):
                enemies_hit.add((current.x, current.y))
            
            for next_state, move_cost in self.get_valid_moves(current):
                if next_state in visited:
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    came_from[next_state] = current
                    g_score[next_state] = tentative_g
                    f_score[next_state] = tentative_g + self.heuristic(next_state, goal_x)
                    heapq.heappush(open_set, (f_score[next_state], next_state))
        
        return {
            'playable': False,
            'reason': 'No path found',
            'path': [],
            'path_length': 0,
            'enemies_encountered': len(enemies_hit),
            'iterations': iterations
        }
    
    def _reconstruct_path(self, came_from: dict, current: MarioState) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to current."""
        path = [(current.x, current.y)]
        while current in came_from:
            current = came_from[current]
            path.append((current.x, current.y))
        return list(reversed(path))


def test_playability(level: np.ndarray) -> dict:
    """
    Convenience function to test if a level is playable.
    
    Args:
        level: 2D numpy array of tile indices
        
    Returns:
        dict with playability results
    """
    agent = AStarAgent(level)
    return agent.find_path()


def test_playability_batch(levels: List[np.ndarray]) -> dict:
    """
    Test playability for multiple levels.
    
    Args:
        levels: List of 2D numpy arrays
        
    Returns:
        dict with aggregate statistics
    """
    results = []
    for level in levels:
        results.append(test_playability(level))
    
    playable_count = sum(1 for r in results if r['playable'])
    total = len(levels)
    
    return {
        'playable_count': playable_count,
        'total': total,
        'playability_rate': playable_count / total if total > 0 else 0,
        'individual_results': results
    }
