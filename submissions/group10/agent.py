"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
"""

import sys
from pathlib import Path
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "BFS Pacman"

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Return the Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """
        Find the shortest path from start to goal using BFS.
 
        Returns:
            List of Move enums from start to goal,
            or [Move.STAY] if no path exists.
        """
        # Each entry: (current_position, path_taken_so_far)
        queue = deque([(start, [])])
        visited = {start}
 
        while queue:
            current_pos, path = queue.popleft()
 
            # Reached the goal – return the path
            if current_pos == goal:
                return path
 
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
 
        # No path found
        return [Move.STAY]
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Ghost's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        
        # use BFS to catch the ghost by calculate the shortest path
        # returns a list of move (up, down, left, right)
        path = self.bfs(my_position, enemy_position, map_state)

        # handle edge case 1: no path found of at goal
        if not path or path == [Move.STAY]:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]: 
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)
            
        # get the first and second move from the path
        first_move = path[0]

        # handle edge case 2: ghost is next to pacman 
        if len(path) == 1:
            return (first_move, 1)

        # get the second move to check for straight line movement
        second_move = path[1]

        # if pacman move in a straight line, he can move 2 steps
        if first_move == second_move and self.pacman_speed >= 2:
            actual_steps = self._max_valid_steps(my_position, first_move, map_state, 2)
            return (first_move, actual_steps)
        
        # default to 1 step if turning of if only 1 step is valid
        return (first_move, 1)
 
    # Helper methods
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Return True if pos is inside the grid and not a wall."""
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0
 
    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Return the new position after applying a move."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)
 
    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        """Return list of (next_pos, move) for all valid moves from pos."""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors
    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Evasive Ghost"
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Return the Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Pacman's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        
        # Example: Simple evasive approach (replace with your algorithm)
        row_diff = my_position[0] - enemy_position[0]
        col_diff = my_position[1] - enemy_position[1]
        
        # Try to move away from Pacman
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        # Check if move is valid
        if self._is_valid_move(my_position, move, map_state):
            return move
        
        # If not valid, try other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move
        
        return Move.STAY
    
    # Helper methods
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Return True if pos is inside the grid and not a wall."""
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0
 
    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Return the new position after applying a move."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)
 
    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        """Return list of (next_pos, move) for all valid moves from pos."""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
