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

# performance benchmark
import time

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
        
        """ # path catching
        self.current_path = []
        self.last_target_pos = None """

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Return the Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        if start == goal:
            return []
        queue = deque([start])
        visited = {start}
        
        # generate a parent map to store 
        parent = {} # next_pos -> (current_pos, move)

        while queue:
            current = queue.popleft()
            
            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)
                    # early stop
                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)
                    
                    queue.append(next_pos)

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
        
        # performance benchmark: 
        """ NOTE: Change from return in every step -> return one final result
                to easier for calculating time
        """
        start_time = time.perf_counter()
        bfs_time = 0 
        
        # ------- starting code -------
        # to catch the ghost when pacman stand beside
        if self._manhattan_distance(my_position, enemy_position) <= 1:
            result = (Move.STAY, 1)
        
        else: 
            bfs_start = time.perf_counter()
            path = self.bfs(my_position, enemy_position, map_state)
            bfs_time = time.perf_counter() - bfs_start
            
            # handle edge case 1: no path found of at goal
            if not path or path == [Move.STAY]:
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]: 
                    if self._is_valid_move(my_position, move, map_state):
                        result = (move, 1)
                        break
                else: 
                    result = (Move.STAY, 1)
            
            else: 
                # get the first and second move from the path
                first_move = path[0]

                # handle edge case 2: ghost is next to pacman 
                if len(path) == 1:
                    result = (first_move, 1)

                # get the second move to check for straight line movement
                if len(path) >= 2:
                    second_move = path[1]
                else:
                    second_move = None

                # if pacman move in a straight line, he can move 2 steps
                if first_move == second_move and self.pacman_speed >= 2:
                    actual_steps = self._max_valid_steps(my_position, first_move, map_state, self.pacman_speed)
                    result = (first_move, actual_steps)
                else: 
                    result = (first_move, 1)
                    # default to 1 step if turning of if only 1 step is valid
        
        # ----- ending code -----
        
        total_time = time.perf_counter() - start_time
        print(f"[Pacman] Step {step_number} | Total: {total_time:.6f}s | BFS: {bfs_time:.6f}s")
        
        return result
 
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
        # use inline to maximize performance
        x, y = pos
        rows, cols = map_state.shape

        neighbors = []

        if x > 0 and map_state[x-1][y] != 1:
            neighbors.append(((x-1, y), Move.UP))

        if x < rows-1 and map_state[x+1][y] != 1:
            neighbors.append(((x+1, y), Move.DOWN))

        if y > 0 and map_state[x][y-1] != 1:
            neighbors.append(((x, y-1), Move.LEFT))

        if y < cols-1 and map_state[x][y+1] != 1:
            neighbors.append(((x, y+1), Move.RIGHT))

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
    
    def _reconstruct_path(self, parent, start, goal):
        path = []
        cur = goal

        while cur != start:
            cur, move = parent[cur]
            path.append(move)

        return path[::-1]
            


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Evasive Ghost"
        
    def bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """
        Find the shortest path from start to goal using BFS.
 
        Returns:
            List of Move enums from start to goal,
            or [Move.STAY] if no path exists.
        """
        if start == goal:
            return []
        
        # Each entry: (current_position, path_taken_so_far)
        queue = deque([start])
        visited = {start}
        parent = {}
 
        while queue:
            current = queue.popleft()
 
            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)

                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)
                    queue.append(next_pos)
 
        # No path found
        return [Move.STAY]
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Return the Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def step(self, map_state, my_position, enemy_position, step_number):
        start_time = time.perf_counter()
        bfs1_start = time.perf_counter()
    
        # BFS of Pacman to calculate its distance to other positions
        queue = deque([(enemy_position, 0)])
        visited = {enemy_position: 0}

        while queue:
            pos, dist = queue.popleft()
            for next_pos, _ in self._get_neighbors(pos, map_state):
                if next_pos not in visited:
                    visited[next_pos] = dist + 1
                    queue.append((next_pos, dist + 1))
                    
        bfs1_time = time.perf_counter() - bfs1_start

        # Finding farthest position that Ghost can reach 
        farthest_pos = my_position
        max_dist = visited.get(my_position, 0)

        for pos, dist in visited.items():
            if dist > max_dist:
                max_dist = dist
                farthest_pos = pos
                
        bfs2_start = time.perf_counter()
        
        # BFS path fron Ghost to that position
        path = self.bfs(my_position, farthest_pos, map_state)
        
        bfs2_time = time.perf_counter() - bfs2_start

        if not path or path[0] == Move.STAY:
            result = Move.STAY
        else:
            result = path[0]
            
        total_time = time.perf_counter() - start_time
        print(f"[Ghost] Step {step_number} | Total: {total_time:.6f}s | BFS1: {bfs1_time:.6f}s | BFS2: {bfs2_time:.6f}s")
        
        return result
    
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
        # use inline to maximize performance
        x, y = pos
        rows, cols = map_state.shape

        neighbors = []

        if x > 0 and map_state[x-1][y] != 1:
            neighbors.append(((x-1, y), Move.UP))

        if x < rows-1 and map_state[x+1][y] != 1:
            neighbors.append(((x+1, y), Move.DOWN))

        if y > 0 and map_state[x][y-1] != 1:
            neighbors.append(((x, y-1), Move.LEFT))

        if y < cols-1 and map_state[x][y+1] != 1:
            neighbors.append(((x, y+1), Move.RIGHT))

        return neighbors
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _reconstruct_path(self, parent, start, goal):
        path = []
        cur = goal

        while cur != start:
            cur, move = parent[cur]
            path.append(move)

        return path[::-1]
    
