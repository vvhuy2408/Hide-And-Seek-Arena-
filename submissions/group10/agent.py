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

        # path catching
        self.current_path = []
        self.last_enemy_pos = None
        self.name = "BFS Pacman"
        
        

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
        Decide the next move for Pacman (Seeker).
        
        Strategy:
            - Use BFS to find shortest path to Ghost
            - Replan every step vì Ghost luôn di chuyển
            - Move 2 steps if going straight, 1 step if turning
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Pacman's current (row, col)
            enemy_position: Ghost's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            (Move, steps): Direction and number of steps to move
        """
        start_time = time.perf_counter()
        bfs_time = 0

        # ------- REPLANNING -------
        # Replan mỗi bước vì Ghost luôn di chuyển → path cũ luôn lỗi thời
        if not self.current_path or enemy_position != self.last_enemy_pos:
            bfs_start = time.perf_counter()
            self.current_path = self.bfs(my_position, enemy_position, map_state)
            bfs_time = time.perf_counter() - bfs_start
            self.last_enemy_pos = enemy_position

        # ------- EDGE CASE: NO PATH -------
        # Xảy ra khi Ghost bị cô lập hoàn toàn bởi wall
        # → đứng yên chờ hết max_steps, Ghost tự thắng
        if not self.current_path or self.current_path == [Move.STAY]:
            self.current_path = []
            result = (Move.STAY, 1)

        else:
            # ------- MULTI-STEP LOGIC -------
            # Rule: đi thẳng (same direction) → 2 bước
            #       quẹo (different direction) → 1 bước
            first_move = self.current_path.pop(0)

            if self.current_path and self.pacman_speed >= 2:
                second_move = self.current_path[0]  # nhìn trước move tiếp theo
                if first_move == second_move:
                    # Đi thẳng → thử đi 2 bước
                    actual_steps = self._max_valid_steps(my_position, first_move, map_state, 2)
                    if actual_steps == 2:
                        self.current_path.pop(0)  # consume thêm 1 move vì đã đi 2 bước
                        result = (first_move, 2)
                    else:
                        # Wall chặn bước 2 → chỉ đi 1
                        result = (first_move, 1)
                else:
                    # Quẹo → chỉ đi 1 bước
                    result = (first_move, 1)
            else:
                # pacman_speed = 1 hoặc path chỉ còn 1 move → đi 1 bước
                result = (first_move, 1)

        # ------- BENCHMARK -------
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
        bfs_pacman_time = 0
        bfs_ghost_time = 0

        # 1. BFS TỪ PACMAN: Tính số ô thực tế từ Pacman tới mọi vị trí
        bfs_pacman_start = time.perf_counter()
        pacman_queue = deque([(enemy_position, 0)])
        pacman_distances = {enemy_position: 0}

        while pacman_queue:
            pos, dist = pacman_queue.popleft()
            for next_pos, _ in self._get_neighbors(pos, map_state):
                if next_pos not in pacman_distances:
                    pacman_distances[next_pos] = dist + 1
                    pacman_queue.append((next_pos, dist + 1))
        bfs_pacman_time = time.perf_counter() - bfs_pacman_start

        # 2. BFS TỪ GHOST: Tìm vùng an toàn có tính đến TỐC ĐỘ X2 CỦA PACMAN
        bfs_ghost_start = time.perf_counter()
        ghost_queue = deque([my_position])
        ghost_distances = {my_position: 0}
        parent = {} 
        
        best_target = my_position
        max_dist_to_pacman = pacman_distances.get(my_position, 0)

        while ghost_queue:
            curr = ghost_queue.popleft()
            ghost_d = ghost_distances[curr]
            
            curr_pacman_d = pacman_distances.get(curr, 0)
            if curr_pacman_d > max_dist_to_pacman:
                max_dist_to_pacman = curr_pacman_d
                best_target = curr
            
            for next_pos, move in self._get_neighbors(curr, map_state):
                if next_pos not in ghost_distances:
                    next_ghost_d = ghost_d + 1
                    next_pacman_d = pacman_distances.get(next_pos, 0)
                    
                    pacman_min_time_to_reach = next_pacman_d / 2.0 
                    
                    if next_ghost_d < pacman_min_time_to_reach: 
                        ghost_distances[next_pos] = next_ghost_d
                        parent[next_pos] = (curr, move)
                        ghost_queue.append(next_pos)
        bfs_ghost_time = time.perf_counter() - bfs_ghost_start

        # 3. RA QUYẾT ĐỊNH & FALLBACK (NÉ NGÕ CỤT)
        best_move = Move.STAY
        
        if best_target != my_position:
            path = self._reconstruct_path(parent, my_position, best_target)
            if path:
                best_move = path[0]
        else:
            best_fallback_score = -float('inf')
            
            for next_pos, move in self._get_neighbors(my_position, map_state):
                p_dist = pacman_distances.get(next_pos, 0)
                score = p_dist
                
                free_neighbors = len(self._get_neighbors(next_pos, map_state))
                
                if free_neighbors <= 1:
                    score -= 100 

                if score > best_fallback_score:
                    best_fallback_score = score
                    best_move = move

        # ------- BENCHMARK -------
        total_time = time.perf_counter() - start_time
        mode = 'Escape' if best_target != my_position else 'Survival'
        print(f"[Ghost] Step {step_number} | Mode: {mode} | Target Dist: {max_dist_to_pacman} | Total: {total_time:.6f}s | BFS Pacman: {bfs_pacman_time:.6f}s | BFS Ghost: {bfs_ghost_time:.6f}s")

        return best_move
    
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
    
