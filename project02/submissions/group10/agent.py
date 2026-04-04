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
import random
# performance benchmark
import time

class MemoryMap:
    def __init__(self):
        self.known_walls = set()       # các ô chắc chắn là tường
        self.known_empty = set()       # các ô chắc chắn là đường đi
        self.last_seen_enemy = None    # vị trí địch lần cuối thấy
        self.step_last_seen = 0        # bước nào thấy lần cuối

    def update(self, map_state, my_pos, enemy_pos, step_number):
        # Cập nhật bản đồ đã biết
        rows, cols = map_state.shape
        for r in range(rows):
            for c in range(cols):
                if map_state[r][c] == 1:
                    self.known_walls.add((r, c))
                elif map_state[r][c] == 0:
                    self.known_empty.add((r, c))
                # map_state[r][c] == -1 -> bỏ qua, không ghi gì

        # Cập nhật vị trí địch
        if enemy_pos is not None:
            self.last_seen_enemy = enemy_pos
            self.step_last_seen = step_number
            
    def get_safe_neighbors(self, pos, map_state):
        x, y = pos
        rows, cols = map_state.shape
        neighbors = []
        for (dx, dy), move in [
            ((-1,0), Move.UP), ((1,0), Move.DOWN),
            ((0,-1), Move.LEFT), ((0,1), Move.RIGHT)
        ]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if map_state[nx][ny] == 0:   # chỉ đi vào ô đã biết là trống
                    neighbors.append(((nx, ny), move))
        return neighbors
    
class PacmanAgent(BasePacmanAgent):
    """
    Example Pacman agent using a simple greedy strategy.
    Students should implement their own search algorithms here.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Example Greedy Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Simple greedy strategy: move towards the ghost.
        
        When enemy_position is None (limited observation mode),
        uses last known position or explores randomly.
        
        Students should implement better search algorithms like:
        - BFS (Breadth-First Search)
        - DFS (Depth-First Search)
        - A* Search
        - Greedy Best-First Search
        - etc.
        """
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            # No information about enemy - explore randomly
            return self._explore(my_position, map_state)
        
        # Calculate direction to target (enemy or last known position)
        row_diff = target[0] - my_position[0]
        col_diff = target[1] - my_position[1]
        
        # List of possible moves in order of preference
        moves = []
        
        # Prioritize vertical movement if needed
        if row_diff > 0:
            moves.append(Move.DOWN)
        elif row_diff < 0:
            moves.append(Move.UP)
        
        # Prioritize horizontal movement if needed
        if col_diff > 0:
            moves.append(Move.RIGHT)
        elif col_diff < 0:
            moves.append(Move.LEFT)
        
        # Try each move in order
        for move in moves:
            desired_steps = self._desired_steps(move, row_diff, col_diff)
            steps = self._max_valid_steps(my_position, move, map_state, desired_steps)
            if steps > 0:
                return (move, steps)
        
        # If no preferred move is valid, try any valid move
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        # If no move is valid, stay
        return (Move.STAY, 1)

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Random exploration when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        return (Move.STAY, 1)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _desired_steps(self, move: Move, row_diff: int, col_diff: int) -> int:
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        if move in (Move.LEFT, Move.RIGHT):
            return abs(col_diff)
        return 1
           

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Competition Ready (Tabu Search + Minimax + Fog Evasion)
    Optimized for massive maps, zero crashes, extreme Minimax depth, and Partial Observability.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Grandmaster Ghost"
        self.pacman_speed = kwargs.get('pacman_speed', 2)
        
        if not hasattr(self, 'memory'):
            self.memory = MemoryMap() 
            
        self.tt = {}
        self.map_hash = None
        self.dead_ends = {}
        self.intersections = set()
        
        # FIX TRIỆT ĐỂ: Ghi nhớ lịch sử mọi ô đã đi qua để chống đi vòng tròn
        self.visit_count = {}

    def step(self, map_state, my_position, enemy_position, step_number):
        self.start_time = time.perf_counter()
        self.TIME_LIMIT = 0.82 
        if len(self.tt) > 50000:
            self.tt.clear()

        # Đánh dấu ô hiện tại đã được dẫm lên thêm 1 lần
        self.visit_count[my_position] = self.visit_count.get(my_position, 0) + 1

        # 1. MEMORY INTEGRATION: Update at the START of step()
        self.memory.update(map_state, my_position, enemy_position, step_number)
        
        # Determine actual threat (current or last seen)
        threat = enemy_position or self.memory.last_seen_enemy

        # 2. Topological Map Analysis (O(V) - Runs strictly ONCE per map layout)
        wall_mask = (map_state == 1).tobytes()
        current_map_hash = hash(wall_mask)
        if self.map_hash != current_map_hash:
            self._analyze_topology(map_state)
            self.map_hash = current_map_hash

        valid_moves = self._get_neighbors(my_position, map_state)
        
        # Emergency Fallback Guarantee
        if not valid_moves:
            return Move.STAY
            
        best_move = valid_moves[0][1]

        # 3. EARLY GAME FALLBACK: Handle Invisible Pacman
        if threat is None:
            return self._early_game_fallback(my_position, valid_moves, map_state)

        # 4. Dual-Root BFS (O(V) - Runs exactly ONCE per step)
        p_root_dist = self._bfs_full(threat, map_state)
        g_root_dist = self._bfs_full(my_position, map_state)

        # 5. Iterative Deepening Minimax
        depth = 1
        while True:
            if time.perf_counter() - self.start_time > self.TIME_LIMIT:
                break

            move, score = self._search_root(my_position, threat, depth, map_state, p_root_dist, g_root_dist)
            
            if time.perf_counter() - self.start_time <= self.TIME_LIMIT:
                best_move = move
                if score >= 900000 or score <= -900000:
                    break 

            depth += 1

        total_time = time.perf_counter() - self.start_time
        print(f"[Ghost] Step {step_number} | Depth: {depth-1} | Total: {total_time:.6f}s")
        
        return best_move

    def _early_game_fallback(self, my_position, valid_moves, map_state):
        """Fallback for when Pacman is completely unknown (start of game)."""
        best_move = Move.STAY
        best_score = -float('inf')
        h, w = map_state.shape
        
        for nxt, move in valid_moves:
            r, c = nxt
            rmin, rmax = max(0, r-2), min(h, r+3)
            cmin, cmax = max(0, c-2), min(w, c+3)
            
            # Đếm sương mù bằng TRÍ NHỚ
            fog_density = 0
            for rr in range(rmin, rmax):
                for cc in range(cmin, cmax):
                    if (rr, cc) not in self.memory.known_walls and (rr, cc) not in self.memory.known_empty:
                        fog_density += 1
            
            trap_depth = self.dead_ends.get(nxt, 0)
            degree = len(self._get_neighbors(nxt, map_state))
            
            score = (fog_density * 50) - (trap_depth * 10000) + (degree * 100)
            
            # TABU PENALTY: Trừ 20,000 điểm cho MỖI LẦN đã từng bước qua ô này
            score -= self.visit_count.get(nxt, 0) * 20000
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def _search_root(self, ghost_pos, pacman_pos, depth, map_state, p_root_dist, g_root_dist):
        alpha = -float('inf')
        beta = float('inf')
        best_move = Move.STAY
        best_score = -float('inf')
        
        moves = self._get_neighbors(ghost_pos, map_state)
        if not moves:
            return Move.STAY, -1000000 - depth
            
        # Root Move Ordering
        moves.sort(key=lambda x: -p_root_dist.get(x[0], 0))

        for next_pos, move in moves:
            if time.perf_counter() - self.start_time > self.TIME_LIMIT:
                break
                
            score = self._minimax(next_pos, pacman_pos, depth - 1, False, alpha, beta, map_state, p_root_dist, g_root_dist)
            
            # TABU PENALTY TẠI MINIMAX: Ép Ghost tìm lối đi mới thay vì quẩn quanh
            score -= self.visit_count.get(next_pos, 0) * 20000
            
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move, best_score

    def _minimax(self, ghost_pos, pacman_pos, depth, is_ghost, alpha, beta, map_state, p_root_dist, g_root_dist):
        if ghost_pos == pacman_pos:
            return -1000000 - depth  

        if depth == 0 or time.perf_counter() - self.start_time > self.TIME_LIMIT:
            return self._evaluate(ghost_pos, pacman_pos, p_root_dist, g_root_dist, map_state)

        tt_key = (ghost_pos, pacman_pos, is_ghost, depth)
        if tt_key in self.tt:
            entry = self.tt[tt_key]
            if entry['type'] == 'exact':
                return entry['val']
            if entry['type'] == 'lower':
                alpha = max(alpha, entry['val'])  
            if entry['type'] == 'upper':
                beta = min(beta, entry['val'])    
            if alpha >= beta:
                return entry['val']

        orig_alpha = alpha

        if is_ghost:
            best_val = -float('inf')
            moves = self._get_neighbors(ghost_pos, map_state)
            if not moves: 
                return -1000000 - depth
                
            # Move Ordering: Ghost maximizes distance from Pacman's origin
            moves.sort(key=lambda x: -p_root_dist.get(x[0], 0))
            
            for next_pos, _ in moves:
                val = self._minimax(next_pos, pacman_pos, depth - 1, False, alpha, beta, map_state, p_root_dist, g_root_dist)
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha: break
        else:
            best_val = float('inf')
            moves = self._get_pacman_next_positions(pacman_pos, map_state)
            
            moves.sort(key=lambda x: g_root_dist.get(x, 9999))
            
            for next_pos in moves:
                val = self._minimax(ghost_pos, next_pos, depth - 1, True, alpha, beta, map_state, p_root_dist, g_root_dist)
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                if beta <= alpha: break

        tt_type = 'exact'
        if best_val <= orig_alpha: tt_type = 'upper'
        elif best_val >= beta: tt_type = 'lower'
        self.tt[tt_key] = {'val': best_val, 'type': tt_type}
        
        return best_val

    def _evaluate(self, ghost_pos, pacman_pos, p_root_dist, g_root_dist, map_state):
        manhattan = self._manhattan_distance(ghost_pos, pacman_pos)
        
        if manhattan <= 1:
            return -500000

        maze_dist = p_root_dist.get(ghost_pos, 9999)
        score = (maze_dist * 100) + (manhattan * 10)
        
        if maze_dist > 8:
            r, c = ghost_pos
            h, w = map_state.shape
            rmin, rmax = max(0, r-2), min(h, r+3)
            cmin, cmax = max(0, c-2), min(w, c+3)
            
            fog_density = 0
            for rr in range(rmin, rmax):
                for cc in range(cmin, cmax):
                    if (rr, cc) not in self.memory.known_walls and (rr, cc) not in self.memory.known_empty:
                        fog_density += 1
                        
            score += (fog_density * 40)
            
        trap_depth = self.dead_ends.get(ghost_pos, 0)
        if trap_depth > 0:
            score -= (50000 - trap_depth * 100)

        if ghost_pos in self.intersections:
            score += 250  

        if manhattan < 8 and (ghost_pos[0] == pacman_pos[0] or ghost_pos[1] == pacman_pos[1]):
            score -= 3000 

        return score

    def _analyze_topology(self, map_state):
        h, w = map_state.shape
        degrees = {}
        self.intersections.clear()
        self.dead_ends.clear()

        for r in range(h):
            for c in range(w):
                if map_state[r, c] != 1:
                    deg = len(self._get_neighbors((r, c), map_state))
                    degrees[(r, c)] = deg
                    if deg >= 3:
                        self.intersections.add((r, c))
        
        dead_ends_init = {pos: 1 for pos, deg in degrees.items() if deg <= 1}
        queue = deque(dead_ends_init.keys())
        self.dead_ends = dead_ends_init.copy()
        
        while queue:
            curr = queue.popleft()
            for nxt, _ in self._get_neighbors(curr, map_state):
                if degrees.get(nxt, 0) == 2 and nxt not in self.dead_ends:
                    self.dead_ends[nxt] = self.dead_ends[curr] + 1
                    queue.append(nxt)

    def _get_pacman_next_positions(self, pacman_pos, map_state):
        positions = set()
        
        for nxt, move in self._get_neighbors(pacman_pos, map_state):
            positions.add(nxt)
            
            if self.pacman_speed > 1:
                dr, dc = move.value
                current = nxt
                for _ in range(1, self.pacman_speed):
                    candidate = (current[0] + dr, current[1] + dc)
                    if self._is_valid_position(candidate, map_state):
                        positions.add(candidate)
                        current = candidate
                    else:
                        break 
                        
        return list(positions) if positions else [pacman_pos]

    def _bfs_full(self, start, map_state):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            d = dist[pos]
            for nxt, _ in self._get_neighbors(pos, map_state):
                if nxt not in dist:
                    dist[nxt] = d + 1
                    queue.append(nxt)
        return dist

    def _manhattan_distance(self, p1: tuple, p2: tuple) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] != 1

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        x, y = pos
        h, w = map_state.shape
        neighbors = []
        if x > 0 and map_state[x-1][y] != 1: neighbors.append(((x-1, y), Move.UP))
        if x < h-1 and map_state[x+1][y] != 1: neighbors.append(((x+1, y), Move.DOWN))
        if y > 0 and map_state[x][y-1] != 1: neighbors.append(((x, y-1), Move.LEFT))
        if y < w-1 and map_state[x][y+1] != 1: neighbors.append(((x, y+1), Move.RIGHT))
        return neighbors

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)
        
    def _reconstruct_path(self, parent, start, goal):
        path = []
        cur = goal
        while cur != start:
            cur, move = parent[cur]
            path.append(move)
        return path[::-1]

    def bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        if start == goal: return []
        queue = deque([start]); visited = {start}; parent = {}
        while queue:
            current = queue.popleft()
            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)
                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)
                    queue.append(next_pos)
        return [Move.STAY]