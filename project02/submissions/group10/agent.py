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
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.memory = MemoryMap() # memory map for project 02
        # path catching
        self.current_path = []
        self.last_enemy_pos = None
        self.dist_map = {}
        self.name = "BFS Pacman"

    def _bfs_distance_map(self, start, map_state):
        queue = deque([(start, 0)])
        dist = {start: 0}

        while queue:
            pos, d = queue.popleft()
            for nxt, _ in self._get_neighbors(pos, map_state):
                if nxt not in dist:
                    dist[nxt] = d + 1
                    queue.append((nxt, d + 1))
        
        return dist

    def bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        if start == goal:
            return []
        queue = deque([start])
        visited = {start}
        
        # generate a parent map to store 
        parent = {} # next_pos -> (current_pos, move)

        while queue:
            current = queue.popleft()
            
            # Get list of neighbors
            neighbors = self._get_neighbors(current, map_state)

            # --- TIE-BREAKING LOGIC: Prioritize unchanged direction ---
            if current in parent:
                _, prev_move = parent[current]
                # Put direction matching previous direction at the start of traversal list
                neighbors.sort(key=lambda x: x[1] != prev_move)

            for next_pos, move in neighbors:
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)
                    # early stop
                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)
                    queue.append(next_pos)

        return []
    
    def predict_enemy_move(self, enemy_pos, my_pos, map_state, steps=2):
        """Predict where Ghost will be after N steps."""
        current = enemy_pos
        for _ in range(steps):
            best_move = Move.STAY
            best_dist = -1
            for next_pos, move in self._get_neighbors(current, map_state):
                dist = self.dist_map.get(next_pos, 0)
                if dist > best_dist:
                    best_dist = dist
                    best_move = move
                    best_next = next_pos
            current = best_next if best_dist > -1 else current
        return current
    
    def step(self, map_state: np.ndarray, 
         my_position: tuple, 
         enemy_position: tuple,
         step_number: int):
        """
        Decide the next move for Pacman (Seeker).
        
        Strategy:
            - Use BFS to find shortest path to Ghost
            - Replan every step because Ghost always moves
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

        # ------- DISTANCE MAP -------
        self.dist_map = self._bfs_distance_map(enemy_position, map_state)

        # ------- GET TARGET -------
        dist_to_ghost = self.dist_map.get(enemy_position, float('inf'))
                    
        # If ghost is 2 steps away -> aim straight at ghost
        if dist_to_ghost <= self.pacman_speed:
            target_pos = enemy_position
        else:
            # ------- PREDICTIVE -------
            # Predict ghost position, aim at predicted position
            target_pos = self.predict_enemy_move(enemy_position, my_position, map_state)

        # ------- REPLANNING -------
        # Replan every step because Ghost always moves -> old path always becomes outdated
        if not self.current_path or self.last_enemy_pos != enemy_position:
            bfs_start = time.perf_counter()  # <- added
            self.current_path = self.bfs(my_position, target_pos, map_state)
            bfs_time = time.perf_counter() - bfs_start  # <- added
            self.last_enemy_pos = enemy_position        
        else:
            bfs_time = 0

        # ------- EDGE CASE: NO PATH -------
        # Occurs when Ghost is completely isolated by walls
        # -> stay still wait for max_steps, Ghost wins automatically
        if not self.current_path or self.current_path == [Move.STAY]:
            # Choose direction with most exits instead of staying still (Move.STAY)
            best_fallback = Move.STAY
            best_score = -1

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_p = self._apply_move(my_position, move)
                if self._is_valid_position(next_p, map_state):
                    # Heuristic avoid dead-end
                    freedom = len(self._get_neighbors(next_p, map_state))
                    ghost_dist = self.dist_map.get(next_p, 0)
                    score = ghost_dist + 0.5 * freedom
                    
                    if score > best_score:
                        best_score = score
                        best_fallback = move

            result = (best_fallback, 1)

        # ------- MULTI-STEP LOGIC -------
        # Rule: go straight (same direction) -> 2 steps
        #       turn (different direction) -> 1 step
        else:
            first_move = self.current_path.pop(0)
            steps_to_move = 1

            if self.current_path and self.pacman_speed >= 2:
                second_move = self.current_path[0]
                if second_move == first_move:
                    actual_steps = self._max_valid_steps(my_position, first_move, map_state, 2)
                    if actual_steps == 2:
                        self.current_path.pop(0)
                        steps_to_move = 2
                
            result = (first_move, steps_to_move)

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
    Ghost (Hider) Agent - Strategic Relocation Edition
    Features: Memory Integration, Global Fog Targeting, Minimax Evasion, Zero Ping-Pong.
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
        
        # MỤC TIÊU KHÁM PHÁ TOÀN CỤC
        self.exploration_target = None

    def step(self, map_state, my_position, enemy_position, step_number):
        self.start_time = time.perf_counter()
        self.TIME_LIMIT = 0.82 
        if len(self.tt) > 50000:
            self.tt.clear()

        # 1. CẬP NHẬT BỘ NHỚ
        self.memory.update(map_state, my_position, enemy_position, step_number)
        threat = enemy_position or self.memory.last_seen_enemy

        # 2. PHÂN TÍCH ĐỊA HÌNH
        wall_mask = (map_state == 1).tobytes()
        current_map_hash = hash(wall_mask)
        if self.map_hash != current_map_hash:
            self._analyze_topology(map_state)
            self.map_hash = current_map_hash

        valid_moves = self._get_neighbors(my_position, map_state)
        
        # Emergency Fallback Guarantee
        if not valid_moves:
            return Move.STAY

        p_root_dist = self._bfs_full(threat, map_state) if threat else {}
        g_root_dist = self._bfs_full(my_position, map_state)
        
        # Kiểm tra khoảng cách nguy hiểm
        danger_dist = p_root_dist.get(my_position, 9999)

        # 3. QUYẾT ĐỊNH CHẾ ĐỘ HOẠT ĐỘNG (Relocate vs Evade)
        # NẾU an toàn (> 8 bước) HOẶC chưa từng thấy Pacman -> Chế độ Chủ động tẩu thoát vào sương mù
        if threat is None or danger_dist > 8:
            
            # Reset target nếu đã đến nơi
            if self.exploration_target == my_position:
                self.exploration_target = None
                
            # Tìm một target sương mù toàn cục mới
            if not self.exploration_target:
                self.exploration_target = self._find_best_fog_target(my_position, threat, map_state, g_root_dist, p_root_dist)
                
            # Đi theo BFS đến target (Kiên định, không ping-pong)
            if self.exploration_target:
                path = self.bfs(my_position, self.exploration_target, map_state)
                if path and path[0] != Move.STAY:
                    total_time = time.perf_counter() - self.start_time
                    print(f"[Ghost] Step {step_number} | RELOCATING to Fog {self.exploration_target} | Time: {total_time:.4f}s")
                    return path[0]

            # [FIX LỖI CRASH TẠI ĐÂY]: Nếu không tìm được đường tới target mà threat vẫn = None
            # Ta KHÔNG được chạy Minimax (vì không có vị trí Pacman). Thay vào đó, chọn hướng an toàn nhất.
            if threat is None:
                best_fallback = Move.STAY
                best_deg = -1
                for nxt, move in valid_moves:
                    deg = len(self._get_neighbors(nxt, map_state))
                    if deg > best_deg:
                        best_deg = deg
                        best_fallback = move
                total_time = time.perf_counter() - self.start_time
                print(f"[Ghost] Step {step_number} | FALLBACK (no threat) | Time: {total_time:.4f}s")
                return best_fallback

        # 4. KÍCH HOẠT MINIMAX (Khi nguy hiểm <= 8 bước)
        # Hủy mục tiêu khám phá vì phải lo giữ mạng
        self.exploration_target = None 
        
        best_move = valid_moves[0][1]
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
        print(f"[Ghost] Step {step_number} | EVADING Minimax Depth {depth-1} | Time: {total_time:.4f}s")
        return best_move

    def _find_best_fog_target(self, my_pos, threat_pos, map_state, g_dist, p_dist):
        """Tìm 1 ô sương mù AN TOÀN và XA PACMAN nhất trên toàn bản đồ làm đích đến."""
        h, w = map_state.shape
        best_target = None
        best_score = -float('inf')

        for r in range(h):
            for c in range(w):
                if map_state[r, c] == 1: continue # Bỏ qua tường
                if (r, c) in self.dead_ends: continue # Tuyệt đối không chọn đích là ngõ cụt

                # Định nghĩa Sương mù: Những ô chưa từng lưu vào known_empty của bộ nhớ
                is_fog = (r, c) not in self.memory.known_empty and (r, c) not in self.memory.known_walls

                if is_fog:
                    dist_from_me = g_dist.get((r, c), 9999)
                    if dist_from_me < 9999: # Đảm bảo đường đi tới đó không bị kẹt
                        # Tính điểm: Càng xa Pacman càng tốt, nhưng ưu tiên điểm gần Ghost để dễ tới
                        dist_from_threat = p_dist.get((r, c), 0) if threat_pos else 0
                        score = (dist_from_threat * 10) - dist_from_me
                        
                        if score > best_score:
                            best_score = score
                            best_target = (r, c)
                            
        # Nếu đã mở sáng toàn bộ bản đồ (hết sương mù), chọn điểm xa Pacman nhất làm target
        if not best_target and threat_pos:
            for r in range(h):
                for c in range(w):
                    if map_state[r, c] == 1 or (r, c) in self.dead_ends: continue
                    dist_from_me = g_dist.get((r, c), 9999)
                    if dist_from_me < 9999:
                        score = p_dist.get((r, c), 0) * 10 - dist_from_me
                        if score > best_score:
                            best_score = score
                            best_target = (r, c)

        return best_target

    def _search_root(self, ghost_pos, pacman_pos, depth, map_state, p_root_dist, g_root_dist):
        alpha = -float('inf')
        beta = float('inf')
        best_move = Move.STAY
        best_score = -float('inf')
        
        moves = self._get_neighbors(ghost_pos, map_state)
        if not moves:
            return Move.STAY, -1000000 - depth
            
        # Root Move Ordering: Ghost wants to MAXIMIZE distance from Pacman's root
        moves.sort(key=lambda x: -p_root_dist.get(x[0], 0))

        for next_pos, move in moves:
            if time.perf_counter() - self.start_time > self.TIME_LIMIT:
                break
                
            score = self._minimax(next_pos, pacman_pos, depth - 1, False, alpha, beta, map_state, p_root_dist, g_root_dist)
            
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move, best_score

    def _minimax(self, ghost_pos, pacman_pos, depth, is_ghost, alpha, beta, map_state, p_root_dist, g_root_dist):
        # Terminal State: Ghost Caught
        if ghost_pos == pacman_pos:
            return -1000000 - depth  

        # Horizon Reached or Time Out Triggered
        if depth == 0 or time.perf_counter() - self.start_time > self.TIME_LIMIT:
            return self._evaluate(ghost_pos, pacman_pos, p_root_dist, g_root_dist, map_state)

        # Transposition Table Lookup
        tt_key = (ghost_pos, pacman_pos, is_ghost, depth)
        if tt_key in self.tt:
            entry = self.tt[tt_key]
            if entry['type'] == 'exact': return entry['val']
            if entry['type'] == 'lower': alpha = max(alpha, entry['val'])  
            if entry['type'] == 'upper': beta = min(beta, entry['val'])    
            if alpha >= beta: return entry['val']

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

        # Transposition Table Store
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
        
        trap_depth = self.dead_ends.get(ghost_pos, 0)
        if trap_depth > 0:
            score -= (50000 - trap_depth * 100)

        if ghost_pos in self.intersections:
            score += 250  

        if manhattan < 8 and (ghost_pos[0] == pacman_pos[0] or ghost_pos[1] == pacman_pos[1]):
            score -= 3000 

        return score

    def _analyze_topology(self, map_state):
        """
        O(V) single-pass map analysis.
        Identifies dead-ends, their depths, and vital intersections.
        """
        h, w = map_state.shape
        degrees = {}
        self.intersections.clear()
        self.dead_ends.clear()

        # Map degrees and intersections
        for r in range(h):
            for c in range(w):
                if map_state[r, c] != 1:
                    deg = len(self._get_neighbors((r, c), map_state))
                    degrees[(r, c)] = deg
                    if deg >= 3:
                        self.intersections.add((r, c))
        
        # Propagate dead-ends
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
    
    