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

class MemoryMap:
    def __init__(self):
        self.known_walls = set()       # các ô chắc chắn là tường
        self.known_empty = set()       # các ô chắc chắn là đường đi
        self.last_seen_enemy = None    # vị trí địch lần cuối thấy
        self.step_last_seen = 0        # bước nào thấy lần cuối
        self.unknown_frontier = set()  # biên giới chưa khám phá

    def update(self, map_state, my_pos, enemy_pos, step_number):
        # Cập nhật bản đồ đã biết - OPTIMIZED: only scan 5x5 local area
        rows, cols = map_state.shape
        newly_found = set()
        
        # Only check 5x5 area around my_pos (MUCH faster than full scan)
        if map_state[my_pos[0], my_pos[1]] == 0:
            self.known_empty.add(my_pos)
        
        for r in range(max(0, my_pos[0]-5), min(rows, my_pos[0]+6)):
            for c in range(max(0, my_pos[1]-5), min(cols, my_pos[1]+6)):
                if map_state[r][c] == 1:
                    self.known_walls.add((r, c))
                    self.unknown_frontier.discard((r, c))
                elif map_state[r][c] == 0:
                    if (r, c) not in self.known_empty:
                        newly_found.add((r, c))
                    self.known_empty.add((r, c))
                    self.unknown_frontier.discard((r, c))

        # Cập nhật vị trí địch
        if enemy_pos is not None:
            self.last_seen_enemy = enemy_pos
            self.step_last_seen = step_number
            
        # Cập nhật biên giới khám phá (neighbor của known_empty là unknown)
        # self.unknown_frontier.clear()
        for pos in newly_found:
            x, y = pos
            for (dx, dy) in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if map_state[nx][ny] == -1:  # ô chưa biết
                        self.unknown_frontier.add((nx, ny))
            
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
    
    def get_exploration_target(self, my_pos, map_state):
        """Tìm mục tiêu khám phá gần nhất trong unknown_frontier."""
        if not self.unknown_frontier:
            # Nếu không có frontier, dùng BFS để tìm ô trống bất kỳ
            return my_pos
        
        # Tìm ô unknown gần nhất bằng Manhattan distance
        best_target = min(
            self.unknown_frontier, 
            key=lambda p: abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1])
        )
        
        return best_target
    
class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory = MemoryMap()  # Khởi tạo memory ngay dòng đầu tiên
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # path catching
        self.capture_distance = int(kwargs.get("capture_distance", 1))
        
        self.current_path = []
        self.last_enemy_pos = None 
        
        self.locked_target = None
        self.target_lock_timer = 0       
        
        self.name = "BFS Pacman"
    
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

        # ------- UPDATE MEMORY -------
        self.memory.update(map_state, my_position, enemy_position, step_number)
        
        # ------- INSTANT CAPTURE -------
        if enemy_position is not None:
            dist = abs(enemy_position[0] - my_position[0]) + abs(enemy_position[1] - my_position[1])
            if dist <= self.capture_distance:
                for nxt, move in self.memory.get_safe_neighbors(my_position, map_state):
                    if nxt == enemy_position:
                        return (move, 1)
            
        # ------- INIT LOCK (chỉ chạy 1 lần) -------
        if not hasattr(self, "locked_target"):
            self.locked_target = None
            self.target_lock_timer = 0

        # ------- DISTANCE MAP -------
        source = my_position
        self.dist_map = self._bfs_distance_map(source, map_state)

        # ------- GET TARGET -------
        # priority: enemy_position > last_seen_enemy > exploration_target
        if enemy_position is not None:
            predicted = self.predict_enemy_move(enemy_position, my_position, map_state, steps=2)
            new_target = self._get_capture_target(my_position, predicted, map_state)
        elif self.memory.last_seen_enemy is not None:
            new_target = self.memory.last_seen_enemy
        else:
            new_target = self.memory.get_exploration_target(my_position, map_state)

        if new_target is None:
            new_target = my_position

        # ------- TARGET LOCK -------
        if self.locked_target is None or self.target_lock_timer <= 0:
            self.locked_target = new_target
            self.target_lock_timer = 3  # giữ target trong 3 step
        else:
            self.target_lock_timer -= 1

        target_pos = self.locked_target

        # ------- REPLANNING -------
        # Replan every step because Ghost always moves -> old path always becomes outdated
        need_replan = False

        if not self.current_path:
            need_replan = True
        elif self.last_enemy_pos != enemy_position:
            need_replan = True
        else:
            # giảm tần suất replan để tránh rung
            if step_number % 2 == 0:
                need_replan = True

        if need_replan:
            bfs_start = time.perf_counter()
            new_path = self.bfs(my_position, target_pos, map_state)
            bfs_time += time.perf_counter() - bfs_start

            # chỉ update nếu path hợp lệ → chống rung
            if new_path:
            # ------- TIE BREAK FIX -------
                if self.current_path:
                    old_move = self.current_path[0]
                    if new_path and new_path[0] == old_move:
                        # giữ hướng cũ → mượt hơn
                        self.current_path = new_path
                    else:
                        # nếu đổi hướng, chỉ đổi khi thực sự tốt hơn
                        # OPTIMIZED: inline score calculation (no nested function)
                        new_p = self._apply_move(my_position, new_path[0])
                        old_p = self._apply_move(my_position, old_move)
                        new_score = -abs(new_p[0] - target_pos[0]) - abs(new_p[1] - target_pos[1])
                        old_score = -abs(old_p[0] - target_pos[0]) - abs(old_p[1] - target_pos[1])
                        if new_score >= old_score:
                            self.current_path = new_path
                else:
                    self.current_path = new_path

            if enemy_position is not None:
                self.last_enemy_pos = enemy_position        

        # ------- EDGE CASE: NO PATH -------
        # Occurs when Ghost is completely isolated by walls, or target is in unknown area
        # -> try to explore or stay still, Ghost wins automatically
        if not self.current_path:
            best_move = Move.STAY
            best_score = -float('inf')
        
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_p = self._apply_move(my_position, move)

                if self._is_valid_position(next_p, map_state):
                    # Ưu tiên các ô còn chưa khám phá
                    freedom = len(self.memory.get_safe_neighbors(next_p, map_state))
                    ghost_dist = self.dist_map.get(next_p, float('inf'))

                    score = -ghost_dist + 0.5 * freedom
                    
                    if score > best_score:
                        best_score = score
                        best_move = move

            result = (best_move, 1)

        # ------- MULTI-STEP LOGIC -------
        # Rule: go straight (same direction) -> 2 steps
        #       turn (different direction) -> 1 step
        else:
            first_move = self.current_path.pop(0)
            
            if not self._is_valid_move(my_position, first_move, map_state):
                neighbors = self.memory.get_safe_neighbors(my_position, map_state)
                if neighbors:
                    return (neighbors[0][1], 1)
                return (Move.STAY, 1)
        
            steps_to_move = 1

            if self.current_path and self.pacman_speed >= 2:
                second_move = self.current_path[0]

                if second_move == first_move:
                    next_pos = self._apply_move(my_position, first_move)
                    next2 = self._apply_move(next_pos, first_move)
                    
                    if self._is_valid_position(next_pos, map_state) and \
                       self._is_valid_position(next2, map_state):
                        steps_to_move = 2
                        self.current_path.pop(0)
            
            result = (first_move, steps_to_move)

        # ------- SAFETY (ANTI-CRASH) -------
        if result is None:
            result = (Move.STAY, 1)

        # ------- BENCHMARK -------
        total_time = time.perf_counter() - start_time
        print(f"[Pacman] Step {step_number} | Total: {total_time:.6f}s | BFS: {bfs_time:.6f}s | Target: {target_pos}")

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
            prev, move = parent[cur]
            path.append(move)
            cur = prev

        return path[::-1]
    
    def _bfs_distance_map(self, start, map_state):
        queue = deque([(start, 0)])
        dist = {start: 0}

        while queue:
            pos, d = queue.popleft()
            # dùng get_safe_neighbors để dist_map phản ánh đúng vùng đã biết
            for nxt, _ in self.memory.get_safe_neighbors(pos, map_state):
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
            
            # Get list of neighbors - chỉ đi qua ô đã biết là đường (get_safe_neighbors)
            neighbors = self.memory.get_safe_neighbors(current, map_state)

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
    
    def _get_capture_target(self, my_pos: tuple, enemy_pos: tuple, map_state: np.ndarray) -> tuple:
        if self.capture_distance <= 1:
            return enemy_pos
 
        capture_zone = {enemy_pos}
        queue = deque([(enemy_pos, 0)])
        visited = {enemy_pos}
 
        while queue:
            pos, d = queue.popleft()
            if d >= self.capture_distance:
                continue
            for nxt, _ in self._get_neighbors(pos, map_state):
                if nxt not in visited:
                    visited.add(nxt)
                    capture_zone.add(nxt)
                    queue.append((nxt, d + 1))
 
        return min(capture_zone,
                   key=lambda p: abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1]))
    
    def predict_enemy_move(self, enemy_pos, my_pos, map_state, steps=2):
        """Predict where Ghost will be after N steps."""
        current = enemy_pos
        for _ in range(steps):
            best_move = Move.STAY
            best_dist = -1
            best_next = current  # Initialize for safety
            for next_pos, move in self._get_neighbors(current, map_state):
                dist = self.dist_map.get(next_pos, 0)
                if dist > best_dist:
                    best_dist = dist
                    best_move = move
                    best_next = next_pos
            current = best_next if best_dist > -1 else current
        return current


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Competition Ready (Dual-Root BFS + Topological Cache)
    Optimized for massive maps, zero crashes, and extreme Minimax depth.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Grandmaster Ghost"
        self.pacman_speed = kwargs.get('pacman_speed', 2)
        self.memory = MemoryMap() # memory map for project 02
        self.tt = {}
        self.map_hash = None
        self.dead_ends = {}
        self.intersections = set()

    def step(self, map_state, my_position, enemy_position, step_number):
        self.start_time = time.perf_counter()
        self.TIME_LIMIT = 0.82 
        if len(self.tt) > 50000:
            self.tt.clear()

        # 1. Topological Map Analysis (O(V) - Runs strictly ONCE per map layout)
        wall_mask = (map_state == 1).tobytes()
        current_map_hash = hash(wall_mask)
        if self.map_hash != current_map_hash:
            self._analyze_topology(map_state)
            self.map_hash = current_map_hash

        # 2. Dual-Root BFS (O(V) - Runs exactly ONCE per step)
        # Provides perfect distance gradients for Alpha-Beta move ordering
        p_root_dist = self._bfs_full(enemy_position, map_state)
        g_root_dist = self._bfs_full(my_position, map_state)

        # 3. Emergency Fallback Guarantee
        valid_moves = self._get_neighbors(my_position, map_state)
        if not valid_moves:
            return Move.STAY
        
        best_move = valid_moves[0][1]

        # 4. Iterative Deepening Minimax
        depth = 1
        while True:
            if time.perf_counter() - self.start_time > self.TIME_LIMIT:
                break

            move, score = self._search_root(my_position, enemy_position, depth, map_state, p_root_dist, g_root_dist)
            
            # Commit the move ONLY if the depth search completed within the time limit
            if time.perf_counter() - self.start_time <= self.TIME_LIMIT:
                best_move = move
                # Early termination if a guaranteed win (survival) or forced loss is found
                if score >= 900000 or score <= -900000:
                    break 

            depth += 1

        total_time = time.perf_counter() - self.start_time
        print(f"[Ghost] Step {step_number} | Total: {total_time:.6f}s")
        
        return best_move

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
            return self._evaluate(ghost_pos, pacman_pos, p_root_dist, g_root_dist)

        # Transposition Table Lookup
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
            
            # Move Ordering: Pacman minimizes distance to Ghost's origin
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

    def _evaluate(self, ghost_pos, pacman_pos, p_root_dist, g_root_dist):
        """
        O(1) Leaf Evaluation. 
        Pure math and dictionary lookups. No loops, no arrays, no BFS.
        """
        manhattan = self._manhattan_distance(ghost_pos, pacman_pos)
        
        # Absolute danger
        if manhattan <= 1:
            return -500000

        # Base strategy: Rely on accurate maze distance from Pacman's origin step
        maze_dist = p_root_dist.get(ghost_pos, 9999)
        
        # Blend Exact Maze Distance and Immediate Manhattan threat
        score = (maze_dist * 100) + (manhattan * 10)
        
        # Voronoi: count cells Ghost controls (reach before Pacman)
        ghost_territory = sum(
            1 for pos, g_d in g_root_dist.items()
            if g_d < p_root_dist.get(pos, 9999)
        )
        score += ghost_territory * 20  # reward control more cells

        # 1. Topological Threat: Massive penalty for dead-ends
        trap_depth = self.dead_ends.get(ghost_pos, 0)
        if trap_depth > 0:
            # Pushes the ghost toward the exit of the dead-end smoothly
            score -= (50000 - trap_depth * 100)

        # 2. Mobility Control: Reward intersections (Voronoi substitute)
        if ghost_pos in self.intersections:
            score += 250  

        # 3. Line-of-Sight (LoS) Raycast Proxy: Break straight speedways
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
                if map_state[r, c] == 0:
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
        """
        Accurate Pacman Speed Simulation.
        Projects straight-line movements up to pacman_speed.
        """
        positions = set()
        
        # Base 1-step moves
        for nxt, move in self._get_neighbors(pacman_pos, map_state):
            positions.add(nxt)
            
            # Speed dash simulation in the continuous straight direction
            if self.pacman_speed > 1:
                dr, dc = move.value
                current = nxt
                for _ in range(1, self.pacman_speed):
                    candidate = (current[0] + dr, current[1] + dc)
                    if self._is_valid_position(candidate, map_state):
                        positions.add(candidate)
                        current = candidate
                    else:
                        break # Stop at walls
                        
        return list(positions) if positions else [pacman_pos]

    def _bfs_full(self, start, map_state):
        """O(V) distance map generated ONCE per step for move ordering."""
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
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

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