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
        self.unknown_frontier = set()  # biên giới chưa khám phá
        
        # Cache for Topology
        self.topology_cache = {} # Save expensive calculations like static distances
        self.fog_cache = {}      # Lưu mật độ sương mù theo bước

    def update(self, map_state, my_pos, enemy_pos, step_number):
        # Cập nhật bản đồ đã biết
        rows, cols = map_state.shape
        newly_found = set()
        
        # Only check 5x5 area around my_pos
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
        for pos in newly_found:
            x, y = pos
            for (dx, dy) in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if map_state[nx][ny] == -1:  # ô chưa biết
                        self.unknown_frontier.add((nx, ny))
                        
        # Delete cache after increasing step to calculate fog_density more properly
        self.fog_cache.clear()
            
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
        """Priotitize areas have highest fog_density near frontier."""
        if not self.unknown_frontier:
            return my_pos
            
        best_target = None
        max_score = -float('inf')
        
       # Limit check to a maximum of 10 frontier cells to save time
        candidates = list(self.unknown_frontier)
        if len(candidates) > 10:
            candidates = random.sample(candidates, 10)
            
        for p in candidates:
            dist = abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1])
            # Score = Surroundings fog density / Distance
            score = self.fog_density(p, 3, map_state) * 100 - dist
            if score > max_score:
                max_score = score
                best_target = p
                
        return best_target or my_pos
    
    def estimate_enemy_pos(self, step_number, map_state):
        """Predict potential enemy locations using BFS with an expanding radius."""
        if self.last_seen_enemy is None:
            return set()
            
        steps_passed = step_number - self.step_last_seen
        # Assume ghost moves 1 cell/step and pacman moves at max_speed. Using a safety radius of steps_passed * 1.2
        radius = int(steps_passed * 1.2) 
        
        possible_positions = {self.last_seen_enemy}
        fringe = deque([(self.last_seen_enemy, 0)])
        visited = {self.last_seen_enemy}
        
        while fringe:
            pos, dist = fringe.popleft()
            if dist >= radius: continue
            
            for (dx, dy) in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = pos[0]+dx, pos[1]+dy
                if (nx, ny) not in visited:
                    # Skip known walls; cells marked as -1 or 0 are potential enemy positions
                    if (nx, ny) not in self.known_walls:
                        visited.add((nx, ny))
                        possible_positions.add((nx, ny))
                        fringe.append(((nx, ny), dist + 1))
        return possible_positions
    
    def fog_density(self, center, radius, map_state):
        """Calculate the ratio of unknown cells (-1) within a given radius."""
        cache_key = (center, radius)
        if cache_key in self.fog_cache:
            return self.fog_cache[cache_key]

        r_min, r_max = max(0, center[0]-radius), min(map_state.shape[0], center[0]+radius+1)
        c_min, c_max = max(0, center[1]-radius), min(map_state.shape[1], center[1]+radius+1)
        
        unknown_count = 0
        total_walkable = 0
        
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                if (r, c) in self.known_walls: continue
                total_walkable += 1
                if (r, c) not in self.known_empty:
                    unknown_count += 1
        
        density = unknown_count / total_walkable if total_walkable > 0 else 0
        self.fog_cache[cache_key] = density
        return density
    
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        self.memory = MemoryMap()
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "BFS + Memory Pacman + Predictive Pacman"
        self.current_path = []
        self.prev_move = None
        self.visited = set()
        self.last_positions = []
        self.dist_map = {}
        self.last_dist_source = None
        self.last_map_hash = None

    def step(self, map_state, my_position, enemy_position, step_number):
        self.memory.update(map_state, my_position, enemy_position, step_number)
        start_time = time.perf_counter()
        self.visited.add(my_position)

        # ---- CACHE dist_map ----
        current_map_hash = hash(map_state.tobytes())

        if self.last_dist_source != my_position or self.last_map_hash != current_map_hash:
            self.dist_map = self._compute_dist_map(my_position, map_state)
            self.last_dist_source = my_position
            self.last_map_hash = current_map_hash

        # ---- LOOP DETECTION ----
        self.last_positions.append(my_position)
        if len(self.last_positions) > 6:
            self.last_positions.pop(0)

        is_looping = len(set(self.last_positions)) <= 2

        # ---- TARGET ----
        if enemy_position is not None:
            target = enemy_position

        # ---- PREDICT ----
        elif self.memory.last_seen_enemy is not None:
            # BFS-based prediction from MemoryMap (uses map_state for wall-aware expansion)
            possible_area = self.memory.estimate_enemy_pos(step_number, map_state)
            
            if possible_area:
                # Filter to reachable candidates using cached dist_map (from pacman branch)
                reachable = [
                    p for p in possible_area
                    if self.dist_map.get(p, float('inf')) < float('inf')
                ]
                
                if reachable:
                    # Prioritize candidates in unknown frontier (fog-aware from main branch)
                    frontier_candidates = [p for p in reachable if p in self.memory.unknown_frontier]
                    
                    if frontier_candidates:
                        # Choose nearest frontier candidate using dist_map (efficient intercept)
                        target = min(frontier_candidates, key=lambda p: self.dist_map.get(p, float('inf')))
                    else:
                        # All known area → choose closest reachable predicted position
                        target = min(reachable, key=lambda p: self.dist_map.get(p, float('inf')))
                else:
                    target = self.memory.last_seen_enemy
            else:
                target = self.memory.last_seen_enemy

        else:
            target = self.memory.get_exploration_target(my_position, map_state)

        # ---- REPLAN ----
        if not self.current_path or is_looping:
            path = self._bfs(my_position, target, map_state)

            if not path:
                if target == self.memory.last_seen_enemy:
                    frontier_candidates = list(self.memory.unknown_frontier)
                    if frontier_candidates:
                        frontier_target = min(
                            frontier_candidates,
                            key=lambda p: abs(p[0] - target[0]) + abs(p[1] - target[1])
                        )
                    else:
                        frontier_target = self._bfs_to_frontier(my_position, map_state)
                else:
                    frontier_target = self._bfs_to_frontier(my_position, map_state)

                path = self._bfs(my_position, frontier_target, map_state)

            self.current_path = path

        # ---- FALLBACK ----
        if not self.current_path:
            return self._explore(my_position, map_state, True)

        move = self.current_path.pop(0)

        steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
        if steps == 0:
            return self._explore(my_position, map_state, True)

        # multi step
        if self.current_path and self.current_path[0] == move:
            next_pos = self._apply_move(my_position, move)
            next2 = self._apply_move(next_pos, move)

            # bước 1 phải hợp lệ
            if not self._is_valid_position(next_pos, map_state):
                steps = 1
            else:
                # bước 2 phải nằm trong known_empty (an toàn tuyệt đối)
                if next2 in self.memory.known_empty:
                    steps = min(2, self.pacman_speed)
                    self.current_path.pop(0)
                else:
                    steps = 1

        self.prev_move = move
        return (move, max(1, steps))
    
    def _compute_dist_map(self, start, map_state):
        dist = {start: 0}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            d = dist[current]

            for next_pos, _ in self.memory.get_safe_neighbors(current, map_state):
                if next_pos not in dist:
                    dist[next_pos] = d + 1
                    queue.append(next_pos)

        return dist
    
    # BFS distance (dùng cho predictive)
    def _bfs_dist(self, start, goal, map_state):
        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, dist = queue.popleft()

            for next_pos, _ in self.memory.get_safe_neighbors(current, map_state):
                if next_pos not in visited:
                    if next_pos == goal:
                        return dist + 1
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))

        return float('inf')
    
    def _bfs(self, start, goal, map_state):
        if start == goal:
            return []
        
        queue = deque([start])
        visited = {start}
        parent = {}

        while queue:
            current = queue.popleft()
            neighbors = self.memory.get_safe_neighbors(current, map_state)

            # tie-break: ưu tiên đi thẳng
            if current in parent:
                _, prev_move = parent[current]
                neighbors.sort(key=lambda x: x[1] != prev_move)

            for next_pos, move in neighbors:
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)

                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)

                    queue.append(next_pos)

        return []

    def _bfs_to_frontier(self, start, map_state):
        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            # nếu gần unknown → target ngon
            if current in self.memory.known_empty:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = current[0]+dx, current[1]+dy
                    if (nx, ny) in self.memory.unknown_frontier:
                        return current

            for next_pos, _ in self.memory.get_safe_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)

        return start  # fallback
    
    def _reconstruct_path(self, parent, start, goal):
        path = []
        cur = goal
        while cur != start:
            prev, move = parent[cur]
            path.append(move)
            cur = prev
        return path[::-1]

    def _explore(self, my_position, map_state, force_random=False):

        neighbors = self.memory.get_safe_neighbors(my_position, map_state)

        if not neighbors:
            return (Move.STAY, 1)

        import random

        if force_random:
            move = random.choice(neighbors)[1]
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            return (move, max(1, steps))

        best_move = Move.STAY
        best_score = -float('inf')

        opposite = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT
        }

        for next_pos, move in neighbors:

            reverse_penalty = -15 if self.prev_move and move == opposite[self.prev_move] else 0
            visited_penalty = -5 if next_pos in self.visited else 3

            # 🔥 gần frontier thì ưu tiên mạnh
            frontier_bonus = 5 if next_pos in self.memory.unknown_frontier else 0

            score = frontier_bonus + visited_penalty + reverse_penalty

            if score > best_score:
                best_score = score
                best_move = move

        steps = self._max_valid_steps(my_position, best_move, map_state, self.pacman_speed)
        self.prev_move = best_move

        return (best_move, max(1, steps))
    
    # Helper methods
    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _max_valid_steps(self, pos, move, map_state, desired_steps):
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos

        for _ in range(max_steps):
            dr, dc = move.value
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Fog Exploitation Edition v3

    Key improvements over v2:
    - TT cleared EVERY STEP (v2 only cleared when >50000 - stale entries caused bad moves)
    - SAFE_DIST scales with pacman_speed (v2 used fixed 8 - too short at speed=2)
    - _corridor_escape: detects same-axis LoS and immediately finds perpendicular path
    - _evaluate: speed-aware maze_dist weight, stronger LOS penalty scaled by speed
    - _find_best_fog_target: approach-axis penalty + speed-aware reachability filter
    - _get_neighbors: still only map_state==0 (critical correctness fix from v2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Grandmaster Ghost v3"
        self.pacman_speed = kwargs.get('pacman_speed', 2)

        if not hasattr(self, 'memory'):
            self.memory = MemoryMap()

        self.tt = {}
        self.map_hash = None
        self.dead_ends = {}
        self.intersections = set()

        self.exploration_target = None

        self._fog_zone_cache = {}
        self._last_fog_cache_step = -1

        # Thresholds scale with Pacman speed
        # At speed=2: SAFE_DIST=16, AMBUSH_DIST=24
        self.SAFE_DIST = max(10, self.pacman_speed * 8)
        self.AMBUSH_DIST = max(20, self.pacman_speed * 12)

    # =========================================================================
    # MAIN STEP
    # =========================================================================

    def step(self, map_state, my_position, enemy_position, step_number):
        self.start_time = time.perf_counter()
        self.TIME_LIMIT = 0.82

        # Clear TT every step (map changes with fog reveals)
        self.tt.clear()

        # Invalidate fog cache each step
        if self._last_fog_cache_step != step_number:
            self._fog_zone_cache.clear()
            self._last_fog_cache_step = step_number

        # 1. UPDATE MEMORY
        self.memory.update(map_state, my_position, enemy_position, step_number)
        threat = enemy_position or self.memory.last_seen_enemy

        # 2. TOPOLOGY ANALYSIS (only when wall layout changes)
        wall_mask = (map_state == 1).tobytes()
        current_map_hash = hash(wall_mask)
        if self.map_hash != current_map_hash:
            self._analyze_topology(map_state)
            self.map_hash = current_map_hash

        valid_moves = self._get_neighbors(my_position, map_state)
        if not valid_moves:
            return Move.STAY

        p_root_dist = self._bfs_full(threat, map_state) if threat else {}
        g_root_dist = self._bfs_full(my_position, map_state)
        danger_dist = p_root_dist.get(my_position, 9999)

        # =========================================================
        # PHASE 1: EARLY-GAME PANIC (new in v4)
        # Direct evasion when spawned adjacent to Pacman (steps 1-20)
        # =========================================================
        if enemy_position is not None and danger_dist <= 6 and step_number <= 20:
            best_move = valid_moves[0][1]
            best_d = -1
            for nxt, mv in valid_moves:
                d = p_root_dist.get(nxt, 0)
                if d > best_d:
                    best_d = d
                    best_move = mv
            self.exploration_target = None
            return best_move

        # =========================================================
        # PHASE 2: CORRIDOR ESCAPE
        # Fires when Pacman is within SAFE_DIST AND Ghost has clear LoS.
        # LoS in a corridor = Pacman speed advantage is decisive.
        # =========================================================
        if threat is not None and danger_dist <= self.SAFE_DIST:
            escape = self._corridor_escape(
                my_position, threat, map_state, g_root_dist, p_root_dist
            )
            if escape is not None:
                return escape

        # =========================================================
        # PHASE 3: ACTIVE HIDING MODE (Pacman far or never seen)
        # =========================================================
        if threat is None or danger_dist > self.SAFE_DIST:

            # AMBUSH: deep in fog + Pacman very far → stay still
            if threat is not None and danger_dist > self.AMBUSH_DIST:
                fog_zone = self._count_connected_fog(my_position, map_state, max_depth=6)
                if fog_zone >= 4:
                    return Move.STAY

            # Invalidate target if reached or revealed
            if self.exploration_target == my_position:
                self.exploration_target = None
            elif self.exploration_target is not None:
                tr, tc = self.exploration_target
                if 0 <= tr < map_state.shape[0] and 0 <= tc < map_state.shape[1]:
                    if map_state[tr, tc] != -1:
                        self.exploration_target = None
                else:
                    self.exploration_target = None

            if not self.exploration_target:
                self.exploration_target = self._find_best_fog_target(
                    my_position, threat, map_state, g_root_dist, p_root_dist
                )

            if self.exploration_target:
                path = self.bfs(my_position, self.exploration_target, map_state)
                if path and path[0] != Move.STAY:
                    # PATH-AHEAD LoS CHECK (new in v4):
                    # If the very next step puts Ghost in a corridor with Pacman,
                    # discard target and fall through to minimax.
                    next_cell = self._apply_move(my_position, path[0])
                    if threat and self._has_clear_los(next_cell, threat, map_state):
                        self.exploration_target = None
                    else:
                        return path[0]

            if threat is None:
                best_fallback = Move.STAY
                best_deg = -1
                for nxt, move in valid_moves:
                    deg = len(self._get_neighbors(nxt, map_state))
                    if deg > best_deg:
                        best_deg = deg
                        best_fallback = move
                return best_fallback

        # =========================================================
        # PHASE 4: MINIMAX EVASION (Pacman within SAFE_DIST steps)
        # =========================================================
        self.exploration_target = None

        best_move = valid_moves[0][1]
        depth = 1
        while True:
            if time.perf_counter() - self.start_time > self.TIME_LIMIT:
                break
            move, score = self._search_root(
                my_position, threat, depth, map_state, p_root_dist, g_root_dist
            )
            if time.perf_counter() - self.start_time <= self.TIME_LIMIT:
                best_move = move
                if score >= 900000 or score <= -900000:
                    break
            depth += 1

        return best_move

    # =========================================================================
    # CORRIDOR TRAP DETECTION & ESCAPE
    # =========================================================================

    def _has_clear_los(self, pos_a, pos_b, map_state):
        """
        True if pos_a and pos_b share a row OR column with no walls between them.
        Clear LoS = worst case for Ghost against a fast Pacman.
        """
        ar, ac = pos_a
        br, bc = pos_b
        if ar == br:
            min_c, max_c = min(ac, bc), max(ac, bc)
            return all(map_state[ar, c] == 0 for c in range(min_c, max_c + 1))
        if ac == bc:
            min_r, max_r = min(ar, br), max(ar, br)
            return all(map_state[r, ac] == 0 for r in range(min_r, max_r + 1))
        return False

    def _corridor_escape(self, my_pos, threat_pos, map_state, g_dist, p_dist):
        """
        REDESIGNED v4: fires on ANY LoS (was: only when danger_dist <= SAFE_DIST).

        Strategy:
          1. Scan all reachable cells for ones that BREAK LoS (highest priority)
          2. Score by: LoS-break bonus + distance from Pacman + intersection bonus
             - directional bias: prefer cells in the 'away from Pacman' direction
             - NO hard speed filter (old filter discarded too many valid escapes)
          3. GUARANTEED FALLBACK: if no cell breaks LoS, simply run AWAY from
             Pacman along the corridor (buys turns to find a turn-off later).
        """
        if not self._has_clear_los(my_pos, threat_pos, map_state):
            return None

        gr, gc = my_pos
        pr, pc = threat_pos

        # Directional bias: cells in the 'away' half score positive
        same_row = (gr == pr)
        if same_row:
            away_sign = 1 if gc > pc else -1   # positive = moving away along row
        else:
            away_sign = 1 if gr > pr else -1   # positive = moving away along col

        best_escape = None
        best_score = -float('inf')

        for pos, g_d in g_dist.items():
            if pos == threat_pos:
                continue
            if pos in self.dead_ends:
                continue

            p_d = p_dist.get(pos, 9999)
            if p_d == 9999:
                continue

            # RELAXED reachability: allow Ghost to target cells even if Pacman
            # gets there "theoretically" first — the path itself matters more.
            # Only hard-filter when Pacman is massively faster (>6 buffer).
            if g_d * self.pacman_speed > p_d + 6:
                continue

            # LoS-break bonus: cell not on same row/col as BOTH agents
            breaks_los = (
                pos[0] != gr or pos[1] != gc  # different from ghost pos
            ) and not self._has_clear_los(my_pos, pos, map_state)

            los_bonus = 600 if breaks_los else 0
            inter_bonus = 400 if pos in self.intersections else 0

            # Directional score: positive if cell is 'behind' ghost (away from pacman)
            if same_row:
                dir_score = (pos[1] - gc) * away_sign * 15
            else:
                dir_score = (pos[0] - gr) * away_sign * 15

            score = los_bonus + p_d * 6 + inter_bonus - g_d * 4 + dir_score

            if score > best_score:
                best_score = score
                best_escape = pos

        # Navigate to best escape cell
        if best_escape is not None:
            path = self.bfs(my_pos, best_escape, map_state)
            if path and path[0] != Move.STAY:
                return path[0]

        # GUARANTEED FALLBACK: run AWAY from Pacman along corridor.
        # Even without breaking LoS this buys turns for Ghost to find a turn-off.
        if same_row:
            away_move = Move.RIGHT if gc > pc else Move.LEFT
        else:
            away_move = Move.DOWN if gr > pr else Move.UP

        dr, dc = away_move.value
        run_pos = (gr + dr, gc + dc)
        if self._is_valid_position(run_pos, map_state):
            return away_move

        # Corridor end: Pacman has us cornered — try perpendicular cells
        for nxt, mv in self._get_neighbors(my_pos, map_state):
            if mv != away_move:
                return mv
        return None

    # =========================================================================
    # FOG ZONE ANALYSIS
    # =========================================================================

    def _count_connected_fog(self, pos, map_state, max_depth=8):
        """
        BFS from pos: count fog cells (-1) reachable within max_depth steps.
        Traverses known-empty (0) and fog (-1); walls (1) block.
        """
        cache_key = (pos, max_depth)
        if cache_key in self._fog_zone_cache:
            return self._fog_zone_cache[cache_key]

        h, w = map_state.shape
        fog_count = 0
        queue = deque([(pos, 0)])
        visited = {pos}

        while queue:
            cur, d = queue.popleft()
            if d >= max_depth:
                continue
            x, y = cur
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in visited:
                    cell = map_state[nx, ny]
                    if cell == 1:
                        continue
                    visited.add((nx, ny))
                    if cell == -1:
                        fog_count += 1
                    queue.append(((nx, ny), d + 1))

        self._fog_zone_cache[cache_key] = fog_count
        return fog_count

    def _entry_bottleneck_score(self, pos, map_state):
        """
        Fewer known-empty neighbors + more fog neighbors = better choke point.
        Ghost here is harder to approach from revealed map.
        """
        x, y = pos
        h, w = map_state.shape
        known_neighbors = 0
        fog_neighbors = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                cell = map_state[nx, ny]
                if cell == 0:
                    known_neighbors += 1
                elif cell == -1:
                    fog_neighbors += 1
        if known_neighbors == 0:
            return 0
        bottleneck = fog_neighbors - known_neighbors
        return max(0, bottleneck) * 120

    # =========================================================================
    # FOG TARGET SELECTION
    # =========================================================================

    def _find_best_fog_target(self, my_pos, threat_pos, map_state, g_dist, p_dist):
        """
        Multi-factor fog target scoring:
          score = dist_from_threat * (10 * pacman_speed)   [far from Pacman]
                + fog_zone * 80                             [deep fog]
                + bottleneck                               [choke point]
                + inter_bonus                              [intersections]
                - dist_from_me * 2                         [easy to reach]
                - axis_penalty                             [avoid Pacman's row/col]

        Speed-aware reachability filter: skip targets Pacman gets to first.
        """
        h, w = map_state.shape
        best_target = None
        best_score = -float('inf')

        p_row = threat_pos[0] if threat_pos else -1
        p_col = threat_pos[1] if threat_pos else -1

        fog_candidates = []
        for r in range(h):
            for c in range(w):
                cell = map_state[r, c]
                if cell == 1:
                    continue
                if (r, c) in self.dead_ends:
                    continue
                if cell == -1:
                    fog_candidates.append((r, c))
                else:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dx, c + dy
                        if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == -1:
                            fog_candidates.append((r, c))
                            break

        if not fog_candidates:
            fog_candidates = [
                (r, c) for r in range(h) for c in range(w)
                if map_state[r, c] != 1 and (r, c) not in self.dead_ends
            ]

        for pos in fog_candidates:
            dist_from_me = g_dist.get(pos, 9999)
            if dist_from_me == 9999:
                continue

            dist_from_threat = p_dist.get(pos, 20) if threat_pos else 20

            # Speed-aware: skip if Pacman clearly gets there first
            if threat_pos and dist_from_me * self.pacman_speed > dist_from_threat + 4:
                continue

            fog_zone = self._count_connected_fog(pos, map_state, max_depth=6)
            bottleneck = self._entry_bottleneck_score(pos, map_state)
            inter_bonus = 300 if pos in self.intersections else 0

            # Penalize targets on Pacman's current attack axis
            axis_penalty = 0
            if threat_pos and (pos[0] == p_row or pos[1] == p_col):
                axis_penalty = 400

            score = (
                (dist_from_threat * 10 * self.pacman_speed)
                + (fog_zone * 80)
                + bottleneck
                + inter_bonus
                - (dist_from_me * 2)
                - axis_penalty
            )

            if score > best_score:
                best_score = score
                best_target = pos

        # Fallback: all fog gone, no threat
        if not best_target and not threat_pos:
            for pos, d in g_dist.items():
                if pos in self.dead_ends:
                    continue
                inter_bonus = 300 if pos in self.intersections else 0
                score = inter_bonus - d * 2
                if score > best_score:
                    best_score = score
                    best_target = pos

        return best_target

    # =========================================================================
    # MINIMAX
    # =========================================================================

    def _search_root(self, ghost_pos, pacman_pos, depth, map_state, p_root_dist, g_root_dist):
        alpha = -float('inf')
        beta = float('inf')
        best_move = Move.STAY
        best_score = -float('inf')

        moves = self._get_neighbors(ghost_pos, map_state)
        if not moves:
            return Move.STAY, -1000000 - depth

        moves.sort(key=lambda x: -p_root_dist.get(x[0], 0))

        for next_pos, move in moves:
            if time.perf_counter() - self.start_time > self.TIME_LIMIT:
                break
            score = self._minimax(
                next_pos, pacman_pos, depth - 1, False,
                alpha, beta, map_state, p_root_dist, g_root_dist
            )
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move, best_score

    def _minimax(self, ghost_pos, pacman_pos, depth, is_ghost,
                 alpha, beta, map_state, p_root_dist, g_root_dist):
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
            moves.sort(key=lambda x: -p_root_dist.get(x[0], 0))
            for next_pos, _ in moves:
                val = self._minimax(next_pos, pacman_pos, depth - 1, False,
                                    alpha, beta, map_state, p_root_dist, g_root_dist)
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
        else:
            best_val = float('inf')
            moves = self._get_pacman_next_positions(pacman_pos, map_state)
            moves.sort(key=lambda x: g_root_dist.get(x, 9999))
            for next_pos in moves:
                val = self._minimax(ghost_pos, next_pos, depth - 1, True,
                                    alpha, beta, map_state, p_root_dist, g_root_dist)
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break

        tt_type = 'exact'
        if best_val <= orig_alpha:
            tt_type = 'upper'
        elif best_val >= beta:
            tt_type = 'lower'
        self.tt[tt_key] = {'val': best_val, 'type': tt_type}

        return best_val

    def _evaluate(self, ghost_pos, pacman_pos, p_root_dist, g_root_dist, map_state):
        """
        Heuristic evaluation for Ghost (maximizer).

        A. Maze distance — weight = 100 + 50*pacman_speed (200 at speed=2)
           At speed=2: maze_dist=8 only buys 4 turns — must weight higher
        B. Connected fog zone (hiding depth, weight 120/cell)
        C. Bottleneck entry score
        D. Dead-end penalty: trap_depth * 400 (deeper = more dangerous)
        E. Intersection bonus: +600 (more escape routes)
        F. LOS penalty: 3000 * pacman_speed (6000 at speed=2), only if clear LoS
        """
        manhattan = self._manhattan_distance(ghost_pos, pacman_pos)

        if manhattan <= 1:
            return -500000

        maze_dist = p_root_dist.get(ghost_pos, 9999)
        speed_weight = 100 + 50 * self.pacman_speed
        score = maze_dist * speed_weight + manhattan * 10

        fog_zone = self._count_connected_fog(ghost_pos, map_state, max_depth=5)
        score += fog_zone * 120

        score += self._entry_bottleneck_score(ghost_pos, map_state)

        trap_depth = self.dead_ends.get(ghost_pos, 0)
        if trap_depth > 0:
            score -= trap_depth * 400

        if ghost_pos in self.intersections:
            score += 600

        # LOS penalty: scaled by speed, only if clear line-of-sight
        los_penalty = 3000 * self.pacman_speed
        if ghost_pos[0] == pacman_pos[0] or ghost_pos[1] == pacman_pos[1]:
            if self._has_clear_los(ghost_pos, pacman_pos, map_state):
                score -= los_penalty

        return score

    # =========================================================================
    # TOPOLOGY ANALYSIS
    # =========================================================================

    def _analyze_topology(self, map_state):
        """
        O(V) topology analysis on known-empty (==0) cells only.
        Computes dead_ends dict {pos: depth} and intersections set.
        """
        h, w = map_state.shape
        degrees = {}
        self.intersections.clear()
        self.dead_ends.clear()

        for r in range(h):
            for c in range(w):
                if map_state[r, c] == 0:
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

    # =========================================================================
    # HELPERS
    # =========================================================================

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
        """Full BFS distance map on known-empty (==0) cells."""
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
        """
        CRITICAL: Only map_state == 0 (known-empty).
        Fog cells (-1) excluded — could be walls in reality.
        """
        x, y = pos
        h, w = map_state.shape
        neighbors = []
        if x > 0 and map_state[x-1][y] == 0:
            neighbors.append(((x-1, y), Move.UP))
        if x < h-1 and map_state[x+1][y] == 0:
            neighbors.append(((x+1, y), Move.DOWN))
        if y > 0 and map_state[x][y-1] == 0:
            neighbors.append(((x, y-1), Move.LEFT))
        if y < w-1 and map_state[x][y+1] == 0:
            neighbors.append(((x, y+1), Move.RIGHT))
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
        """
        BFS to goal. If goal is fog (-1), stops at adjacent known-empty cell.
        """
        if start == goal:
            return []

        h, w = map_state.shape
        queue = deque([start])
        visited = {start}
        parent = {}

        while queue:
            current = queue.popleft()
            cx, cy = current

            for dx, dy, move in [(-1, 0, Move.UP), (1, 0, Move.DOWN),
                                   (0, -1, Move.LEFT), (0, 1, Move.RIGHT)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) == goal and 0 <= nx < h and 0 <= ny < w:
                    parent[(nx, ny)] = (current, move)
                    return self._reconstruct_path(parent, start, goal)

            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)
                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)
                    queue.append(next_pos)

        return [Move.STAY]
        