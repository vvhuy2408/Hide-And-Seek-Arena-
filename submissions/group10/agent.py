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
            self.start_time = time.perf_counter()
            self.TIME_LIMIT = 0.85 # Dành 0.15s dự phòng để không bao giờ bị quá 1 giây
            self.map_state = map_state

            # 1. TÍNH BFS GỐC ĐỂ SẮP XẾP ƯU TIÊN (Sort Candidates)
            # Giúp Alpha-Beta Pruning cắt tỉa nhánh cực nhanh
            root_pacman_distances = self._bfs_full(enemy_position, map_state)

            candidates = []
            for next_pos, move in self._get_neighbors(my_position, map_state):
                candidates.append((next_pos, move))
            candidates.append((my_position, Move.STAY))

            # Sắp xếp: Ưu tiên thử những bước chạy xa Pacman nhất trước
            candidates.sort(key=lambda x: -root_pacman_distances.get(x[0], 0))

            best_move = Move.STAY
            best_score = -float('inf')
            alpha = -float('inf')
            beta = float('inf')

            # ĐỘ SÂU (DEPTH): Nhìn trước 4 lượt (Ghost -> Pacman -> Ghost -> Pacman)
            MAX_DEPTH = 4

            for next_pos, move in candidates:
                # Time Guard: Nếu tính toán sắp lố 1 giây, dừng ngay và dùng kết quả tốt nhất hiện tại
                if time.perf_counter() - self.start_time > self.TIME_LIMIT:
                    print(f"[Ghost] Step {step_number} | Timeout Guard Triggered!")
                    break

                # 2. CHẠY MINIMAX DỰ ĐOÁN TƯƠNG LAI
                score = self._minimax(
                    ghost_pos = next_pos,
                    pacman_pos = enemy_position,
                    depth = MAX_DEPTH - 1,
                    is_ghost = False, # Lượt tiếp theo là của Pacman
                    alpha = alpha,
                    beta = beta
                )

                if score > best_score:
                    best_score = score
                    best_move = move
                
                if best_score > alpha:
                    alpha = best_score

            elapsed = time.perf_counter() - self.start_time
            print(f"[Ghost] Step {step_number} | Move: {best_move} | Score: {best_score} | Time: {elapsed:.4f}s")
            return best_move

    # ════════════════════════════════════════════════════
    #  THUẬT TOÁN MINIMAX & ALPHA-BETA PRUNING
    # ════════════════════════════════════════════════════
    def _minimax(self, ghost_pos, pacman_pos, depth, is_ghost, alpha, beta):
        # Base case 1: Pacman bắt được Ghost -> Điểm âm cực nặng, nhưng cộng thêm depth để "chết muộn nhất có thể"
        if ghost_pos == pacman_pos:
            return -1000000 + depth 

        # Base case 2: Hết độ sâu hoặc hết thời gian -> Gọi hàm đánh giá 11-step của bạn
        if depth == 0 or time.perf_counter() - self.start_time > self.TIME_LIMIT:
            return self._evaluate_11_step(ghost_pos, pacman_pos)

        if is_ghost:
            # LƯỢT CỦA GHOST (Cố gắng MAXIMIZE điểm số)
            best = -float('inf')
            moves = self._get_neighbors(ghost_pos, self.map_state)
            moves.append((ghost_pos, Move.STAY))
            
            for next_pos, _ in moves:
                val = self._minimax(next_pos, pacman_pos, depth - 1, False, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break # Alpha-Beta Pruning
            return best
        else:
            # LƯỢT CỦA PACMAN (Cố gắng MINIMIZE điểm số của Ghost)
            best = float('inf')
            pacman_nexts = self._get_pacman_next_positions(pacman_pos)
            
            for next_pac in pacman_nexts:
                val = self._minimax(ghost_pos, next_pac, depth - 1, True, alpha, beta)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    break # Alpha-Beta Pruning
            return best

    # ════════════════════════════════════════════════════
    #  HÀM ĐÁNH GIÁ (HEURISTIC) TỪ BẢN 11-STEP CỦA BẠN
    # ════════════════════════════════════════════════════
    def _evaluate_11_step(self, ghost_pos, pacman_pos):
        """
        Đây là linh hồn giúp Ghost đạt 11-step. 
        Được gọi ở các "lá" của cây Minimax để chấm điểm cục diện.
        """
        # 1. Khoảng cách BFS thực tế (Chính xác 100%, đập tan sai lầm dùng Manhattan của Claude)
        p_dist = self._bfs_dist(pacman_pos, ghost_pos)
        
        if p_dist <= 1:
            return -500000

        # 2. Quét vùng an toàn (Flood Fill) để né ngõ cụt
        ghost_queue = deque([ghost_pos])
        visited = {ghost_pos, pacman_pos} # Cấm đi xuyên qua người Pacman
        safe_area = 0

        while ghost_queue:
            curr = ghost_queue.popleft()
            safe_area += 1
            
            # Quét tối đa 25 ô là đủ biết ngõ cụt hay không gian mở, tiết kiệm thời gian
            if safe_area >= 25:
                break
                
            for nxt, _ in self._get_neighbors(curr, self.map_state):
                if nxt not in visited:
                    visited.add(nxt)
                    ghost_queue.append(nxt)

        # 3. CÔNG THỨC 11-STEP HUYỀN THOẠI
        score = (p_dist * 1000) + safe_area

        # Khóa chặn ngõ cụt: Nếu vùng an toàn < 15, đây chắc chắn là ngõ cụt chết người!
        if safe_area < 15:
            score -= 50000

        return score

    # ════════════════════════════════════════════════════
    #  CÁC HÀM TIỆN ÍCH HỖ TRỢ MINIMAX
    # ════════════════════════════════════════════════════
    def _get_pacman_next_positions(self, pacman_pos):
        """Mô phỏng chính xác khả năng đi 2 ô/bước trên đường thẳng của Pacman"""
        positions = set()
        for nxt, move in self._get_neighbors(pacman_pos, self.map_state):
            positions.add(nxt)
            # Khả năng nhảy 2 ô thẳng
            dr, dc = move.value
            pos2 = (nxt[0] + dr, nxt[1] + dc)
            if self._is_valid_position(pos2, self.map_state):
                positions.add(pos2)
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

    def _bfs_dist(self, start, target):
        """Chạy mini-BFS siêu tốc để tìm khoảng cách giữa 2 điểm bất kỳ"""
        if start == target: return 0
        queue = deque([start])
        dist = {start: 0}
        while queue:
            curr = queue.popleft()
            d = dist[curr]
            if curr == target: 
                return d
            for nxt, _ in self._get_neighbors(curr, self.map_state):
                if nxt not in dist:
                    dist[nxt] = d + 1
                    queue.append(nxt)
        return 999
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
    
