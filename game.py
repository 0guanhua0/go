import torch
import numpy as np
from collections import deque

class GoGameState:
    """
    A functional Go game state implementation with proper capture and suicide rules.
    This class manages the board, move history, and game-over conditions.
    """
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1 for Black, -1 for White
        # History deque stores past board states. It's pre-filled with empty boards
        # to ensure the input tensor shape is always consistent.
        self.history = deque([np.zeros_like(self.board) for _ in range(15)], maxlen=15)
        self.consecutive_passes = 0
        self.max_moves = board_size * board_size * 2 # Cap game length
        self.move_count = 0
        self.ko_point = None # Stores a (y, x) tuple of a forbidden ko point

    def clone(self):
        """Creates a deep copy of the game state for simulations."""
        new_state = GoGameState(self.board_size)
        new_state.board = np.copy(self.board)
        new_state.current_player = self.current_player
        new_state.history = self.history.copy() # deque.copy() is a shallow copy, which is efficient and sufficient here
        new_state.consecutive_passes = self.consecutive_passes
        new_state.move_count = self.move_count
        new_state.ko_point = self.ko_point
        return new_state

    def _get_group(self, y, x, board_to_check):
        """
        Finds the group of connected stones and its liberties on a given board.
        This is a helper function used for capture, suicide, and scoring checks.

        Args:
            y, x (int): Coordinates of a stone to start the search from.
            board_to_check (np.array): The board state to analyze.

        Returns:
            tuple: A tuple containing (set of group stone coordinates, set of liberty coordinates).
        """
        color = board_to_check[y, x]
        if color == 0:
            return None, None

        q = deque([(y, x)])
        visited = set([(y, x)])
        group_stones = set([(y, x)])
        liberties = set()

        while q:
            cy, cx = q.popleft()
            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                ny, nx = cy + dy, cx + dx

                if not (0 <= ny < self.board_size and 0 <= nx < self.board_size):
                    continue

                neighbor = board_to_check[ny, nx]
                if neighbor == 0:
                    liberties.add((ny, nx))
                elif neighbor == color and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    group_stones.add((ny, nx))
                    q.append((ny, nx))
        return group_stones, liberties


    def get_legal_moves(self):
        """
        Returns a list of legal moves as integer actions,
        respecting suicide and ko rules.
        """
        legal_moves = []
        empty_spots = np.argwhere(self.board == 0)

        for y, x in empty_spots:
            action = y * self.board_size + x

            # Ko check
            if (y, x) == self.ko_point:
                continue

            # Suicide check
            temp_board = np.copy(self.board)
            temp_board[y, x] = self.current_player

            # Check liberties of the newly placed stone's group
            _, liberties = self._get_group(y, x, temp_board)
            if len(liberties) > 0:
                legal_moves.append(action)
                continue

            # If no liberties, check if it captures opponent stones
            captures_made = False
            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < self.board_size and 0 <= nx < self.board_size):
                    continue
                if temp_board[ny, nx] == -self.current_player:
                    _, opponent_liberties = self._get_group(ny, nx, temp_board)
                    if len(opponent_liberties) == 0:
                        captures_made = True
                        break
            if captures_made:
                legal_moves.append(action)

        legal_moves.append(self.board_size * self.board_size) # Pass move
        return legal_moves

    def apply_move(self, action):
        """Applies a move, handles captures, updates history, and switches player."""
        self.ko_point = None # Reset ko point each turn
    
        if action == self.board_size * self.board_size:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
            y, x = divmod(action, self.board_size)
            if self.board[y,x] != 0:
                raise ValueError("Invalid move: trying to play on an occupied stone.")
            self.board[y, x] = self.current_player
    
            # ... (capture logic remains the same) ...
            captured_stones_total = 0
            single_captured_stone_group = None
            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < self.board_size and 0 <= nx < self.board_size):
                    continue
                if self.board[ny, nx] == -self.current_player:
                    group, liberties = self._get_group(ny, nx, self.board)
                    if group and not liberties:
                        for stone_y, stone_x in group:
                            self.board[stone_y, stone_x] = 0
                        captured_stones_total += len(group)
                        if len(group) == 1:
                            single_captured_stone_group = group
    
            my_group, my_liberties = self._get_group(y, x, self.board)
            if captured_stones_total == 1 and my_group and len(my_group) == 1 and len(my_liberties) == 1:
                self.ko_point = single_captured_stone_group.pop()
    
        self.history.appendleft(np.copy(self.board))
    
        self.current_player *= -1
        self.move_count += 1

    def is_game_over(self):
        """Checks if the game is over by passes or move limit."""
        if self.consecutive_passes >= 2 or self.move_count >= self.max_moves:
            winner = self._get_winner()
            return True, winner
        return False, None



    def get_current_player(self):
        return self.current_player

    def get_representation(self):
        """
        Converts the board state into the multi-plane tensor representation
        for the neural network. (17 planes)
        Planes 0-7: Current player's stones (T=0, T-1, ..., T-7)
        Planes 8-15: Opponent's stones (T=0, T-1, ..., T-7)
        Plane 16: Color to play (1 for black, 0 for white)
        """
        state_tensor = np.zeros((1, 17, self.board_size, self.board_size), dtype=np.float32)

        state_tensor[0, 0, :, :] = (self.board == self.current_player)
        state_tensor[0, 8, :, :] = (self.board == -self.current_player)

        # Planes 1-7 and 9-15 for history (7 past states: T-1, T-2, ...)
        for i in range(7):
            if i < len(self.history):
                past_board = self.history[i]
                state_tensor[0, i + 1, :, :] = (past_board == self.current_player)
                state_tensor[0, 8 + i + 1, :, :] = (past_board == -self.current_player)

        # Color plane (indicates whose turn it is)
        if self.current_player == 1: # Black to play
            state_tensor[0, 16, :, :] = 1.0
        # If White to play, it remains 0.0, so no `else` needed.

        return torch.from_numpy(state_tensor)


    def _get_winner(self):
        """A simplified scoring method (area scoring)."""
        black_score, white_score = np.sum(self.board == 1), np.sum(self.board == -1)
        white_score += 7.5 # Komi

        # Use a temporary board to flood-fill and find territory
        territory_mask = np.copy(self.board)
        empty_points = list(zip(*np.where(territory_mask == 0)))

        for y, x in empty_points:
            if territory_mask[y,x] != 0: continue # Already part of a found territory

            q = deque([(y, x)])
            visited = set([(y,x)])
            borders = {'black': False, 'white': False}

            while q:
                cy, cx = q.popleft()
                for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < self.board_size and 0 <= nx < self.board_size):
                        continue

                    neighbor_val = self.board[ny,nx]
                    if neighbor_val == 1: borders['black'] = True
                    elif neighbor_val == -1: borders['white'] = True
                    elif neighbor_val == 0 and (ny,nx) not in visited:
                        visited.add((ny,nx))
                        q.append((ny,nx))

            owner = 0
            if borders['black'] and not borders['white']: owner = 1
            elif borders['white'] and not borders['black']: owner = -1

            if owner != 0:
                for ry,rx in visited:
                    territory_mask[ry,rx] = owner

        # Recalculate scores based on stones + final territory
        # This is Chinese/Area scoring
        final_black_score = np.sum(territory_mask == 1)
        final_white_score = np.sum(territory_mask == -1) + 7.5 # Komi

        if final_black_score > final_white_score: return 1
        elif final_white_score > final_black_score: return -1
        else: return 0
