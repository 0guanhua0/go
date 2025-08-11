import torch
import numba
import numpy as np
from collections import deque

# --- JIT-COMPILED HELPER FUNCTIONS ---
# These functions are designed to be fast and operate only on NumPy arrays
# and primitive types, making them ideal for Numba's nopython mode.

@numba.jit(nopython=True, cache=True)
def _get_group_numba(y, x, board, board_size):
    """
    Finds the group of connected stones and its liberties.
    (This function remains the same as your original)
    """
    group_stones = set([(0, 0)])
    group_stones.clear()
    liberties = set([(0, 0)])
    liberties.clear()

    color = board[y, x]
    if color == 0:
        return group_stones, liberties

    q = [(y, x)]
    visited = set([(y, x)])
    group_stones.add((y, x))

    while len(q) > 0:
        cy, cx = q.pop(0)
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < board_size and 0 <= nx < board_size):
                continue

            neighbor = board[ny, nx]
            if neighbor == 0:
                liberties.add((ny, nx))
            elif neighbor == color and (ny, nx) not in visited:
                visited.add((ny, nx))
                group_stones.add((ny, nx))
                q.append((ny, nx))
    return group_stones, liberties

@numba.jit(nopython=True, cache=True)
def _get_legal_moves_numba(board, current_player, ko_point, board_size):
    """
    A fast, JIT-compiled version to find legal moves.
    Returns a boolean array where True indicates a legal move.
    """
    # Create a boolean mask for legal moves, initialized to False.
    # The size is board_size^2 for board positions + 1 for the pass move.
    legal_mask = np.full(board_size * board_size + 1, False)

    # The pass move is always legal.
    legal_mask[board_size * board_size] = True

    for y in range(board_size):
        for x in range(board_size):
            # If the spot is not empty, it's not a legal move.
            if board[y, x] != 0:
                continue

            # Ko check
            if ko_point is not None and y == ko_point[0] and x == ko_point[1]:
                continue

            # Simulate placing a stone and check for suicide.
            temp_board = board.copy()
            temp_board[y, x] = current_player

            # Check liberties of the newly placed stone's group.
            _, liberties = _get_group_numba(y, x, temp_board, board_size)
            if len(liberties) > 0:
                legal_mask[y * board_size + x] = True
                continue

            # If no liberties, it's a suicide unless it captures opponent stones.
            captures_made = False
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < board_size and 0 <= nx < board_size):
                    continue

                if temp_board[ny, nx] == -current_player:
                    _, opponent_liberties = _get_group_numba(ny, nx, temp_board, board_size)
                    if len(opponent_liberties) == 0:
                        captures_made = True
                        break # Found a capture, so the move is legal.

            if captures_made:
                legal_mask[y * board_size + x] = True

    return legal_mask


@numba.jit(nopython=True, cache=True)
def _apply_move_numba(board, action, current_player, board_size):
    """
    A fast, JIT-compiled version of applying a move.
    Returns the new board state and the new ko_point tuple (or None).
    """
    # Pass move does not change the board or create a ko.
    if action == board_size * board_size:
        return board, None

    new_board = board.copy()
    y, x = divmod(action, board_size)

    # It's assumed the move is legal, so we don't check for occupation.
    new_board[y, x] = current_player

    captured_stones_total = 0
    single_captured_group = set([(0,0)])
    single_captured_group.clear()

    # Check for captures of opponent groups
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < board_size and 0 <= nx < board_size):
            continue
        if new_board[ny, nx] == -current_player:
            group, liberties = _get_group_numba(ny, nx, new_board, board_size)
            if len(group) > 0 and len(liberties) == 0:
                for stone_y, stone_x in group:
                    new_board[stone_y, stone_x] = 0
                captured_stones_total += len(group)
                if len(group) == 1:
                    single_captured_group = group

    # Ko Rule Check: A ko is created if a single stone is captured,
    # which results in the board state returning to the previous position.
    new_ko_point = None
    my_group, my_liberties = _get_group_numba(y, x, new_board, board_size)
    if (captured_stones_total == 1 and len(my_group) == 1 and
        len(my_liberties) == 0 and len(single_captured_group) == 1):
            # This is a potential ko situation. The captured stone's position is the new ko point.
            # Numba can't return a set, so we extract the tuple.
            for ko_pos in single_captured_group:
                new_ko_point = ko_pos
                break # there's only one

    return new_board, new_ko_point


@numba.jit(nopython=True, cache=True)
def _find_territory_numba(board, board_size):
    """
    Calculates territory using a flood-fill algorithm.
    (This function remains the same as your original)
    """
    territory_mask = np.copy(board)
    for y in range(board_size):
        for x in range(board_size):
            if territory_mask[y, x] == 0:
                q = [(y, x)]
                visited = set([(y, x)])
                borders = {'black': False, 'white': False}

                while len(q) > 0:
                    cy, cx = q.pop(0)
                    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ny, nx = cy + dy, cx + dx
                        if not (0 <= ny < board_size and 0 <= nx < board_size):
                            continue

                        neighbor_val = board[ny, nx]
                        # Numba can't use dictionary literals directly inside the loop like this
                        # for type inference, so pre-defining helps.
                        if neighbor_val == 1:
                            borders['black'] = True
                        elif neighbor_val == -1:
                            borders['white'] = True
                        elif neighbor_val == 0 and (ny, nx) not in visited:
                            visited.add((ny, nx))
                            q.append((ny, nx))

                owner = 0
                if borders['black'] and not borders['white']:
                    owner = 1
                elif borders['white'] and not borders['black']:
                    owner = -1

                if owner != 0:
                    for ry, rx in visited:
                        territory_mask[ry, rx] = owner
    return territory_mask

@numba.jit(nopython=True, cache=True)
def _get_representation_numba(board, history_list, current_player, board_size):
    """
    A fast, JIT-compiled version to create the 17-plane network input.
    """
    state_tensor = np.zeros((17, board_size, board_size), dtype=np.float32)

    state_tensor[0, :, :] = (board == current_player)
    state_tensor[8, :, :] = (board == -current_player)

    # History states (T-1 to T-7)
    # The history list is ordered from most recent (T-1) to oldest.
    history_len = min(7, len(history_list))
    for i in range(history_len):
        past_board = history_list[i]
        state_tensor[i + 1, :, :] = (past_board == current_player)
        state_tensor[8 + i + 1, :, :] = (past_board == -current_player)

    # Color plane
    if current_player == 1: # Black to play
        state_tensor[16, :, :] = 1.0

    return state_tensor


# --- GoGameState Class ---
# The main class now acts as a wrapper around the fast JIT-compiled functions.

class GoGameState:
    """
    A functional Go game state implementation with proper capture and suicide rules.
    This class manages the board, move history, and game-over conditions.
    """
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1 for Black, -1 for White
        # We need a history of raw numpy arrays for Numba
        self.history_boards = deque([np.zeros_like(self.board) for _ in range(8)], maxlen=8)
        self.consecutive_passes = 0
        self.max_moves = board_size * board_size * 2
        self.move_count = 0
        self.ko_point = None # Stores a (y, x) tuple of a forbidden ko point

    def clone(self):
        """Creates a deep copy of the game state for simulations."""
        new_state = GoGameState(self.board_size)
        new_state.board = np.copy(self.board)
        new_state.current_player = self.current_player
        new_state.history_boards = self.history_boards.copy()
        new_state.consecutive_passes = self.consecutive_passes
        new_state.move_count = self.move_count
        new_state.ko_point = self.ko_point
        return new_state

    def get_legal_moves(self):
        """
        Returns a list of legal moves by calling the fast Numba helper.
        """
        legal_mask = _get_legal_moves_numba(self.board, self.current_player, self.ko_point, self.board_size)
        # Convert the boolean mask to a list of integer actions
        return np.where(legal_mask)[0].tolist()

    def apply_move(self, action):
        """Applies a move by calling the fast Numba helper, then updates state."""
        # The Numba function handles board changes and ko calculation.
        new_board, new_ko_point = _apply_move_numba(self.board, action, self.current_player, self.board_size)

        self.board = new_board
        self.ko_point = new_ko_point

        # The rest of the state update remains in Python.
        if action == self.board_size * self.board_size:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        # Update history with the state *before* the player switch
        self.history_boards.appendleft(np.copy(self.board))

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
        Calls the fast Numba helper to create the NumPy representation,
        then converts it to a PyTorch tensor.
        """
        # Numba can't handle deques, so we convert it to a list of arrays.
        # We pass T-1, T-2, ... boards. self.history_boards[0] is the current board.
        history_list = list(self.history_boards)[1:]

        state_numpy = _get_representation_numba(
            self.board, history_list, self.current_player, self.board_size
        )
        return torch.from_numpy(state_numpy)

    def _get_winner(self):
        """A simplified scoring method (area scoring)."""
        territory_mask = _find_territory_numba(self.board, self.board_size)
        final_black_score = np.sum(territory_mask == 1)
        final_white_score = np.sum(territory_mask == -1) + 7.5 # Komi

        if final_black_score > final_white_score:
            return 1
        elif final_white_score > final_black_score:
            return -1
        else:
            return 0
