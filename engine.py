import torch
import numpy as np
import sys
import argparse
from safetensors.torch import load_file

# --- New Imports for TPU Support ---
try:
    import torch_xla.core.xla_model as xm
    _TPU_AVAILABLE = True
except ImportError:
    _TPU_AVAILABLE = False

# Local project imports
import config
from game import GoGameState
from mcts_wrapper import NetworkWrapper
from search import MCTS, MCTSNode
from network import AlphaGoZeroNet


class GTPEngine:
    def __init__(self, model_path, device, num_simulations):
        self.board_size = config.BOARD_SIZE
        self.game_state = GoGameState(board_size=self.board_size)
        self.komi = 7.5
        self.GTP_COORD = "ABCDEFGHJKLMNOPQRST"

        # Load the model with the architecture from the config file
        self.model = AlphaGoZeroNet(
            board_size=config.BOARD_SIZE,
            num_res_blocks=config.NUM_RES_BLOCKS,
            in_channels=config.IN_CHANNELS,
            num_filters=config.NUM_FILTERS
        ).to(device)
        # Use safetensors for loading the model state
        self.model.load_state_dict(load_file(model_path, device=device))
        self.model.eval()
        self.device = device

        # MCTS attributes
        self.num_simulations = num_simulations
        wrapped_network = NetworkWrapper(self.model, self.device)
        self.mcts = MCTS(wrapped_network, config.C_PUCT, config.DIRICHLET_ALPHA, config.DIRICHLET_EPSILON)
        self.root_node = MCTSNode()

    def run(self):
        """Main loop to read GTP commands from stdin."""
        while True:
            line = sys.stdin.readline().strip()
            if not line:
                continue
            parts = line.split()
            command = parts[0]
            args = parts[1:]

            handler = getattr(self, f"handle_{command}", self.handle_unknown)
            if command == "quit":
                self.respond("")
                break
            handler(args)

    def respond(self, response):
        sys.stdout.write(f"= {response}\n\n")
        sys.stdout.flush()

    def fail(self, response):
        sys.stdout.write(f"? {response}\n\n")
        sys.stdout.flush()

    def handle_unknown(self, args):
        self.respond("") # Respond empty to unknown commands

    def handle_genmove(self, args):
        """Generates a move using MCTS."""
        color_str = args[0].lower()
        if color_str == 'b' or color_str == 'black':
            self.game_state.current_player = 1
        else:
            self.game_state.current_player = -1


        self.mcts.run_simulations(self.root_node, self.game_state.clone(), self.num_simulations)

        # The value of the root node is the expected outcome for the current player.
        # If this value is below the resignation threshold, the engine should resign.
        move_probs = self.mcts.get_move_probs(self.root_node, temp=0)
        best_action = max(move_probs, key=move_probs.get) if move_probs else self.board_size * self.board_size

        # Get the value from the perspective of the current player for the chosen move.
        # Fallback to a very low value if the move isn't in the explored actions for some reason.
        root_value = self.root_node.mean_action_value.get(best_action, -1.0)

        if root_value < config.RESIGNATION_THRESHOLD:
            self.respond("resign")
            self.game_state.apply_move(self.board_size * self.board_size) # Apply a pass to advance state
            self.root_node = MCTSNode() # Reset tree after resignation
            return

        self.game_state.apply_move(best_action)
        self.root_node = self.root_node.children.get(best_action, MCTSNode())
        if self.root_node.parent:
            self.root_node.parent = None # Prune the tree

        if best_action == self.board_size * self.board_size:
            self.respond("pass")
        else:
            y, x = divmod(best_action, self.board_size)
            move_gtp = self.GTP_COORD[x] + str(self.board_size - y)
            self.respond(move_gtp)

    def handle_play(self, args):
        """Handles a move played by the opponent and updates the MCTS tree."""
        color_str, move = args[0].lower(), args[1].lower()
        legal_moves = self.game_state.get_legal_moves()

        if move == "pass":
            action = self.board_size * self.board_size
        else:
            try:
                if not move or move[0].upper() not in self.GTP_COORD:
                     self.fail(f"invalid coordinate {move}")
                     return
                x = self.GTP_COORD.find(move[0].upper())
                y = self.board_size - int(move[1:])
                if not (0 <= x < self.board_size and 0 <= y < self.board_size):
                    self.fail(f"invalid coordinate {move}")
                    return
                action = y * self.board_size + x
            except (ValueError, IndexError):
                self.fail(f"invalid coordinate {move}")
                return

        if action not in legal_moves:
            self.fail(f"illegal move: {move}")
            return

        self.game_state.apply_move(action)

        # Advance the MCTS tree to the new state
        if action in self.root_node.children:
            self.root_node = self.root_node.children[action]
            self.root_node.parent = None
        else:
            # If the opponent's move was not in our search tree, we must start a new tree
            self.root_node = MCTSNode()
        self.respond("")

    def handle_clear_board(self, args):
        self.game_state = GoGameState(board_size=self.board_size)
        self.root_node = MCTSNode()
        self.respond("")

    def handle_boardsize(self, args):
        size = int(args[0])
        if size != self.board_size:
            self.fail(f"Only boardsize {self.board_size} is supported")
        else:
            self.handle_clear_board(args)

    def handle_komi(self, args):
        self.komi = float(args[0])
        self.respond("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Go engine using AlphaGo Zero principles.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.safetensors file).")
    parser.add_argument("--num_simulations", type=int, default=config.NUM_SIMULATIONS_PLAY, help="Number of MCTS simulations per move.")
    args = parser.parse_args()

    # --- Modified Device Selection for TPU ---
    if _TPU_AVAILABLE:
        device = xm.xla_device()
    else:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    sys.stderr.write(f"GoZero Engine using device: {device}\n")
    sys.stderr.write(f"MCTS simulations per move: {args.num_simulations}\n")
    sys.stderr.flush()

    engine = GTPEngine(model_path=args.model_path, device=device, num_simulations=args.num_simulations)
    engine.run()
