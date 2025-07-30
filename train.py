import torch.nn.functional as F
import wandb
import os
import math
import random
from collections import deque
import numpy as np
import time

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from safetensors.torch import save_file, load_file

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
from network import AlphaGoZeroNet
from search import MCTS, MCTSNode

def get_model(run, device):
    """Initializes a new model or resumes from a W&B artifact."""
    model = AlphaGoZeroNet(
        board_size=config.BOARD_SIZE,
        num_res_blocks=config.NUM_RES_BLOCKS,
        in_channels=config.IN_CHANNELS,
        num_filters=config.NUM_FILTERS
    ).to(device)

    # In a generational system, we always load the 'best' model checkpoint
    # The run state (generation number) is recovered from wandb's step
    if run.resumed or run.step > 0:
        print("Resuming run. Attempting to load 'best' model from W&B...")
        try:
            artifact_path = f'{run.entity}/{config.WANDB_PROJECT_NAME}/{config.WANDB_ARTIFACT_NAME}:latest'
            artifact = run.use_artifact(artifact_path, type='model')
            artifact_dir = artifact.download()
            model_path = os.path.join(artifact_dir, "alphago-zero.safetensors")
            model.load_state_dict(load_file(model_path, device=device))
            print("Successfully loaded 'best' model from previous checkpoint.")
        except Exception as e:
            print(f"Could not find artifact. Starting with a fresh model. Error: {e}")
    else:
        print("Starting a new run. Using a freshly initialized model.")
    return model

def play_game(network_wrapper, generation_num):
    """
    Simulates one full game of self-play using the provided network, returning training data.
    """
    game_state = GoGameState(config.BOARD_SIZE)
    # The MCTS instance is recreated for each game to ensure a clean state
    mcts = MCTS(network_wrapper, config.C_PUCT, config.DIRICHLET_ALPHA, config.DIRICHLET_EPSILON)
    root_node = MCTSNode()

    game_history = []

    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            break

        mcts.run_simulations(root_node, game_state.clone(), config.NUM_SIMULATIONS_TRAIN)

        # Check for resignation only after a certain number of generations
        # to allow the model to learn basic concepts first.
        if generation_num > 5 and root_node.children:
            best_action = max(root_node.visit_count, key=root_node.visit_count.get)
            root_value = root_node.mean_action_value.get(best_action, -1.0)
            if root_value < config.RESIGNATION_THRESHOLD:
                winner = -game_state.get_current_player()
                break # Game ends by resignation

        # Temperature is used for exploration in early moves
        temp = 1.0 if game_state.move_count < 30 else 0.0
        move_probs = mcts.get_move_probs(root_node, temp)

        policy_target = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE + 1)
        for action, prob in move_probs.items():
            policy_target[action] = prob

        game_history.append((
            game_state.get_representation(),
            policy_target,
            game_state.get_current_player()
        ))

        action_to_play = np.random.choice(len(policy_target), p=policy_target)
        game_state.apply_move(action_to_play)
        root_node = root_node.children.get(action_to_play, MCTSNode())
        if root_node.parent:
            root_node.parent = None # Prune the tree

    training_data = []
    for state_repr, policy, player in game_history:
        z = 0
        if winner is not None:
             z = 1 if player == winner else -1
        training_data.append((state_repr, policy, z))

    return training_data

def train_step(model, optimizer, replay_buffer, device):
    """Performs one training step on a batch of data from the replay buffer."""
    if len(replay_buffer) < config.BATCH_SIZE:
        return None

    mini_batch = random.sample(replay_buffer, config.BATCH_SIZE)

    # Unzip and convert to tensors.
    # We do this here to keep the replay_buffer with numpy arrays for efficiency.
    state_reprs, policy_targets, value_targets = zip(*mini_batch)
    state_tensors = torch.cat(state_reprs).to(device)
    policy_targets_tensor = torch.tensor(np.array(policy_targets), dtype=torch.float32).to(device)
    value_targets_tensor = torch.tensor(np.array(value_targets), dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    optimizer.zero_grad()

    policy_pred_logits, value_pred = model(state_tensors)

    value_loss = F.mse_loss(value_pred, value_targets_tensor)
    policy_loss = F.cross_entropy(policy_pred_logits, policy_targets_tensor)
    total_loss = policy_loss + value_loss

    total_loss.backward()

    if _TPU_AVAILABLE and 'xla' in str(device):
        xm.optimizer_step(optimizer, barrier=True)
    else:
        optimizer.step()

    return total_loss.item()

def evaluate_game(candidate_wrapper, best_wrapper, is_candidate_black):
    """
    Plays a single game between the candidate and best models.
    Returns 1 if the candidate wins, -1 if the best model wins, 0 for a draw.
    """
    if is_candidate_black:
        black_player, white_player = candidate_wrapper, best_wrapper
    else:
        black_player, white_player = best_wrapper, candidate_wrapper

    game_state = GoGameState(config.BOARD_SIZE)

    # Each player gets their own MCTS instance
    black_mcts = MCTS(black_player, config.C_PUCT)
    white_mcts = MCTS(white_player, config.C_PUCT)

    black_root = MCTSNode()
    white_root = MCTSNode()

    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            if winner == 0: return 0
            candidate_won = (winner == 1 and is_candidate_black) or \
                            (winner == -1 and not is_candidate_black)
            return 1 if candidate_won else -1

        if game_state.get_current_player() == 1: # Black's turn
            mcts, root = black_mcts, black_root
        else: # White's turn
            mcts, root = white_mcts, white_root

        mcts.run_simulations(root, game_state.clone(), config.NUM_SIMULATIONS_PLAY)
        move_probs = mcts.get_move_probs(root, temp=0) # temp=0 for deterministic evaluation
        action_to_play = max(move_probs, key=move_probs.get) if move_probs else (config.BOARD_SIZE**2)

        game_state.apply_move(action_to_play)

        # Update both trees to the new game state
        black_root = black_root.children.get(action_to_play, MCTSNode())
        white_root = white_root.children.get(action_to_play, MCTSNode())
        if black_root.parent: black_root.parent = None
        if white_root.parent: white_root.parent = None


def main():
    """Main training loop with generational self-play, training, and evaluation."""
    # Note: To use multiple TPU cores, you would use torch_xla.distributed.xla_multiprocessing.spawn
    if _TPU_AVAILABLE:
        device = xm.xla_device()
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    run = wandb.init(project=config.WANDB_PROJECT_NAME, id=config.WANDB_RUN_ID, resume="allow", job_type="training")
    # Update config with any local changes, useful for hyperparameter sweeps
    run.config.update({k: v for k, v in config.__dict__.items() if k.isupper()}, allow_val_change=True)

    # --- Initialize Best and Candidate Models ---
    best_model = get_model(run, device)
    candidate_model = AlphaGoZeroNet(
        board_size=config.BOARD_SIZE, num_res_blocks=config.NUM_RES_BLOCKS,
        in_channels=config.IN_CHANNELS, num_filters=config.NUM_FILTERS
    ).to(device)
    candidate_model.load_state_dict(best_model.state_dict()) # Start candidate from best

    best_network_wrapper = NetworkWrapper(best_model, device)
    candidate_network_wrapper = NetworkWrapper(candidate_model, device)

    optimizer = optim.SGD(candidate_model.parameters(), lr=run.config.INITIAL_LR, momentum=0.9, weight_decay=run.config.L2_REGULARIZATION)
    scheduler = MultiStepLR(optimizer, milestones=run.config.LR_MILESTONES, gamma=0.1)

    replay_buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)

    start_generation = run.step + 1 if run.resumed else 1

    for generation in range(start_generation, run.config.MAX_GENERATIONS + 1):
        print(f"\n--- Generation {generation}/{run.config.MAX_GENERATIONS} ---")

        # === PHASE A: SELF-PLAY using the BEST model ===
        print(f"  [1/3] Generating {run.config.GAMES_PER_GENERATION} games with the 'best' model...")
        best_model.eval()
        start_time = time.time()
        for i in range(run.config.GAMES_PER_GENERATION):
            game_data = play_game(best_network_wrapper, generation)
            replay_buffer.extend(game_data)
            print(f"\r    Game {i+1}/{run.config.GAMES_PER_GENERATION} generated...", end="")
        elapsed_time = time.time() - start_time
        print(f"\n    Self-play finished in {elapsed_time:.2f} seconds.")

        # === PHASE B: TRAINING the CANDIDATE model ===
        print(f"  [2/3] Training the 'candidate' model on replay buffer (size: {len(replay_buffer)})...")
        last_loss = None
        if len(replay_buffer) >= config.BATCH_SIZE:
            start_time = time.time()
            for i in range(run.config.TRAINING_UPDATES_PER_GENERATION):
                loss = train_step(candidate_model, optimizer, replay_buffer, device)
                if loss: last_loss = loss
                print(f"\r    Update {i+1}/{run.config.TRAINING_UPDATES_PER_GENERATION}, Loss: {loss:.4f if loss else 'N/A'}...", end="")
            scheduler.step()
            elapsed_time = time.time() - start_time
            print(f"\n    Training finished in {elapsed_time:.2f} seconds.")
        else:
            print("    Skipping training, replay buffer not full enough.")

        # === PHASE C: EVALUATING the CANDIDATE against the BEST ===
        print(f"  [3/3] Evaluating candidate vs. best over {run.config.NUM_EVAL_GAMES} games...")
        candidate_model.eval()
        best_model.eval()
        candidate_wins = 0
        start_time = time.time()
        for i in range(run.config.NUM_EVAL_GAMES):
            # Alternate who plays black for fairness
            winner = evaluate_game(candidate_network_wrapper, best_network_wrapper, is_candidate_black=(i % 2 == 0))
            if winner == 1:
                candidate_wins += 1
            print(f"\r    Eval game {i+1}/{run.config.NUM_EVAL_GAMES}, Candidate wins: {candidate_wins}", end="")

        win_rate = candidate_wins / run.config.NUM_EVAL_GAMES
        elapsed_time = time.time() - start_time
        print(f"\n    Evaluation finished in {elapsed_time:.2f} seconds. Candidate win rate: {win_rate:.2f}")

        # === PHASE D: PROMOTION ===
        if win_rate > run.config.EVAL_WIN_THRESHOLD:
            print("  >>> New best model found! Promoting candidate. <<<")
            best_model.load_state_dict(candidate_model.state_dict())

            print("  Saving new best model as W&B Artifact.")
            model_filename = "alphago-zero.safetensors"
            save_file(best_model.state_dict(), model_filename)

            model_artifact = wandb.Artifact(
                name=config.WANDB_ARTIFACT_NAME, type="model",
                description=f"Model from generation {generation} promoted with win rate {win_rate:.2f}.",
                metadata={"generation": generation, "loss": last_loss, "win_rate": win_rate}
            )
            model_artifact.add_file(model_filename)
            aliases = [f"generation-{generation}", "latest"]
            run.log_artifact(model_artifact, aliases=aliases)
            print(f"  Logged new model version to W&B Artifacts: {aliases}")
        else:
            print("  Candidate not strong enough. Discarding weights.")
            candidate_model.load_state_dict(best_model.state_dict()) # Reset candidate

        # Log metrics for the generation
        run.log({
            "candidate_win_rate": win_rate,
            "loss": last_loss,
            "replay_buffer_size": len(replay_buffer),
            "learning_rate": scheduler.get_last_lr()[0]
        }, step=generation)

    print("\nMaximum generations reached. Training complete!")
    run.finish()

if __name__ == "__main__":
    main()
