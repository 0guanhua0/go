import torch.nn.functional as F
import wandb
import os
import math
import random
from collections import deque
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from safetensors.torch import save_file, load_file

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

    if run.resumed:
        print("Resuming run. Attempting to load latest model from W&B...")
        try:
            # Note: wandb artifact paths are 'entity/project/name:alias'
            artifact_path = f'{run.entity}/{config.WANDB_PROJECT_NAME}/{config.WANDB_ARTIFACT_NAME}:latest'
            artifact = run.use_artifact(artifact_path, type='model')
            artifact_dir = artifact.download()
            # Use safetensors for loading
            model_path = os.path.join(artifact_dir, "alphago-zero.safetensors")
            model.load_state_dict(load_file(model_path, device=device))
            print("Successfully loaded model from previous checkpoint.")
        except Exception as e:
            print(f"Could not find artifact. Starting with a fresh model. Error: {e}")
    else:
        print("Starting a new run. Using a freshly initialized model.")
    return model

def play_game(network_wrapper):
    """
    Simulates one full game of self-play, returning training data.
    This function implements the self-play and data generation part of the AlphaGo Zero algorithm.
    """
    print("  Simulating one game of self-play...")
    game_state = GoGameState(config.BOARD_SIZE)
    mcts = MCTS(network_wrapper, config.C_PUCT, config.DIRICHLET_ALPHA, config.DIRICHLET_EPSILON)
    root_node = MCTSNode()

    game_history = [] # Stores (state_representation, policy_target, current_player)

    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            break

        mcts.run_simulations(root_node, game_state.clone(), config.NUM_SIMULATIONS_TRAIN)

        # To check for resignation, we need the value of the best move (most visited).
        # We can get this by finding the most visited child.
        if root_node.children:
            best_action = max(root_node.visit_count, key=root_node.visit_count.get)
            root_value = root_node.mean_action_value.get(best_action, -1.0)

            if root_value < config.RESIGNATION_THRESHOLD:
                winner = -game_state.get_current_player()
                print(f"  Game resigned. Winner: {winner}")
                break

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

        # Choose action based on the probability distribution (with temperature)
        action_to_play = np.random.choice(len(policy_target), p=policy_target)
        game_state.apply_move(action_to_play)
        root_node = root_node.children.get(action_to_play, MCTSNode()) # Reuse subtree
        if root_node.parent:
            root_node.parent = None # Prune tree

    # Game is over, create the final training data with the outcome (z)
    training_data = []
    for state_repr, policy, player in game_history:
        z = 0
        if winner is not None:
             z = 1 if player == winner else -1
        training_data.append((state_repr, policy, z))

    print(f"  Game finished. Winner: {winner}. Generated {len(training_data)} training samples.")
    return training_data

def train_step(model, optimizer, replay_buffer, device):
    """Performs one training step on a batch of data from the replay buffer."""
    if len(replay_buffer) < config.BATCH_SIZE:
        return None, None

    mini_batch = random.sample(replay_buffer, config.BATCH_SIZE)
    state_tensors, policy_targets, value_targets = zip(*mini_batch)

    state_tensors = torch.cat(state_tensors).to(device)
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32).to(device)
    value_targets = torch.tensor(np.array(value_targets), dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    optimizer.zero_grad()

    policy_pred_logits, value_pred = model(state_tensors)

    value_loss = F.mse_loss(value_pred, value_targets)
    policy_loss = F.cross_entropy(policy_pred_logits, policy_targets)
    total_loss = policy_loss + value_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item()

def main():
    """Main training loop."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    run = wandb.init(project=config.WANDB_PROJECT_NAME, id=config.WANDB_RUN_ID, resume="allow", job_type="training")

    run.config.update({k: v for k, v in config.__dict__.items() if k.isupper()}, allow_val_change=True)

    model = get_model(run, device)
    optimizer = optim.SGD(model.parameters(), lr=run.config.INITIAL_LR, momentum=0.9, weight_decay=run.config.L2_REGULARIZATION)
    scheduler = MultiStepLR(optimizer, milestones=run.config.LR_MILESTONES, gamma=0.1)

    network_wrapper = NetworkWrapper(model, device)
    replay_buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)

    start_game = run.step if run.resumed else 0

    for game_num in range(start_game + 1, run.config.MAX_GAMES + 1):
        print(f"\n--- Game {game_num}/{run.config.MAX_GAMES} ---")

        model.eval() # Set model to eval mode for self-play
        game_data = play_game(network_wrapper)
        replay_buffer.extend(game_data)

        print(f"  Training on replay buffer (size: {len(replay_buffer)})...")
        last_loss = None
        if len(replay_buffer) >= config.BATCH_SIZE:
            # Train for a number of steps proportional to new data
            num_training_steps = len(game_data) // config.BATCH_SIZE + 1
            for _ in range(num_training_steps):
                loss = train_step(model, optimizer, replay_buffer, device)
                if loss:
                    last_loss = loss
            scheduler.step() # Step scheduler once per training phase

            if last_loss is not None:
                 print(f"  Training complete. Last loss: {last_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        else:
            print("  Skipping training, replay buffer not full enough.")

        # Log all metrics once per game for clear, consistent plotting
        log_data = {"replay_buffer_size": len(replay_buffer), "learning_rate": scheduler.get_last_lr()[0]}
        if last_loss is not None:
            log_data["loss"] = last_loss
        run.log(log_data, step=game_num)

        if game_num % config.CHECKPOINT_INTERVAL == 0:
            print(f"Checkpoint at game {game_num}. Saving model to W&B.")
            model_filename = "alphago-zero.safetensors"

            save_file(model.state_dict(), model_filename)

            model_artifact = wandb.Artifact(
                name=config.WANDB_ARTIFACT_NAME, type="model",
                description=f"Model checkpoint after {game_num} games.",
                metadata={"game_num": game_num, "loss": last_loss}
            )
            model_artifact.add_file(model_filename)
            aliases = [f"game-{game_num}", "latest"]
            run.log_artifact(model_artifact, aliases=aliases)
            print(f"  Logged new model version to W&B Artifacts: {aliases}")

    print("\nTraining complete!")
    run.finish()

if __name__ == "__main__":
    main()
