import argparse
import hashlib
import logging
import os
import time

import torch
import torch.nn.functional as F
import json
import numpy as np
from types import SimpleNamespace

with open("config.json") as f:
    config = SimpleNamespace(**json.load(f))

from network import AlphaGoZero

LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(processName)s - %(message)s",
}


def weight_hash(weight):
    hasher = hashlib.sha256()
    for w in weight:
        b = w.detach().cpu().contiguous().numpy().tobytes()
        hasher.update(b)
    return hasher.hexdigest()


class Trainer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self._init_model()

    def _init_model(self):
        net = (
            config.board,
            config.history,
            config.conv_filter,
            config.res_block,
        )
        self.model = AlphaGoZero(*net).to(self.device)
        self.model.eval()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.initial_lr,
            momentum=0.9,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.lr_milestones, gamma=0.1
        )

    def save_checkpoint(self):
        self.model.eval()
        model_id = weight_hash(self.model.state_dict().values())
        os.makedirs("models", exist_ok=True)
        filename = f"models/{model_id}.pt"

        example_input = torch.zeros(
            1, config.history * 2 + 1, config.board, config.board
        ).to(self.device)

        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(filename)
        logging.info(f"saved checkpoint: {filename}")
        return model_id

    def train_step(self, state, policy, value):
        self.model.train()
        state = state.to(self.device)
        policy = policy.to(self.device)
        value = value.to(self.device)

        self.optimizer.zero_grad()
        policy_next, value_next = self.model(state)

        policy_loss = F.cross_entropy(policy_next, policy)
        value_loss = F.mse_loss(value_next, value)

        l2_penalty = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            if p.requires_grad and p.dim() > 1:
                l2_penalty += torch.sum(p.pow(2))

        loss = policy_loss + value_loss + config.l2_regularization * l2_penalty
        loss.backward()
        self.optimizer.step()
        return loss.item()


def main(args):
    logging.basicConfig(**LOGGING_CONFIG)
    trainer = Trainer(args.device)

    if not os.listdir("models") if os.path.exists("models") else True:
        trainer.save_checkpoint()

    data_path = "data/shuffle/latest.npz"

    while True:
        if not os.path.exists(data_path):
            logging.warning(f"waiting for {data_path}...")
            time.sleep(10)
            continue

        logging.info(f"loading data from {data_path}")
        try:
            data = np.load(data_path)
            features = torch.from_numpy(data["board"])
            policies = torch.from_numpy(data["policy"])
            values = torch.from_numpy(data["value"])
        except Exception as e:
            logging.error(f"failed to load data: {e}")
            time.sleep(5)
            continue

        num_samples = features.shape[0]
        logging.info(f"training on {num_samples} samples")

        total_loss = 0.0
        batches_done = 0

        # Shuffle indices for this epoch
        perm = torch.randperm(num_samples)

        for i in range(0, num_samples - config.batch_size + 1, config.batch_size):
            indices = perm[i : i + config.batch_size]
            loss = trainer.train_step(
                features[indices], policies[indices], values[indices]
            )
            total_loss += loss
            batches_done += 1

            if batches_done % 100 == 0:
                logging.info(f"step {batches_done}: loss={loss:.4f}")

            if batches_done >= config.training_updates_per_generation:
                break

        trainer.scheduler.step()
        logging.info(f"LR: {trainer.scheduler.get_last_lr()[0]}")

        trainer.save_checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args)
