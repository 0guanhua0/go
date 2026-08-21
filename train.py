import argparse
from collections import Counter
import hashlib
import logging
import os

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

from network import AlphaGoZero


class StreamingDataset(IterableDataset):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

    def __iter__(self):
        data = np.load(self.path, allow_pickle=True).item()

        board = torch.from_numpy(data["board"])
        policy = torch.from_numpy(data["policy"])
        value = torch.from_numpy(data["value"])

        sample_cnt = board.shape[0]
        for i in range(0, sample_cnt - self.batch_size + 1, self.batch_size):
            yield (
                board[i : i + self.batch_size],
                policy[i : i + self.batch_size],
                value[i : i + self.batch_size],
            )


BATCH = int(os.environ["BATCH"])
BOARD = int(os.environ["BOARD"])
CONV_FILTER = int(os.environ["CONV_FILTER"])
DEVICE = os.environ["DEVICE"]
HISTORY = int(os.environ["HISTORY"])
INITIAL_LR = float(os.environ["INITIAL_LR"])
L2_REGULARIZATION = float(os.environ["L2_REGULARIZATION"])
LR_MILESTONES = eval(os.environ["LR_MILESTONES"])
RES_BLOCK = int(os.environ["RES_BLOCK"])

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
    def __init__(self):
        self.device = torch.device(DEVICE)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self._init_model()

    def _init_model(self):
        net = (
            BOARD,
            HISTORY,
            CONV_FILTER,
            RES_BLOCK,
        )
        self.model = AlphaGoZero(*net).to(self.device)
        self.model.eval()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=INITIAL_LR,
            momentum=0.9,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=LR_MILESTONES, gamma=0.1
        )

    def save_model(self, path):
        self.model.eval()
        model_id = weight_hash(self.model.state_dict().values())
        os.makedirs(path, exist_ok=True)
        model_input = torch.zeros(1, HISTORY * 2 + 1, BOARD, BOARD).to(self.device)
        torch.jit.trace(self.model, model_input).save(f"{path}/{model_id}.pt")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            f"{path}/{model_id}.state",
        )
        logging.info(f"save {model_id}")

    def load_model(self, path):
        pt = [f for f in os.listdir(path) if f.endswith(".pt")]
        model = max(pt, key=lambda f: os.path.getmtime(os.path.join(path, f)))
        self.model.load_state_dict(
            torch.jit.load(os.path.join(path, model)).state_dict()
        )
        state = torch.load(os.path.join(path, model.replace(".pt", ".state")))
        self.optimizer.load_state_dict(state["optimizer"])
        state["scheduler"]["milestones"] = Counter(LR_MILESTONES)
        self.scheduler.load_state_dict(state["scheduler"])
        logging.info(f"load {model}")

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
        loss = policy_loss + value_loss + L2_REGULARIZATION * l2_penalty
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, state, policy, value):
        self.model.eval()
        with torch.no_grad():
            state = state.to(self.device)
            policy = policy.to(self.device)
            value = value.to(self.device)
            policy_next, value_next = self.model(state)
            return F.cross_entropy(policy_next, policy).item(), F.mse_loss(
                value_next, value
            ).item()


def main(args):
    logging.basicConfig(**LOGGING_CONFIG)
    trainer = Trainer()
    if args.init:
        trainer.save_model("model")
        exit(0)
    trainer.load_model("model")
    train_dataset = StreamingDataset(args.data_train, BATCH)
    train_loader = DataLoader(train_dataset, batch_size=None)
    step = 0
    for board, policy, value in train_loader:
        loss = trainer.train_step(board, policy, value)
        step += 1
        if step % 100 == 0:
            logging.info(f"step {step} loss {loss:.4f}")
    valid_dataset = StreamingDataset(args.data_valid, BATCH)
    valid_loader = DataLoader(valid_dataset, batch_size=None)
    step = 0
    policy_loss, value_loss = 0.0, 0.0
    for board, policy, value in valid_loader:
        p, v = trainer.eval_step(board, policy, value)
        policy_loss += p
        value_loss += v
        step += 1
    logging.info(
        f"validation policy loss {policy_loss / step:.4f} value loss {value_loss / step:.4f}"
    )
    trainer.scheduler.step()
    logging.info(f"LR: {trainer.scheduler.get_last_lr()[0]}")
    trainer.save_model("eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--data-train")
    parser.add_argument("--data-valid")
    args = parser.parse_args()
    main(args)
