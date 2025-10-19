import torch

import dihedral


class Ring:
    def __init__(self, data, feature, board):
        self.lock = torch.multiprocessing.Lock()
        self.data = data
        self.board = board
        self.size = torch.multiprocessing.Value("i", 0)
        self.head = torch.multiprocessing.Value("i", 0)

        self.state = torch.zeros(data, feature, board, board, dtype=torch.bool)
        self.policy = torch.zeros(data, board * board + 1)
        self.value = torch.zeros(data, 1)

        self.state.share_memory_()
        self.policy.share_memory_()
        self.value.share_memory_()

    def add(self, data):
        state, policy, value = zip(*data)
        s = torch.stack(state).to(dtype=torch.bool)
        p = torch.stack(policy)
        v = torch.tensor(value).unsqueeze(-1)

        with self.lock:
            idx = torch.arange(self.head.value, self.head.value + len(data)) % self.data

            self.state[idx] = s
            self.policy[idx] = p
            self.value[idx] = v

            self.head.value = (self.head.value + len(data)) % self.data
            self.size.value = min(self.data, self.size.value + len(data))

    def sample(self, batch_size):
        with self.lock:
            idx = torch.randint(0, self.size.value, (batch_size,))
            state = self.state[idx].clone()
            policy = self.policy[idx].clone()
            value = self.value[idx].clone()

        transform = torch.randint(0, len(dihedral.apply), (batch_size,))

        state_dihedral = []
        policy_dihedral = []
        for s, p, t in zip(state, policy, transform):
            s = dihedral.apply[int(t.item())](s)

            policy_2d = p[:-1].view(self.board, self.board)
            policy_2d = dihedral.apply[int(t.item())](policy_2d)
            p = torch.cat(
                (
                    policy_2d.reshape(-1),
                    p[-1:].clone(),
                )
            )

            state_dihedral.append(s)
            policy_dihedral.append(p)

        state = (
            torch.stack(state_dihedral).to(dtype=torch.get_default_dtype()).contiguous()
        )
        policy = torch.stack(policy_dihedral).contiguous()
        value = value.contiguous()

        return state, policy, value

    def __len__(self):
        return self.size.value
