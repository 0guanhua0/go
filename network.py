import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, conv_filter):
        super().__init__()
        self.conv1 = nn.Conv2d(
            conv_filter, conv_filter, 3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(conv_filter)
        self.conv2 = nn.Conv2d(
            conv_filter, conv_filter, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(conv_filter)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = F.relu(x)
        return x


class AlphaGoZeroNet(nn.Module):
    def __init__(self, board, history, conv_filter, res_block):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(history * 2 + 1, conv_filter, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv_filter),
            nn.ReLU(),
        )
        self.res_tower = nn.Sequential(
            *(ResBlock(conv_filter) for _ in range(res_block))
        )

        self.policy_conv = nn.Conv2d(conv_filter, 2, 1, stride=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_l = nn.Linear(2 * board * board, board * board + 1)

        self.value_conv = nn.Conv2d(conv_filter, 1, 1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_l1 = nn.Linear(board * board, conv_filter)
        self.value_l2 = nn.Linear(conv_filter, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_tower(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_l(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_l1(value))
        value = torch.tanh(self.value_l2(value))

        return policy, value
