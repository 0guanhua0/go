import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    A single residual block as described in the AlphaGo Zero paper.
    Each block consists of two convolutional layers with batch normalization
    and ReLU activations, and a skip connection.
    """
    def __init__(self, num_filters=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        """
        Forward pass for the residual block.
        """
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class AlphaGoZeroNet(nn.Module):
    """
    The full neural network for AlphaGo Zero. It consists of a residual tower
    followed by a policy head and a value head.

    Args:
        board_size (int): The side length of the Go board (e.g., 19).
        num_res_blocks (int): The number of residual blocks in the tower (e.g., 19 or 39).
        in_channels (int): The number of input feature planes (17 in the paper).
        num_filters (int): The number of filters used in convolutional layers (256 in the paper).
    """
    def __init__(self, board_size=19, num_res_blocks=19, in_channels=17, num_filters=256):
        super(AlphaGoZeroNet, self).__init__()
        self.board_size = board_size

        # Initial convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        # Residual tower
        res_tower = [ResBlock(num_filters) for _ in range(num_res_blocks)]
        self.res_tower = nn.Sequential(*res_tower)

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass for the full network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, board_size, board_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing policy logits and the predicted value.
                - policy_logits: Tensor of shape (batch_size, board_size*board_size + 1)
                - value: Tensor of shape (batch_size, 1)
        """
        # Pass through the initial convolutional block and residual tower
        x = self.conv_block(x)
        x = self.res_tower(x)

        # Policy head path
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)

        # Value head path
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

if __name__ == '__main__':
    # The paper describes two main versions of the network:
    # 1. 20-block network (1 conv block + 19 res blocks) used for the 3-day run.
    # 2. 40-block network (1 conv block + 39 res blocks) used for the 40-day run.

    # Example: Create the 20-block network
    print("Creating the 20-block network (1 conv block + 19 residual blocks)...")
    net_20_blocks = AlphaGoZeroNet(board_size=19, num_res_blocks=19, in_channels=17)

    # Let's test the 20-block network with a dummy input
    batch_size = 8
    board_size = 19
    in_channels = 17 # 8 planes for player stones, 8 for opponent, 1 for color

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, in_channels, board_size, board_size)

    print(f"\n--- Testing the 20-block network ---")
    print(f"Input shape: {dummy_input.shape}")

    # Pass the input through the network
    # In a real scenario, you might use `with torch.no_grad():` for inference
    policy_logits, value = net_20_blocks(dummy_input)

    # The policy head outputs logits for each of the 19x19=361 moves + 1 pass move
    policy_size = board_size * board_size + 1

    print(f"Output policy logits shape: {policy_logits.shape} (Expected: ({batch_size}, {policy_size}))")
    print(f"Output value shape: {value.shape} (Expected: ({batch_size}, 1))")

    # The policy logits would then be passed through a softmax function to get probabilities.
    # The value is already in the range [-1, 1] due to the tanh activation.
    policy_probs = F.softmax(policy_logits, dim=1)
    print(f"\nPolicy probabilities shape after softmax: {policy_probs.shape}")
    print(f"Sum of probabilities for the first sample in batch: {torch.sum(policy_probs[0]).item():.2f}")
    print(f"Example value output for the first sample: {value[0].item():.4f}")

    # --- Parameter Count ---
    # Quick check on the number of parameters
    num_params_20 = sum(p.numel() for p in net_20_blocks.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters in 20-block network: {num_params_20:,}")

    # Create and check the 40-block network
    net_40_blocks = AlphaGoZeroNet(board_size=19, num_res_blocks=39, in_channels=17)
    num_params_40 = sum(p.numel() for p in net_40_blocks.parameters() if p.requires_grad)
    print(f"Total trainable parameters in 40-block network: {num_params_40:,}")
