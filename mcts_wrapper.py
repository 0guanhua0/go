import torch
import torch.nn.functional as F

class NetworkWrapper:
    """Wraps the PyTorch model to be used by the MCTS."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, state_tensor):
        """
        Predicts policy and value from a board representation.
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = state_tensor.to(self.device)
            policy_logits, value = self.model(input_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
            value = value.cpu().numpy()
        return policy_probs, value
