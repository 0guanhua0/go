import torch
import torch.nn.functional as F

class NetworkWrapper:
    """
    Acts as a client to the InferenceServer.
    Instead of running the model locally, it sends inference requests
    to the central server process via multiprocessing queues/pipes.
    """
    def __init__(self, worker_id, request_queue, result_pipe):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_pipe = result_pipe
        # No model or device is stored here anymore.

    def predict(self, state_tensor_batch):
        """
        Sends a batch of states to the InferenceServer and waits for the result.
        The Rust MCTS implementation conveniently already batches the states for us.
        """
        # The state_tensor_batch is already on the CPU.
        # We send our worker_id and the tensor batch.
        # The tensor is automatically pickled when sent through the queue.
        self.request_queue.put((self.worker_id, state_tensor_batch))

        # Block and wait for the result to come back on our personal pipe.
        policy_probs, value = self.result_pipe.recv()

        # The result is already a NumPy array, ready for the MCTS.
        return policy_probs, value
