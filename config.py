# --- W&B Configuration ---
WANDB_PROJECT_NAME = "go-zero"
WANDB_ARTIFACT_NAME = "go-zero-model"
WANDB_RUN_ID = None # Set to a specific ID to resume a run

# --- MCTS and Game Configuration ---
# Values from the AlphaGo Zero paper for the smaller run
C_PUCT = 1.0
DIRICHLET_ALPHA = 0.03
DIRICHLET_EPSILON = 0.25

# --- Self-Play Training Configuration ---
NUM_SIMULATIONS_TRAIN = 128
REPLAY_BUFFER_SIZE = 500_000
BATCH_SIZE = 256 #2048
CHECKPOINT_INTERVAL = 8
MAX_GAMES = 10**6
INITIAL_LR = 1e-2
LR_MILESTONES = [400_000, 600_000] # Steps at which to decay learning rate
L2_REGULARIZATION = 1e-4
RESIGNATION_THRESHOLD = -0.95

# --- GTP Engine Configuration ---
NUM_SIMULATIONS_PLAY = 800 # Higher number for stronger competitive play

# --- Network Architecture ---
BOARD_SIZE = 19
IN_CHANNELS = 17 # 8 past boards for each player + 1 color plane
NUM_FILTERS = 128 # 256
NUM_RES_BLOCKS = 9 #19
