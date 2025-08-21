# --- W&B Configuration ---
WANDB_PROJECT_NAME = "go-zero"
WANDB_RUN_ID = None

# --- MCTS and Game Configuration ---
# Values from the AlphaGo Zero paper. MCTS specific values (C_PUCT)
# are from the smaller run as they were not specified for the larger run.
C_PUCT = 1.0
DIRICHLET_ALPHA = 0.03
DIRICHLET_EPSILON = 0.25

MAX_GENERATIONS = 2 ** 20
TRAINING_UPDATES_PER_GENERATION = 1000
NUM_EVAL_GAMES = 400
EVAL_WIN_THRESHOLD = 0.55

# --- ELO Configuration ---
ELO_INITIAL = 1000
ELO_K_FACTOR = 32
CHECKPOINT_FREQUENCY = 1000
CHECKPOINT_NAME = "alphago-zero-checkpoint"

# --- Self-Play Training Configuration ---
NUM_SIMULATIONS = 16#1600
REPLAY_BUFFER_SIZE = 500_000
BATCH_SIZE = 2048
INITIAL_LR = 1e-2
LR_MILESTONES = [400_000, 600_000]
L2_REGULARIZATION = 1e-4
RESIGNATION_THRESHOLD = -0.95

# --- Network Architecture ---
BOARD_SIZE = 19
IN_CHANNELS = 17
NUM_FILTERS = 64#256
NUM_RES_BLOCKS = 20#40
