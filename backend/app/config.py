import os

# Base directory (backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----- Model Architecture -----
SEQ_LEN = 256        # lookback window (v3: increased from 128 for better trend capture)
PRED_LEN = 30        # prediction horizon (overridden to 1 when USE_RECURSIVE_FORECAST=True)

# ----- Architecture Options -----
USE_CHANNEL_MIXER = True       # cross-channel linear mixing before decomposition
USE_RESIDUAL_BLOCK = True      # nonlinear residual temporal projection
DROPOUT_RATE = 0.1             # dropout after channel mixer

# ----- v3 Upgrades -----
USE_LOG_RETURNS = True         # train on log(p[t+1]/p[t]) instead of raw prices
USE_RECURSIVE_FORECAST = True  # 1-step recursive prediction loop
LOSS_TYPE = "mse"              # "mse" or "huber"
USE_BIAS_CORRECTION = True     # compute + apply mean prediction bias
BLEND_WEIGHT = 0.85            # persistence blending: blend*model + (1-blend)*last_price

# ----- Training -----
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 32
PATIENCE = 10                  # early stopping patience
WEIGHT_DECAY = 1e-5            # L2 regularization via optimizer
GRAD_CLIP_NORM = 1.0           # gradient clipping max norm
SEED = 42                      # random seed for reproducibility

# ----- Paths -----
DATA_PATH = os.path.join(BASE_DIR, "data", "uploaded.csv")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "dlinear.pth")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")
