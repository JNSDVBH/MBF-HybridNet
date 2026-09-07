"""
Configuration file for MBF-HybridNet

This file contains the general training
and model hyperparameters.
"""


# =========================
# Training settings
# =========================

BATCH_SIZE = 1

EPOCHS = 100

LR = 0.002

WEIGHT_DECAY = 1e-4


# =========================
# Optimization
# =========================

OPTIMIZER = "Adam"

LOSS_FUNCTION = "MSELoss"


# =========================
# Validation strategy
# =========================

VALIDATION_METHOD = "Leave-One-Group-Out"


# =========================
# Model settings
# =========================

OUTPUT_DIM = 1


# =========================
# Random seed
# =========================

RANDOM_SEED = 42