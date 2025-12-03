import torch

#
ROI_NUM = 116
NUM_WINDOWS = 54
TIMEPOINTS = 30

BATCH_SIZE = 4
EPOCHS = 5

#  GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


NUM_CLUSTERS = 2

SEED=3407
W_D= 5e-4
LR = 1e-4
PATIENCE=70
