
import torch
import torch.nn as nn
import torch.optim as optim
import src.optuna as optuna
import matplotlib.pyplot as plt
import src.helper_utils as helper_utils
import torch.nn.functional as F
from pprint import pprint


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(device)


