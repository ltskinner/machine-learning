
import torch

torch.manual_seed(1)
torch.cuda.current_device()

"""
N       is batch_size
D_in    is input dimension
H       is hidden dimension
D_out   is output dimension
"""

N, D_in, H, D_out = 64, 1000, 100, 10
