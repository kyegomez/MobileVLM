import torch
from mobilevlm.model import LDP


ldp = LDP(in_channels=128, out_channels=128, depth=3)
input_tensor = torch.randn(1, 128, 64, 64)  # Example input
output = ldp(input_tensor)
print(output.shape)
