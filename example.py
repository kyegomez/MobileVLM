# Import the necessary libraries
import torch
from mobilevlm import LDP

# Create an instance of the LDP model
ldp = LDP(in_channels=128, out_channels=128)

# Create an example input tensor
input_tensor = torch.randn(1, 128, 64, 64)

# Pass the input tensor through the LDP model to get the output
output = ldp(input_tensor)

# Print the shape of the output tensor
print(output.shape)
