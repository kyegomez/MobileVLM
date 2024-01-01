import torch
import torch.nn as nn
import torch.nn.functional as F


class LDP(nn.Module):
    """
    Lightweight Downsample Projection (LDP) layer implementation in PyTorch with a depth parameter.

    The LDP layer applies a sequence of depthwise separable convolutions with GELU activation function.
    The sequence can be repeated multiple times based on the 'depth' parameter.
    The final output of the sequence is added to the original input via a pixel-wise addition.

    Parameters:
    in_channels (int): Number of channels in the input image
    out_channels (int): Number of channels produced by the convolutions
    depth (int): Number of times the sequence of operations is repeated
    
    
    Example usage:
    ldp = LDP(in_channels=128, out_channels=128, depth=3)
    input_tensor = torch.randn(1, 128, 64, 64)  # Example input
    output = ldp(input_tensor)
    
    """

    def __init__(
        self, in_channels: int, out_channels: int, depth: int = 1
    ):
        super(LDP, self).__init__()
        self.depth = depth

        # Define the operations for each depth in nn.ModuleList
        self.operations = nn.ModuleList()
        for _ in range(depth):
            layers = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    groups=in_channels,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_channels,
                ),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
            )
            self.operations.append(layers)

    def forward(self, x):
        # Save the input for pixel-wise addition later
        identity = x

        # Apply the sequence of operations for each depth
        for operation in self.operations:
            x = operation(x)

            # Downsample the identity to match the size of x if needed
            if identity.shape != x.shape:
                identity = F.interpolate(
                    identity,
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

            x += identity
            identity = x  # Update identity to the output of the current depth

        return x


# Example usage:
ldp = LDP(in_channels=128, out_channels=128, depth=3)
input_tensor = torch.randn(1, 128, 64, 64)  # Example input
output = ldp(input_tensor)
print(output.shape)
