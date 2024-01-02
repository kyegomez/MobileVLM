import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LDP(nn.Module):
    """
    Lightweight Downsample Projection (LDP) layer as described in the diagram.

    The LDP layer applies two sequences of depthwise and pointwise convolutions.
    After the first sequence of depthwise and pointwise convolution, a pixel-wise addition is performed with the input of the first depthwise convolution.
    After the second depthwise convolution, another pixel-wise addition is performed with the input to the LDP block.
    """

    def __init__(
        self, in_channels: int, out_channels: int, depth: int
    ):
        super(LDP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # First pointwise convolution followed by GELU
        self.pointwise1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
        self.gelu1 = nn.GELU()

        # First depthwise convolution with stride=1
        self.depthwise1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
        )

        # Second pointwise convolution
        self.pointwise2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1
        )

        # Second depthwise convolution with stride=2
        self.depthwise2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
        )

        # Third pointwise convolution
        self.pointwise3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1
        )

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MobileVLM model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        # Save the input for pixel-wise addition later
        B, C, H, W = x.shape
        identity = x

        for i in range(self.depth):
            # First pointwise and GELU activation
            x = self.pointwise1(x)
            x = self.gelu1(x)
            x = self.pointwise1(x)

            # Save the output for pixel-wise addition after the first depthwise
            skip_connection = x

            # First depthwise convolution
            x = self.depthwise1(x)
            x = self.pointwise1(x)
            # x = self.norm(x)

            # Pixel-wise addition from the first pointwise convolution
            x = x + skip_connection

            # Second depthwise convolution
            x = self.depthwise2(x)

            # Pixel-wise addition from the input to the LDP block
            if identity.shape != x.shape:
                # Resize identity to match x's dimensions
                identity = F.interpolate(
                    identity,
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            # x = x + identity

            # Third pointwise convolution
            x = self.pointwise3(x)

        return x


# Example usage:
# ldp = LDP(in_channels=128, out_channels=128)
# input_tensor = torch.randn(1, 128, 64, 64)  # Example input
# output = ldp(input_tensor)
# print(output.shape)
