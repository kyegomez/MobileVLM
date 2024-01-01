[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MobileVLM
Implementation of the LDP module block in PyTorch and Zeta from the paper: "MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices"


# Install
`pip3 install mobilevlm`


## Usage
```python
# Import the necessary libraries
import torch
from mobilevlm import LDP

# Create an instance of the LDP model
ldp = LDP(in_channels=128, out_channels=128, depth=3)

# Create an example input tensor
input_tensor = torch.randn(1, 128, 64, 64)

# Pass the input tensor through the LDP model to get the output
output = ldp(input_tensor)

# Print the shape of the output tensor
print(output.shape)

```


## Lightweight Downsample Projection (LDP) Layer

The Lightweight Downsample Projection (LDP) Layer is a component designed for efficient feature extraction and dimensionality reduction in convolutional neural networks. The LDP layer is particularly suited for mobile and edge devices where computational resources are limited. 

The LDP layer combines depthwise separable convolutions with pointwise convolutions and skip connections, allowing for a reduced number of parameters while maintaining a rich feature representation. The incorporation of Layer Normalization stabilizes the training process and allows for faster convergence.

### Architecture

The LDP layer is structured as follows:

1. **Initial Pointwise Convolution**: This is a 1x1 convolution that transforms the input feature map to the desired number of channels. It is computationally efficient and serves as a channel-wise feature transformation.

2. **GELU Activation**: After the initial pointwise convolution, we apply a Gaussian Error Linear Unit (GELU) activation function. GELU provides non-linearity to the model, allowing it to learn more complex patterns.

3. **First Depthwise Convolution**: A depthwise convolution with a stride of 1 follows, which applies a single filter per input channel. It is used for spatial feature extraction without altering the dimensionality of the feature map.

4. **First Skip Connection**: The output of the first depthwise convolution is added back to the output of the initial pointwise convolution. This skip connection allows gradients to flow directly through the network, mitigating the vanishing gradient problem and enabling deeper architectures.

5. **Second Pointwise Convolution**: Another 1x1 convolution is applied to further mix the channel-wise features.

6. **Layer Normalization**: Normalization is applied over the channel dimension to stabilize the mean and variance of activations, leading to improved training dynamics.

7. **Second GELU Activation**: A second GELU activation function is applied for additional non-linearity.

8. **Second Depthwise Convolution**: This depthwise convolution has a stride of 2, halving the spatial dimensions of the feature map and effectively downsampling the input.

9. **Second Skip Connection**: A pixel-wise addition combines the downsampled input to the block with the output of the second depthwise convolution. This connection helps to preserve information lost due to downsampling.

10. **Third Pointwise Convolution**: A final 1x1 convolution adjusts the channel dimensions if necessary and refines the features before passing them to subsequent layers.

11. **Layer Normalization**: Another layer normalization is applied to the output of the final pointwise convolution.

## Why It Works

The LDP layer is designed to capture the essence of the input features while reducing the spatial resolution in a computationally efficient manner. The use of depthwise separable convolutions significantly decreases the number of parameters compared to standard convolutions, reducing both the computational cost and the risk of overfitting.

Skip connections not only help to preserve information throughout the layer but also improve gradient flow during backpropagation, allowing for deeper network architectures. Layer Normalization is known to accelerate training and make the model less sensitive to initialization and learning rate choices.

This combination of efficiency and robustness makes the LDP layer a versatile component in designing neural networks for resource-constrained environments.



# Citation
```bibtex
@misc{chu2023mobilevlm,
    title={MobileVLM : A Fast, Reproducible and Strong Vision Language Assistant for Mobile Devices}, 
    author={Xiangxiang Chu and Limeng Qiao and Xinyang Lin and Shuang Xu and Yang Yang and Yiming Hu and Fei Wei and Xinyu Zhang and Bo Zhang and Xiaolin Wei and Chunhua Shen},
    year={2023},
    eprint={2312.16886},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


# License
MIT

