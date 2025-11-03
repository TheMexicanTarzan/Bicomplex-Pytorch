# BiComplex PyTorch

Neural networks with bicomplex numbers using idempotent representation for an elegant adaptation with PyTorch.

## Overview

BiComplex PyTorch enables building neural networks that operate on bicomplex numbers (a 4-dimensional extension of complex numbers). By leveraging the idempotent representation, bicomplex operations decompose into pairs of independent complex operations, ensuring:

- **Numerical stability** - Avoids zero divisor issues
- **Efficient computation** - Leverages optimized complex arithmetic
- **Seamless integration** - Compatible with PyTorch autograd
- **Flexible architectures** - Shared or independent weight branches

## What are Bicomplex Numbers?

Bicomplex numbers are a 4-dimensional hypercomplex number system:
```
z = a + bi + cj + dij
```

where `i² = j² = -1` and `ij = ji`.

Using the **idempotent representation**, any bicomplex number can be expressed as:
```
z = z₁e₁ + z₂e₂
```

where `z₁, z₂ ∈ ℂ` and `e₁, e₂` are idempotent elements. This allows treating bicomplex operations as pairs of independent complex operations.

## Installation

### From PyPI (recommended)
...

### From source
...

### Dependencies
...

## Features
...

### Available Layers

- **Linear Layers**: `BiComplexLinear`
- **Convolutional Layers**: `BiComplexConv1d`, `BiComplexConv2d`, `BiComplexConv3d`
- **Normalization**: `BiComplexBatchNorm1d`, `BiComplexBatchNorm2d`
- **Pooling**: `BiComplexMaxPool2d`, `BiComplexAvgPool2d`
- **Dropout**: `BiComplexDropout`

### Activation Functions

- `BiComplexReLU`
- `BiComplexTanh`
- `BiComplexSigmoid`
- `BiComplexLeakyReLU`

### Loss Functions

- `BiComplexMSELoss`
- `BiComplexCrossEntropyLoss`
- `BiComplexL1Loss`

### Weight Initialization
```python
from bicomplex_pytorch.nn.init import bicomplex_kaiming_uniform_, bicomplex_xavier_uniform_

# Initialize layer weights
layer = BiComplexLinear(10, 20)
bicomplex_kaiming_uniform_(layer.branch1.weight)
bicomplex_xavier_uniform_(layer.branch2.weight)
```

## Advanced Usage

### Shared vs Independent Weights
```python
# Independent weights (default) - more expressive
layer_independent = BiComplexLinear(10, 20, shared_weights=False)

# Shared weights - fewer parameters, enforces symmetry
layer_shared = BiComplexLinear(10, 20, shared_weights=True)
```

### Working with Idempotent Representation
```python
from bicomplex_pytorch import to_idempotent, from_idempotent

# Convert bicomplex tensor to idempotent form
x = torch.randn(32, 10, 4)  # bicomplex
z1, z2 = to_idempotent(x)   # two complex tensors

# Perform operations on complex components
z1_processed = complex_operation(z1)
z2_processed = complex_operation(z2)

# Convert back to bicomplex
result = from_idempotent(z1_processed, z2_processed)
```

### Custom Architectures
```python
class BiComplexResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BiComplexConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BiComplexBatchNorm2d(64)
        self.relu = BiComplexReLU()
        self.pool = BiComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        # ... more layers
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        # ... more operations
        return x
```

## Examples

Check out the [`examples/`](examples/) directory for complete examples:

...

## Documentation

Full documentation is available at ...

## Contributing

Contributions are welcome! Please see our [Contributing Guide] ...

### Development Setup
...

### Running Tests
```bash
pytest tests/ -v --cov=bicomplex_pytorch
```

### Code Style

We use `black` for formatting and `flake8` for linting:
```bash
black bicomplex_pytorch/
flake8 bicomplex_pytorch/
```

## Citation
...


