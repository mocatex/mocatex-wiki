# PyTorch

![PyTorch Logo](../assets/images/python/pytorch-logo.png){.center width="90%"}

> PyTorch is an open-source machine learning library. It is widely used for applications such as computer vision and natural language processing, and is known for its flexibility and ease of use.

```py title="import_pytorch.py"
import torch                    # Main PyTorch library
import torch.nn as nn           # Neural Networks
import torch.optim as optim     # Optimization algorithms
```

## 1. General Information

### 1.1 Comon Practices

- use GPU if available (Windows/Linux -> CUDA, Mac -> MPS)

    ```py
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    ```

- fixed random seed for reproducibility
   
    ```py
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch.mps.manual_seed(42)
    else:
        torch.manual_seed(42)
    ```

- set operations to deterministic (may slow down training)

    ```py
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    ```

### 1.2 Fundamental Concepts

#### Tensors

Tensors are the fundamental building blocks in PyTorch. They are similar to NumPy arrays but can also be used on GPUs for faster computation.

```py
# Create a 2D tensor (matrix) of shape (3, 4)
tensor = torch.Tensor(3, 4)  # Uninitialized
tensor = torch.zeros(3, 4)   # Filled with zeros
tensor = torch.ones(3, 4)    # Filled with ones
tensor = torch.rand(3, 4)    # Filled with random values between 0 and 1
tensor = torch.randn(3, 4)   # Filled with random values from a _normal distribution_
```

You can manipulate tensors using various operations:

```py
x = x.view(4, 3)  # Reshape tensor to shape (4, 3)
x = x.t()         # Transpose the tensor
x = x +,-,*,/ y  # Element-wise operations
x = x.permute(1, 0)  # Change the order of dimensions
```

#### matmul vs. mm vs. bmm vs. einsum

- `matmul` or `@`: General matrix multiplication that supports broadcasting. It can handle 1D, 2D, and higher-dimensional tensors.
- `mm`: Specifically for 2D tensors (matrices). It does not support broadcasting.
- `bmm`: Batch matrix multiplication for 3D tensors. It multiplies batches of matrices.
- `einsum`: Einstein summation convention, which provides a way to specify complex tensor operations in a concise manner.

> **broadcasting**: Automatically expands the dimensions of tensors to make their shapes compatible for element-wise operations.

!!! tip
    Torch Tensors can be converted to NumPy arrays and vice versa:

    ```py
    # PyTorch tensor to NumPy array
    tensor = torch.ones(3, 4)
    array = tensor.numpy()
    # NumPy array to PyTorch tensor
    array = np.ones((3, 4))
    tensor = torch.from_numpy(array)
    ```

    You can also use standard Python indexing and slicing on tensors like you would with NumPy arrays.

#### Autograd

Autograd is PyTorch's automatic differentiation engine that powers neural network training. It tracks operations on tensors to compute gradients automatically.

```py
# Create a tensor with requires_grad=True to track computations
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

# Compute gradients
out.backward()
print(x.grad)  # Gradient of out with respect to x
```

> The value in `x.grad` mean: if x[i] increases by t then out increases by x.grad[i] * t.
> If you want to stop tracking history on a tensor, you can use `with torch.no_grad():` or `tensor.detach()`.

