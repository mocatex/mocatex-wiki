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

## 2. Vanilla Neural Networks

To build a neural network in PyTorch, you typically follow these steps:

1. Define a model by subclassing `nn.Module`.
2. Define a loss function and an optimizer.
3. Train the model using a training loop.
4. Evaluate the model on a validation/test set.
5. Save and load the model.

You normaly use `torch.nn` module which provides various layers and loss functions, and `torch.optim` module which provides optimization algorithms.

```py
import torch
import torch.nn as nn
import torch.optim as optim
```

### 2.1 Define a Model

You can define a neural network model by subclassing `nn.Module` and defining the *layers* in the `__init__` method and the *forward pass* in the `forward` method.

```py
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)    # First fully connected layer
        self.act_fn = nn.ReLU()         # ReLU activation function
        self.fc2 = nn.Linear(50, 1)     # Second fully connected layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x
```

> a Layer is built: `nn.Linear(in_features, out_features)`

We then create an instance of the model and move it to the appropriate device (CPU or GPU).

```py
model = SimpleNN()
model.to(device)  # Move model to the appropriate device
```

### 2.2 Datasets and DataLoaders

PyTorch provides utilities to handle datasets and data loading through `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.

```py
from torch.utils.data import Dataset, DataLoader
```

#### 2.2.3 Custom Dataset

You can create a custom dataset by subclassing `Dataset` and implementing the `__len__` and `__getitem__` methods.

```py
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # or create a data-generation logic here
    
    # Return the size of the dataset
    def __len__(self):
        return len(self.data)
    
    # Return a single sample and its label
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
```

#### 2.2.4 DataLoader

You can use `DataLoader` to create batches of data and shuffle the dataset.

```py
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

!!! info
    The most important parameters of `DataLoader` are:
    - `batch_size`: Number of samples per batch.
    - `shuffle`: Whether to shuffle the data at every epoch. (use it for training set!)
    - `num_workers`: Number of subprocesses to use for data loading. (Use ~ number of CPU cores)
    - `pin_memory`: will copy Tensors into CUDA pinned memory before returning. (use it when using Nvidia GPU)
    - `drop_last`: If `True`, the last incomplete batch will be dropped. (use it for training set!)

### 2.3. Activation Functions

The activation functions introduce non-linearity into the neural network, allowing it to learn complex patterns. PyTorch provides several activation functions in the `torch.nn` module.

#### 2.3.1 ReLU (Rectified Linear Unit)

### 2.4 Loss Functions

Loss functions measure **how wrong** a model’s predictions are compared to the true labels.
They tell the optimizer *how much and in which direction to adjust the weights*.

Conceptual Calculation of Loss over a batch:

$$
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \text{error}(y_i, \hat{y}_i)
$$

where

- $y_i$ = true value
- $\hat{y}_i$ = predicted value
- `error()` = some function that gets **larger when prediction is worse** (e.g. squared difference, log loss, etc.)
- $N$ = number of samples in the batch

The optimizer then tries to **minimize** this loss.

PyTorch provides various loss functions in the `torch.nn` module.

#### 2.4.1 Binary Cross Entropy Loss

Used for binary classification tasks.
So the model output should be in the range [0, 1] (use Sigmoid activation).

```py
criterion = nn.BCELoss()
# or for logits (more stable)
criterion = nn.BCEWithLogitsLoss()
```

> BCEWithLogitsLoss = Sigmoid + BCELoss
> It is advised to always prefer BCEWithLogitsLoss over BCELoss for numerical stability.
> Since it uses the log-sum-exp trick to compute the loss in a more stable way.
> -> $log(e^a + e^b) = max(a, b) + log(1 + e^{-|a-b|})$

#### More Loss Functions are following...

### 2.5 Optimizers

Optimizer have theese main functions:

- `optimizer.zero_grad()`: Clears old gradients from the last batch (otherwise they will accumulate)
- `loss.backward()`: Computes the gradient of the loss w.r.t. the model parameters
- `optimizer.step()`: Updates the model parameters based on the computed gradients

PyTorch provides various optimization algorithms in the `torch.optim` module.

#### 2.5.1 Stochastic Gradient Descent (SGD)

Updates the parameters using the gradient of the loss function.

We can add hyperparameters:

- `lr`: Learning rate (step size for each update). -> Always required.
- `momentum`: Accelerates SGD by adding a fraction of the previous update to the current update.
- `weight_decay`: L2 regularization term to prevent overfitting. It adds a penalty proportional to the square of the magnitude of the parameters.

```py
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
```

!!! tip "good starting values"
    - `lr`: 0.001 to 0.1
    - `momentum`: 0.9
    - `weight_decay`: 1e-4 to 1e-3

### 2.6 Training Loop

A typical training loop in PyTorch involves iterating over the data, performing forward and backward passes, and updating the model parameters.

```py
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

            optimizer.zero_grad()               # Clear previous gradients
            outputs = model(inputs)             # Forward pass
            loss = criterion(outputs, labels)   # Compute loss
            loss.backward()                     # Backward pass (compute gradients)
            optimizer.step()                    # Update model parameters

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

### 2.7 Saving and Loading Models

You can save and load model weights using `torch.save()` and `torch.load()`.

```py
# Save model weights
torch.save(model.state_dict(), 'model.pth')

# Create evaluation model and load weights
model = SimpleNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode
```

> You can theoretically use any file extension, but `.pth` or `.pt` are commonly used for PyTorch models.

### 2.8 Evaluation

To evaluate the model, you typically switch to evaluation mode and disable gradient computation.

```py
def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
```