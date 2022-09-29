# %% [markdown]
# ## Getting Started
# Import the pytorch library
# %%
import torch

# %% [markdown]
# Create a new tensor by arrange
# %%
x = torch.arange(12, dtype=torch.float32)
x

# %% [markdown]
# Number of elements
# %%
x.numel()

# %% [markdown]
# And shape of the element
# %%
x.shape

# %%
x.reshape([3, 4])

# %% [markdown]
# We can provide only one dim to reshape our tensor
# to infer the dim provide -1 to the place
# %%
x.reshape([-1, 4])

# %% [markdown]
# Or you can put dims as an arguments
# %%
X = x.reshape(-1, 4)
X

# %% [markdown]
# Array of zeroes
# %%
torch.zeros(2, 3, 4)

# %% [markdown]
# Array of ones
# %%
torch.ones(2, 3, 4)

# %% [markdown]
# Random init or the tensor
# %%
torch.randn(3, 4)

# %% [markdown]
# Create a tensor from python array
# %%
torch.tensor([
    [2, 3, 1, 0],
    [4, 5, 6, 7],
    [8, 7, 6, 5]
])

# %% [markdown]
## Indexing and Slicing
# %%
X

# %% [markdown]
# Select a row by slice
# %%
X[-1], X[1:3]

# %% [markdown]
# Rewrite elements of a matrix by index
# %%
X[1, 2] = 17
X

# %% [markdown]
# Very similar you can assign data to some slice
# %%
X[:2, :] = 12
X

# %% [markdown]
# ## Operations
# Element wise operations
# %%
torch.exp(x)

# %%
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

x + y, x - y, x * y, x / y, x ** y

# %% [markdown]
# Tensor concatenation
# %%
X = torch.arange(12, dtype=torch.float32).reshape(3, -1)
Y = torch.tensor([
    [2.0, 1, 4, 3],
    [1, 2, 3, 4],
    [4, 3, 2, 1]
])

# %% [markdown]
# Concat along rows
# %%
torch.cat((X, Y), dim=0)

# %% [markdown]
# Concat along cols
# %%
torch.cat((X, Y), dim=1)

# %% [markdown]
# Binary tensor via logical statement
# %%
X == Y

# %% [markdown]
# Summing all elements in the tensor
# %%
X.sum()

# %% [markdown]
# Broadcasting
# %%
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)

a, b
# %% [markdown]
# Because of the dimension are different,
# broadcasting makes this matrix 3 x 2 by
# duplicating cols for the first and rows
# for the second
# %%
a + b

# %% [markdown]
# Saving memory
# When you reassign the variable, you allocate the
# memory
# %%
before = id(Y)
Y = X + Y
before == id(Y)

# %% [markdown]
# In-place update
# %%
Z = torch.zeros_like(Y)
before = id(Z)

Z[:] = X + Y

before == id(Z)

# %% [markdown]
# Conversion to Other Python Objects
# %%
A = X.numpy()
B = torch.from_numpy(A)

type(A), type(B)

# %% [markdown]
# To convert a size-1 tensor to scalar
# %%
a = torch.tensor([3.5])

a, a.item(), float(a), int(a)

# %% [markdown]
# ## Exercises
# 1.
# %%
X < Y

# %%
X > Y

# %% [markdown]
# ## Exercises
# 2.
# %%
a = torch.arange(1, 5).reshape((4, 1))
b = torch.arange(1, 3).reshape((1, 2))
a, b

a + b

# %%
