# %% [markdown]
# Scalars
# %%
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x ** y

# %% [markdown]
# Vectors implemented as 1-st order tensors
# %%
x = torch.arange(3)
x

# %% [markdown]
# Access to the elements via indexing
# %%
x[2]

# %% [markdown]
# You can check `x.shape` and `len(x)`
# %%
x.shape, len(x)

# %% [markdown]
# Matrices
# 2-dims tensors
# %%
A = torch.arange(6).resize(3, 2)
A

# %% [markdown]
# Matrix Transposition
# %%
A.T

# %% [markdown]
# Symmetric matrices $A = A^T$
# %%
A = torch.tensor([
    [1, 2, 3],
    [2, 5, 2],
    [3, 2, 9]
])

A == A.T

# %% [markdown]
# Tensors
# %%
A = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
A

# %% [markdown]
# Basic properties of tensor arithmetics
# %%
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()

A, A + B

# %% [markdown]
# The elementwise product or Hadamard product
# (denoted by $\odot$)
# $$A \odot B = \begin{bmatrix}
#   a_{11}b_{11} & a_{12}b_{12} & \cdots & a_{1n}b_{1n} \\
#   a_{21}b_{21} & a_{22}b_{22} & \cdots & a_{2n}b_{2n} \\
#   \vdots       & \vdots       & \ddots & \vdots       \\
#   a_{m1}b_{m1} & a_{m2}b_{m2} & \cdots & a_{mn}b_{mn}
# \end{bmatrix}
# $$
# %%
A * B

# %% [markdown]
# Adding or multiplying scalar to tensor
# produce the same result shape
# %%
X = torch.arange(24).reshape(2, 4, 3)
X

# %%
a = 2
X + a, X * a

# %%
(X + a).shape == X.shape

# %% [markdown]
# Reduction
# To express the sum of the elements in the vector
# $x$ of length $n$ we write $\sum_{i=1}^{n} x_{i}$
# %%
x = torch.arange(3, dtype=torch.float32)
x, x.sum()

# %% [markdown]
# Sum of the $m \times n$ matrix $A$ could be written as
# $\sum_{i=1}^{m} \sum_{j=1}^{m} a_{ij}$
# %%
A, A.sum()

# %% [markdown]
# Sum along the rows / cols
# %%
A.sum(axis=0), A.sum(axis=1)

# %% [markdown]
# Sum along all of the axis the same as
# regular sum
# %%
A.sum(axis=[0, 1]) == A.sum()

# %% [markdown]
# Compute the mean
# %%
A.mean(dtype=torch.float32)

# %% [markdown]
# The same result as for the previous
# %%
A.sum() / A.numel()

# %% [markdown]
# We can reduce the tensor along specific axis
# %%
A.mean(axis=0, dtype=torch.float32), A.sum(axis=0) / A.shape[0]

# %% [markdown]
# Non-reduction sum
# We won't change the dims with `keepdims=True`
# param
# %%
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape

# %%
A / sum_A

# %%
A, A.cumsum(axis=0), A.cumsum(axis=1)

# %% [markdown]
# Vector dot product definition.
# Is the scalar in this case
# $$x^Ty = \sum_{i=1}^{d} x_i y_i$$
# %%
y = torch.ones(3, dtype=torch.float32)
x, y, torch.dot(x, y)

# %% [markdown]
# Hadamard product and sum give the same result
# %%
torch.sum(x * y)

# %% [markdown]
# Matrix-vector product definition
# $$ Ax = \begin{bmatrix}
#   a_1^Tx \\
#   a_2^Tx \\
#   \vdots \\
#   a_m^Tx
# \end{bmatrix} $$
# With the `torch.mv` or `@`
# %%
A, x, torch.mv(A, x), A @ x

# %% [markdown]
# Matrix-matrix multiplication
# %%
B = torch.ones(3, 4)
B

# %%
torch.mm(A, B), A @ B

# %% [markdown]
# Norms 
#
# $\ell_2$ norm:
# $$ \| x \|_2 = \sqrt{\sum_{i=1}^n x_i^2} $$
# %%
u = torch.tensor([3.0, -4.0])
torch.norm(u)

# %% [markdown]
# Manhattan distance $\| x \|_1 = \sum_{i=1}^{n} | x_i |$
# %%
torch.abs(u).sum()

# %% [markdown]
# Invoking the following function will
# calculate the Frobenius norm of a matrix
# %%
torch.norm(torch.ones((4, 9)))

# %%
