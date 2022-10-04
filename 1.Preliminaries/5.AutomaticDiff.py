# %% [markdown]
# Simple function:
# $ y = 2 x^T x $
# %%
import torch

x = torch.arange(4.0, requires_grad=True)
x

# %% [markdown]
# Create a function
# %%
y = 2 * torch.dot(x, x)
y

# %%
y.backward()
x.grad

# %% [markdown]
# Based on definition we know that
# $$ y = 2 x^T x $$
# $$ \frac{dy}{dx} = 4x $$
# So we can check it by:
# %%
x.grad == 4 * x

# %%
x.grad.zero_()  # Reset the gradient

y = x.sum()
y.backward()
x.grad

# %% [markdown]
# Detaching Computation
# %%
x.grad.zero_()
y = x * x

u = y.detach()
z = u * x
z.sum().backward()

x.grad == u

# %%
