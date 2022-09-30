# %% [markdown]
# Let's create a test CSV file
# %%
import pandas as pd
import os

data_path = os.path.join('..', 'data')

os.makedirs(data_path, exist_ok=True)
data_file = os.path.join(data_path, 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')

# %% [markdown]
# Let's import pandas and load the dataset
# %%
data = pd.read_csv(data_file)
print(data)

# %%
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs, targets

# %% [markdown]
# Fill NA values and add a new col (dummy_na) that mark
# all NA
# %%
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# %% [markdown]
# Fill NA values by the mean
# %%
inputs = inputs.fillna(inputs.mean())
print(inputs)

# %% [markdown]
# Conversion to the Tensor Format
# %%
import torch

X, y = torch.tensor(inputs.values), torch.tensor(targets.values)
X, y

# %% [markdown]
# ### Exercises
# 1.
# %%
path_to_the_data = os.path.join(data_path, 'abalone.data')

abalone_data = pd.read_csv(path_to_the_data, names=[
    'Sex',
    'Length',
    'Diameter',
    'Height',
    'WholeWeight',
    'ShuckedWeight',
    'VisceraWeight',
    'ShellWeight',
    'Rings'
])

abalone_data.head()

# %%
abalone_data.describe()

# %% [markdown]
# There is no NA rows!
# %%
abalone_data[abalone_data.isna().any(axis=1)]

# %%
abalone_data[['Sex', 'Length']]

# %%

# %%
