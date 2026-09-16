import torch
import matplotlib.pyplot as plt

# Make the random split repeatable
torch.manual_seed(42)

# Create 200 input and target pairs
x = torch.linspace(
    -2 * torch.pi,
    2 * torch.pi,
    steps=200
).reshape(-1, 1)

y = torch.sin(x)

# Generate the numbers 0 to 199 in random order
indices = torch.randperm(len(x))

# Use 80% for training
train_size = int(0.8 * len(x))

train_indices = indices[:train_size]
validation_indices = indices[train_size:]

# Select the corresponding examples
x_train = x[train_indices]
y_train = y[train_indices]

x_validation = x[validation_indices]
y_validation = y[validation_indices]

plt.scatter(
    x_train[:, 0],
    y_train[:, 0],
    color="blue",
    label="Training"
)

plt.scatter(
    x_validation[:, 0],
    y_validation[:, 0],
    color="orange",
    label="Validation"
)

plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Training and validation data")
plt.legend()
plt.grid(True)
plt.show()