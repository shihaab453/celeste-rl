from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


torch.manual_seed(42)

x = torch.linspace(-2 * torch.pi, 2 * torch.pi, steps=200).reshape(-1, 1)
y = torch.sin(x)

indices = torch.randperm(len(x))
train_size = int(0.8 * len(x))
train_indices = indices[:train_size]
validation_indices = indices[train_size:]

x_train = x[train_indices]
y_train = y[train_indices]
x_validation = x[validation_indices]
y_validation = y[validation_indices]


class SineRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, inputs):
        return self.network(inputs)


def evaluate_loss(model, inputs, targets, loss_function):
    model.eval()
    with torch.no_grad():
        return loss_function(model(inputs), targets)


model = SineRegressor()

print("Model:")
print(model)

print("\nParameter shapes:")
for name, parameter in model.named_parameters():
    print(f"{name}: {tuple(parameter.shape)}")

sample_inputs = x_train[:5]
sample_targets = y_train[:5]

model.eval()
with torch.no_grad():
    sample_predictions = model(sample_inputs)

print("\nSample training inputs:")
print(sample_inputs)
print("\nSample targets:")
print(sample_targets)
print("\nUntrained predictions:")
print(sample_predictions)

print(f"\nInput shape: {tuple(sample_inputs.shape)}")
print(f"Target shape: {tuple(sample_targets.shape)}")
print(f"Prediction shape: {tuple(sample_predictions.shape)}")

assert sample_predictions.shape == sample_targets.shape, (
    "Predictions and targets must have the same shape, but got "
    f"{tuple(sample_predictions.shape)} and {tuple(sample_targets.shape)}"
)

print("Shape assertion passed.")

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 600
training_losses = []
validation_losses = []

for epoch in range(1, epochs + 1):
    model.train()

    # Prediction
    predictions = model(x_train)

    # Loss calculation
    loss = loss_function(predictions, y_train)
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"Training loss is not finite at epoch {epoch}")

    # Clearing gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Parameter update
    optimizer.step()

    training_loss = evaluate_loss(model, x_train, y_train, loss_function)
    validation_loss = evaluate_loss(model, x_validation, y_validation, loss_function)

    if not torch.isfinite(training_loss).item():
        raise FloatingPointError(f"Recorded training loss is not finite at epoch {epoch}")
    if not torch.isfinite(validation_loss).item():
        raise FloatingPointError(f"Validation loss is not finite at epoch {epoch}")

    training_losses.append(training_loss.item())
    validation_losses.append(validation_loss.item())

    if epoch == 1 or epoch % 100 == 0:
        print(
            f"Epoch {epoch:3d}: "
            f"training loss = {training_losses[-1]:.6f}, "
            f"validation loss = {validation_losses[-1]:.6f}"
        )

print(f"\nRecorded {len(training_losses)} finite training losses")
print(f"Recorded {len(validation_losses)} finite validation losses")

model.eval()
x_plot = torch.linspace(-2 * torch.pi, 2 * torch.pi, steps=500).reshape(-1, 1)
with torch.no_grad():
    true_y = torch.sin(x_plot)
    predicted_y = model(x_plot)

results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(parents=True, exist_ok=True)

curve_path = results_dir / "sine_predictions.png"
fig, ax = plt.subplots()
ax.plot(x_plot[:, 0].numpy(), true_y[:, 0].numpy(), label="True sine")
ax.plot(x_plot[:, 0].numpy(), predicted_y[:, 0].numpy(), label="Trained model")
ax.set(xlabel="x", ylabel="y", title="Sine curve and trained predictions")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(curve_path, dpi=150)
plt.close(fig)

loss_path = results_dir / "sine_loss_history.png"
fig, ax = plt.subplots()
epoch_numbers = range(1, epochs + 1)
ax.plot(epoch_numbers, training_losses, label="Training loss")
ax.plot(epoch_numbers, validation_losses, label="Validation loss")
ax.set(xlabel="Epoch", ylabel="Mean squared error", title="Training and validation loss")
ax.set_yscale("log")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(loss_path, dpi=150)
plt.close(fig)

print(f"Saved prediction plot: {curve_path}")
print(f"Saved loss plot: {loss_path}")
