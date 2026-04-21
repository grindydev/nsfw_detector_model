
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
import optuna
import matplotlib.pyplot as plt
import helper_utils
import torch.nn.functional as F
from pprint import pprint

helper_utils.set_seed(15)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(device)


# ==================== STEP 1: FLEXIBLE CNN ARCHITECTURE ====================
# KEY CONCEPT: Unlike previous models with fixed architectures, this CNN's
# structure is DYNAMIC -- the number of layers, filters, and kernel sizes
# are determined by hyperparameters that Optuna will tune.

class FlexibleCNN(nn.Module):
    def __init__(self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size, num_classes):
        super(FlexibleCNN, self).__init__()

        blocks = []
        in_channels = 3

        for i in range(n_layers):
            out_channels = n_filters[i]
            kernel_size = kernel_sizes[i]
            padding = (kernel_size - 1) // 2

            block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            blocks.append(block)
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.dropout_rate = dropout_rate
        self.fc_size = fc_size
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(n_filters[-1], self.fc_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.fc_size, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==================== STEP 2: DEFINE THE OBJECTIVE FUNCTION ====================
# KEY CONCEPT: This is the function Optuna will call repeatedly.
# Each call:
#   1. Samples new hyperparameters from the search space
#   2. Builds a model with those hyperparameters
#   3. Trains the model
#   4. Returns the validation accuracy (what Optuna tries to maximize)
def objective(trial, device, total_trials):
    """
    Optuna objective: sample hyperparameters, train, return accuracy.
    """
    # --- Sample architecture hyperparameters ---
    # trial.suggest_int: pick an integer in the given range
    n_layers = trial.suggest_int("n_layers", 2, 5)

    # Each layer can have a different number of filters
    n_filters = [trial.suggest_int(f"n_filters_{i}", 16, 128) for i in range(n_layers)]

    # Each layer can have kernel size 3 or 5
    kernel_sizes = [trial.suggest_categorical(f"kernel_size_{i}", [3, 5]) for i in range(n_layers)]

    # --- Sample classifier hyperparameters ---
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_size = trial.suggest_int("fc_size", 64, 256)

    # -- Load Data
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader, val_loader, _, num_classes = get_dataloaders(
    batch_size=batch_size,
    val_fraction=0.2,
    test_fraction=0.0
    )

    # --- Build model ---
    model = FlexibleCNN(n_layers, n_filters, kernel_sizes, dropout_rate, fc_size, num_classes).to(device)

    # --- Train ---
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    n_epochs = 10

    print(f"\n{'='*60}")
    print(f"Trial {trial.number + 1}/{total_trials}")
    print(f"  layers={n_layers} | filters={n_filters} | kernels={kernel_sizes}")
    print(f"  lr={learning_rate:.5f} | dropout={dropout_rate:.3f} | fc_size={fc_size} | batch_size={batch_size}")
    print(f"  train={len(train_loader.dataset)} images | val={len(val_loader.dataset)} images")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_fcn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"  Epoch [{epoch+1:2d}/{n_epochs}] Loss: {avg_loss:.4f}")

    # --- Evaluate ---
    accuracy = helper_utils.evaluate_accuracy(model, val_loader, device)
    print(f"  → Val Accuracy: {accuracy*100:.2f}%")
    return accuracy



# ==================== STEP 3: RUN THE OPTIMIZATION ====================
# Create an Optuna study that tries to MAXIMIZE accuracy
study = optuna.create_study(direction='maximize')

# Run 20 trials (each trial trains a model from scratch)
# In practice, you'd use more trials (50-100+) for better results
n_trials = 20
study.optimize(lambda trial: objective(trial, device, n_trials), n_trials=n_trials)

# View all trial results
df = study.trials_dataframe()

# Print the best result
best_trial = study.best_trial
print("Best trial:")
print(f"  Value (Accuracy): {best_trial.value:.4f}")
print("  Hyperparameters:")
pprint(best_trial.params)


# ==================== STEP 4: VISUALIZE RESULTS ====================
# How accuracy improved over trials
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.show()

# Which hyperparameters mattered most?
optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()

# Parallel coordinate plot: see all hyperparameter combinations at once
ax = optuna.visualization.matplotlib.plot_parallel_coordinate(
    study, params=['n_layers', 'n_filters_0', 'kernel_size_0', 'dropout_rate', 'fc_size']
)
fig = ax.figure
fig.set_size_inches(12, 6, forward=True)
fig.tight_layout()

# ==================== SAVE RESULTS ====================
best = study.best_trial.params

# Save Optuna study database (can reload later)
study.save("models/optuna_study.db")
print("✅ Optuna study saved to models/optuna_study.db")

# Rebuild with best params
n_layers = best["n_layers"]
n_filters = [best[f"n_filters_{i}"] for i in range(n_layers)]
kernel_sizes = [best[f"kernel_size_{i}"] for i in range(n_layers)]

model = FlexibleCNN(
    n_layers, n_filters, kernel_sizes,
    best["dropout_rate"], best["fc_size"], num_classes=5
).to(device)

# Retrain with best hyperparameters
train_loader, val_loader, _, _ = get_dataloaders(
    batch_size=best["batch_size"],
    val_fraction=0.1,
    test_fraction=0.5
)

optimizer = optim.Adam(model.parameters(), lr=best["lr"])
loss_fcn = nn.CrossEntropyLoss()
n_epochs = 10

print(f"\n{'='*60}")
print(f"Retraining best model for {n_epochs} epochs...")
print(f"{'='*60}")

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = loss_fcn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / len(train_loader.dataset)
    print(f"  Epoch [{epoch+1:2d}/{n_epochs}] Loss: {avg_loss:.4f}")

best_accuracy = helper_utils.evaluate_accuracy(model, val_loader, device)

# Save checkpoint (same format as main.py so evaluate.py can load it)
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': 5,
    'val_accuracy': best_accuracy * 100,
    'epoch': n_epochs,
    'best_params': best,
}, 'models/best_flexible_cnn.pth')
print(f"✅ Best model saved to models/best_flexible_cnn.pth (accuracy: {best_accuracy*100:.2f}%)")