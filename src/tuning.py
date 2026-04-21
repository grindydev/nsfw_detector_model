
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
import optuna
import matplotlib.pyplot as plt
import helper_utils
import joblib
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
                    padding=padding,
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

    # Each layer picks from GPU-friendly powers of 2
    FILTER_CHOICES = [16, 32, 64, 128]
    n_filters = [trial.suggest_categorical(f"n_filters_{i}", FILTER_CHOICES) for i in range(n_layers)]

    # Fixed kernel=3 everywhere — simpler, faster, sufficient for NSFW patterns
    kernel_sizes = [3] * n_layers

    # --- Sample classifier hyperparameters ---
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_size = trial.suggest_categorical("fc_size", [64, 128, 256])

    # -- Load Data (subset for fast Optuna trials)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_loader, val_loader, _, num_classes = get_dataloaders(
        batch_size=batch_size,
        val_fraction=0.15,
        test_fraction=0.0,
        train_fraction=0.3,
        num_workers=4
    )

    # --- Build model ---
    model = FlexibleCNN(n_layers, n_filters, kernel_sizes, dropout_rate, fc_size, num_classes).to(device)

    # --- Train + evaluate each epoch, track best ---
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    n_epochs = 10

    print(f"\n{'='*60}")
    print(f"Trial {trial.number + 1}/{total_trials}")
    print(f"  layers={n_layers} | filters={n_filters} | kernels={kernel_sizes}")
    print(f"  lr={learning_rate:.5f} | dropout={dropout_rate:.3f} | fc_size={fc_size} | batch_size={batch_size}")
    print(f"  train={len(train_loader.dataset)} images | val={len(val_loader.dataset)} images")
    print(f"{'='*60}")

    best_trial_accuracy = 0.0

    for epoch in range(n_epochs):
        helper_utils.train_model(
            model=model,
            train_dataloader=train_loader,
            n_epochs=1,
            loss_fcn=loss_fcn,
            optimizer=optimizer,
            device=device
        )

        val_accuracy = helper_utils.evaluate_accuracy(model, val_loader, device)
        marker = " ← best so far" if val_accuracy > best_trial_accuracy else ""
        print(f"  Epoch [{epoch+1:2d}/{n_epochs}] Val Acc: {val_accuracy*100:.2f}%{marker}")

        if val_accuracy > best_trial_accuracy:
            best_trial_accuracy = val_accuracy

    print(f"  → Best Val Accuracy: {best_trial_accuracy*100:.2f}%")
    return best_trial_accuracy



# ==================== STEP 3: RUN THE OPTIMIZATION ====================
# Create an Optuna study that tries to MAXIMIZE accuracy
study = optuna.create_study(direction='maximize')

# Run 20 trials with subset data for fast iteration
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
    study, params=['n_layers', 'n_filters_0', 'dropout_rate', 'fc_size']
)
fig = ax.figure
fig.set_size_inches(12, 6, forward=True)
fig.tight_layout()

# ==================== SAVE RESULTS ====================
best = study.best_trial.params

# Save Optuna study database (can reload later)
import joblib
joblib.dump(study, "models/optuna_study.pkl")
print("✅ Optuna study saved to models/optuna_study.pkl")

# Rebuild with best params
n_layers = best["n_layers"]
FILTER_CHOICES = [16, 32, 64, 128]
n_filters = [best[f"n_filters_{i}"] for i in range(n_layers)]
kernel_sizes = [3] * n_layers

model = FlexibleCNN(
    n_layers, n_filters, kernel_sizes,
    best["dropout_rate"], best["fc_size"], num_classes=5
).to(device)

# Retrain with best hyperparameters on FULL data
train_loader, val_loader, _, _ = get_dataloaders(
    batch_size=best["batch_size"],
    val_fraction=0.15,
    test_fraction=0.0,
    train_fraction=1.0,
    num_workers=4
)

optimizer = optim.AdamW(model.parameters(), lr=best["lr"], weight_decay=best.get("weight_decay", 0.01))
loss_fcn = nn.CrossEntropyLoss(label_smoothing=best.get("label_smoothing", 0.0))
n_epochs = 10

print(f"\n{'='*60}")
print(f"Retraining best model for {n_epochs} epochs on FULL data ({len(train_loader.dataset)} images)...")
print(f"  lr={best['lr']:.5f} | weight_decay={best.get('weight_decay', 0.01):.5f} | label_smoothing={best.get('label_smoothing', 0.0):.3f}")
print(f"  dropout={best['dropout_rate']:.3f} | fc_size={best['fc_size']} | batch_size={best['batch_size']}")
print(f"{'='*60}")

helper_utils.train_model(
    model=model,
    train_dataloader=train_loader,
    n_epochs=n_epochs,
    loss_fcn=loss_fcn,
    optimizer=optimizer,
    device=device
)

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