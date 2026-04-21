import math
import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from directory_tree import DisplayTree
import torch
from torchvision import transforms
import matplotlib.ticker as mticker

# Global plot style
PLOT_STYLE = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.family": "sans",  # "sans-serif",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 3,
    "lines.markersize": 6,
}

mpl.rcParams.update(PLOT_STYLE)

# Custom colors (reusable)
BLUE_COLOR_TRAIN = "#237B94"  # Blue
PINK_COLOR_TEST = "#F65B66"  # Pink

def set_seed(seed=42):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed: The integer value to use as the random seed.
    """
    # Set the seed for PyTorch CPU operations
    torch.manual_seed(seed)
    # Set the seed for PyTorch CUDA operations on all GPUs
    torch.cuda.manual_seed_all(seed)
    # Set the seed for NumPy's random number generator
    np.random.seed(seed)
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Configure CuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable the CuDNN benchmark mode, which can be non-deterministic
    torch.backends.cudnn.benchmark = False


def print_data_folder_structure(root_dir, max_depth=1):
    """Print the folder structure of the dataset directory."""
    config_tree = {
        "dirPath": root_dir,
        "onlyDirs": False,
        "maxDepth": max_depth,
        "sortBy": 1,  # Sort by type (files first, then folders)
    }
    DisplayTree(**config_tree)


def explore_extensions(root_dir):
    """Explore and print the file extensions in the dataset directory."""
    extensions = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                extensions[ext] = []
            extensions[ext].append(os.path.join(dirpath, filename))
    return extensions



def plot_training_metrics(metrics):
    """
    Plots the training and validation metrics from a model training process.

    This function generates two side-by-side plots:
    1. Training Loss vs. Validation Loss.
    2. Validation Accuracy.

    Args:
        metrics (list): A list or tuple containing three lists:
                        [train_losses, val_losses, val_accuracies].
    """
    # Unpack the metrics into their respective lists
    train_losses, val_losses, val_accuracies = metrics
    
    # Determine the number of epochs from the length of the training losses list
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configure the first subplot for training and validation loss ---
    # Select the first subplot
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    # Select the second subplot
    ax2 = axes[1]
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # --- Apply dynamic and consistent styling to both subplots ---
    # Calculate a suitable interval for the x-axis ticks to avoid clutter
    x_interval = (num_epochs - 1) // 10 + 1

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        
        # Set the major tick locator for the x-axis using the dynamic interval
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        # Set the font size for the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust subplot parameters for a tight layout
    plt.tight_layout()
    # Display the plots
    plt.show()
    
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Final Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    filename = 'data/confusion_matrix.png'
    plt.savefig(filename, dpi=200)
    plt.show()
    return filename


class NestedProgressBar:
    """
    Manages nested tqdm progress bars for loops like epochs and batches.

    This class provides a convenient way to display and control separate
    progress bars for outer and inner loops (e.g., training epochs and
    data batches). It supports both terminal and notebook environments
    and allows for conditional message logging at specified intervals.
    """
    def __init__(
        self,
        total_epochs,
        total_batches,
        g_epochs=None,
        g_batches=None,
        epoch_message_freq=None,
        batch_message_freq=None,
        mode="train",
    ):
        """
        Initializes the nested progress bars.

        Args:
            total_epochs: The total number of epochs for the process.
            total_batches: The total number of batches in one epoch.
            g_epochs: The visual granularity for the epoch bar. If None,
                      it defaults to total_epochs.
            g_batches: The visual granularity for the batch bar. If None,
                       it defaults to total_batches.
            epoch_message_freq: The frequency (in epochs) to log messages.
            batch_message_freq: The frequency (in batches) to log messages.
            mode: The operational mode, either 'train' or 'eval'. 'train'
                  mode shows both epoch and batch bars, while 'eval'
                  mode only shows the batch bar.
        """
        # Set the operational mode ('train' or 'eval')
        self.mode = mode

        # Dynamically import the appropriate tqdm implementation
        from tqdm.auto import tqdm as tqdm_impl

        self.tqdm_impl = tqdm_impl

        # Store the actual total counts for epochs and batches
        self.total_epochs_raw = total_epochs
        self.total_batches_raw = total_batches

        # Determine the visual granularity for the progress bars
        self.g_epochs = min(g_epochs or total_epochs, total_epochs)
        self.g_batches = min(g_batches or total_batches, total_batches)

        # Set the total steps for the progress bars based on granularity
        self.total_epochs = self.g_epochs
        self.total_batches = self.g_batches

        # Initialize the tqdm progress bar instances based on the mode
        if self.mode == "train":
            # Outer bar for epochs
            self.epoch_bar = self.tqdm_impl(
                total=self.total_epochs, desc="Current Epoch", position=0, leave=True
            )
            # Inner bar for batches
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Current Batch", position=1, leave=False
            )
        elif self.mode == "eval":
            # No epoch bar needed for evaluation
            self.epoch_bar = None
            # A single bar for evaluation progress
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Evaluating", position=0, leave=False
            )

        # Keep track of the last updated visual step to avoid redundant updates
        self.last_epoch_step = -1
        self.last_batch_step = -1

        # Store the frequency for logging messages
        self.epoch_message_freq = epoch_message_freq
        self.batch_message_freq = batch_message_freq

    def update_epoch(self, epoch, postfix_dict=None):
        """
        Updates the epoch progress bar and resets the batch bar.

        Args:
            epoch: The current epoch number (1-indexed).
            postfix_dict: An optional dictionary of metrics to display
                          on the epoch bar.
        """
        # Calculate the visual step based on the current epoch and granularity
        epoch_step = math.floor((epoch - 1) * self.g_epochs / self.total_epochs_raw)

        # Update the bar only when the visual step changes
        if epoch_step != self.last_epoch_step:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step
        # Ensure the bar completes on the final epoch
        elif epoch == self.total_epochs_raw and self.epoch_bar.n < self.g_epochs:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step

        # Update the description and postfix for the epoch bar in train mode
        if self.mode == "train":
            self.epoch_bar.set_description(f"Training - Current Epoch: {epoch}")
        if postfix_dict:
            self.epoch_bar.set_postfix(postfix_dict)

        # Reset the batch bar for the new epoch
        self.batch_bar.reset()
        self.last_batch_step = -1

    def update_batch(self, batch, postfix_dict=None):
        """
        Updates the batch progress bar.

        Args:
            batch: The current batch number (1-indexed).
            postfix_dict: An optional dictionary of metrics to display
                          on the batch bar.
        """
        # Calculate the visual step based on the current batch and granularity
        batch_step = math.floor((batch - 1) * self.g_batches / self.total_batches_raw)

        # Update the bar only when the visual step changes
        if batch_step != self.last_batch_step:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step
        # Ensure the bar completes on the final batch
        elif batch == self.total_batches_raw and self.batch_bar.n < self.g_batches:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step

        # Update the description of the batch bar based on the mode
        if self.mode == "train":
            self.batch_bar.set_description(f"Training - Current Batch: {batch}")
        elif self.mode == "eval":
            self.batch_bar.set_description(f"Evaluation - Current Batch: {batch}")

        # Set any provided metrics on the batch bar
        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def maybe_log_epoch(self, epoch, message):
        """
        Prints a message at a specified epoch frequency.

        Args:
            epoch: The current epoch number.
            message: The message to print.
        """
        # Check if logging is enabled and if the current epoch is a logging interval
        if self.epoch_message_freq and epoch % self.epoch_message_freq == 0:
            print(message)

    def maybe_log_batch(self, batch, message):
        """
        Prints a message at a specified batch frequency.

        Args:
            batch: The current batch number.
            message: The message to print.
        """
        # Check if logging is enabled and if the current batch is a logging interval
        if self.batch_message_freq and batch % self.batch_message_freq == 0:
            print(message)

    def close(self, last_message=None):
        """
        Closes the progress bars and optionally prints a final message.

        Args:
            last_message: An optional final message to print after closing
                          the bars.
        """
        # Close the epoch bar if it exists (in 'train' mode)
        if self.mode == "train":
            self.epoch_bar.close()
        # Close the batch bar
        self.batch_bar.close()

        # Print a final message if one is provided
        if last_message:
            print(last_message)
            


def train_model(model, train_dataloader, n_epochs, loss_fcn, optimizer, device):
    """
    Trains a model with nested progress bars showing epoch + batch progress.

    Args:
        model: The neural network to train.
        train_dataloader: The DataLoader providing the training dataset.
        n_epochs: Number of epochs to train.
        loss_fcn: The loss function.
        optimizer: The optimizer.
        device: The device to run training on.

    Returns:
        List of average loss per epoch.
    """
    pbar = NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=len(train_dataloader),
        mode="train",
    )

    epoch_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            pbar.update_batch(batch_idx + 1)

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_fcn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        avg_loss = running_loss / len(train_dataloader.dataset)
        epoch_losses.append(avg_loss)

        pbar.update_epoch(epoch, postfix_dict={"loss": f"{avg_loss:.4f}"})

    pbar.close(last_message="Training complete.")
    return epoch_losses


def evaluate_accuracy(model, data_loader, device):
    """
    Calculates the accuracy of a model on a given dataset.

    This function iterates through the provided data loader, performs a
    forward pass with the model, and compares the predicted labels to the
    true labels to compute the overall accuracy. It operates in evaluation
    mode and disables gradient calculations for efficiency.

    Args:
        model: The neural network model to be evaluated.
        data_loader: The DataLoader providing the evaluation dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.

    Returns:
        The accuracy of the model on the dataset as a float.
    """
    # Initialize a progress bar for the evaluation process
    pbar = NestedProgressBar(
        total_epochs=1,
        total_batches=len(data_loader),
        mode="eval",
    )

    # Set the model to evaluation mode
    model.eval()
    # Initialize counters for correct predictions and total samples
    total_correct = 0
    total_samples = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate over the batches in the data loader
        for batch_idx, (inputs, labels) in enumerate(data_loader):

            # Update the progress bar for the current batch
            pbar.update_batch(batch_idx + 1)

            # Move the input and label tensors to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            # Perform a forward pass to get the model's outputs
            outputs = model(inputs)

            # Get the predicted class by finding the index of the maximum logit
            _, predicted = outputs.max(1)
            # Tally the number of correct predictions in the batch
            total_correct += (predicted == labels).sum().item()
            # Tally the total number of samples in the batch
            total_samples += labels.size(0)

    # Close the progress bar and display a completion message
    pbar.close(last_message="Evaluation complete.")

    # Calculate the final accuracy
    accuracy = total_correct / total_samples
    # Return the computed accuracy
    return accuracy
