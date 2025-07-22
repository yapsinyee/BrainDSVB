import torch
import matplotlib.pyplot as plt
import argparse
import os

def load_checkpoint(path):
    """
    Loads a PyTorch checkpoint from the given path.
    
    Args:
        path (str): Path to the checkpoint file.
        
    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
    print(f"Loading checkpoint from: {path}")
    # Load checkpoint, mapping to CPU to avoid device issues if GPU is not available
    checkpoint = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
    return checkpoint

def plot_loss_curves(losses, labels, title, xlabel='Epochs', ylabel='Loss'):
    """
    Plots loss curves for training and validation/testing losses.
    
    Args:
        losses (list of torch.Tensor): A list of loss tensors (e.g., [train_loss_tensor, val_loss_tensor]).
        labels (list of str): Labels for each loss curve (e.g., ['Train', 'Validation']).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(10, 6)) # Create a new figure for each plot
    for loss, label in zip(losses, labels):
        # Ensure loss is on CPU and convert to NumPy for plotting
        plt.plot(torch.stack(loss, dim=0).cpu().numpy(), label=label)
    
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True) # Add a grid for better readability
    plt.show()

def analyze_checkpoint(loadPATH):
    """
    Loads and analyzes the given checkpoint by plotting loss curves and accuracy.
    Prints key training settings and final metrics.
    
    Args:
        loadPATH (str): Path to the checkpoint file.
    """
    checkpoint = load_checkpoint(loadPATH)
    setting = checkpoint['training_setting']
    
    # Plot KL Divergence Loss (z_KLD)
    plot_loss_curves(
        [checkpoint['train_losses']['z_KLD'], checkpoint['test_losses']['z_KLD']],
        ['Train', 'Test'], # Assuming 'test_losses' is used for validation/test
        'KL Divergence Loss (Latent Variable z)'
    )
    
    # Plot Adjacency Negative Log-Likelihood Loss (a_NLL)
    plot_loss_curves(
        [checkpoint['train_losses']['a_NLL'], checkpoint['test_losses']['a_NLL']],
        ['Train', 'Test'],
        'Adjacency Negative Log-Likelihood Loss (Graph Reconstruction)'
    )
    
    # Plot Binary Cross-Entropy Loss (y_BCE) for classification
    plot_loss_curves(
        [checkpoint['train_losses']['y_BCE'], checkpoint['test_losses']['y_BCE']],
        ['Train', 'Test'],
        'Binary Cross-Entropy Loss (Classification)'
    )
    
    # Plot Accuracy (y_ACC)
    plot_loss_curves(
        [checkpoint['train_losses']['y_ACC'], checkpoint['test_losses']['y_ACC']],
        ['Train', 'Test'],
        'Accuracy', ylabel='Accuracy'
    )
    
    # Print training settings and final metrics
    print(f"\n--- Analysis of Checkpoint: {loadPATH} ---")
    print("Training Settings:")
    for key, value in setting.items():
        print(f"  {key}: {value}")
    
    # Get the last recorded test losses/metrics
    if checkpoint['test_losses']['y_BCE']:
        final_bce = checkpoint['test_losses']['y_BCE'][-1].item()
        final_acc = checkpoint['test_losses']['y_ACC'][-1].item()
        print(f"\nFinal BCE Loss (Test): {final_bce:.4f}")
        print(f"Final Accuracy (Test): {final_acc:.4f}")
        print(f"Final Error Rate (1 - Accuracy, Test): {1 - final_acc:.4f}")
    else:
        print("\nNo test loss records found in the checkpoint.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a PyTorch model checkpoint.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file (e.g., ./saved_models/VGRNN_softmax_adv_fold11_mac-1.pth).")
    
    args = parser.parse_args()
    analyze_checkpoint(args.checkpoint_path)

