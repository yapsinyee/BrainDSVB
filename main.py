#%% 0-Importing Libraries and Modules
# NOTE for MPS split: in model.py/train.py, keep adjacency/edge_index/sparse tensors on CPU; move only dense tensors/parameters to DEVICE.
# Ensure any `.cuda()` calls are replaced with `.to(DEVICE)` and guard unsupported ops with try/except.
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt # For potential future plotting directly in main
# from mpl_toolkits import mplot3d # Uncomment if 3D plotting is needed

# Import necessary functions and classes from custom modules
from model import VGRNN, get_device # VGRNN model and device utility
from train import load_data, myDataset, padseq, loadCheckpoint, train # Data loading, Dataset, DataLoader collate, checkpointing, training loop

# Set environment variable to allow fallback to CPU for unsupported MPS operations.
# This MUST be set before torch is imported for the first time.
# This is crucial for running on Mac Mini with MPS (Metal Performance Shaders).
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch # Import torch after setting the environment variable

# ---- Device selection: prefer Apple MPS, safe CPU fallback ----
def pick_device(prefer_mps=True):
    try:
        if prefer_mps and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

DEVICE = pick_device(prefer_mps=True)
print(f"\n[Device] Using device: {DEVICE}. MPS available: {getattr(torch.backends.mps, 'is_available', lambda: False)()}")
# NOTE: Keep sparse adjacency and PyG scatter/gather on CPU inside model/train.
# Only move dense tensors (features, labels, GRU/MLP params) to DEVICE.

from torch.utils.data import DataLoader # Explicitly import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Train BrainDSVB using static or dynamic connectivity.")
    parser.add_argument("--dataset", choices=["abide", "cobre_static", "cobre_fmri"], default="abide")
    parser.add_argument("--connectivity-mode", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--outer-loop", type=int, default=1)
    parser.add_argument("--inner-loop", type=int, default=1)
    parser.add_argument("--version", default="v3")
    return parser.parse_args()


args = parse_args()

# Define the path for the saved_model folder
saved_model_path = './saved_models'
os.makedirs(saved_model_path, exist_ok=True) # Create the directory if it doesn't exist

#%% 1-Loading and Preparing Datasets

# Define the outer and inner loop indices for data loading
# Change outer_loop only if you want to run different cross-validation folds.
# inner_loop typically iterates from 1 to 5 within the outer loop for nested CV.
outer_loop = args.outer_loop
inner_loop = args.inner_loop
version = args.version
connectivity_mode = args.connectivity_mode
dataset = args.dataset

print(
    f"Loading {dataset} {connectivity_mode} data for outer loop {outer_loop}, "
    f"inner loop {inner_loop}..."
)
# Load train, test, and validation graph sequences
train_graphs, test_graphs, val_graphs = load_data(
    outer_loop,
    inner_loop,
    connectivity_mode=connectivity_mode,
    dataset=dataset,
)
print("Data loaded.")

# Creating Datasets & Initializing DataLoaders
# Concatenate train and validation graphs for the training dataset
# This is a common practice if validation is used solely for early stopping,
# and the model is ultimately trained on the combined train+val set.
train_dataset = myDataset(np.concatenate([train_graphs, val_graphs], axis=0))
# DataLoader for training data
train_loader = DataLoader(
    train_dataset, 
    batch_size=4, # REDUCED BATCH SIZE TO PREVENT MPS OUT OF MEMORY
    shuffle=True, 
    num_workers=0, # Set to 0 for debugging, can be higher for faster data loading on multi-core CPUs
    drop_last=True, # Avoid single-sample batches that break BatchNorm in the classifier
    collate_fn=padseq, # Custom collate function to handle graph sequences
    pin_memory=False # Speeds up data transfer to GPU (if using CUDA/MPS)
)

val_dataset = myDataset(val_graphs)
# DataLoader for validation data (batch size usually full validation set for single evaluation)
val_loader = DataLoader(
    val_dataset, 
    batch_size=len(val_dataset), # Evaluate on the entire validation set at once
    shuffle=False, # No need to shuffle validation data
    num_workers=0, 
    collate_fn=padseq, 
    pin_memory=False
)

test_dataset = myDataset(test_graphs)
# DataLoader for testing data (similar to validation)
test_loader = DataLoader(
    test_dataset, 
    batch_size=len(test_dataset), # Evaluate on the entire test set at once
    shuffle=False, # No need to shuffle test data
    num_workers=0, 
    collate_fn=padseq, 
    pin_memory=False
)

# Printing Dataset Sizes
partition = [len(train_dataset), len(val_dataset), len(test_dataset)]
print(f"Number of subjects in train_graphs: {len(train_graphs)}")
print(f"Number of subjects in val_graphs: {len(val_graphs)}")
print(f"Number of subjects in test_graphs: {len(test_graphs)}")
print(f"Dataset partition (train+val, val, test subjects): {partition}")

sample_graph = None
for graph_collection in (train_graphs, val_graphs, test_graphs):
    if len(graph_collection) > 0 and len(graph_collection[0]) > 0 and graph_collection[0][0] is not None:
        sample_graph = graph_collection[0][0]
        break

if sample_graph is None:
    raise RuntimeError("Unable to infer graph dimensions from the loaded dataset.")

num_nodes = int(sample_graph.num_nodes)
x_dim = int(sample_graph.num_node_features)
num_classes = int(sample_graph.num_classes)
print(f"Inferred graph shape: num_nodes={num_nodes}, x_dim={x_dim}, num_classes={num_classes}")

#%% 2-Initializing Training and Model Parameters

# Setting Paths for saving and loading model checkpoints
# The path includes outer_loop and inner_loop for specific fold checkpoints
savePATH = os.path.join(
    saved_model_path,
    f'VGRNN_{dataset}_{connectivity_mode}_fold{outer_loop}{inner_loop}_{version}.pth'
)
loadPATH = savePATH # By default, load from the same path where it will be saved

# Model Parameters for the VGRNN architecture
model_params = {
    'num_nodes': num_nodes, # Number of nodes (ROIs) in the brain graphs
    'num_classes': num_classes, # Number of output classes
    'x_dim': x_dim, # Dimension of input node features
    'y_dim': num_classes, # Dimension of output classification
    'z_hidden_dim': 32, # Hidden dimension for latent variable z processing
    'z_dim': 16, # Dimension of the latent variable z
    'z_phi_dim': 8, # Dimension after transformation of z
    'x_phi_dim': 64, # Dimension after transformation of x
    'rnn_dim': 16, # Dimension of the recurrent hidden state
    'y_hidden_dim': [32], # Hidden layer dimensions for the classifier
    'x_hidden_dim': 64, # Hidden dimension for x decoder (if used)
    'layer_dims': [] # General hidden layer dimensions for GCN/Dense layers (empty means no hidden layers)
}

# Learning Rate Annealing settings
# 'ReduceLROnPlateau' is a common choice, reducing LR when a metric stops improving
lr_annealType = 'ReduceLROnPlateau'
lr_annealType = [lr_annealType, lr_annealType] # Applies to both optimizers

# Training Settings dictionary
setting = {
    'device': str(DEVICE),          # 'mps' or 'cpu' as resolved above
    'prefer_mps': True,             # hint for train/model code
    'sparse_on_cpu': True,          # keep adjacency / edge_index on CPU
    'rngPATH': savePATH,            # Reuse the active checkpoint path for RNG state
    'model_params': model_params,
    'recurrent': connectivity_mode == 'dynamic', # Disable recurrence for static connectivity
    'learnRate': [1e-4, 1e-4], # Learning rates for the two optimizers
    'yBCEMultiplier': [1, 1], # Multiplier for BCE loss in adversarial training (if DAT is True)
    'l2factor': [0.005, 0.005], # L2 regularization factor for the two optimizers
    'lr_annealType': lr_annealType,
    'lr_annealFactor': [0.8, 0.8], # Factor by which LR is reduced
    'lr_annealPatience': [30, 30], # Number of epochs with no improvement after which LR will be reduced
    'variational': True, # Whether to use variational inference (VAE part)
    'DAT': False, # Domain Adversarial Training (set to True for adversarial classification)
    'graphRNN': connectivity_mode == 'dynamic', # Only meaningful for dynamic sequences
    'partition': partition # Dataset partition sizes
}

# Loading Model, Optimizer, and Scheduler
# This function initializes the model or loads a checkpoint if it exists.
model, optimizers, schedulers, epochStart, train_losses, val_losses, test_losses = loadCheckpoint(setting, loadPATH, savePATH)

print("\nModel Architecture:")
print(model)
print("\nModel Parameters:")
for param_name, param_value in model_params.items():
    print(f"  {param_name}: {param_value}")
print("\nOptimizers:")
for i, optimizer in enumerate(optimizers):
    print(f"  Optimizer {i}: {optimizer}")
print("\nSchedulers:")
for i, scheduler in enumerate(schedulers):
    print(f"  Scheduler {i}: {scheduler}")
print(f"\nRNG State Path: {setting['rngPATH']}")
print(f"Checkpoint Save Path: {savePATH}")
print(f"Starting Epoch: {epochStart}")
print(f"Resolved DEVICE for dense ops: {DEVICE} | sparse_on_cpu={setting['sparse_on_cpu']}")

#%% 3-Training the Model and Evaluating Performance

print("\nStarting model training...")
# Call the main training function
model, train_losses, val_losses, test_losses = train(
    model, optimizers, schedulers, setting, savePATH,
    train_losses, val_losses, test_losses,
    train_loader, val_loader, test_loader, 
    device=DEVICE,
    epochStart=epochStart, # Start from the loaded epoch
    numEpochs=100, # Total number of epochs to run
    gradThreshold=1, # Gradient clipping threshold
    gradientClip=True, # Enable gradient clipping
    verboseFreq=1, # Print verbose output every 1 iteration (can be set higher)
    verbose=True, # Enable verbose printing
    valFreq=1, # Perform validation/testing every 1 epoch
    validation=True, # Enable validation
    testing=True, # Enable testing
    earlyStopPatience=30, # Patience for early stopping
    earlyStop=True # Enable early stopping
)

print("\nTraining finished. You can now use visualize.py to plot the results.")
