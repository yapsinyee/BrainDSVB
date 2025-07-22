#%% 0-Importing Libraries and Modules
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

# Define the path for the saved_model folder
saved_model_path = './saved_models'
os.makedirs(saved_model_path, exist_ok=True) # Create the directory if it doesn't exist

#%% 1-Loading and Preparing Datasets

# Define the outer and inner loop indices for data loading
# Change outer_loop only if you want to run different cross-validation folds.
# inner_loop typically iterates from 1 to 5 within the outer loop for nested CV.
outer_loop = 1
inner_loop = 1

print(f"Loading data for outer loop {outer_loop}, inner loop {inner_loop}...")
# Load train, test, and validation graph sequences
train_graphs, test_graphs, val_graphs = load_data(outer_loop, inner_loop)
print("Data loaded.")

# Creating Datasets & Initializing DataLoaders
# Concatenate train and validation graphs for the training dataset
# This is a common practice if validation is used solely for early stopping,
# and the model is ultimately trained on the combined train+val set.
train_dataset = myDataset(np.concatenate([train_graphs, val_graphs], axis=0))
# DataLoader for training data
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, # Number of subjects per batch
    shuffle=True, 
    num_workers=0, # Set to 0 for debugging, can be higher for faster data loading on multi-core CPUs
    collate_fn=padseq, # Custom collate function to handle graph sequences
    pin_memory=True # Speeds up data transfer to GPU (if using CUDA/MPS)
)

val_dataset = myDataset(val_graphs)
# DataLoader for validation data (batch size usually full validation set for single evaluation)
val_loader = DataLoader(
    val_dataset, 
    batch_size=len(val_dataset), # Evaluate on the entire validation set at once
    shuffle=False, # No need to shuffle validation data
    num_workers=0, 
    collate_fn=padseq, 
    pin_memory=True
)

test_dataset = myDataset(test_graphs)
# DataLoader for testing data (similar to validation)
test_loader = DataLoader(
    test_dataset, 
    batch_size=len(test_dataset), # Evaluate on the entire test set at once
    shuffle=False, # No need to shuffle test data
    num_workers=0, 
    collate_fn=padseq, 
    pin_memory=True
)

# Printing Dataset Sizes
partition = [len(train_dataset), len(val_dataset), len(test_dataset)]
print(f"Number of subjects in train_graphs: {len(train_graphs)}")
print(f"Number of subjects in val_graphs: {len(val_graphs)}")
print(f"Number of subjects in test_graphs: {len(test_graphs)}")
print(f"Dataset partition (train+val, val, test subjects): {partition}")

#%% 2-Initializing Training and Model Parameters

# Setting Paths for saving and loading model checkpoints
# The path includes outer_loop and inner_loop for specific fold checkpoints
savePATH = os.path.join(saved_model_path, f'VGRNN_softmax_adv_fold{outer_loop}{inner_loop}_mac-1.pth')
loadPATH = savePATH # By default, load from the same path where it will be saved

# Model Parameters for the VGRNN architecture
model_params = {
    'num_nodes': 264, # Number of nodes (ROIs) in the brain graphs
    'num_classes': 2, # Number of output classes (e.g., ASD vs. TD)
    'x_dim': 264, # Dimension of input node features (e.g., correlation values for each ROI)
    'y_dim': 2, # Dimension of output classification (2 for binary classification)
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
    'rngPATH': os.path.join(saved_model_path, "VGRNN_softmax_adv_fold11.pth"), # Path for RNG state (can be same as savePATH)
    'model_params': model_params,
    'recurrent': True, # Whether to use recurrent connections
    'learnRate': [1e-4, 1e-4], # Learning rates for the two optimizers
    'yBCEMultiplier': [1, 1], # Multiplier for BCE loss in adversarial training (if DAT is True)
    'l2factor': [0.005, 0.005], # L2 regularization factor for the two optimizers
    'lr_annealType': lr_annealType,
    'lr_annealFactor': [0.8, 0.8], # Factor by which LR is reduced
    'lr_annealPatience': [30, 30], # Number of epochs with no improvement after which LR will be reduced
    'variational': True, # Whether to use variational inference (VAE part)
    'DAT': False, # Domain Adversarial Training (set to True for adversarial classification)
    'graphRNN': True, # Whether to use GRU-GCN (True) or standard GRU (False)
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


#%% 3-Training the Model and Evaluating Performance

print("\nStarting model training...")
# Call the main training function
model, train_losses, val_losses, test_losses = train(
    model, optimizers, schedulers, setting, savePATH,
    train_losses, val_losses, test_losses,
    train_loader, val_loader, test_loader, 
    epochStart=epochStart, # Start from the loaded epoch
    numEpochs=300, # Total number of epochs to run
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

