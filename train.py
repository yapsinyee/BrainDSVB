#%%
import os
import time
import pickle
import dill
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData # For type checking in padseq

import matplotlib.pyplot as plt # Although imported, plotting is mainly in visualize.py
# from IPython.display import Audio, display, clear_output # For interactive environments like Jupyter - REMOVED

from model import VGRNN # Import the VGRNN model definition

#%%
def save_obj(obj, path):
    """
    Saves a Python object to a file using pickle.
    Ensures the file has a .pkl extension.
    """
    if not path.endswith('.pkl'):
        path = path + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    """
    Loads a Python object from a file using pickle.
    Ensures the file has a .pkl extension.
    """
    if not path.endswith('.pkl'):
        path = path + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def check_path(path):
    """
    Checks if a file exists at the given path, appending .pkl if necessary.
    """
    if not path.endswith('.pkl'):
        path = path + '.pkl'
    return os.path.exists(path)

def load_data(outer_loop, inner_loop):
    """
    Loads preprocessed graph data (train, validation, test sets) for a specific
    cross-validation fold from a .pkl file.
    
    Args:
        outer_loop (int): Index of the outer cross-validation fold.
        inner_loop (int): Index of the inner cross-validation fold.
        
    Returns:
        tuple: (train_graphs, test_graphs, val_graphs)
            train_graphs (np.ndarray): Array of graph sequences for training.
            test_graphs (np.ndarray): Array of graph sequences for testing.
            val_graphs (np.ndarray): Array of graph sequences for validation.
    """
    saveTo = './data/folds_data/' 
    file_path = os.path.join(saveTo, f'graphs_outer{outer_loop}_inner{inner_loop}.pkl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}. Please run step2_prepare_data.py first.")

    with open(file_path, 'rb') as f:
        # torch.load is used because the graphs were saved using torch.save in step2_prepare_data.py
        # weights_only=False is important if the object contains non-tensor data
        f_loaded = torch.load(f, weights_only=False)
        train_graphs = f_loaded['train_graphs']
        val_graphs = f_loaded['val_graphs']
        test_graphs = f_loaded['test_graphs']
    return train_graphs, test_graphs, val_graphs

class myDataset(Dataset):
    """
    Custom PyTorch Dataset for handling the graph sequences.
    """
    def __init__(self, data):
        """
        Args:
            data (np.ndarray): A 2D NumPy array (subjects x windows) of PyTorch Geometric Data objects.
        """
        self.data = data
        # Assuming all graphs have consistent properties, take from the first graph
        if data.size > 0 and data[0,0] is not None:
            self.num_node_features = data[0,0].num_node_features
            self.num_classes = data[0,0].num_classes
            self.num_nodes = data[0,0].num_nodes
        else:
            # Handle empty dataset case
            self.num_node_features = 0
            self.num_classes = 0
            self.num_nodes = 0

    def __len__(self):
        """Returns the number of subjects (sequences) in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single subject's sequence of graphs.
        
        Args:
            idx (int): Index of the subject.
            
        Returns:
            np.ndarray: A NumPy array containing the sequence of PyTorch Geometric Data objects for the subject.
        """
        sample = self.data[idx]
        return sample

def padseq(batch):
    """
    Custom collate function for PyTorch DataLoader.
    It takes a batch of sequences (each sequence is a list of graphs) and
    stacks them into a format suitable for the VGRNN model.
    
    Args:
        batch (list): A list of subject samples, where each sample is a
                      NumPy array of `torch_geometric.data.Data` objects.
                      
    Returns:
        list: A list of `torch_geometric.data.Batch` objects, where each
              `Batch` object corresponds to a time step across all subjects
              in the batch.
    """
    # Transpose the batch: from list of (subject_graphs) to list of (graphs_at_time_t_across_subjects)
    # batch is initially [ [graph_s1_t1, graph_s1_t2, ...], [graph_s2_t1, graph_s2_t2, ...], ... ]
    # After transpose, it becomes [ [graph_s1_t1, graph_s2_t1, ...], [graph_s1_t2, graph_s2_t2, ...], ... ]
    batch = np.asarray(batch).T
    
    new_batch = []
    if batch.size > 0 and isinstance(batch[0, 0], BaseData):
        # For each time step, create a Batch object from the list of graphs at that time step
        # This combines graphs from different subjects at the same time point into a single batch
        new_batch = [Batch.from_data_list(graphs_at_t.tolist()) for graphs_at_t in batch]
    return new_batch

def loadRNG(loadPATH):
    """
    Loads the PyTorch random number generator state from a checkpoint.
    This helps in reproducing training results.
    """
    if os.path.exists(loadPATH):
        print(f'Loading RNG state from: {loadPATH}')
        checkpoint = torch.load(loadPATH, weights_only=False)
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
    else:
        print(f'RNG state checkpoint not found at: {loadPATH}. Starting with new RNG state.')


def loadCheckpoint(setting, loadPATH, savePATH):
    """
    Initializes the VGRNN model, optimizers, and learning rate schedulers.
    Loads a saved checkpoint if it exists to resume training.
    
    Args:
        setting (dict): Dictionary containing model and training settings.
        loadPATH (str): Path to load the checkpoint from.
        savePATH (str): Path to save the checkpoint to (also used for initial save).
        
    Returns:
        tuple: (model, optimizers, schedulers, epochStart, train_losses, val_losses, test_losses)
            model (VGRNN): The initialized or loaded VGRNN model.
            optimizers (list): List of PyTorch optimizers.
            schedulers (list): List of PyTorch learning rate schedulers.
            epochStart (int): The epoch to start training from (0 if new, or next epoch if loaded).
            train_losses (dict): Dictionary to store training losses.
            val_losses (dict): Dictionary to store validation losses.
            test_losses (dict): Dictionary to store testing losses.
    """
    epochStart = 0
    # Initialize loss dictionaries
    train_losses = {'x_NLL': [], 'z_KLD': [], 'a_NLL': [], 'y_BCE': [], 'y_ACC': [], 'Total': []}
    val_losses = {'x_NLL': [], 'z_KLD': [], 'a_NLL': [], 'y_BCE': [], 'y_ACC': [], 'Total': []}
    test_losses = {'x_NLL': [], 'z_KLD': [], 'a_NLL': [], 'y_BCE': [], 'y_ACC': [], 'Total': []}

    # Load RNG state if specified in settings
    if 'rngPATH' in setting and setting['rngPATH']:
        loadRNG(setting['rngPATH'])

    # Initialize the VGRNN model and move it to the determined device (CPU, CUDA, or MPS)
    model = VGRNN(setting)
    model.to(model.device)

    # Define optimizers: one for the main model parameters, one for the classifier parameters
    # This allows different learning rates or weight decays for different parts of the model
    new_named_parameters = {}
    for key, item in model.named_parameters():
        if 'classifier' not in key: # Exclude classifier parameters
            new_named_parameters[key] = item
    
    optimizers = []
    optimizers.append(optim.AdamW(new_named_parameters.values(), lr=setting['learnRate'][0], weight_decay=setting['l2factor'][0]))
    optimizers.append(optim.AdamW(model.classifier.parameters(), lr=setting['learnRate'][1], weight_decay=setting['l2factor'][1]))

    # Define learning rate schedulers
    schedulers = []
    for i in range(2): # For each optimizer
        if setting['lr_annealType'][i] == 'StepLR':
            schedulers.append(optim.lr_scheduler.StepLR(optimizers[i], step_size=2, gamma=0.96))
        elif setting['lr_annealType'][i] == 'ReduceLROnPlateau':
            schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(optimizers[i], mode='min', patience=setting['lr_annealPatience'][i], factor=setting['lr_annealFactor'][i]))
        elif setting['lr_annealType'][i] == None:
            schedulers.append(None) # No scheduler
        else: 
            raise Exception("Learning rate annealing type not supported.")

    # Ensure the parent directory for saving the checkpoint exists
    # This is a crucial addition to prevent "Parent directory does not exist" errors
    os.makedirs(os.path.dirname(savePATH), exist_ok=True)

    # Load checkpoint if it exists
    if os.path.exists(loadPATH):
        print(f'Loading checkpoint from: {loadPATH}')
        checkpoint = torch.load(loadPATH, weights_only=False, map_location=model.device) # Map to current device
        
        # Load model state dictionary
        incompatible_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            print("---")
            print("Warning: Mismatched keys found when loading checkpoint.")
            print("This is expected if you have changed the model architecture (e.g., from TransformerConv to GCNConv).")
            print("The model will load the weights for matching layers and initialize the rest randomly.")
            print(f"Missing keys: {incompatible_keys.missing_keys}")
            print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
            print("---")

        # If loading from the same path as saving, resume training parameters
        if loadPATH == savePATH:
            print('Resuming training parameters from checkpoint.')
            epochStart = checkpoint['epoch'] + 1 # Start from the next epoch
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            test_losses = checkpoint['test_losses']
            
            # Load optimizer and scheduler states
            for i in range(2):
                optimizers[i].load_state_dict(checkpoint['optimizer_state_dicts'][i])
                if schedulers[i] is not None and checkpoint['scheduler_state_dicts'][i] is not None:
                    schedulers[i].load_state_dict(checkpoint['scheduler_state_dicts'][i])
        else:
            print('Checkpoint loaded for inference or fine-tuning. Training will start from epoch 0.')
    else:
        print('No checkpoint found. Initializing new model and training from scratch.')
        # If no checkpoint, save the initial state of the model and optimizers
        model_state_dict = copy.deepcopy(model.state_dict())
        optimizer_state_dicts = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        scheduler_state_dicts = [copy.deepcopy(sch.state_dict()) if sch is not None else None for sch in schedulers]

        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dicts': optimizer_state_dicts,
            'scheduler_state_dicts': scheduler_state_dicts,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'epoch': 0,
            'training_setting': setting,
            'rng_state': torch.get_rng_state() # Save current RNG state
        }, savePATH)

    return model, optimizers, schedulers, epochStart, train_losses, val_losses, test_losses


def train(model, optimizers, schedulers, setting, checkpointPATH,
          train_losses, val_losses, test_losses, 
          train_loader, val_loader, test_loader,
          epochStart=0, numEpochs=100, gradThreshold=1, gradientClip=True,
          verboseFreq=1, verbose=True, valFreq=0, 
          validation=False, testing=False, 
          earlyStopPatience=1, earlyStop=True):
    """
    Main training loop for the VGRNN model.
    
    Args:
        model (VGRNN): The VGRNN model to train.
        optimizers (list): List of PyTorch optimizers.
        schedulers (list): List of PyTorch learning rate schedulers.
        setting (dict): Dictionary of training settings.
        checkpointPATH (str): Path to save model checkpoints.
        train_losses (dict): Dictionary to accumulate training losses.
        val_losses (dict): Dictionary to accumulate validation losses.
        test_losses (dict): Dictionary to accumulate testing losses.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for testing data.
        epochStart (int): Starting epoch number.
        numEpochs (int): Total number of epochs to train.
        gradThreshold (float): Gradient clipping threshold.
        gradientClip (bool): Whether to apply gradient clipping.
        verboseFreq (int): Frequency (in iterations) to print verbose output.
        verbose (bool): Whether to print verbose output during training.
        valFreq (int): Frequency (in epochs) to perform validation/testing.
        validation (bool): Whether to perform validation.
        testing (bool): Whether to perform testing.
        earlyStopPatience (int): Number of epochs to wait for improvement before early stopping.
        earlyStop (bool): Whether to enable early stopping.
        
    Returns:
        tuple: (model, train_losses, val_losses, test_losses)
            model (VGRNN): The trained model (or best model if early stopping).
            train_losses (dict): Accumulated training losses.
            val_losses (dict): Accumulated validation losses.
            test_losses (dict): Accumulated testing losses.
    """
    print(f'Current device: {model.device}')
    # torch.autograd.set_detect_anomaly(True) # Uncomment for debugging gradient issues

    # Initialize current epoch losses (will be updated per epoch)
    x_NLL, z_KLD, a_NLL, y_BCE, y_ACC, total_loss = (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    # Initialize validation and testing losses (will hold last computed values for display)
    valLoss = {'x_NLL': torch.zeros(1).to(model.device), 'z_KLD': torch.zeros(1).to(model.device),
               'a_NLL': torch.zeros(1).to(model.device), 'y_BCE': torch.zeros(1).to(model.device),
               'y_ACC': torch.zeros(1).to(model.device), 'Total': torch.zeros(1).to(model.device)}

    testLoss = {'x_NLL': torch.zeros(1).to(model.device), 'z_KLD': torch.zeros(1).to(model.device),
                'a_NLL': torch.zeros(1).to(model.device), 'y_BCE': torch.zeros(1).to(model.device),
                'y_ACC': torch.zeros(1).to(model.device), 'Total': torch.zeros(1).to(model.device)}

    # Initialize variables for early stopping and best model tracking
    best_params = copy.deepcopy(model.state_dict()) # Store the best model parameters
    best_valLoss = math.inf # Metric to track for early stopping (initialized to infinity)
    best_testLoss = math.inf # Test loss corresponding to the best validation loss
    best_atEpoch = epochStart # Epoch at which the best model was found
    patience_count = 0 # Counter for early stopping patience
    lr_anneal_metric = 0 # Metric used by ReduceLROnPlateau scheduler
    num_bad_epochs = [None, None] # For ReduceLROnPlateau scheduler status
    
    # Initialize patience_metric and test_metric to avoid UnboundLocalError
    patience_metric = float('inf') 
    test_metric = float('nan') # Use nan as a placeholder, as it's only meaningful if testing is enabled and valFreq is met

    start_time = time.time() # Start time of training

    for epoch in range(epochStart, numEpochs):
        # Reset epoch training losses
        trainLoss = {'x_NLL': torch.zeros(1).to(model.device),
					 'z_KLD': torch.zeros(1).to(model.device),
					 'a_NLL': torch.zeros(1).to(model.device),
					 'y_BCE': torch.zeros(1).to(model.device),
					 'y_ACC': torch.zeros(1).to(model.device),
					 'Total': torch.zeros(1).to(model.device)}

        numIter_train = len(train_loader) # Number of iterations per training epoch

        for idxTrain, batch in enumerate(train_loader):
            model.train() # Set model to training mode

            # Joint Training: Iterate through optimizers (one for VGRNN core, one for classifier)
            for i, (optimizer, multiplier) in enumerate(zip(optimizers, setting['yBCEMultiplier'])):
                optimizer.zero_grad() # Clear gradients for the current optimizer

                # Compute batch training losses
                if i == 0: # For the VGRNN core (x_NLL, z_KLD, a_NLL)
                    x_NLL, z_KLD, a_NLL, Readout, Target = model(batch, setting, sample=False)
                    # Classifier loss is also computed here, but its gradients are applied by the second optimizer
                    y_BCE, y_ACC = model.classifier(Readout, Target, sample=False)
                    loss = x_NLL + z_KLD + a_NLL # Total loss for the VGRNN core
                    if setting['DAT']: # Domain Adversarial Training (if enabled)
                        loss = loss - multiplier * y_BCE # Subtract BCE to make it adversarial
                else: # For the classifier
                    # Detach Readout from the graph model to prevent gradients flowing back
                    y_BCE, y_ACC = model.classifier(Readout.detach(), Target, sample=False)
                    loss = multiplier * y_BCE # Only BCE loss for the classifier

                # Backpropagate
                loss.backward()
            
                # Gradient clipping to prevent exploding gradients
                if gradientClip: 
                    nn.utils.clip_grad_norm_(model.parameters(), gradThreshold, norm_type=2, error_if_nonfinite=False)

                # Gradient descent step
                optimizer.step()

            # Accumulate training losses for the current epoch (without gradient tracking)
            with torch.no_grad():
                total_loss = x_NLL + z_KLD + a_NLL + y_BCE # Overall total loss
                trainLoss['x_NLL'] += x_NLL / numIter_train
                trainLoss['z_KLD'] += z_KLD / numIter_train
                trainLoss['a_NLL'] += a_NLL / numIter_train
                trainLoss['y_BCE'] += y_BCE / numIter_train
                trainLoss['y_ACC'] += y_ACC / numIter_train
                trainLoss['Total'] += total_loss / numIter_train

            # Print verbose output at specified frequency
            if verbose and (idxTrain + 1 == 1 or (idxTrain + 1) % verboseFreq == 0):
                current_time = time.time()
                # Format and print training progress
                Print = (
                    f'Time Elapsed: {current_time - start_time:.0f}s  Epoch: {epoch+1}/{numEpochs}  Iteration: {idxTrain+1}/{numIter_train}  '
                    f'Sequence Length: {len(batch)}  Batch Size: {batch[0].num_graphs} \n'
                    f'Variational Bayes: {setting["variational"]}  Domain Adversarial: {setting["DAT"]}  Graph RNN: {setting["graphRNN"]}  Recurrent: {setting["recurrent"]} \n'
                    f'Learning Rates: {optimizers[0].param_groups[0]["lr"]:.2e}, {optimizers[1].param_groups[0]["lr"]:.2e}  '
                    f'BCE Multipliers: {setting["yBCEMultiplier"][0]:.0e}, {setting["yBCEMultiplier"][1]:.0e}  '
                    f'Anneal Metric: {lr_anneal_metric:.4f} \n'
                    f'Patience: {patience_count}/{earlyStopPatience}  No. of Bad Epochs: {num_bad_epochs[0]}, {num_bad_epochs[1]}  '
                    f'Patience Metric: {patience_metric:.4f}  Test Metric at Best Validation = {best_testLoss:.4f}  Best at Epoch: {best_atEpoch} \n' # Updated line
                    f'Training -- x_NLL = {x_NLL:.4f}  z_KLD = {z_KLD:.4f}  a_NLL = {a_NLL:.4f}  y_BCE = {y_BCE:.4f}  y_ACC = {y_ACC:.4f}  Total = {total_loss:.4f} \n'
                    f'Validation -- x_NLL = {valLoss["x_NLL"].item():.4f}  z_KLD = {valLoss["z_KLD"].item():.4f}  a_NLL = {valLoss["a_NLL"].item():.4f}  '
                    f'y_BCE = {valLoss["y_BCE"].item():.4f}  y_ACC = {valLoss["y_ACC"].item():.4f}  Total = {valLoss["Total"].item():.4f} \n'
                    f'Testing -- x_NLL = {testLoss["x_NLL"].item():.4f}  z_KLD = {testLoss["z_KLD"].item():.4f}  a_NLL = {testLoss["a_NLL"].item():.4f}  '
                    f'y_BCE = {testLoss["y_BCE"].item():.4f}  y_ACC = {testLoss["y_ACC"].item():.4f}  Total = {testLoss["Total"].item():.4f}'
                )
                # clear_output(wait=True) # REMOVED: This function requires IPython.display
                print(Print)

        # Collect epoch-level training losses
        train_losses['x_NLL'].append(trainLoss['x_NLL'].cpu())
        train_losses['z_KLD'].append(trainLoss['z_KLD'].cpu())
        train_losses['a_NLL'].append(trainLoss['a_NLL'].cpu())
        train_losses['y_BCE'].append(trainLoss['y_BCE'].cpu())
        train_losses['y_ACC'].append(trainLoss['y_ACC'].cpu())
        train_losses['Total'].append(trainLoss['Total'].cpu())

        # Perform validation and testing at specified frequency
        if (epoch + 1) % valFreq == 0:
            if validation:
                valLoss = validate(model, setting, val_loader) # Compute validation losses
                # Collect epoch-level validation losses
                val_losses['x_NLL'].append(valLoss['x_NLL'].cpu())
                val_losses['z_KLD'].append(valLoss['z_KLD'].cpu())
                val_losses['a_NLL'].append(valLoss['a_NLL'].cpu())
                val_losses['y_BCE'].append(valLoss['y_BCE'].cpu())
                val_losses['y_ACC'].append(valLoss['y_ACC'].cpu())
                val_losses['Total'].append(valLoss['Total'].cpu())

            if testing:
                testLoss = validate(model, setting, test_loader) # Compute testing losses
                # Collect epoch-level testing losses
                test_losses['x_NLL'].append(testLoss['x_NLL'].cpu())
                test_losses['z_KLD'].append(testLoss['z_KLD'].cpu())
                test_losses['a_NLL'].append(testLoss['a_NLL'].cpu())
                test_losses['y_BCE'].append(testLoss['y_BCE'].cpu())
                test_losses['y_ACC'].append(testLoss['y_ACC'].cpu())
                test_losses['Total'].append(testLoss['Total'].cpu())
                # Metric for early stopping, using test accuracy in this case
                test_metric = testLoss['y_ACC'].item()
        
            if earlyStop:
                # Early stop patience metric: sum of training BCE and accuracy (can be adjusted)
                patience_metric = (trainLoss['y_BCE'] + trainLoss['y_ACC']).item()
                # Save model params if current patience metric is better than best_valLoss
                if patience_metric <= best_valLoss:
                    best_valLoss = patience_metric
                    best_testLoss = test_metric # Store test performance at this best validation point
                    best_params = copy.deepcopy(model.state_dict()) # Save model state
                    best_atEpoch = epoch + 1 # Record epoch
                    patience_count = 0 # Reset patience counter
                else:
                    patience_count += 1 # Increment patience counter
                # print(f'Patience: {patience_count}/{earlyStopPatience}')
            else:
                # If no early stopping, always save the last model's parameters
                best_params = copy.deepcopy(model.state_dict())
        
        # Learning rate annealing
        lr_anneal_metric = trainLoss['y_BCE'].item() # Metric for ReduceLROnPlateau
        for i, scheduler in enumerate(schedulers):
            if isinstance(scheduler, optim.lr_scheduler.StepLR):
                scheduler.step()
            elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(lr_anneal_metric)
                num_bad_epochs[i] = scheduler.state_dict()['num_bad_epochs']
            elif scheduler is None: 
                pass # No scheduler
            else: 
                raise Exception("Learning rate annealing type not supported.")

        # Prepare optimizer and scheduler states for saving
        optimizer_state_dicts = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        scheduler_state_dicts = [copy.deepcopy(sch.state_dict()) if sch is not None else None for sch in schedulers]

        # Save training checkpoint after every epoch
        torch.save({
            'model_state_dict': best_params, # Save the best parameters found so far
            'optimizer_state_dicts': optimizer_state_dicts,
            'scheduler_state_dicts': scheduler_state_dicts,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'epoch': epoch,
            'training_setting': setting,
            'rng_state': torch.get_rng_state() # Save current RNG state
        }, checkpointPATH)

        # Early stopping check
        if earlyStop and patience_count > earlyStopPatience:
            print('Early Stopped.')
            break # Exit the training loop

    # Optional: Play a sound when training is done (requires IPython.display.Audio) - REMOVED
    # Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg", autoplay=True)
    print('Training done!')
    return model, train_losses, val_losses, test_losses


def validate(model, setting, val_loader):
    """
    Performs validation or testing of the model on a given DataLoader.
    Calculates average losses and accuracy without gradient computation.
    
    Args:
        model (VGRNN): The VGRNN model to evaluate.
        setting (dict): Training settings.
        val_loader (DataLoader): DataLoader for validation or test data.
        
    Returns:
        dict: A dictionary containing average losses and accuracy for the evaluation set.
    """
    numIter = len(val_loader) # Number of batches in the loader
    # Initialize validation/test losses
    valLoss = {'x_NLL': torch.zeros(1).to(model.device), 'z_KLD': torch.zeros(1).to(model.device),
               'a_NLL': torch.zeros(1).to(model.device), 'y_BCE': torch.zeros(1).to(model.device),
               'y_ACC': torch.zeros(1).to(model.device), 'Total': torch.zeros(1).to(model.device)}

    for idxVal, batch in enumerate(val_loader):
        model.eval() # Set model to evaluation mode (disables dropout, batch norm updates)
        with torch.no_grad(): # Disable gradient computation
            # Forward pass
            x_NLL, z_KLD, a_NLL, Readout, Target = model(batch, setting, sample=False)
            y_BCE, y_ACC = model.classifier(Readout, Target, sample=False)
            total_loss = x_NLL + z_KLD + a_NLL + y_BCE # Calculate total loss

        # Accumulate losses for the current evaluation set
        with torch.no_grad():
            valLoss['x_NLL'] += x_NLL / numIter
            valLoss['z_KLD'] += z_KLD / numIter
            valLoss['a_NLL'] += a_NLL / numIter
            valLoss['y_BCE'] += y_BCE / numIter
            valLoss['y_ACC'] += y_ACC / numIter
            valLoss['Total'] += total_loss / numIter

    return valLoss

