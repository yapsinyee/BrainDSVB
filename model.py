#%%
import os
import time
import dill
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import det, svd, cholesky, cholesky_ex, inv, matrix_rank
from torch_geometric.nn import Sequential, GCNConv, TransformerConv # Using TransformerConv as per original code
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import negative_sampling, batched_negative_sampling, to_dense_batch, unbatch 

def get_device():
    """
    Determines and returns the appropriate PyTorch device (CUDA, MPS, or CPU).
    Prioritizes CUDA if available, then MPS for Apple Silicon Macs, otherwise defaults to CPU.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear CUDA cache to free up memory
        torch.backends.cudnn.enabled = True # Enable CuDNN for performance
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # MPS (Metal Performance Shaders) for Apple Silicon GPUs
        return torch.device('mps')
    else:
        # Fallback to CPU if no GPU is available
        return torch.device('cpu')

def get_activation(type='relu'):
    """
    Returns a PyTorch activation function module given its string name.
    
    Args:
        type (str): Name of the activation function ('relu', 'elu', 'lelu', 'tanh', 'sigmoid', 'softplus', None).
        
    Returns:
        nn.Module: The corresponding PyTorch activation function module.
    
    Raises:
        Exception: If the activation function type is not supported.
    """
    if type == 'relu':
        return nn.ReLU()
    elif type == 'elu':
        return nn.ELU()
    elif type == 'lelu':
        return nn.LeakyReLU()
    elif type == 'tanh':
        return nn.Tanh()
    elif type == 'sigmoid':
        return nn.Sigmoid()
    elif type == 'softplus':
        return nn.Softplus()
    elif type == None:
        return None
    else:
        raise Exception("Activation function not supported.")

def dense_vary(input_dim, layer_dims, output_dim, activation='relu', dropout=0., batch_norm=False, last_act=None):
    """
    Creates a sequential neural network with dense (fully connected) layers.
    Allows for varying hidden layer dimensions, activation functions, dropout, and batch normalization.
    
    Args:
        input_dim (int): Dimension of the input features.
        layer_dims (list): List of integers, where each integer is the dimension of a hidden layer.
        output_dim (int): Dimension of the output features.
        activation (str): Activation function to use for hidden layers.
        dropout (float): Dropout probability.
        batch_norm (bool): Whether to apply batch normalization.
        last_act (str): Activation function for the last layer (e.g., 'sigmoid', 'softmax', or None).
        
    Returns:
        nn.Sequential: A PyTorch Sequential model.
    """
    layers = nn.Sequential()
    
    if len(layer_dims) != 0:
        # First layer
        layers.append(nn.Linear(input_dim, layer_dims[0]))
        layers.append(get_activation(activation))
        if batch_norm: layers.append(nn.BatchNorm1d(layer_dims[0]))
        layers.append(nn.Dropout(p=dropout))
        
        # Intermediate layers
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(get_activation(activation))
            if batch_norm: layers.append(nn.BatchNorm1d(layer_dims[i+1]))
            layers.append(nn.Dropout(p=dropout))
            
        # Last layer
        layers.append(nn.Linear(layer_dims[len(layer_dims) - 1], output_dim))
        if last_act is not None:
            layers.append(get_activation(last_act))
    else: 
        # If no hidden layers, directly connect input to output
        layers.append(nn.Linear(input_dim, output_dim))
        if last_act is not None:
            layers.append(get_activation(last_act))
            
    return layers

def GCN_vary(input_dim, layer_dims, output_dim, activation='relu', last_act=None):
    """
    Creates a sequential Graph Convolutional Network (GCN) using TransformerConv layers.
    
    Args:
        input_dim (int): Dimension of the input node features.
        layer_dims (list): List of integers, dimensions of hidden GCN layers.
        output_dim (int): Dimension of the output node features.
        activation (str): Activation function for hidden layers.
        last_act (str): Activation function for the last layer.
        
    Returns:
        torch_geometric.nn.Sequential: A PyTorch Geometric Sequential model.
    """
    modules = []
    
    if len(layer_dims) != 0:
        # First GCN layer
        modules.append((TransformerConv(input_dim, layer_dims[0]), 'x, edge_index -> x'))
        modules.append((get_activation(activation)))
        
        # Intermediate GCN layers
        for i in range(len(layer_dims) - 1):
            modules.append((TransformerConv(layer_dims[i], layer_dims[i+1]), 'x, edge_index -> x'))
            modules.append((get_activation(activation)))
            
        # Last GCN layer
        modules.append((TransformerConv(layer_dims[len(layer_dims) - 1], output_dim), 'x, edge_index -> x'))
        if last_act is not None:
            modules.append(get_activation(last_act))
    else: 
        # If no hidden layers, directly connect input to output with one GCN layer
        modules.append((TransformerConv(input_dim, output_dim), 'x, edge_index -> x'))
        if last_act is not None:
            modules.append(get_activation(last_act))
            
    # Sequential model for PyTorch Geometric
    layers = Sequential('x, edge_index', modules)
    return layers

def recurrent_cell(input_dim, hidden_dim, rnn_type):
    """
    Returns a PyTorch recurrent cell (LSTMCell or GRUCell).
    
    Args:
        input_dim (int): Dimension of the input to the RNN cell.
        hidden_dim (int): Dimension of the hidden state of the RNN cell.
        rnn_type (str): Type of RNN cell ('lstm' or 'gru').
        
    Returns:
        nn.Module: The corresponding PyTorch RNN cell.
        
    Raises:
        Exception: If the RNN type is not supported.
    """
    if rnn_type == 'lstm':
        return nn.LSTMCell(input_dim, hidden_dim)
    elif rnn_type == 'gru':
        return nn.GRUCell(input_dim, hidden_dim)
    else:
        raise Exception("No such rnn type.")

class GraphDecoder(nn.Module):
    """
    Decodes latent representations into graph adjacency matrices.
    It can reconstruct edges based on inner product of node embeddings or
    reconstruct a full adjacency matrix.
    """
    def __init__(self, activation=None, dropout=0.5):
        super().__init__()
        self.act = get_activation(activation) # Activation for the output
        self.dropout = dropout # Dropout rate

    def forward(self, z, edge_index):
        """
        Forward pass for decoding edges from node embeddings (used for link prediction).
        
        Args:
            z (torch.Tensor): Node embeddings (latent representations).
            edge_index (torch.Tensor): Edge indices (2xNumEdges).
            
        Returns:
            torch.Tensor: Predicted edge probabilities/scores.
        """
        z = F.dropout(z, self.dropout, training=self.training) # Apply dropout
        # Compute inner product between connected nodes' embeddings
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        if self.act is not None:
            value = self.act(value) # Apply activation
        return value
    
    def forward_sb(self, z):
        """
        Forward pass for decoding a full adjacency matrix from batched node embeddings.
        'sb' likely stands for 'separated batch' or 'sequence batch'.
        
        Args:
            z (torch.Tensor): Batched node embeddings (BatchSize x NumNodes x FeatureDim).
            
        Returns:
            torch.Tensor: Reconstructed adjacency matrix.
        """
        z = F.dropout(z, self.dropout, training=self.training) # Apply dropout
        # Compute outer product (matrix multiplication) to get adjacency-like matrix
        adj = z @ torch.transpose(z, dim0=-2, dim1=-1)
        if self.act is not None:
            adj = self.act(adj) # Apply activation (e.g., sigmoid for probabilities)
        return adj

    def cov2corr(self, cov):
        """
        Converts a covariance matrix to a correlation matrix (PyTorch version).
        (Note: This function is defined but not used in the provided forward/loss methods).
        """
        v = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
        outer_v = v.unsqueeze(dim=-1) @ v.unsqueeze(dim=-2)
        corr = cov / outer_v
        return corr

    def loss(self, input, target):
        """
        Binary Cross-Entropy loss for graph reconstruction (for link prediction scenario).
        Applies a positive weight to handle class imbalance (more non-edges than edges).
        (Note: This function is defined but `loss_sb` is used in VGRNN forward).
        """
        temp_sum = target.sum()
        temp_size = target.shape[0]
        # Calculate positive weight to balance classes (non-edges vs edges)
        weight = float(temp_size * temp_size - temp_sum) / temp_sum
        # Normalization factor
        norm = (temp_size * temp_size) / float((temp_size * temp_size - temp_sum) * 2)
        # Binary Cross-Entropy with logits (numerically stable)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=input, target=target, pos_weight=weight, reduction='none')
        nll_loss = norm * torch.mean(nll_loss_mat)
        return nll_loss

    def loss_sb(self, input, target, reduce=False):
        """
        Binary Cross-Entropy loss for batched graph reconstruction (full adjacency matrix).
        Applies a positive weight to handle class imbalance.
        
        Args:
            input (torch.Tensor): Predicted adjacency logits (BatchSize x NumNodes x NumNodes).
            target (torch.Tensor): True adjacency matrices (BatchSize x NumNodes x NumNodes).
            reduce (bool): If True, returns the mean loss over the batch.
            
        Returns:
            torch.Tensor: Reconstruction loss.
        """
        # Sum of elements in input (logits, not probabilities)
        temp_sum = input.sum(dim=[-1,-2])
        temp_size = input.shape[1] # Number of nodes
        
        # Calculate positive weight per batch item
        weight = (temp_size * temp_size - temp_sum) / (temp_sum + 1e-6) # Add epsilon to prevent division by zero
        # Normalization factor per batch item
        norm = (temp_size * temp_size) / ((temp_size * temp_size - temp_sum + 1e-6) * 2)
        
        # Binary Cross-Entropy with logits
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='none')
        
        # Apply weight and normalization, then mean over nodes
        nll_loss = norm * weight * torch.mean(nll_loss_mat, dim=[-1,-2])
        if reduce: nll_loss = torch.mean(nll_loss) # Mean over batch if reduce is True
        return nll_loss

class GraphClassifier(nn.Module):
    """
    A classifier for graphs, typically used after a global pooling/readout layer.
    """
    def __init__(self, input_dim, layer_dims, output_dim, dropout=0., batch_norm=False):
        super().__init__()
        self.dropout = dropout
        # Linear layers for classification
        self.linear = dense_vary(input_dim, layer_dims, output_dim, last_act=None, dropout=dropout, batch_norm=batch_norm)

    def forward(self, input, target, sample=False):
        """
        Forward pass for graph classification.
        
        Args:
            input (torch.Tensor): Graph-level features (readout from VGRNN).
            target (torch.Tensor): True class labels.
            sample (bool): If True, returns probabilities and predictions in addition to loss and accuracy.
            
        Returns:
            tuple: (y_BCE, y_ACC) or (y_BCE, y_ACC, y_prob, y_pred)
                y_BCE (torch.Tensor): Binary Cross-Entropy loss.
                y_ACC (torch.Tensor): Accuracy.
                y_prob (torch.Tensor, optional): Predicted probabilities.
                y_pred (torch.Tensor, optional): Predicted classes.
        """
        y_logit = self.linear(input) # Get logits from linear layers
        y_bce = self.loss(y_logit, target) # Calculate BCE loss
        
        with torch.no_grad(): # No gradient computation for accuracy and predictions
            y_acc, y_prob, y_pred = self.acc(y_logit, target) # Calculate accuracy, probabilities, predictions
        
        y_BCE = y_bce.mean() # Mean BCE loss over the batch
        y_ACC = y_acc.mean() # Mean accuracy over the batch
        
        if sample:
            return y_BCE, y_ACC, y_prob, y_pred
        else:
            return y_BCE, y_ACC

    def loss(self, logit, target, reduce=False):
        """
        Calculates the classification loss (Cross-Entropy for multi-class, BCE for binary).
        
        Args:
            logit (torch.Tensor): Predicted logits.
            target (torch.Tensor): True labels.
            reduce (bool): If True, returns the mean loss.
            
        Returns:
            torch.Tensor: Classification loss.
        """
        if logit.shape[-1] > 1:
            # Multi-class classification
            nll_loss = F.cross_entropy(input=logit, target=target, reduction='none')
        else:
            # Binary classification
            nll_loss = F.binary_cross_entropy_with_logits(input=logit.squeeze(-1), target=target.float(), reduction='none')
        if reduce: nll_loss = torch.mean(nll_loss)
        return nll_loss

    def acc(self, logit, target, reduce=False):
        """
        Calculates classification accuracy, probabilities, and predicted classes.
        
        Args:
            logit (torch.Tensor): Predicted logits.
            target (torch.Tensor): True labels.
            reduce (bool): If True, returns the mean accuracy and probabilities.
            
        Returns:
            tuple: (acc, prob, pred)
                acc (torch.Tensor): Accuracy (0 or 1 for each sample).
                prob (torch.Tensor): Predicted probabilities.
                pred (torch.Tensor): Predicted classes.
        """
        if logit.shape[-1] > 1:
            # Multi-class: get probabilities via softmax, then predicted class
            prob = F.softmax(logit, dim=-1)
            pred = torch.argmax(prob, dim=-1).float()
            acc = (target == pred).float() # Accuracy for each sample
        else:
            # Binary: get probabilities via sigmoid, then round to get predicted class
            prob = torch.sigmoid(logit.squeeze(-1))
            pred = torch.round(prob)
            acc = (target == pred).float() # Accuracy for each sample
        if reduce: 
            acc = torch.mean(acc)
            prob = torch.mean(prob)
        return acc, prob, pred

class GRU_GCN(nn.Module):
    """
    A Gated Recurrent Unit (GRU) cell with Graph Convolutional Network (GCN) operations
    (specifically TransformerConv) for processing graph-structured data over time.
    """
    def __init__(self, input_size, hidden_size, n_layer=1, bias=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer # Number of layers (not used in current implementation, defaults to 1)
        
        # GRU weights implemented with TransformerConv layers
        # z (update gate): z_t = sigmoid(W_xz * x_t + W_hz * h_{t-1})
        self.weight_xz = TransformerConv(input_size, hidden_size, bias=bias)
        self.weight_hz = TransformerConv(hidden_size, hidden_size, bias=bias)
        
        # r (reset gate): r_t = sigmoid(W_xr * x_t + W_hr * h_{t-1})
        self.weight_xr = TransformerConv(input_size, hidden_size, bias=bias)
        self.weight_hr = TransformerConv(hidden_size, hidden_size, bias=bias)
        
        # h_tilde (candidate hidden state): h_tilde_t = tanh(W_xh * x_t + W_hh * (r_t * h_{t-1}))
        self.weight_xh = TransformerConv(input_size, hidden_size, bias=bias)
        self.weight_hh = TransformerConv(hidden_size, hidden_size, bias=bias)
    
    def forward(self, input, edge_index, h):
        """
        Forward pass of the GRU-GCN cell.
        
        Args:
            input (torch.Tensor): Current input node features (x_t).
            edge_index (torch.Tensor): Edge indices of the graph.
            h (torch.Tensor): Previous hidden state (h_{t-1}).
            
        Returns:
            torch.Tensor: New hidden state (h_t).
        """
        # Calculate update gate
        z_g = torch.sigmoid(self.weight_xz(input, edge_index) + self.weight_hz(h, edge_index))
        # Calculate reset gate
        r_g = torch.sigmoid(self.weight_xr(input, edge_index) + self.weight_hr(h, edge_index))
        # Calculate candidate hidden state
        h_tilde_g = torch.tanh(self.weight_xh(input, edge_index) + self.weight_hh(r_g * h, edge_index))
        # Calculate new hidden state
        h_out = z_g * h + (1 - z_g) * h_tilde_g    
        return h_out

def stack_truncate(Sample, Mask):
    """
    Stacks a list of tensors (representing sequences) and then truncates
    the padded parts based on a mask.
    
    Args:
        Sample (list of torch.Tensor): List of tensors, where each tensor is
                                      (TimeStep x BatchSize x NumNodes x FeatureDim).
        Mask (torch.Tensor): Mask indicating valid time steps (TimeStep x BatchSize).
        
    Returns:
        list: A list of tensors, where each tensor is a truncated sequence for a batch item.
    """
    Container = []
    # Permute dimensions (TxBxNxD -> BxTxNxD)
    Sample = Sample.permute(1, 0, 2, 3)
    Mask = Mask.permute(1, 0)
    for sample, mask in zip(Sample, Mask):
        # Truncate padding of T in each B
        sample = sample[mask]
        # Collect samples (B[TxNxD])
        Container.append(sample)
    return Container

#%%
class VGRNN(torch.nn.Module):
    """
    Variational Graph Recurrent Neural Network (VGRNN) model.
    This model learns dynamic graph representations by combining recurrent neural networks
    with graph convolutional networks and variational autoencoders.
    """
    def __init__(self, setting):
        """
        Initializes the VGRNN model with parameters from the provided setting dictionary.
        
        Args:
            setting (dict): A dictionary containing model parameters and training settings.
        """
        super().__init__()

        model_params = setting['model_params']
        # Check if 'recurrent' is specified, default to True
        self.recurrent = setting.get('recurrent', True)
        # Check if 'graphRNN' is specified, default to True
        self.graphRNN = setting.get('graphRNN', True)

        # Extract dimensions from model_params
        x_dim = model_params['x_dim'] # Input feature dimension (e.g., number of ROIs)
        z_dim = model_params['z_dim'] # Latent space dimension
        y_dim = model_params['y_dim'] # Output classification dimension (number of classes)
        x_phi_dim = model_params['x_phi_dim'] # Dimension after input feature transformation
        z_phi_dim = model_params['z_phi_dim'] # Dimension after latent state transformation
        x_hidden_dim = model_params['x_hidden_dim'] # Hidden dimension for x decoder
        z_hidden_dim = model_params['z_hidden_dim'] # Hidden dimension for z encoder/prior
        y_hidden_dim = model_params['y_hidden_dim'] # Hidden dimensions for classifier
        rnn_dim = model_params['rnn_dim'] # Hidden dimension for RNN cell
        layer_dims = model_params['layer_dims'] # General hidden layer dimensions

        self.num_nodes = model_params['num_nodes'] # Number of nodes in each graph
        self.num_classes = model_params['num_classes'] # Number of output classes
        self.rnn_dim = rnn_dim
        self.device = get_device() # Get the device (CPU, CUDA, or MPS)
        self.EPS = 1e-15 # Epsilon for numerical stability
        self.rng_state = torch.get_rng_state() # Store initial RNG state

        # --- Model Components ---

        # Data (x) extraction: Transforms input node features
        self.phi_x = dense_vary(x_dim, [], x_phi_dim, last_act='relu')

        # Latent state (z) prior: Defines the prior distribution p(z_t | h_{t-1})
        self.prior_z_hidden = GCN_vary(rnn_dim, [], z_hidden_dim, last_act='relu')
        self.prior_z_mean = dense_vary(z_hidden_dim, [], z_dim, last_act=None)
        self.prior_z_std = dense_vary(z_hidden_dim, [], z_dim, last_act='softplus') # Softplus ensures positive std

        # Latent state (z) encoder: Encodes input and hidden state into posterior q(z_t | x_t, h_{t-1})
        enc_in_dim = x_phi_dim + rnn_dim if self.recurrent else x_phi_dim
        self.enc_z_hidden = GCN_vary(enc_in_dim, layer_dims, z_hidden_dim, last_act='relu')
        self.enc_z_mean = dense_vary(z_hidden_dim, [], z_dim, last_act=None)
        self.enc_z_std = dense_vary(z_hidden_dim, [], z_dim, last_act='softplus')

        # Latent state (z) extraction: Transforms sampled latent state
        self.phi_z = dense_vary(z_dim, [], z_phi_dim, last_act='relu')

        # Latent recurrent update: Updates hidden state h_t based on x_phi and z_phi
        if self.recurrent:
            rnn_in_dim = x_phi_dim + z_phi_dim
            if self.graphRNN: 
                self.rnn_cell = GRU_GCN(rnn_in_dim, rnn_dim) # Custom GRU with GCN
            else: 
                self.rnn_cell = recurrent_cell(rnn_in_dim, rnn_dim, 'gru') # Standard GRU

        # Graph (adjacency matrix) decoder: Reconstructs the graph structure
        self.dec_graph = GraphDecoder()

        # Data (x) decoder: Reconstructs input node features (currently commented out/not used in forward pass)
        # If used, it would decode x from z_phi and h
        # dec_in_dim = z_phi_dim + rnn_dim if self.recurrent else z_phi_dim
        # self.dec_x_hidden = GCN_vary(dec_in_dim, [], x_hidden_dim, last_act='relu')
        # self.dec_x_mean = dense_vary(x_hidden_dim, [], x_dim, last_act=None)
        # self.dec_x_std = dense_vary(x_hidden_dim, [], x_dim, last_act='softplus')

        # Graph classifier: Classifies the entire graph sequence based on a readout
        # Readout dimension depends on whether recurrent and which features are concatenated
        readout_dim = self.num_nodes * (z_phi_dim + rnn_dim) if self.recurrent else self.num_nodes * z_phi_dim
        self.classifier = GraphClassifier(readout_dim, y_hidden_dim, y_dim, dropout=0.5, batch_norm=True)

    def forward(self, graphs, setting, sample=False):
        """
        Forward pass of the VGRNN model. Processes a sequence of graphs.
        
        Args:
            graphs (list of torch_geometric.data.Data): A list of graph objects,
                                                        representing a sequence for a batch.
            setting (dict): Training settings, including 'variational' flag.
            sample (bool): If True, returns sampled latent variables and reconstructions.
            
        Returns:
            tuple: (x_NLL, z_KLD, a_NLL, Readout, Target) or
                   (x_NLL, z_KLD, a_NLL, Readout, Target, z_Sample, adj_Sample, h_Sample, zh_Sample)
                x_NLL (torch.Tensor): Negative Log-Likelihood for x (reconstruction loss for node features).
                                      (Currently returns zero as x decoder is not fully utilized).
                z_KLD (torch.Tensor): KL Divergence loss for latent variable z.
                a_NLL (torch.Tensor): Negative Log-Likelihood for adjacency (graph reconstruction loss).
                Readout (torch.Tensor): Final graph-level representation for classification.
                Target (torch.Tensor): True labels for the graphs.
                z_Sample (list, optional): Sampled latent variables z for each time step.
                adj_Sample (list, optional): Reconstructed adjacency matrices for each time step.
                h_Sample (list, optional): Hidden states h for each time step.
                zh_Sample (list, optional): Concatenation of z and h for each time step.
        """
        variational = setting['variational'] # Flag to enable/disable variational inference

        # Initiate containers for losses and intermediate values across time steps
        Mask = []; Last = [] # For handling padded sequences
        x_NLL = []; z_KLD = []; adj_NLL = [] # Loss components
        Readout = []; Target = [] # For classifier input and true labels
        
        # Containers for sampled outputs if 'sample' is True
        if sample:
            z_Sample = []; adj_Sample = []; h_Sample = []; zh_Sample = []

        # Initialize recurrent hidden state (h) to zeros
        # h is shared across nodes within a graph and across batch items for the first time step
        batch_size = graphs[0].num_graphs # Number of subjects in the batch
        h = torch.zeros(batch_size * self.num_nodes, self.rnn_dim).to(self.device, dtype=torch.float)

        # Iterate through each graph in the sequence (time step)
        for step, graph in enumerate(graphs):
            # Move graph data to the appropriate device
            x = graph.x.to(self.device) # Node features
            edge_index = graph.edge_index.to(self.device) # Graph connectivity
            batch = graph.batch.to(self.device) # Batch assignment for nodes
            adj = graph.adj.to(self.device) # Adjacency matrix (dense)
            y = graph.y.to(self.device) # Graph-level label

            # Mask and last_step indicator for sequence handling
            mask = (graph.pad.to(self.device) == False) # True for valid graphs, False for padding
            last = graph.last.to(self.device) # True if this is the last valid graph in a subject's sequence

            # 1. Data (x) extraction: Transform raw node features
            x_phi = self.phi_x(x)

            # 2. Latent state (z) encoder: Infer posterior q(z_t | x_t, h_{t-1})
            # Input to encoder depends on whether recurrent connections are used
            enc_in = torch.cat([x_phi, h], dim=-1) if self.recurrent else x_phi
            z_enc_hidden = self.enc_z_hidden(enc_in, edge_index) # GCN layer
            z_enc_mean_sb = self.sep_batch(self.enc_z_mean(z_enc_hidden), batch) # Mean of posterior
            
            if variational:
                z_enc_std_sb = self.sep_batch(self.enc_z_std(z_enc_hidden), batch) # Std of posterior
                
                # Latent state (z) prior: p(z_t | h_{t-1})
                if self.recurrent:
                    z_prior_hidden = self.prior_z_hidden(h, edge_index) # GCN layer
                    z_prior_mean_sb = self.sep_batch(self.prior_z_mean(z_prior_hidden), batch)
                    z_prior_std_sb = self.sep_batch(self.prior_z_std(z_prior_hidden), batch)
                else:
                    # If not recurrent, prior is standard normal (mean=0, std=1, log_std=0)
                    z_prior_mean_sb = torch.zeros_like(z_enc_mean_sb).to(self.device)
                    z_prior_std_sb = torch.zeros_like(z_enc_mean_sb).to(self.device)
                
                # Latent state (z) KL divergence loss (between posterior and prior)
                z_kld_sb = self.kld_normal_sb(z_enc_mean_sb, z_enc_std_sb, z_prior_mean_sb, z_prior_std_sb, reduce=False)    
                
                # Latent state (z) reparameterization trick for training
                if self.training:
                    z_sample_sb = self.reparameterize_normal(z_enc_mean_sb, z_enc_std_sb)
                else:
                    z_sample_sb = z_enc_mean_sb # Use mean during evaluation
            else:
                # If not variational, KL divergence is zero, and z is just the mean
                z_kld_sb = torch.zeros(batch_size).to(self.device)
                z_sample_sb = z_enc_mean_sb

            # 3. Latent state (z) extraction: Transform sampled z
            z_phi = self.phi_z(torch.flatten(z_sample_sb, end_dim=1)) # Flatten for dense layer

            # 4. Graph (adjacency matrix) decoder: Reconstruct graph from z and h
            h_sb = self.sep_batch(h, batch) # Separate hidden state back into batch format
            # Input to graph decoder depends on recurrence
            adj_in_sb = torch.cat([z_sample_sb, h_sb], dim=-1) if self.recurrent else z_sample_sb
            adj_dec_sb = self.dec_graph.forward_sb(adj_in_sb) # Reconstructed adjacency logits
            
            # Graph (adjacency matrix) reconstruction loss
            adj_sb = self.sep_batch(adj, batch) # True adjacency matrix in batch format
            adj_nll_sb = self.dec_graph.loss_sb(adj_dec_sb, adj_sb, reduce=False) # BCE loss

            # 5. Latent recurrent update: Update hidden state for next time step
            if self.recurrent:
                rnn_in = torch.cat([x_phi, z_phi], dim=-1) # Input to RNN cell
                if self.graphRNN: 
                    h = self.rnn_cell(rnn_in, edge_index, h) # GRU-GCN
                else: 
                    h = self.rnn_cell(rnn_in, h) # Standard GRU

            # 6. Data (x) decoder: (Currently commented out in the original code, returns zero loss)
            # If enabled, it would reconstruct node features 'x'
            x_NLL_step = torch.zeros(1).to(self.device) # Placeholder for x_NLL

            # 7. Readout layer: Prepare graph-level representation for classifier
            z_readout_sb = self.phi_z(z_enc_mean_sb) # Use mean of z for readout
            readout_sb = torch.cat([z_readout_sb, h_sb], dim=-1) if self.recurrent else z_readout_sb
            readout_flatten = readout_sb.flatten(start_dim=1, end_dim=2) # Flatten to a single vector per graph

            # Append step-wise results to containers
            Mask.append(mask)
            Last.append(last)
            x_NLL.append(x_NLL_step) # Placeholder
            z_KLD.append(z_kld_sb)
            adj_NLL.append(adj_nll_sb)
            Readout.append(readout_flatten)
            Target.append(y)

            # Store sampled outputs if requested
            if sample: 
                z_Sample.append(z_sample_sb)
                adj_Sample.append(torch.sigmoid(adj_dec_sb)) # Sigmoid to get probabilities
                h_Sample.append(h_sb)
                zh_Sample.append(adj_in_sb) # This was the input to the adj decoder, useful for debugging

        # --- Aggregate results across time steps ---
        Mask = torch.stack(Mask) # Stack masks (Time x Batch)
        SeqLen = Mask.sum(dim=0) # Calculate actual sequence length for each subject

        # Average losses over valid time steps for each subject, then mean over batch
        x_NLL_final = ((torch.stack(x_NLL) * Mask).sum(dim=0) / SeqLen).mean()
        z_KLD_final = ((torch.stack(z_KLD) * Mask).sum(dim=0) / SeqLen).mean()
        adj_NLL_final = ((torch.stack(adj_NLL) * Mask).sum(dim=0) / SeqLen).mean()

        Last = torch.stack(Last) # Stack 'last' indicators (Time x Batch)
        assert Last.sum() == batch_size, "Sum of 'last' indicators must equal batch size."
        
        # Select the readout and target only from the last valid time step of each sequence
        Readout_final = torch.stack(Readout)[Last]
        Target_final = torch.stack(Target)[Last]

        if sample: 
            # Stack and truncate sampled sequences for visualization/analysis
            z_Sample = stack_truncate(torch.stack(z_Sample), Mask)
            adj_Sample = stack_truncate(torch.stack(adj_Sample), Mask)
            h_Sample = stack_truncate(torch.stack(h_Sample), Mask)
            zh_Sample = stack_truncate(torch.stack(zh_Sample), Mask)
            return x_NLL_final, z_KLD_final, adj_NLL_final, Readout_final, Target_final, z_Sample, adj_Sample, h_Sample, zh_Sample
        else:
            return x_NLL_final, z_KLD_final, adj_NLL_final, Readout_final, Target_final

    def sep_batch(self, input, batch):
        """
        Separates a flattened batch of node-level features into a batched tensor
        (BatchSize x NumNodes x FeatureDim).
        
        Args:
            input (torch.Tensor): Flattened input tensor (TotalNodesInBatch x FeatureDim).
            batch (torch.Tensor): Batch assignment vector from PyTorch Geometric.
            
        Returns:
            torch.Tensor: Reshaped tensor (BatchSize x NumNodes x FeatureDim).
        """
        # Uses torch_geometric.utils.to_dense_batch for efficient unbatching and padding
        output, _ = to_dense_batch(input, batch)
        return output

    def reparameterize_normal(self, mean, std):
        """
        Performs the reparameterization trick for a normal distribution.
        z = mean + eps * std, where eps is sampled from N(0, 1).
        
        Args:
            mean (torch.Tensor): Mean of the distribution.
            std (torch.Tensor): Standard deviation of the distribution.
            
        Returns:
            torch.Tensor: Sampled latent variable.
        """
        eps = torch.randn(mean.shape).to(self.device, dtype=torch.float)
        return mean + torch.mul(eps, std)

    def kld_normal_sb(self, mean_q, std_q, mean_p, std_p, reduce=False):
        """
        Calculates the Kullback-Leibler Divergence (KLD) between two normal distributions
        (q || p) for batched data.
        
        Args:
            mean_q (torch.Tensor): Mean of the posterior distribution q.
            std_q (torch.Tensor): Standard deviation of the posterior distribution q.
            mean_p (torch.Tensor): Mean of the prior distribution p.
            std_p (torch.Tensor): Standard deviation of the prior distribution p.
            reduce (bool): If True, returns the mean KLD over the batch.
            
        Returns:
            torch.Tensor: KLD loss.
        """
        # KLD formula for two normal distributions:
        # 0.5 * (2 * log(std_p/std_q) + (std_q^2 + (mean_q - mean_p)^2) / std_p^2 - 1)
        kld_element = (2 * (torch.log(std_p + self.EPS) - torch.log(std_q + self.EPS)) +
                      (torch.pow(std_q + self.EPS , 2) + torch.pow(mean_q - mean_p, 2)) / 
                      torch.pow(std_p + self.EPS , 2) - 1)
        # Sum over feature dimensions and nodes, then average by number of nodes
        kld = (0.5 / self.num_nodes) * torch.sum(kld_element, dim=[-1,-2])
        if reduce: kld = torch.mean(kld) # Mean over batch if reduce is True
        return kld

    # The following kld_mvnormal and nll_normal_sb (for x) functions are present in the original
    # file but not directly called in the main VGRNN forward pass as currently implemented.
    # They might be remnants or for alternative model configurations.
    
    def kld_mvnormal(self, mean_q, node_sqm_q, feat_std_q, mean_p, std_p):
        """
        Calculates KLD for multivariate normal distributions (not used in current VGRNN forward).
        """
        n_size, m_size = mean_q.shape[1:]

        node_cov_q = node_sqm_q @ torch.transpose(node_sqm_q, dim0=-2, dim1=-1)

        nll_q = n_size*(2*torch.log(feat_std_q + self.EPS)).sum(dim=-1) + m_size*torch.log(det(node_cov_q) + self.EPS) + n_size*m_size
        nll_p_det = (2*torch.log(std_p + self.EPS)).sum(dim=[-1,-2])

        UpVp = (std_p.pow(2) + self.EPS).pow(-1)
        UqVq = torch.diagonal(node_cov_q, dim1=-2, dim2=-1).unsqueeze(-1) @ feat_std_q.pow(2).unsqueeze(-2)
        nll_p_tr = (UqVq * UpVp).sum(dim=[-1,-2])

        Xd =  (mean_q - mean_p) * std_p
        nll_p_wls = (Xd * Xd).sum(dim=[-1,-2])

        nll_p = nll_p_det + nll_p_tr + nll_p_wls
        kld = (0.5 / (n_size*m_size)) * torch.mean(nll_p - nll_q, dim=0)
        
        return kld if kld > 0 else torch.zeros(1).to(self.device, dtype=torch.float)

    def nll_normal_sb(self, x, mean, std, reduce=False):
        """
        Calculates Negative Log-Likelihood (NLL) for a normal distribution for batched data
        (used for x reconstruction if enabled).
        """
        constant = math.log(2*math.pi)
        xd = x - mean
        nll_element = 2*torch.log(std + self.EPS) + torch.div(xd, std + self.EPS).pow(2) + constant
        # Average over feature dimensions and nodes, then by number of nodes
        nll = (0.5 / (self.num_nodes * mean.shape[-1])) * torch.sum(nll_element, dim=[-1,-2])
        if reduce: nll = torch.mean(nll)
        return nll

    def edge_recon_loss(self, z, batch, pos_edge_index, neg_edge_index=None):
        """
        Calculates edge reconstruction loss (for link prediction, not used in current VGRNN forward).
        """
        num_pos_edges = pos_edge_index.shape[-1]
        pos_edges_dec = self.dec_graph(z, pos_edge_index)
        norm = (num_pos_edges * num_pos_edges) / float((num_pos_edges * num_pos_edges - pos_edges_dec.sum()) * 2)
        pos_loss = -torch.log(pos_edges_dec + self.EPS).mean()
        if neg_edge_index is None: neg_edge_index = batched_negative_sampling(pos_edge_index, batch)
        neg_edges_dec = self.dec_graph(z, neg_edge_index)
        neg_loss = - torch.log(1 - neg_edges_dec + self.EPS).mean()
        return norm * (pos_loss + neg_loss)

