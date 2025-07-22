#%%
from sklearn.covariance import LedoitWolf
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure
import math
import os
import pickle
import dill

#%%
# Load data from .npy files
def load_data():
    """
    Loads ASD and TD fMRI time-series data from specified paths.
    Concatenates them and creates corresponding labels.
    Performs initial data cleaning by removing subjects with all-zero ROIs.
    """
    # Load Autism Spectrum Disorder (ASD) data
    asd_data = np.load('./data/timeseries/power_asd.npy', allow_pickle=True)
    # Load Typically Developing (TD) data
    td_data = np.load('./data/timeseries/power_td.npy', allow_pickle=True)
    
    # Concatenate data from both groups
    data = np.concatenate((asd_data, td_data))
    # Create labels: 1 for ASD, 0 for TD
    labels = np.concatenate(([np.ones(len(asd_data)), np.zeros(len(td_data))])).astype(int) # 1 : ASD, 0: TD
    
    # Check for missing ROIs data (all zeros in a column or row)
    nROIs = 264 # Expected number of Regions of Interest
    to_remove = [] # List to store indices of subjects to be removed

    # Iterate through each subject's data
    for i, x in enumerate(data):
        # Check if the number of ROIs matches the expected number
        if x.shape[1] == nROIs:
            # Check for columns (ROIs) that are all zeros
            results = np.all((x == 0), axis=0)
            if np.any(results):
                to_remove.append(i)
                print(f'Data of subject {i} is removed due to missing column ROI/s observations')
        else:
            # If ROI count doesn't match, check for rows (time points) that are all zeros
            results = np.all((x == 0), axis=1)
            if np.any(results):
                to_remove.append(i)
                print(f'Data of subject {i} is removed due to missing row ROI/s observations')
    
    # Remove identified subjects from data and labels
    if to_remove:
        data = np.delete(data, to_remove, 0)
        labels = np.delete(labels, to_remove, 0)
    
    return data, labels

# Compute the correlation & do thresholding
## Function for Converting Covariance to Correlation
def cov2corr(covariance):
    """
    Converts a covariance matrix to a correlation matrix.
    """
    v = np.sqrt(np.diag(covariance)) # Standard deviations
    outer_v = np.outer(v, v)        # Outer product of standard deviations
    correlation = covariance / outer_v # Correlation formula
    correlation[covariance == 0] = 0 # Handle cases where covariance is zero
    return correlation

## Function for Proportional Thresholding
def threshold_proportional(W, p, copy=True):
    """
    Thresholds the connectivity matrix by preserving a proportion 'p' of the strongest weights.
    All other weights and diagonal elements are set to 0.
    
    Args:
        W (np.ndarray): Weighted or binary connectivity matrix.
        p (float): Proportion of weights to preserve (0 < p < 1).
        copy (bool): If True, a copy of W is made to avoid modifying in place.
    
    Returns:
        np.ndarray: Thresholded connectivity matrix.
    """
    assert 0 < p < 1, "Proportion p must be between 0 and 1."
    if copy:
        W = W.copy()
    n = len(W)                        # number of nodes
    np.fill_diagonal(W, 0)            # clear diagonal (self-connections)
    
    # Determine if matrix is symmetric to handle upper/lower triangle efficiently
    if np.all(W == W.T):                # if symmetric matrix
        W[np.tril_indices(n)] = 0        # set lower triangle to 0 to avoid double counting
        ud = 2                        # factor for symmetric matrix (links counted twice)
    else:
        ud = 1
    
    ind = np.where(W)                    # find all non-zero link indices
    I = np.argsort(W[ind])[::-1]        # sort indices by magnitude in descending order
    
    # Number of links to be preserved
    en = round((n * n - n) * p / ud)
    
    # Set weights of weaker links to 0
    W[(ind[0][I][en:], ind[1][I][en:])] = 0    # apply threshold
    
    if ud == 2:                        # if symmetric matrix
        W[:, :] = W + W.T                        # reconstruct symmetry
    
    # Ensure the highest correlation coefficient is 1 (or close to it)
    # This line seems to be a specific heuristic, might need review based on data characteristics.
    W[W > 0.9999] = 1                          
    return W

def extract_ldw_corr(data, wSize, shift):
    """
    Extracts Ledoit-Wolf optimal shrinkage covariance, converts to correlation,
    and applies proportional thresholding using a sliding window approach.
    
    Args:
        data (list): List of subject time-series data (each element is a np.ndarray).
        wSize (int): Sliding window size.
        shift (int): Shift (step size) for the sliding window.
    
    Returns:
        tuple: (node_feats, LDW_adj_mat, nWin)
            node_feats (list): List of lists, where each inner list contains
                               correlation matrices (node features) for each window of a subject.
            LDW_adj_mat (list): List of lists, where each inner list contains
                               thresholded adjacency matrices for each window of a subject.
            nWin (list): List of number of windows for each subject.
    """
    nSub = len(data)
    nROI = data[0].shape[1] # Number of ROIs
    tpLen = [item.shape[0] for item in data] # Time points length for each subject
    
    overlap = wSize - shift # Overlap between consecutive windows
    # Calculate number of windows for each subject
    nWin = [int((l - overlap) / (wSize - overlap)) for l in tpLen]
    
    node_feats = [] # Container for node features (correlation matrices)
    LDW_adj_mat = [] # Container for adjacency matrices

    for sub in tqdm(range(len(data)), desc="Processing subjects"):    # For each subject
        corr_mat_subject = [] # Correlation matrices for current subject
        adj_mat_subject = [] # Adjacency matrices for current subject
        
        for wi in range(nWin[sub]): # Iterate through windows for the current subject
            st = wi * (wSize - overlap) # Start index of the window
            en = st + wSize             # End index of the window
            w_data = data[sub][st:en, :] # Extract data for the current window
            
            # Apply Ledoit-Wolf covariance estimation
            lw = LedoitWolf(assume_centered=False)
            cov = lw.fit(w_data.squeeze())
            covariance_matrix = cov.covariance_
            
            # Convert covariance to correlation
            corr_neg = cov2corr(covariance_matrix)
            corr = np.abs(corr_neg) # Take absolute value for thresholding
            corr_mat_subject.append(corr_neg) # Store original correlation matrix as node features

            # Apply proportional thresholding to create adjacency matrix
            th_corr = threshold_proportional(corr, 0.40) # Keep top 40% coefficients
            
            # Fill diagonal with ones to avoid zero-degree nodes (common in graph analysis)
            np.fill_diagonal(th_corr, 1)
            adj_mat_subject.append(th_corr) # Store thresholded adjacency matrix

            # Assertions for data integrity (optional, but good for debugging)
            assert not np.all(np.all((th_corr == 0), axis=1)), 'adjacency matrix contains rows of all zeros'
            assert not np.all(np.all((th_corr == 0), axis=0)), 'adjacency matrix contains columns of all zeros'
            assert np.all(th_corr >= 0), 'adjacency matrix contains negative values'
        
        node_feats.append(corr_mat_subject)
        LDW_adj_mat.append(adj_mat_subject)
        
    return node_feats, LDW_adj_mat, nWin

# Main execution block
if __name__ == "__main__":
    # Ensure data directory exists
    data_timeseries_path = './data/timeseries'
    os.makedirs(data_timeseries_path, exist_ok=True)

    # --- IMPORTANT ---
    # The following lines assume you have 'power_asd.npy' and 'power_td.npy'
    # in the './data/timeseries/' directory.
    # If you don't have them, you need to create dummy files or get the actual data.
    # Refer to the "Dummy Data Placeholder" section in the instructions for creating dummy files.
    # --- IMPORTANT ---

    print("Loading raw fMRI time-series data...")
    data, labels = load_data()
    # Ensure data is a list of numpy arrays, as expected by extract_ldw_corr
    data = [np.array(item) for item in data]
    print(f"Loaded data from {len(data)} subjects.")

    # Sliding window parameters
    wSize = 20  # Window size (number of time points in each window)
    shift = 10  # Shift size (number of time points to move for the next window)

    print(f"Extracting Ledoit-Wolf correlations and adjacency matrices with window size {wSize} and shift {shift}...")
    node_feats, adj_mats, nWin = extract_ldw_corr(data, wSize, shift)
    print("Extraction complete.")

    # Prepare data for saving
    LDW_data = {}
    LDW_data['adj_mat'] = adj_mats
    LDW_data['node_feat'] = node_feats
    LDW_data['labels'] = labels

    win_info = {}
    win_info['wSize'] = wSize
    win_info['shift'] = shift
    win_info['nWin'] = nWin

    # Define path to save processed data
    saveTo = './data/ldw_data/'
    os.makedirs(saveTo, exist_ok=True) # Create directory if it doesn't exist
    
    print(f"Saving processed data to {saveTo}...")
    # Save the processed data using pickle
    with open(os.path.join(saveTo, 'LDW_abide_data.pkl'), 'wb') as f:
        pickle.dump(LDW_data, f, protocol=4) # protocol=4 for compatibility
        
    # Save window information
    with open(os.path.join(saveTo, 'win_info.pkl'), 'wb') as f:
        pickle.dump(win_info, f, protocol=4)
    print("Processed data saved successfully.")

