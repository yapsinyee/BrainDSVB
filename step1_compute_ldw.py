#%%
import argparse
import os
import pickle
import re

import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from scipy.io import loadmat
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

#%%
def build_output_dir(dataset, connectivity_mode):
    return os.path.join('./data/ldw_data', dataset, connectivity_mode)


def build_output_filename(dataset):
    return f'LDW_{dataset}_data.pkl'


DEFAULT_COBRE_FMRI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cobre_fmri"))
DEFAULT_COBRE_LABELS_MAT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "cobre_static_fc", "cobre_resolution_122.mat")
)
DEFAULT_SCHAEFER_ATLAS = os.path.expanduser(
    "~/nilearn_data/schaefer_2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
)


# Load data from .npy files
def load_abide_timeseries():
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


def infer_num_nodes_from_triangular_length(num_edges):
    # n(n-1)/2 = num_edges
    num_nodes = int((1 + np.sqrt(1 + 8 * num_edges)) / 2)
    if num_nodes * (num_nodes - 1) // 2 != num_edges:
        raise ValueError(f"Cannot infer node count from upper-triangular vector length {num_edges}.")
    return num_nodes


def vector_to_symmetric_matrix(vector, num_nodes):
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    upper = np.triu_indices(num_nodes, k=1)
    matrix[upper] = vector
    matrix[(upper[1], upper[0])] = vector
    np.fill_diagonal(matrix, 1.0)
    return matrix


def labels_from_cobre_subject_ids(subj_ids):
    labels = []
    for raw_id in subj_ids:
        subject_id = str(raw_id).lower()
        if subject_id.startswith('cont'):
            labels.append(0)
        elif subject_id.startswith('sz'):
            labels.append(1)
        else:
            raise ValueError(
                f"Unsupported COBRE subject prefix in '{raw_id}'. "
                "Expected control IDs starting with 'cont' or schizophrenia IDs starting with 'sz'."
            )
    return np.asarray(labels, dtype=int)


def extract_numeric_subject_id(raw_id):
    match = re.search(r"(\d{7})$", str(raw_id))
    if match is None:
        raise ValueError(f"Unable to extract 7-digit subject ID from '{raw_id}'.")
    return match.group(1)


def load_cobre_label_lookup(mat_path):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"COBRE labels/connectivity .mat file not found: {mat_path}")

    mat = loadmat(mat_path)
    if "subj_id" not in mat:
        raise KeyError("Expected key 'subj_id' in COBRE .mat file.")

    subj_ids = [str(item[0]) for item in mat["subj_id"][0]]
    labels = labels_from_cobre_subject_ids(subj_ids)
    return {
        extract_numeric_subject_id(subject_id): int(label)
        for subject_id, label in zip(subj_ids, labels)
    }


def resolve_cobre_atlas_path(atlas_path):
    if atlas_path and os.path.exists(atlas_path):
        return atlas_path
    if os.path.exists(DEFAULT_SCHAEFER_ATLAS):
        return DEFAULT_SCHAEFER_ATLAS
    raise FileNotFoundError(
        "No atlas file found. Provide --atlas-path or place the cached Schaefer atlas at "
        f"{DEFAULT_SCHAEFER_ATLAS}."
    )


def load_cobre_fmri_timeseries(
    fmri_dir,
    atlas_path,
    labels_mat_path,
    standardize=True,
    detrend=True,
    smoothing_fwhm=None,
    subject_limit=None,
):
    if not os.path.isdir(fmri_dir):
        raise FileNotFoundError(f"COBRE fMRI directory not found: {fmri_dir}")

    atlas_path = resolve_cobre_atlas_path(atlas_path)
    label_lookup = load_cobre_label_lookup(labels_mat_path)
    fmri_files = sorted(
        file_name for file_name in os.listdir(fmri_dir)
        if file_name.startswith("fmri_") and file_name.endswith(".nii.gz")
    )
    if subject_limit is not None:
        fmri_files = fmri_files[:subject_limit]
    if not fmri_files:
        raise FileNotFoundError(f"No fmri_*.nii.gz files found in {fmri_dir}")

    subject_ids = []
    data = []
    labels = []
    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize="zscore_sample" if standardize else False,
        detrend=detrend,
        smoothing_fwhm=smoothing_fwhm,
    )

    for file_name in tqdm(fmri_files, desc="Extracting COBRE atlas time series"):
        subject_id = extract_numeric_subject_id(file_name.removeprefix("fmri_").replace(".nii.gz", ""))
        if subject_id not in label_lookup:
            raise KeyError(
                f"Subject {subject_id} from {file_name} is missing from label lookup {labels_mat_path}."
            )

        fmri_path = os.path.join(fmri_dir, file_name)
        img = nib.load(fmri_path)
        timeseries = np.asarray(masker.fit_transform(img), dtype=np.float32)
        if timeseries.ndim != 2 or timeseries.shape[0] < 2 or timeseries.shape[1] < 2:
            raise ValueError(f"Invalid extracted time series shape {timeseries.shape} for {fmri_path}")

        zero_var = np.isclose(timeseries.std(axis=0), 0)
        if np.any(zero_var):
            keep = ~zero_var
            timeseries = timeseries[:, keep]
            print(
                f"Warning: subject {subject_id} had {int(zero_var.sum())} zero-variance parcels; "
                f"keeping {int(keep.sum())} parcels."
            )

        subject_ids.append(subject_id)
        data.append(timeseries)
        labels.append(label_lookup[subject_id])

    return data, np.asarray(labels, dtype=int), subject_ids, atlas_path


def load_cobre_static_connectivity(mat_path, threshold=0.40):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"COBRE .mat file not found: {mat_path}")

    mat = loadmat(mat_path)
    if 'data' not in mat or 'subj_id' not in mat:
        raise KeyError("Expected keys 'data' and 'subj_id' in COBRE .mat file.")

    flat_connectivity = np.asarray(mat['data'], dtype=np.float32)
    subj_ids = [str(item[0]) for item in mat['subj_id'][0]]

    num_subjects, num_edges = flat_connectivity.shape
    num_nodes = infer_num_nodes_from_triangular_length(num_edges)
    labels = labels_from_cobre_subject_ids(subj_ids)

    node_feats = []
    adj_mats = []
    nWin = []

    for i in tqdm(range(num_subjects), desc="Processing COBRE subjects"):
        corr = vector_to_symmetric_matrix(flat_connectivity[i], num_nodes)
        adj = threshold_proportional(np.abs(corr), threshold)
        np.fill_diagonal(adj, 1.0)
        node_feats.append([corr])
        adj_mats.append([adj.astype(np.float32)])
        nWin.append(1)

    return node_feats, adj_mats, nWin, labels

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

def compute_subject_ldw_graph(subject_data, threshold=0.40):
    """
    Computes a single Ledoit-Wolf correlation matrix and thresholded adjacency matrix
    for one subject using the full time series.
    """
    lw = LedoitWolf(assume_centered=False)
    cov = lw.fit(subject_data.squeeze())
    covariance_matrix = cov.covariance_

    corr_neg = cov2corr(covariance_matrix)
    corr = np.abs(corr_neg)
    th_corr = threshold_proportional(corr, threshold)
    np.fill_diagonal(th_corr, 1)

    assert not np.all(np.all((th_corr == 0), axis=1)), 'adjacency matrix contains rows of all zeros'
    assert not np.all(np.all((th_corr == 0), axis=0)), 'adjacency matrix contains columns of all zeros'
    assert np.all(th_corr >= 0), 'adjacency matrix contains negative values'

    return corr_neg, th_corr


def extract_ldw_corr_dynamic(data, wSize, shift, threshold=0.40):
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
            corr_neg, th_corr = compute_subject_ldw_graph(w_data, threshold=threshold)
            corr_mat_subject.append(corr_neg)
            adj_mat_subject.append(th_corr)
        
        node_feats.append(corr_mat_subject)
        LDW_adj_mat.append(adj_mat_subject)
        
    return node_feats, LDW_adj_mat, nWin


def extract_ldw_corr_static(data, threshold=0.40):
    """
    Extracts one static Ledoit-Wolf correlation graph per subject using the full time series.
    The downstream pipeline still receives a sequence, but with length 1 for each subject.
    """
    node_feats = []
    LDW_adj_mat = []
    nWin = []

    for subject_data in tqdm(data, desc="Processing subjects"):
        corr_neg, th_corr = compute_subject_ldw_graph(subject_data, threshold=threshold)
        node_feats.append([corr_neg])
        LDW_adj_mat.append([th_corr])
        nWin.append(1)

    return node_feats, LDW_adj_mat, nWin


def compute_num_windows_per_subject(data, wSize, shift):
    overlap = wSize - shift
    return [int((item.shape[0] - overlap) / (wSize - overlap)) for item in data]


def filter_subjects_by_min_windows(data, labels, subject_ids, wSize, shift, min_windows):
    nWin = compute_num_windows_per_subject(data, wSize, shift)
    keep_idx = [i for i, n_win in enumerate(nWin) if n_win >= min_windows]
    removed = [(subject_ids[i], nWin[i]) for i in range(len(nWin)) if i not in keep_idx]

    if len(keep_idx) == len(data):
        return data, labels, subject_ids, removed

    filtered_data = [data[i] for i in keep_idx]
    filtered_labels = labels[keep_idx]
    filtered_subject_ids = [subject_ids[i] for i in keep_idx]
    return filtered_data, filtered_labels, filtered_subject_ids, removed


def parse_args():
    parser = argparse.ArgumentParser(description="Compute static or dynamic brain connectivity graphs.")
    parser.add_argument(
        "--dataset",
        choices=["abide", "cobre_static", "cobre_fmri"],
        default="abide",
        help="Input dataset to preprocess.",
    )
    parser.add_argument(
        "--connectivity-mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Connectivity estimation mode.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Sliding window size used only for dynamic connectivity.",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=10,
        help="Sliding window shift used only for dynamic connectivity.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Proportional threshold applied to the absolute correlation matrix.",
    )
    parser.add_argument(
        "--input-mat",
        default=DEFAULT_COBRE_LABELS_MAT,
        help="Path to the COBRE connectivity .mat file when --dataset cobre_static is used.",
    )
    parser.add_argument(
        "--fmri-dir",
        default=DEFAULT_COBRE_FMRI_DIR,
        help="Directory containing COBRE raw fMRI NIfTI files for --dataset cobre_fmri.",
    )
    parser.add_argument(
        "--atlas-path",
        default=DEFAULT_SCHAEFER_ATLAS,
        help="Atlas NIfTI used to extract parcel time series for --dataset cobre_fmri.",
    )
    parser.add_argument(
        "--labels-mat",
        default=DEFAULT_COBRE_LABELS_MAT,
        help="COBRE .mat file providing subject IDs and labels for --dataset cobre_fmri.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable per-parcel standardization during atlas time-series extraction.",
    )
    parser.add_argument(
        "--no-detrend",
        action="store_true",
        help="Disable detrending during atlas time-series extraction.",
    )
    parser.add_argument(
        "--smoothing-fwhm",
        type=float,
        default=None,
        help="Optional spatial smoothing in mm before atlas extraction for --dataset cobre_fmri.",
    )
    parser.add_argument(
        "--subject-limit",
        type=int,
        default=None,
        help="Optional cap on the number of COBRE fMRI subjects to preprocess for debugging.",
    )
    parser.add_argument(
        "--min-windows",
        type=int,
        default=None,
        help="Optional minimum dynamic window count; subjects below this are excluded.",
    )
    return parser.parse_args()

# Main execution block
if __name__ == "__main__":
    args = parse_args()

    # Ensure data directory exists
    data_timeseries_path = './data/timeseries'
    os.makedirs(data_timeseries_path, exist_ok=True)

    # --- IMPORTANT ---
    # The following lines assume you have 'power_asd.npy' and 'power_td.npy'
    # in the './data/timeseries/' directory.
    # If you don't have them, you need to create dummy files or get the actual data.
    # Refer to the "Dummy Data Placeholder" section in the instructions for creating dummy files.
    # --- IMPORTANT ---

    wSize = args.window_size
    shift = args.shift

    subject_ids = None
    extraction_info = {}

    if args.dataset == "cobre_static":
        if args.connectivity_mode != "static":
            raise ValueError("COBRE connectivity vectors only support --connectivity-mode static.")
        print(f"Loading COBRE static connectivity from {args.input_mat}...")
        node_feats, adj_mats, nWin, labels = load_cobre_static_connectivity(
            args.input_mat,
            threshold=args.threshold,
        )
        print(f"Loaded COBRE static connectivity for {len(node_feats)} subjects.")
    elif args.dataset == "cobre_fmri":
        print(f"Loading COBRE raw fMRI from {args.fmri_dir}...")
        data, labels, subject_ids, atlas_path = load_cobre_fmri_timeseries(
            fmri_dir=args.fmri_dir,
            atlas_path=args.atlas_path,
            labels_mat_path=args.labels_mat,
            standardize=not args.no_standardize,
            detrend=not args.no_detrend,
            smoothing_fwhm=args.smoothing_fwhm,
            subject_limit=args.subject_limit,
        )
        extraction_info = {
            "atlas_path": atlas_path,
            "fmri_dir": os.path.abspath(args.fmri_dir),
            "labels_mat": os.path.abspath(args.labels_mat),
            "standardize": not args.no_standardize,
            "detrend": not args.no_detrend,
            "smoothing_fwhm": args.smoothing_fwhm,
        }
        print(
            f"Loaded {len(data)} COBRE raw fMRI subjects with "
            f"{data[0].shape[1]} parcels and {data[0].shape[0]} time points each."
        )

        if args.connectivity_mode == "dynamic":
            if args.min_windows is not None:
                data, labels, subject_ids, removed_subjects = filter_subjects_by_min_windows(
                    data=data,
                    labels=labels,
                    subject_ids=subject_ids,
                    wSize=wSize,
                    shift=shift,
                    min_windows=args.min_windows,
                )
                if not data:
                    raise ValueError(
                        f"No COBRE fMRI subjects remain after applying --min-windows {args.min_windows}."
                    )
                extraction_info["min_windows"] = args.min_windows
                extraction_info["removed_subjects"] = removed_subjects
                if removed_subjects:
                    print(
                        f"Removed {len(removed_subjects)} subjects with fewer than "
                        f"{args.min_windows} windows: {removed_subjects}"
                    )

            print(
                f"Extracting dynamic Ledoit-Wolf connectivity with window size {wSize}, "
                f"shift {shift}, threshold {args.threshold}..."
            )
            node_feats, adj_mats, nWin = extract_ldw_corr_dynamic(data, wSize, shift, threshold=args.threshold)
        else:
            print(f"Extracting static Ledoit-Wolf connectivity with threshold {args.threshold}...")
            node_feats, adj_mats, nWin = extract_ldw_corr_static(data, threshold=args.threshold)
    else:
        print("Loading raw fMRI time-series data...")
        data, labels = load_abide_timeseries()
        # Ensure data is a list of numpy arrays, as expected by extract_ldw_corr
        data = [np.array(item) for item in data]
        print(f"Loaded data from {len(data)} subjects.")

        if args.connectivity_mode == "dynamic":
            print(
                f"Extracting dynamic Ledoit-Wolf connectivity with window size {wSize}, "
                f"shift {shift}, threshold {args.threshold}..."
            )
            node_feats, adj_mats, nWin = extract_ldw_corr_dynamic(data, wSize, shift, threshold=args.threshold)
        else:
            print(f"Extracting static Ledoit-Wolf connectivity with threshold {args.threshold}...")
            node_feats, adj_mats, nWin = extract_ldw_corr_static(data, threshold=args.threshold)
    print("Extraction complete.")

    # Prepare data for saving
    LDW_data = {}
    LDW_data['adj_mat'] = adj_mats
    LDW_data['node_feat'] = node_feats
    LDW_data['labels'] = labels
    if subject_ids is not None:
        LDW_data['subject_ids'] = subject_ids

    win_info = {}
    win_info['connectivity_mode'] = args.connectivity_mode
    win_info['wSize'] = wSize if args.connectivity_mode == 'dynamic' else None
    win_info['shift'] = shift if args.connectivity_mode == 'dynamic' else None
    win_info['threshold'] = args.threshold
    win_info['nWin'] = nWin
    win_info['dataset'] = args.dataset
    win_info['subject_ids'] = subject_ids
    win_info.update(extraction_info)

    # Define path to save processed data
    saveTo = build_output_dir(args.dataset, args.connectivity_mode)
    os.makedirs(saveTo, exist_ok=True) # Create directory if it doesn't exist
    
    print(f"Saving processed data to {saveTo}...")
    # Save the processed data using pickle
    with open(os.path.join(saveTo, build_output_filename(args.dataset)), 'wb') as f:
        pickle.dump(LDW_data, f, protocol=4) # protocol=4 for compatibility
        
    # Save window information
    with open(os.path.join(saveTo, 'win_info.pkl'), 'wb') as f:
        pickle.dump(win_info, f, protocol=4)
    print("Processed data saved successfully.")
