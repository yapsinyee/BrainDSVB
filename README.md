DSVB: A Deep Probabilistic Spatiotemporal Framework for Dynamic Graph Representation Learning with Application to Brain Disorder Identification

This repository contains the implementation of the DSVB framework, a deep probabilistic spatiotemporal model designed for dynamic graph representation learning, with a specific application to brain disorder identification.

IJCAI 2024 Publication: https://www.ijcai.org/proceedings/2024/0592.pdf
Features

    Dynamic Graph Construction: Transforms fMRI time-series data into dynamic brain graphs using sliding windows and Ledoit-Wolf covariance estimation with proportional thresholding.

    Nested Cross-Validation: Prepares data into stratified K-fold cross-validation splits for robust model evaluation.

    Variational Graph Recurrent Neural Network (VGRNN): Implements a novel VGRNN architecture that combines recurrent neural networks with graph convolutional layers for learning spatiotemporal graph representations.

    Probabilistic Modeling: Utilizes variational inference for robust latent space learning.

    MPS Compatibility: Optimized for Apple Silicon Macs leveraging Metal Performance Shaders (MPS) for accelerated computation.

    Comprehensive Training & Evaluation: Includes modules for model training, checkpointing, and performance visualization.

Prerequisites

Before running the framework, ensure you have Python 3.8+ installed. The following Python libraries are required:

scikit-learn
pandas
tqdm
nilearn
dill
torch
torchvision
torchaudio
torch_geometric
matplotlib

Installation

    Clone the repository (or ensure all provided files are in your working directory).

    Create a virtual environment (recommended):

    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate

    Install dependencies:
    Create a requirements.txt file with the content listed in the "Prerequisites" section above, then run:

    pip install -r requirements.txt

        Note on PyTorch for Mac Mini (MPS): When installing torch, torchvision, and torchaudio, ensure you install the MPS-enabled version. Typically, pip install torch torchvision torchaudio will automatically handle this for Apple Silicon. If you encounter issues, refer to the official PyTorch website for specific macOS installation instructions.

Data Preparation

This framework expects fMRI time-series data in .npy format. Specifically, it looks for power_asd.npy (for Autism Spectrum Disorder subjects) and power_td.npy (for Typically Developing subjects) in the ./data/timeseries/ directory.

Important: These data files are not included in this repository. You will need to obtain or create them. For demonstration or testing purposes, you can generate dummy data:

    Create data directories:

    mkdir -p data/timeseries
    mkdir -p data/ldw_data
    mkdir -p data/folds_data
    mkdir -p saved_models

    Generate Dummy Data (Optional, for testing):
    Create a Python script (e.g., create_dummy_data.py) with the following content and run it:

    import numpy as np
    import os

    num_subjects_asd = 5
    num_subjects_td = 5
    time_points_min = 100
    time_points_max = 200
    num_ROIs = 264 # As specified in step1_compute_ldw.py

    data_dir = './data/timeseries'
    os.makedirs(data_dir, exist_ok=True)

    # Dummy ASD data
    asd_data = []
    for _ in range(num_subjects_asd):
        time_points = np.random.randint(time_points_min, time_points_max + 1)
        asd_data.append(np.random.rand(time_points, num_ROIs))
    np.save(os.path.join(data_dir, 'power_asd.npy'), np.array(asd_data, dtype=object), allow_pickle=True)

    # Dummy TD data
    td_data = []
    for _ in range(num_subjects_td):
        time_points = np.random.randint(time_points_min, time_points_max + 1)
        td_data.append(np.random.rand(time_points, num_ROIs))
    np.save(os.path.join(data_dir, 'power_td.npy'), np.array(td_data, dtype=object), allow_pickle=True)

    print(f"Dummy data saved to {data_dir}")

    Run: python create_dummy_data.py

Usage

Follow these steps to run the framework:
Step 1: Compute Ledoit-Wolf Correlations and Adjacency Matrices

This script processes the raw fMRI time-series data into dynamic brain graphs.

python step1_compute_ldw.py

This will create LDW_abide_data.pkl and win_info.pkl in the ./data/ldw_data/ directory.
Step 2: Prepare Data for Training

This script organizes the dynamic graphs into cross-validation folds and converts them into torch_geometric.data.Data objects.

python step2_prepare_data.py

This will create graphs_outerX_innerY.pkl files (e.g., graphs_outer1_inner1.pkl) in the ./data/folds_data/ directory, representing different cross-validation splits.
Step 3: Train the Model

The main.py script orchestrates the model training. It loads the prepared data, initializes the VGRNN model, and starts the training loop.

python main.py

During execution, you will see real-time training progress. Model checkpoints will be saved in the ./saved_models/ directory.

MPS Compatibility: The main.py script includes os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' at the beginning, which ensures PyTorch can fall back to CPU if certain MPS operations are not supported, making it more robust for Mac Mini environments.
Step 4: Visualize Training Results

After training, use visualize.py to plot the loss curves and accuracy over epochs.

python visualize.py ./saved_models/VGRNN_softmax_adv_fold11_mac-1.pth

Note: Replace ./saved_models/VGRNN_softmax_adv_fold11_mac-1.pth with the actual path to your saved checkpoint file (e.g., if you changed outer_loop or inner_loop in main.py).
File Structure

    main.py: The main script to run the training and evaluation pipeline.

    model.py: Defines the VGRNN architecture, including various graph convolutional layers and recurrent units.

    step1_compute_ldw.py: Handles initial data preprocessing, including sliding window, Ledoit-Wolf covariance, and graph construction.

    step2_prepare_data.py: Prepares the constructed graphs for training, including cross-validation splitting and converting to PyTorch Geometric format.

    train.py: Contains the core training loop, data loading utilities (myDataset, padseq), checkpointing logic, and evaluation functions.

    visualize.py: Provides utilities to load trained model checkpoints and visualize training/validation curves.

    verify_torch.py: A utility script to check PyTorch's MPS (Metal Performance Shaders) availability on your system.

    requirements.txt: Lists all Python dependencies.

    data/: Directory to store raw and processed data.

        timeseries/: Expected location for power_asd.npy and power_td.npy.

        ldw_data/: Stores intermediate processed data from step1_compute_ldw.py.

        folds_data/: Stores cross-validation splits of graph data from step2_prepare_data.py.

    saved_models/: Directory to store trained model checkpoints.