# BrainDSVB
A Deep Probabilistic Spatiotemporal Framework for Dynamic Graph Representation Learning with Application to Brain Disorder Identification

This repository contains the implementation of the Brain DSVB framework, a deep probabilistic spatiotemporal model designed for dynamic graph representation learning, with a specific application to brain disorder identification.

The repo supports two dataset entry paths:

- `abide`: original time-series preprocessing path using `data/timeseries/power_asd.npy` and `data/timeseries/power_td.npy`
- `cobre_static`: static-connectivity path using a MATLAB `.mat` file with one upper-triangular connectivity vector per subject, such as `../data/cobre/cobre_resolution_122.mat`
- `cobre_fmri`: raw COBRE NIfTI path using per-subject fMRI volumes in `../data/cobre_fmri/`, parcel time-series extraction with an atlas, then graph construction

IJCAI 2024 Publication: https://www.ijcai.org/proceedings/2024/0592.pdf

## Environment Setup
Bash command:
```
# Create a new conda environment
conda create -n braindsvb python=3.10

# Activate the environment
conda activate braindsvb

# Installation command
pip install -r requirements.txt
```

## 1) Data Preprocessing - `step1_compute_ldw.py`
This script supports both connectivity modes for the original `abide` time-series path, static connectivity import for `cobre_static`, and raw-NIfTI preprocessing for `cobre_fmri`.

- `dynamic`: applies a sliding window approach, estimates Ledoit-Wolf covariance per window, converts it to correlation, and thresholds it to create a graph sequence.
- `static`: estimates a single Ledoit-Wolf covariance matrix on the full subject time series and creates one graph per subject.

How to run:
```
python step1_compute_ldw.py --connectivity-mode dynamic --window-size 20 --shift 10
python step1_compute_ldw.py --connectivity-mode static
python step1_compute_ldw.py --dataset cobre_static --connectivity-mode static --input-mat ../data/cobre_static_fc/cobre_resolution_122.mat
python step1_compute_ldw.py --dataset cobre_fmri --connectivity-mode static --fmri-dir ../data/cobre_fmri --atlas-path ~/nilearn_data/schaefer_2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz
python step1_compute_ldw.py --dataset cobre_fmri --connectivity-mode dynamic --fmri-dir ../data/cobre_fmri --window-size 20 --shift 10
```

This will create dataset- and mode-specific outputs under `data/ldw_data/<dataset>/<mode>/`.

## 2) Data Preparation - `step2_prepare_data.py`
This script takes the graphs generated in `step1_compute_ldw.py`, applies stratified K-fold cross-validation, pads graph sequences when needed, and converts them into `torch_geometric.data.Data` objects. These `Data` objects are then saved, organized by cross-validation folds.

How to run:
```
python step2_prepare_data.py --connectivity-mode dynamic
python step2_prepare_data.py --connectivity-mode static
python step2_prepare_data.py --dataset cobre_static --connectivity-mode static
python step2_prepare_data.py --dataset cobre_fmri --connectivity-mode static
python step2_prepare_data.py --dataset cobre_fmri --connectivity-mode dynamic
```

This will create dataset- and mode-specific outputs under `data/folds_data/<dataset>/<mode>/`.

## 3) Model Definition - `model.py`
This file defines the core neural network architecture, `VGRNN`, which is a Variational Graph Recurrent Neural Network. It includes various helper layers and the logic for the forward pass, including variational inference and graph reconstruction.

## 4) Training Utilities - `train.py`
This file contains the PyTorch `Dataset` and `DataLoader` setup, functions for saving and loading model checkpoints, and the main training and validation loops. It handles optimization, learning rate scheduling, and early stopping.

## 5) Main Execution - `main.py`
This is the main script that ties everything together. It sets up the environment, loads data, initializes the model, and starts the training process.

How to run:
```
python main.py --connectivity-mode dynamic --outer-loop 1 --inner-loop 1
python main.py --connectivity-mode static --outer-loop 1 --inner-loop 1
python main.py --dataset cobre_static --connectivity-mode static --outer-loop 1 --inner-loop 1
python main.py --dataset cobre_fmri --connectivity-mode static --outer-loop 1 --inner-loop 1
python main.py --dataset cobre_fmri --connectivity-mode dynamic --outer-loop 1 --inner-loop 1
```

This will start the training process. Dynamic mode keeps the recurrent graph model enabled; static mode disables the recurrent components and trains on one graph per subject. The training script now infers `num_nodes`, `x_dim`, and `num_classes` from the prepared graphs, so 122-node COBRE folds do not require manual code edits. Checkpoints will be saved in the `./saved_models/` directory.

## 6) Optional: Visualization - `visualize.py`
This script helps you analyze the training process by loading a saved checkpoint and plotting the loss curves and accuracy over epochs.

How to run:
After training, you can analyze the results by running:

```
python visualize.py ./saved_models/VGRNN_softmax_adv_fold11.pth
```

Replace `./saved_models/VGRNN_softmax_adv_fold11.pth` with the actual path to your saved checkpoint file.

## Cite
```
@inproceedings{ijcai2024p592,
  title     = {A Deep Probabilistic Spatiotemporal Framework for Dynamic Graph Representation Learning with Application to Brain Disorder Identification},
  author    = {Yap, Sin-Yee and Loo, Junn Yong and Ting, Chee-Ming and Noman, Fuad and Phan, Raphaël C.-W. and Razi, Adeel and Dowe, David L.},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {5353--5361},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/592},
  url       = {https://doi.org/10.24963/ijcai.2024/592},
}
```
