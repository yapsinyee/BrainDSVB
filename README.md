# BrainDSVB
A Deep Probabilistic Spatiotemporal Framework for Dynamic Graph Representation Learning with Application to Brain Disorder Identification

This repository contains the implementation of the Brain DSVB framework, a deep probabilistic spatiotemporal model designed for dynamic graph representation learning, with a specific application to brain disorder identification.

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
This script takes raw fMRI time-series data, applies a sliding window approach, estimates Ledoit-Wolf covariance, converts it to correlation, and then thresholds it to create binary adjacency matrices (graphs). It also extracts node features (the correlation matrices themselves).

How to run:
```
python step1_compute_ldw.py
```

This will create a `data/ldw_data/` directory containing `LDW_abide_data.pkl` and `win_info.pkl`.

## 2) Data Preparation - `step2_prepare_data.py`
This script takes the dynamic graphs generated in `step1_compute_ldw.py`, applies stratified K-fold cross-validation, pads the sequences of graphs, and converts them into `torch_geometric.data.Data` objects. These `Data` objects are then saved, organized by cross-validation folds.

How to run:
```
python step2_prepare_data.py
```

This will create a `data/folds_data/` directory containing `graphs_outerX_innerY.pkl` files for each fold.

## 3) Model Definition - `model.py`
This file defines the core neural network architecture, `VGRNN`, which is a Variational Graph Recurrent Neural Network. It includes various helper layers and the logic for the forward pass, including variational inference and graph reconstruction.

## 4) Training Utilities - `train.py`
This file contains the PyTorch `Dataset` and `DataLoader` setup, functions for saving and loading model checkpoints, and the main training and validation loops. It handles optimization, learning rate scheduling, and early stopping.

## 5) Main Execution - `main.py`
This is the main script that ties everything together. It sets up the environment, loads data, initializes the model, and starts the training process.

How to run:
```
python main.py
```

This will start the training process. You will see progress updates in your console. Checkpoints will be saved in the `./saved_models/` directory.

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