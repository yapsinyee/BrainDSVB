# BrainDSVB
A Deep Probabilistic Spatiotemporal Framework for Dynamic Graph Representation Learning with Application to Brain Disorder Identification

This repository contains the implementation of the Brain DSVB framework, a deep probabilistic spatiotemporal model designed for dynamic graph representation learning, with a specific application to brain disorder identification.

IJCAI 2024 Publication: https://www.ijcai.org/proceedings/2024/0592.pdf

## Environment Setup
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
```
python step1_compute_ldw.py
```
This will create a `data/ldw_data/` directory containing `LDW_abide_data.pkl` and `win_info.pkl`.