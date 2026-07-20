# Hunting Leopards with Half a Brain

## Overview
This project aims to protect Saudi Arabia's endangered Arabian leopard using real-time edge AI. We are deploying a YOLO object detection model across constrained edge hardware (Raspberry Pi Zero) and a more powerful local edge server (Raspberry Pi 3).

By implementing a Split Inference Pipeline, we offload partial computation from the Pi Zero to the Pi 3, overcoming hardware bottlenecks, unstable communication, and power constraints in remote wildlife reserves.

## Quick Start
1. Clone this repository.
2. Create your virtual environment and install dependencies.
3. Run `python setup_data.py` to automatically download and extract the dataset into the correct local directories.

## Project File Tree
```text
Hunting-Leopard-with-half-a-brain/
├── models/
│   ├── v1_model/                      # Older model (Single class only)
│   │   
│   └── v2_model/                      # New, updated model (Mutliple classes)
│       
│ 
├── notebooks/                   # Jupyter notebooks
│   ├── v1_train.ipynb
│   └── v2_train.ipynb
│  
├── README.md
├── requirements.txt
│
├── -------------- [ IGNORED BY GIT ] --------------
├── .git/                        # Git tracking (Hidden)
├── data/                        # Leopard datasets & negative samples 
│   ├── test/
│   ├── train/
│   └── val/
├── .venv/ ...etc                # Python virtual environment 
└── *.zip / *.rar                # Compressed data archives
```