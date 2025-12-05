Simulation of Health Time Series with Nonstationarity

Full Reproduction of Toye et al. (2024)

This repository provides a complete, modular implementation for replicating the paper:

“Simulation of Health Time Series with Nonstationarity” — Toye et al., 2024

It includes pipelines to simulate physiological time-series (glucose, EDA) under various distribution shifts, evaluate forecasting and classification performance, and benchmark against real datasets.

CS598_DLH_Project/
│
├── eda/                # EDA (electrodermal activity) simulation functions
│   ├── process_wesad.py
│   └── simulate_eda.py
│
├── glucose/            # Glucose simulation + forecasting pipelines
│   ├── simulate_glucose.py
│   ├── forecasting_models.py
│   └── utils.py
│
├── ohio/               # OHIO-T1DM dataset processing scripts for glucose data
│   ├── load_ohio.py
│   ├── preprocess_ohio.py
│   └── ohio_utils.py
│
└── wesad/              # WESAD dataset utilities for EDA/stress classification
    ├── load_wesad.py
    ├── preprocess_wesad.py
    └── wesad_utils.py
