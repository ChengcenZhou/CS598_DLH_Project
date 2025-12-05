import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

def load_wesad_data(base_path):
    """Load WESAD dataset and extract EDA features"""
    subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 
                'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
    
    all_data = []
    all_labels = []
    
    for subject in subjects:
        # Load pickle file
        data = pd.read_pickle(f"{base_path}/{subject}/{subject}.pkl")
        
        # Extract EDA signal (chest-worn device)
        eda = data['signal']['chest']['EDA']
        labels = data['label']
        
        # Extract features from windows
        window_size = 700  # ~1 minute at 700Hz
        stride = 350  # 50% overlap
        
        features = []
        window_labels = []
        
        for i in range(0, len(eda) - window_size, stride):
            window = eda[i:i+window_size]
            label_window = labels[i:i+window_size]
            
            # Get most common label in window
            window_label = stats.mode(label_window, keepdims=True)[0][0]
            
            # Skip transient/not defined states (0, 4, 5, 6, 7)
            if window_label in [0, 4, 5, 6, 7]:
                continue
            
            # Extract features
            feat = extract_eda_features(window)
            features.append(feat)
            
            # Binary classification: stress (2) vs baseline (1)
            window_labels.append(1 if window_label == 2 else 0)
        
        all_data.extend(features)
        all_labels.extend(window_labels)
    
    return np.array(all_data), np.array(all_labels)

def extract_eda_features(window):
    """Extract statistical features from EDA window"""
    features = [
        np.mean(window),
        np.std(window),
        np.min(window),
        np.max(window),
        np.percentile(window, 25),
        np.percentile(window, 75),
        stats.skew(window),
        stats.kurtosis(window),
        # First and second derivatives
        np.mean(np.diff(window)),
        np.std(np.diff(window)),
        # Power in different frequency bands (simple approximation)
        np.sum(np.abs(np.fft.fft(window)[:50])),
        np.sum(np.abs(np.fft.fft(window)[50:100]))
    ]
    return features

def split_data_by_time(X, y, train_ratio=0.6, val_ratio=0.2):
    """Split data chronologically to preserve temporal order"""
    n = len(X)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    X_train = X[:train_idx]
    y_train = y[:train_idx]
    X_val = X[train_idx:val_idx]
    y_val = y[train_idx:val_idx]
    X_test = X[val_idx:]
    y_test = y[val_idx:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
