import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_ohio_data(data_path, patient_id='559'):
    """Load OHIO-T1DM dataset"""
    train_file = f"{data_path}/OhioT1DM-training/{patient_id}-ws-training.xml"
    test_file = f"{data_path}/OhioT1DM-testing/{patient_id}-ws-testing.xml"
    
    train_data = parse_ohio_xml(train_file)
    test_data = parse_ohio_xml(test_file)
    
    return train_data, test_data

def parse_ohio_xml(xml_file):
    """Parse OHIO XML file"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract glucose readings
    glucose_data = []
    for event in root.findall('.//glucose_level'):
        ts = event.get('ts')
        value = float(event.get('value'))
        glucose_data.append({'timestamp': pd.to_datetime(ts), 'glucose': value})
    
    df = pd.DataFrame(glucose_data)
    df = df.set_index('timestamp').sort_index()
    
    return df

def create_sequences(df, history_len=12, horizon=6):
    """Create sequences for forecasting
    history_len: number of past 5-min readings (12 = 1 hour)
    horizon: prediction horizon (6 = 30 minutes ahead)
    """
    X, y = [], []
    
    for i in range(history_len, len(df) - horizon):
        # Input: past glucose values
        X.append(df['glucose'].iloc[i-history_len:i].values)
        # Target: glucose value at horizon
        y.append(df['glucose'].iloc[i+horizon])
    
    return np.array(X), np.array(y)

def extract_additional_features(X):
    """Extract statistical features from glucose history"""
    features = []
    for seq in X:
        feat = [
            np.mean(seq),
            np.std(seq),
            np.min(seq),
            np.max(seq),
            seq[-1],  # Most recent value
            seq[-1] - seq[-2],  # Rate of change
            np.mean(np.diff(seq)),  # Average rate of change
            np.std(np.diff(seq))  # Variability in rate of change
        ]
        features.append(feat)
    return np.array(features)
