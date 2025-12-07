# WESAD Dataset (Wearable Stress and Affect Detection)

PyHealth supports the WESAD dataset using a Kaggle-backed loader.

## Kaggle Source
Dataset URL:  
https://www.kaggle.com/datasets/qiriro/stress

## Usage Example

```python
from pyhealth.datasets import WESADDataset

ds = WESADDataset(
    root="path/to/cache", 
    window_size=256,
    stride=128
)
samples = ds.parse()

print(ds.info())
