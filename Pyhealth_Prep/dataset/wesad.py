import os
import zipfile
import pandas as pd
import numpy as np
from pyhealth.datasets.kaggle_downloader import KaggleDownloader
from pyhealth.data import BaseDataset


class WESADDataset(BaseDataset):
    """
    WESAD (Wearable Stress and Affect Detection) Dataset Loader for PyHealth.
    Supports: automatic Kaggle download, CSV parsing, sliding windows.

    Kaggle dataset used:
        https://www.kaggle.com/datasets/qiriro/stress

    Expected Kaggle file structure:
        stress/
            s2.csv
            s3.csv
            ...

    Each CSV should have:
        timestamp, acc_x, acc_y, acc_z, eda, temp, label
    """

    kaggle_dataset = "qiriro/stress"

    def __init__(
        self,
        root: str,
        window_size: int = 256,
        stride: int = 128,
        **kwargs
    ):
        super().__init__(root=root, dataset_name="WESAD", **kwargs)
        self.window_size = window_size
        self.stride = stride
        self._ensure_data()
        self.load()

    # ----------------------------------------------------
    # Download dataset from Kaggle if missing
    # ----------------------------------------------------
    def _ensure_data(self):
        if len(os.listdir(self.root)) > 0:
            return

        print("Downloading WESAD from Kaggle…")

        downloader = KaggleDownloader(
            dataset_name=self.kaggle_dataset,
            download_dir=self.root,
        )
        archive_path = downloader.download()

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

    # ----------------------------------------------------
    # Load all CSV subjects
    # ----------------------------------------------------
    def load(self):
        files = [
            os.path.join(self.root, f)
            for f in os.listdir(self.root)
            if f.endswith(".csv")
        ]
        if not files:
            raise FileNotFoundError("No CSV files found in WESAD dataset directory.")

        self.raw_subjects = {}
        for f in files:
            sid = os.path.basename(f).replace(".csv", "")
            df = pd.read_csv(f)
            self.raw_subjects[sid] = df

    # ----------------------------------------------------
    # Parse → sliding windows
    # ----------------------------------------------------
    def parse(self):
        samples = []

        for sid, df in self.raw_subjects.items():
            X = df[["acc_x", "acc_y", "acc_z", "eda", "temp"]].values
            y = df["label"].values

            N = len(df)
            for start in range(0, N - self.window_size, self.stride):
                window = X[start:start + self.window_size]
                label_window = y[start:start + self.window_size]
                final_label = int(np.argmax(np.bincount(label_window)))

                samples.append({
                    "subject_id": sid,
                    "signal": window.astype(np.float32),
                    "label": final_label
                })

        self.samples = samples
        return samples

    def info(self):
        return {
            "dataset": "WESAD",
            "subjects": len(self.raw_subjects),
            "samples": len(self.samples),
            "window_size": self.window_size,
            "stride": self.stride
        }
