"""
Data loader for MTSC benchmark datasets
Supports both .npy and .ts file formats
"""

from pathlib import Path

import numpy as np
import torch


class MTSCDataLoader:
    """Multivariate Time Series Classification Data Loader"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.datasets = self._find_datasets()

    def _find_datasets(self):
        """Find available datasets"""
        datasets = []

        # Look for .npy files
        for file in self.data_dir.glob("*.npy"):
            datasets.append(file.stem)

        # Look for .ts files
        for folder in self.data_dir.iterdir():
            if folder.is_dir():
                train_file = folder / f"{folder.name}_TRAIN.ts"
                test_file = folder / f"{folder.name}_TEST.ts"
                if train_file.exists() and test_file.exists():
                    datasets.append(folder.name)

        return sorted(list(set(datasets)))

    def load_dataset(self, name):
        """Load a dataset by name"""
        # Try .npy first
        npy_file = self.data_dir / f"{name}.npy"
        if npy_file.exists():
            return self._load_npy(npy_file)

        # Try .ts files
        folder = self.data_dir / name
        train_file = folder / f"{name}_TRAIN.ts"
        test_file = folder / f"{name}_TEST.ts"

        if train_file.exists() and test_file.exists():
            return self._load_ts(train_file, test_file)

        raise FileNotFoundError(f"Dataset {name} not found")

    def _load_npy(self, filepath):
        """Load .npy file"""
        data = np.load(filepath, allow_pickle=True).item()

        X_train = np.array(data["train"]["X"], dtype=np.float32)
        y_train = np.array(data["train"]["y"], dtype=int)
        X_test = np.array(data["test"]["X"], dtype=np.float32)
        y_test = np.array(data["test"]["y"], dtype=int)

        return self._preprocess_data(X_train, y_train, X_test, y_test)

    def _load_ts(self, train_file, test_file):
        """Load .ts files"""
        try:
            # Try with aeon/sktime
            from aeon.datasets import load_from_tsfile

            X_train, y_train = load_from_tsfile(
                str(train_file), return_data_type="numpy3d"
            )
            X_test, y_test = load_from_tsfile(
                str(test_file), return_data_type="numpy3d"
            )
        except:
            try:
                from sktime.datasets import load_from_tsfile_to_dataframe

                X_train, y_train = load_from_tsfile_to_dataframe(str(train_file))
                X_test, y_test = load_from_tsfile_to_dataframe(str(test_file))
                X_train, X_test = self._convert_dataframe_to_array(X_train, X_test)
            except:
                raise ImportError("Please install: pip install aeon-toolkit or sktime")

        # Convert labels to integers
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_train = np.array([label_map[label] for label in y_train])
        y_test = np.array([label_map[label] for label in y_test])

        return self._preprocess_data(X_train, y_train, X_test, y_test)

    def _convert_dataframe_to_array(self, X_train, X_test):
        """Convert pandas dataframe to numpy array"""

        def df_to_array(df):
            n_instances = len(df)
            n_channels = len(df.columns)
            max_len = max(
                len(df.iloc[i][col]) for i in range(n_instances) for col in df.columns
            )

            X = np.zeros((n_instances, n_channels, max_len))
            for i in range(n_instances):
                for j, col in enumerate(df.columns):
                    series = df.iloc[i][col]
                    series_len = len(series)
                    X[i, j, :series_len] = series
                    if series_len < max_len:
                        X[i, j, series_len:] = series[-1]  # Pad with last value
            return X

        X_train_array = df_to_array(X_train)
        X_test_array = df_to_array(X_test)
        return X_train_array, X_test_array

    def _preprocess_data(self, X_train, y_train, X_test, y_test):
        """Preprocess the data"""
        # Ensure float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Normalize
        mean = np.mean(X_train, axis=(0, 2), keepdims=True)
        std = np.std(X_train, axis=(0, 2), keepdims=True)
        std = np.where(std == 0, 1.0, std)

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Ensure labels start from 0
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        if unique_labels.min() != 0:
            label_map = {old: new for new, old in enumerate(unique_labels)}
            y_train = np.array([label_map[label] for label in y_train])
            y_test = np.array([label_map[label] for label in y_test])

        return X_train, y_train, X_test, y_test

    def get_dataset_info(self, name):
        """Get dataset information"""
        X_train, y_train, X_test, y_test = self.load_dataset(name)

        return {
            "name": name,
            "n_channels": X_train.shape[1],
            "seq_length": X_train.shape[2],
            "n_classes": len(np.unique(y_train)),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
