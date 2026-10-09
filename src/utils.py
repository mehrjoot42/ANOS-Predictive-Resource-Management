"""
Utility functions for ANOS.

Provides:
    - set_seed         : Fix random seeds for reproducibility.
    - compute_metrics  : Compute MAE and RMSE between predictions and truth.
    - baseline_ma      : Moving-average baseline for comparison.
    - create_sequences : Convert a time series into (X, Y) sequence pairs.
"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set all random seeds to guarantee full reproducibility.

    This covers Python's built-in random, NumPy, and PyTorch (CPU + GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behavior on cuDNN-backed operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(pred, truth):
    """
    Compute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

    Args:
        pred  : array-like of predictions
        truth : array-like of ground-truth values

    Returns:
        (mae, rmse) as floats
    """
    pred = np.asarray(pred).flatten()
    truth = np.asarray(truth).flatten()
    mae = float(np.mean(np.abs(pred - truth)))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    return mae, rmse


def baseline_ma(series, w=15):
    """
    Moving-average baseline.

    Predicts the next value as the mean of the last `w` values.

    Args:
        series : 1-D array-like time series
        w      : window size

    Returns:
        Array of predictions (length = len(series) - 1)
    """
    series = np.asarray(series).flatten()
    return np.array([
        np.mean(series[max(0, i - w + 1): i + 1])
        for i in range(len(series) - 1)
    ])


def create_sequences(data, seq_len):
    """
    Convert a multivariate time series into (X, Y) pairs.

    Args:
        data    : 2-D array of shape (N, features)
        seq_len : number of past steps used to predict the next step

    Returns:
        X : array of shape (N - seq_len, seq_len, features)
        Y : array of shape (N - seq_len,) — target is feature 0 (download speed)
    """
    data = np.asarray(data, dtype=np.float32)
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        Y.append(data[i + seq_len, 0])  # predict feature 0 (dl_mbps)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
