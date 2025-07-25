import numpy as np


def data_norm(data, norm_type=4, axis=0):
    data = np.asarray(data, dtype=float)
    eps = np.finfo(float).eps

    if axis != 0:
        raise NotImplementedError("Only axis=0 (column-wise) supported.")

    if norm_type == 0:
        counts = np.count_nonzero(data, axis=0)
        scale = np.where(counts > 0, counts, 1)
        data = data / scale
    elif norm_type == 1:
        sums = np.sum(np.abs(data), axis=0) + eps
        data = data / sums
    elif norm_type == 2:
        l2 = np.sqrt(np.sum(data**2, axis=0)) + eps
        data = data / l2
    elif norm_type == 3:
        max_val = np.max(data, axis=0)
        data = data / max_val
    elif norm_type == 4:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=0)
        std = np.where(std > 0, std, 1)
        data = (data - mean) / std
    elif norm_type == 5:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        rng = max_val - min_val + eps
        data = (data - min_val) / rng
    else:
        raise ValueError("Wrong configuration: norm_type must be 0–5")

    return data
