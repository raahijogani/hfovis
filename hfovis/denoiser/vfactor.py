import numpy as np


def compute_vfactor(data, start=0, end=None):
    """
    Compute V-Factor (range / std deviation) for each column in 2D data.

    Parameters:
    -----------
        data: shape (n_samples, n_events)
        start, end: optional slicing range
    Returns:
    --------
        v_factors: shape (n_events,)
    """
    if end is None:
        end = data.shape[0]
    sliced = data[start:end, :]

    std = np.std(sliced, axis=0, ddof=1)
    rng = np.ptp(sliced, axis=0)  # peak-to-peak (max - min)

    # Avoid division by zero
    v_factors = np.where(std > 0, rng / std, 0.0)
    return v_factors
