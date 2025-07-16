import os
import numpy as np
from scipy.signal import butter, filtfilt, detrend, resample
from scipy.signal.windows import tukey
from .normalize import data_norm
from .vfactor import compute_vfactor
from .omp import batch_omp
import joblib

bL, aL = butter(2, 1, fs=2048, btype="high")
bH, aH = butter(4, (80, 600), fs=2048, btype="bandpass")
data_dir = os.path.join(os.path.dirname(__file__), "data")
gabor_data = joblib.load(os.path.join(data_dir, "gabor_dictionary_full.pkl"))
dictionary = gabor_data["Dictionary"]
dh_freq = gabor_data["DHfrq"]


def extract_omp_features(
    signal: np.ndarray,
    n_iter: int = 50,
    bL: np.ndarray = bL,
    aL: np.ndarray = aL,
    bH: np.ndarray = bH,
    aH: np.ndarray = aH,
):
    """
    Extract features using OMP.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal to extract features from. Expected shape is (n_samples, n_events).
        This will be resampled to 512 samples.
    n_iter : int, optional
        Number of iterations for OMP. Default is 50.
    bL : np.ndarray, optional
        Coefficients for the high-pass filter. Default is a 1st order Butterworth filter
        at 1 Hz.
    aL : np.ndarray, optional
        Denominator coefficients for the high-pass filter. Default is a 1st order
        Butterworth filter at 1 Hz.
    bH : np.ndarray, optional
        Coefficients for the band-pass filter. Default is a 4th order Butterworth filter
        between 80 Hz and 600 Hz.
    aH : np.ndarray, optional
        Denominator coefficients for the band-pass filter. Default is a 4th order
        Butterworth filter between 80 Hz and 600 Hz.

    Returns:
    --------
    np.ndarray
        Extracted features of shape (n_events, 155).
    """

    # === Preprocessing ===
    data = resample(signal, 512)
    n_samples, n_events = data.shape

    data_unnormalized = detrend(data, type="constant", axis=0)
    data_detrended = detrend(data, type="linear", axis=0)
    data_normalized = data_norm(data_detrended, norm_type=2, axis=0)

    data_dc_filt = filtfilt(
        bL,
        aL,
        data_normalized,
        axis=0,
        method="pad",
        padlen=3 * (max(len(aL), len(bL)) - 1),
    )
    hfo_unormalized = filtfilt(
        bH,
        aH,
        data_unnormalized,
        axis=0,
        method="pad",
        padlen=3 * (max(len(aH), len(bH)) - 1),
    )

    # === OMP ===
    supports, residual_history, errors = batch_omp(dictionary, data_dc_filt, n_iter)

    # === Feature 1: Residual Errors (shape: n_iter × n_events)
    # Transpose to shape (n_events, n_iter)
    res_errors = errors[:n_iter, :].T

    # === Feature 2: V-Factors (shape: (n_events, n_iter+1))
    window = (tukey(n_samples, 0.25) + 0.5) / 1.5
    vfactors = np.zeros((n_events, n_iter + 1))
    for k in range(n_iter + 1):
        windowed_res = residual_history[k] * window[:, None]
        vfactors[:, k] = compute_vfactor(windowed_res)

    # === Feature 3: Line noise count up to each iteration (n_iter × n_events)
    # Result shape: (n_events, n_iter)
    line_noise_counts = np.zeros((n_events, n_iter), dtype=int)
    for k in range(1, n_iter + 1):
        freqs_selected = dh_freq[supports[:k, :]]
        # Check line noise in range 50–70 Hz
        is_line_noise = (freqs_selected > 50) & (freqs_selected < 70)
        line_noise_counts[:, k - 1] = np.sum(is_line_noise, axis=0)

    # === Feature 4: Signal stats (4 values per event)
    sd = np.std(data_unnormalized, axis=0, ddof=1)
    rng = np.ptp(data_unnormalized, axis=0)
    sd_h = np.std(hfo_unormalized, axis=0, ddof=1)
    rng_h = np.ptp(hfo_unormalized, axis=0)

    # === Concatenate all features ===
    # Each event: [residuals (n_iter), vfactors (n_iter+1), line_noise (n_iter), sd, rng, sd_h, rng_h]
    features = np.concatenate(
        [
            res_errors,
            vfactors,
            line_noise_counts,
            sd[None, :].T,
            rng[None, :].T,
            sd_h[None, :].T,
            rng_h[None, :].T,
        ],
        axis=1,
    )  # shape (n_events, 3*n_iter + 1 + 4)

    return features
