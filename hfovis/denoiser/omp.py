import numpy as np
from numba import njit, prange
from scipy.linalg import cho_factor, cho_solve


@njit(
    "void(float64[:, :], int64[:], int64[:, :], int64, int64)",
    parallel=True,
    fastmath=True,
    cache=True,
)
def project_and_select(
    DtR: np.ndarray,
    support_lens: np.ndarray,
    supports: np.ndarray,
    E: int,
    K: int,
):
    """
    Selects the best atom for each event based on the projections.

    Parameters:
    ----------
    DtR : np.ndarray
        The projections of the residuals onto the dictionary atoms, shape (K, E).
    support_lens : np.ndarray
        Current lengths of the supports for each event, shape (E,).
    supports : np.ndarray
        Indices of selected atoms for each event, shape (max_atoms, E).
    E : int
        Number of events.
    K : int
        Number of atoms in the dictionary.
    """
    # DtR = D.T @ residuals (passed in)
    # Updates supports, computes new coefficients via small-matrix solves (optional here)
    # Returns updated support_lens and supports and flags to stop
    for e in prange(E):
        # select largest projection
        best = 0.0
        argmax = 0
        for k in range(K):
            val = abs(DtR[k, e])
            if val > best:
                best = val
                argmax = k
        idx = argmax
        supports[support_lens[e], e] = idx
        support_lens[e] += 1
    return


def batch_omp(D, X, max_atoms, err_thresh=-1, err_diff_thresh=-1, min_iters=5):
    """
    Batch Orthogonal Matching Pursuit (OMP) algorithm.

    Parameters:
    -----------
    D : np.ndarray
        Dictionary matrix of shape (n_samples, K), where K is the number of atoms.
    X : np.ndarray
        Input data matrix of shape (n_samples, E), where E is the number of events.
    max_atoms : int
        Maximum number of atoms to select.
    err_thresh : float, optional
        Error threshold for stopping criteria. Default is -1 (no threshold).
    err_diff_thresh : float, optional
        Error difference threshold for stopping criteria. Default is -1 (no threshold).
    min_iters : int, optional
        Minimum number of iterations to run. Default is 5.

    Returns:
    --------
    supports : np.ndarray
        Indices of selected atoms for each event, shape (max_atoms, E).
    residual_history : np.ndarray
        History of residuals at each iteration, shape (max_atoms + 1, n_samples, E).
    errors : np.ndarray
        Relative errors at each iteration, shape (max_atoms, E).
    """
    n, K = D.shape
    _, E = X.shape
    DtX = D.T @ X
    residuals = X.copy()
    coeffs = np.zeros((K, E))
    supports = np.zeros((max_atoms, E), dtype=int)
    support_lens = np.zeros(E, dtype=int)
    errors = np.zeros((max_atoms, E))

    # Store residuals per iteration: shape (max_atoms + 1, n, E)
    residual_history = np.zeros((max_atoms + 1, n, E))
    residual_history[0] = residuals

    for it in range(max_atoms):
        DtR = D.T @ residuals
        project_and_select(DtR, support_lens, supports, E, K)

        # Solve least squares for each event
        for e in range(E):
            sel = supports[: support_lens[e], e]
            if len(sel) == 0:
                continue
            Gs = D[:, sel].T @ D[:, sel]
            rhs = DtX[sel, e]
            cfac, low = cho_factor(Gs, lower=True, overwrite_a=True, check_finite=False)
            sol = cho_solve((cfac, low), rhs, overwrite_b=True)
            coeffs[sel, e] = sol

        Y = D @ coeffs
        residuals = X - Y
        residual_history[it + 1] = residuals  # save this step's residual

        curr_err = np.linalg.norm(residuals, axis=0) / np.linalg.norm(X, axis=0)
        errors[it, :] = curr_err

        if it + 1 >= min_iters:
            prev_err = errors[it - 1, :]
            err_diff = prev_err - curr_err
            stop = np.all(
                (err_thresh > 0) & (curr_err < err_thresh)
                | (err_diff_thresh > 0) & (err_diff < err_diff_thresh)
            )
            if stop:
                break

    return supports, residual_history[: it + 2], errors[: it + 1]
