import numpy as np
from scipy.linalg import cho_factor, cho_solve
from numba import njit, prange
from numpy.linalg import norm, pinv


@njit(
    "void(float64[:, :], int64[:], int64[:, :], int64, int64)",
    parallel=True,
    fastmath=True,
    cache=True,
)
def project_and_select(
    DtR,
    support_lens,
    supports,
    E,
    K,
):
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


def omp_visualize(D, X, L, S1=-1, S2=-1, S3=5):
    """
    Exact implementation of MATLAB's OMP_Visualize function.

    Parameters:
        D : np.ndarray
            The dictionary (shape: [n_features, n_atoms]), columns must be L2-normalized.
        X : np.ndarray
            The signal (shape: [n_features,]), a single column.
        L : int
            Maximum number of atoms to use.
        S1 : float
            Minimum error threshold to stop.
        S2 : float
            Minimum error difference to stop.
        S3 : int
            Minimum number of iterations.

    Returns:
        y : np.ndarray
            Reconstructed signals at each iteration (shape: [n_features, j]).
        coeff : np.ndarray
            Full sparse coefficient vector (shape: [n_atoms,]).
        loc : list
            Indices of selected atoms.
        residual : np.ndarray
            Residuals at each iteration (shape: [n_features, j+1]).
        Error : list
            Normalized residual errors at each iteration.
    """
    n, K = D.shape
    assert X.ndim == 1 and X.shape[0] == n, (
        "X must be a 1D vector of same row size as D"
    )

    residual = np.zeros((n, L + 1))
    y = np.zeros((n, L))
    Error = []
    Error_diff = []
    loc = []

    residual[:, 0] = X
    x_norm = norm(X)

    for j in range(L):
        proj = D.T @ residual[:, j]
        pos = np.argmax(np.abs(proj))
        loc.append(pos)

        D_selected = D[:, loc]
        a = pinv(D_selected) @ X
        y[:, j] = D_selected @ a
        residual[:, j + 1] = X - y[:, j]

        err = norm(residual[:, j + 1]) / x_norm
        Error.append(err)

        temp = np.zeros(K)
        temp[loc] = a

        if j > 0:
            diff = Error[j - 1] - Error[j]
            Error_diff.append(diff)

            if j + 1 > S3:
                if (S1 > 0 and Error[j] < S1) or (S2 > 0 and diff < S2):
                    break

    return loc, residual[:, : j + 2], Error[: j + 1]
