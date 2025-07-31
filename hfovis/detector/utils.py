import numpy as np
import pandas as pd
from scipy.signal import convolve
from numpy.lib.stride_tricks import as_strided


def get_adaptive_threshold(
    data, frame, overlap, h_frame, h_overlap, threshold_type, param, user_input=None
):
    """
    Computes an adaptive threshold based on signal statistics.

    Parameters:
        data (ndarray): 1D array of single-channel raw data.
        frame (int): Window length for rectangular window, or pre-defined window.
        overlap (int): Overlapping samples when calculating the variance.
        h_frame (int): Number of adaptive frames when recalculating threshold.
        h_overlap (int): Number of adaptive overlap frames when recalculating threshold.
        threshold_type (str): Type of operator to use ('Std', 'Var', 'Manual').
        param (float): Scaling parameter for the threshold.
        user_input (any, optional): Used only if threshold_type is 'Manual'.

    Returns:
        v (ndarray): Computed variance or standard deviation.
        th (ndarray): Adaptive threshold for the signal.
    """

    if threshold_type != "Manual" or user_input == 0:
        user_input = None

    # Compute variance or standard deviation over buffered windows
    if threshold_type == "Std":
        v, _ = buffered_stats(data, frame, overlap, "std")

    else:
        v, _ = buffered_stats(data, frame, overlap, "var")

    # Compute adaptive threshold over buffer frames
    h_th, bs = buffered_stats(v, h_frame, h_overlap, "median")
    N = len(data)
    Nv = len(v)

    # Adjust the last buffer (handle boundary case)
    dt = Nv - (bs - 1) * (h_frame - h_overlap)
    if dt < 0.75 * h_frame:
        h_th[-1] = h_th[-2]  # Extend last threshold value

    # Interpolate threshold across the full signal length
    x_samples = np.linspace(0, len(h_th) - 1, N)  # Adjusted for Python indexing
    th = np.interp(x_samples, np.arange(len(h_th)), h_th) * param

    return v, th


def buffer_signal(data, frame_size, overlap_size):
    """Creates overlapping frames from the input data, handling edge cases properly."""
    data = np.asarray(data)
    step = frame_size - overlap_size
    L = data.shape[0]

    # Calculate number of frames. Ensure at least one frame.
    num_frames = int(np.ceil(max(L - frame_size, 0) / step)) + 1

    # Compute total length needed, and pad with zeros at the end if necessary.
    required_len = step * (num_frames - 1) + frame_size
    padded_data = np.pad(data, (0, max(0, required_len - L)), mode="constant")

    # Use stride_tricks to extract overlapping frames.
    s = padded_data.strides[0]
    shape = (num_frames, frame_size)
    strides = (step * s, s)
    result = as_strided(padded_data, shape=shape, strides=strides).copy()

    return result


def buffered_stats(data, frame, overlap, stat_type="std"):
    """
    Computes statistical measures on buffered data.

    Parameters:
        data (ndarray): Input 1D array.
        frame (int): Frame size.
        overlap (int): Overlap size between frames.
        stat_type (str, optional): Type of statistic to compute ('std', 'mean', 'median', 'var', 'max'). Default is 'std'.

    Returns:
        y (ndarray): Computed statistical measure for each frame.
        bs (int): Number of buffered frames.
    """

    # Buffering the signal like MATLAB
    buffered_segments = buffer_signal(data, frame, overlap)
    bs = buffered_segments.shape[0]  # Number of buffered windows

    # Apply statistical function
    if stat_type == "std":
        y = np.std(buffered_segments, axis=1, ddof=1)  # ddof=1 for unbiased std
    elif stat_type == "mean":
        y = np.mean(buffered_segments, axis=1)
    elif stat_type == "median":
        y = np.median(buffered_segments, axis=1)
    elif stat_type == "var":
        y = np.var(buffered_segments, axis=1, ddof=1)
    elif stat_type == "max":
        y = np.max(np.abs(buffered_segments), axis=1)
    else:
        raise ValueError(
            "Invalid stat_type. Choose from 'std', 'mean', 'median', 'var', 'max'."
        )

    return y, bs


def find_burst_events(data_raw, data_filtered, thresholds, config):
    """
    Detects burst events in a signal.

    Parameters:
        data_raw (ndarray): Raw input signal (1D array).
        data_filtered (ndarray): Filtered input signal (1D array).
        thresholds (dict): Contains threshold values.
        config (dict): Configuration parameters.

    Returns:
        pd.DataFrame: DataFrame containing detected burst events with start, end, center, threshold, and duration.
    """
    # -- 1. Extract and validate threshold parameters  --

    # Ensure the detection band is set correctly
    detection_band = config.get("detectionBand")
    if detection_band is None:
        raise KeyError("Missing 'detectionBand' in config.")

    # Decode the thresholds safely
    th1 = thresholds.get("th1")
    th2 = thresholds.get("th2", None)  # Used only in "Dual" mode
    th_rej = thresholds.get("th_rej")

    if th1 is None or th_rej is None:
        raise KeyError(
            "Thresholds dictionary is missing required keys: 'th1' or 'th_rej'."
        )
    if config["threshold_type"] == "Dual" and th2 is None:
        raise KeyError(
            "Thresholds dictionary is missing 'th2' for Dual threshold mode."
        )

    # -- 2. Choose main threshold (Single vs. Dual) and find candidate indices --
    if config["threshold_type"] == "Single":
        main_threshold = th1
    elif config["threshold_type"] == "Dual":
        main_threshold = th2
    else:
        raise ValueError("Invalid threshold type! Use 'Single' or 'Dual'.")

    # Indices where data_filtered exceeds threshold (in absolute value)
    index = np.where(np.abs(data_filtered) > main_threshold)[0]
    # print(f"Above-threshold samples: {len(index)}")
    if len(index) <= 1:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "center",
                "threshold",
                "duration",
                "event_raw",
                "event_filtered",
            ]
        )

    # MATLAB-equivalent grouping code
    li = len(index)
    # Exit early if index has only one element
    if li <= 1:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "center",
                "threshold",
                "duration",
                "event_raw",
                "event_filtered",
            ]
        )

    # Find indices where the difference between consecutive indices is within stitchTime
    F = (
        np.where(np.diff(index) <= config["stitchTime"])[0] + 1
    )  # adjust to mimic MATLAB 1-indexing
    # Exit early if no valid groups exist
    if F.size == 0:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "center",
                "threshold",
                "duration",
                "event_raw",
                "event_filtered",
            ]
        )

    # Identify breaks in consecutive sequences by splitting F where the gap is >1
    splits = np.where(np.diff(F) > 1)[0]
    groups_F = np.split(F, splits + 1)

    groups = []
    for grp in groups_F:
        # For each group, mimic MATLAB: group = index( F(i_start) - 1 : F(i_end) + 1 )
        start_idx = grp[0] - 1 if (grp[0] - 1) >= 0 else grp[0]
        end_idx = (
            grp[-1] + 1
        )  # slice end is non-inclusive, so this works as MATLAB's end index
        burst_group = index[start_idx:end_idx]
        # Remove groups with only one element
        if burst_group.size > 1:
            groups.append(burst_group)

    # print(f"Groups: {len(groups)}")

    # Compute duration of each group (assuming index represents time steps)
    group_durations = [grp[-1] - grp[0] for grp in groups]

    # Create logical mask for rejection conditions
    atf = [
        (
            duration < config["minBurst_duration"]
            or duration > config["maxBurst_duration"]
        )
        for duration in group_durations
    ]

    # Remove rejected groups (mimicking MATLAB's: group(atf) = [])
    valid_groups = [grp for grp, reject in zip(groups, atf) if not reject]

    if not valid_groups:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "center",
                "threshold",
                "duration",
                "event_raw",
                "event_filtered",
            ]
        )

    # -- 5. Iterate over valid groups, apply artifact/rejection checks, pick center --
    accepted_events = []
    rejected_groups = []

    halfBurst = config["burstWindow"] // 2
    halfMaxDur = config["maxBurst_duration"] // 2

    # for group in [valid_groups[15]]:
    # for group in [valid_groups[490]]:

    for group in valid_groups:
        start, end = group[0], group[-1]
        x = data_filtered[start : end + 1]

        # Select burst center based on criteria
        if config["burstWindow_center"] == "max":
            center = group[np.argmax(np.abs(data_filtered[group]))]
        elif config["burstWindow_center"] == "best" and len(group) >= 8:
            diff_group = np.gradient(np.diff(group))
            smooth_indices = np.where(np.abs(diff_group) < 2)[0]
        if config["burstWindow_center"] == "best" and len(group) >= 8:
            diff_group = np.gradient(np.diff(group))
            smooth_indices = np.where(np.abs(diff_group) < 2)[0]
            center = (
                group[int(np.median(smooth_indices))]
                if smooth_indices.size > 0
                else group[np.argmax(np.abs(data_filtered[group]))]
            )
        else:
            center = group[np.argmax(np.abs(data_filtered[group]))]

        # Extract the burst window

        # Calculate beginning and end indices of the event as in MATLAB
        start_window = int(center - halfBurst + 1)
        end_window = int(center + halfBurst + 1)

        # Ensure the indices are within valid bounds
        if start_window >= 0 and end_window <= len(data_filtered):
            y = data_filtered[start_window + halfMaxDur : end_window - halfMaxDur]
        else:
            rejected_groups.append(
                {
                    "group": group,
                    "reason": "window_out_of_bounds",
                }
            )
            continue

        # Rejection threshold check
        if np.max(np.abs(x)) > np.mean(th_rej[start : end + 1]):
            rejected_groups.append(
                {
                    "group": group,
                    "reason": "rejection_threshold",
                }
            )
            continue

        # Assign threshold values safely
        threshold = th1[center]

        # Remove large artifacts in FR component
        if config["Band_of_Interest"] == "FR":
            if threshold > config["max_threshold_FR"]:
                rejected_groups.append(
                    {
                        "group": group,
                        "reason": "max_threshold_FR",
                    }
                )
                continue

        # Check global and local swing
        try:
            p, info = amp_crossing(y, threshold, config)
            if p == 0:
                rejected_groups.append({"group": group, "reason": "amp_crossing"})
                continue
        except Exception as e:
            rejected_groups.append(
                {
                    "group": group,
                    "reason": "amp_crossing_exception",
                    "error": str(e),
                }
            )
            continue

        accepted_events.append(
            [
                int(start),
                int(end),
                int(center),
                threshold,
                int(end) - int(start),
            ]
        )

    # Convert to DataFrame
    df_accepted_events = pd.DataFrame(
        accepted_events, columns=["start", "end", "center", "threshold", "duration"]
    )
    # print(f"Accepted events 1 : {df_accepted_events.shape[0]}")

    # If no events detected, return an empty DataFrame
    if df_accepted_events.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "center",
                "threshold",
                "duration",
                "event_raw",
                "event_filtered",
            ]
        )

    # -- 6. Side oscillation and centralization checks --
    rej_idx = np.zeros(len(df_accepted_events), dtype=bool)

    for k in range(len(df_accepted_events)):
        c = df_accepted_events.iloc[k]["center"]
        start_window = int(c - halfBurst + 1)
        end_window = int(c + halfBurst + 1)

        if start_window <= 1 or end_window >= len(data_filtered):
            rej_idx[k] = True
            rejected_groups.append({"group": c, "reason": "window_out_of_bounds_2"})
            continue

        tmp = data_filtered[start_window:end_window]

        # Side oscillation removal
        if config["removeSide"]:
            left_limit = halfBurst - halfMaxDur - 1
            right_limit = halfBurst + halfMaxDur - 1

            p_side_left = hfo_amp_detector(
                tmp[: left_limit + 1],
                None,
                df_accepted_events.iloc[k]["threshold"],
                config["fs"],
                detection_band[0],
                config["removeSide"],
            )
            p_side_right = hfo_amp_detector(
                tmp[right_limit:],
                None,
                df_accepted_events.iloc[k]["threshold"],
                config["fs"],
                detection_band[0],
                config["removeSide"],
            )
            if p_side_right == 1 or p_side_left == 1:
                rej_idx[k] = True
                rejected_groups.append({"group": c, "reason": "side_oscillation"})
                continue

        # Centralization check
        if config["checkCentralized"]:
            centralized, _ = check_centralized_component(tmp)
            rej_idx[k] = not centralized
            if centralized == False:
                rejected_groups.append({"group": c, "reason": "not_centralized"})

    df_accepted_events = df_accepted_events[~rej_idx]
    # print(f"Accepted events 2: {df_accepted_events.shape[0]}")

    if df_accepted_events.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "center",
                "threshold",
                "duration",
                "event_raw",
                "event_filtered",
            ]
        )

    # -- 7. Attach the raw & filtered event waveforms for each accepted event --
    centers = df_accepted_events["center"].astype(int)
    start_indices = np.maximum(centers - halfBurst + 1, 0)
    end_indices = np.minimum(centers + halfBurst + 1, len(data_raw))

    df_accepted_events["event_raw"] = [
        data_raw[start:end] for start, end in zip(start_indices, end_indices)
    ]
    df_accepted_events["event_filtered"] = [
        data_filtered[start:end] for start, end in zip(start_indices, end_indices)
    ]

    return df_accepted_events


def amp_crossing(x, th, config, plot=False):
    """
    Detects amplitude crossings in a signal.

    Parameters:
        x (ndarray): Input signal.
        th (float): Amplitude threshold for crossings.
        config (dict): Configuration parameters with the following keys:
            - 'detectionBand': List with detection band frequency [Hz].
            - 'fs': Sampling frequency [Hz].
            - 'min_symmetricGlobalSwing': Minimum global swings required.
            - 'symmetricGlobalSwing_type': 'Full' or other.
            - 'symmetricLocalSwing_type': 'Full-Symmetric', 'Half-Symmetric', or 'No-Symmetric'.
            - 'min_symmetricLocalSwing': Minimum local swings required.

    Returns:
        p (int): Output decision based on crossing events.
        info (str): Detailed information about detected crossings.
    """

    if not all(
        k in config
        for k in [
            "detectionBand",
            "fs",
            "min_symmetricGlobalSwing",
            "symmetricGlobalSwing_type",
            "symmetricLocalSwing_type",
            "min_symmetricLocalSwing",
        ]
    ):
        raise ValueError("Check the input parameters in config.")

    N = round(
        1 / config["detectionBand"][0] * config["fs"]
    )  # Minimum distance between crossings

    p = 0
    info = ""

    pp = pn = pp_nonSym = pn_nonSym = 0  # Counters for crossings

    # Detect crossings
    _, ixp = zerocross_count((x - th))
    _, ixn = zerocross_count((x + th))
    _, ixo = zerocross_count(x)

    # # If any of the crossings are empty, return p = 0 and info = {}
    # if len(ixp) == 0 or len(ixn) == 0 or len(ixo) == 0:
    #     return 0, {}

    loc = np.sort(np.concatenate([ixp, ixn]))
    loc_full = np.sort(np.concatenate([ixp, ixn, ixo]))
    # Exact MATLAB logic: loc_full(loc_full < min([ixp,ixn])) = []; loc_full(loc_full > max([ixp,ixn])) = [];
    min_val = min(
        np.min(ixp) if len(ixp) > 0 else float("inf"),
        np.min(ixn) if len(ixn) > 0 else float("inf"),
    )
    max_val = max(
        np.max(ixp) if len(ixp) > 0 else float("-inf"),
        np.max(ixn) if len(ixn) > 0 else float("-inf"),
    )
    loc_full = loc_full[(loc_full >= min_val) & (loc_full <= max_val)]

    logic_loc = np.zeros(len(loc), dtype=int)
    logic_loc[np.isin(loc, ixp)] = +1
    logic_loc[np.isin(loc, ixn)] = -1

    diff_logic_loc = np.diff(logic_loc)
    GSw_np = np.sum(diff_logic_loc == 2)  # number of oscillations negative to positive
    GSw_pn = np.sum(diff_logic_loc == -2)  # number of oscillations positive to negative

    logic_loc_full = np.zeros(len(loc_full), dtype=int)
    logic_loc_full[np.isin(loc_full, ixp)] = +1
    logic_loc_full[np.isin(loc_full, ixn)] = -1
    change_indices = np.where(
        np.diff(np.insert(logic_loc_full, [0, len(logic_loc_full)], 0)) != 0
    )[0]

    group_ind_st, group_ind_et, group_logic = [], [], []
    for i in range(len(change_indices) - 1):
        start_index, end_index = change_indices[i], change_indices[i + 1] - 1
        if logic_loc_full[start_index] != 0:
            group_ind_st.append(loc_full[start_index])
            group_ind_et.append(loc_full[end_index])
            group_logic.append(logic_loc_full[start_index])

    group_ind_st, group_ind_et, group_logic = (
        np.array(group_ind_st),
        np.array(group_ind_et),
        np.array(group_logic),
    )

    m = np.array(
        [
            np.median(range(s, e + 1))
            if len(range(s, e + 1)) % 2 != 0
            else (
                range(s, e + 1)[len(range(s, e + 1)) // 2 - 1]
                + range(s, e + 1)[len(range(s, e + 1)) // 2]
            )
            / 2
            for s, e in zip(group_ind_st, group_ind_et)
        ]
    )

    ixp_Corrected = m[group_logic == 1]
    ixn_Corrected = m[group_logic == -1]

    h = np.ones(config["min_symmetricLocalSwing"] - 1)

    if config["symmetricGlobalSwing_type"] == "Full":
        if min(GSw_pn, GSw_np) >= config["min_symmetricGlobalSwing"]:
            if config["symmetricLocalSwing_type"] == "Full-Symmetric":
                if len(ixp_Corrected) >= config["min_symmetricLocalSwing"]:
                    ix_diff = np.diff(ixp_Corrected)
                    if ix_diff.size > 0 and np.sum(
                        convolve(ix_diff < N, h, mode="full")
                        >= config["min_symmetricLocalSwing"] - 1
                    ):
                        pp = 1
                if len(ixn_Corrected) >= config["min_symmetricLocalSwing"]:
                    ix_diff = np.diff(ixn_Corrected)
                    if ix_diff.size > 0 and np.sum(
                        convolve(ix_diff < N, h, mode="full")
                        >= config["min_symmetricLocalSwing"] - 1
                    ):
                        pn = 1
                p = pn * pp
                info = f"P:{p} GSw:{GSw_pn}-{GSw_np} LSw:{pp}-{pn} - Full Local Symmetricity with Global Swing"
                if plot == True:
                    plot_swing(x, ixp, ixn, ixo, th, info)

            elif config["symmetricLocalSwing_type"] == "Half-Symmetric":
                if len(ixp_Corrected) >= config["min_symmetricLocalSwing"]:
                    ix_diff = np.diff(ixp_Corrected)
                    if ix_diff.size > 0 and np.sum(
                        convolve(ix_diff < N, h, mode="full")
                        >= config["min_symmetricLocalSwing"] - 1
                    ):
                        pp = 1
                elif len(ixp_Corrected) == config["min_symmetricLocalSwing"] - 1:
                    h2 = np.ones(config["min_symmetricLocalSwing"] - 2)
                    ix_diff = np.diff(ixp_Corrected)
                    if ix_diff.size > 0 and np.sum(
                        convolve(ix_diff < N, h2, mode="full")
                        >= config["min_symmetricLocalSwing"] - 2
                    ):
                        pp_nonSym = 1

                if len(ixn_Corrected) >= config["min_symmetricLocalSwing"]:
                    ix_diff = np.diff(ixn_Corrected)
                    if ix_diff.size > 0 and np.sum(
                        convolve(ix_diff < N, h, mode="full")
                        >= config["min_symmetricLocalSwing"] - 1
                    ):
                        pn = 1
                elif len(ixn_Corrected) == config["min_symmetricLocalSwing"] - 1:
                    h2 = np.ones(config["min_symmetricLocalSwing"] - 2)
                    ix_diff = np.diff(ixn_Corrected)
                    if ix_diff.size > 0 and np.sum(
                        convolve(ix_diff < N, h2, mode="full")
                        >= config["min_symmetricLocalSwing"] - 2
                    ):
                        pn_nonSym = 1

                p = max([pn * pp, pn * pp_nonSym, pp * pn_nonSym])
                info = f"P:{p} GSw:{GSw_pn}-{GSw_np} LSw:{pp}({pp_nonSym})-{pn}({pn_nonSym}) - Half Local Symmetricity with Global Swing"
                if plot == True:
                    plot_swing(x, ixp, ixn, ixo, th, info)

            elif config["symmetricLocalSwing_type"] == "No-Symmetric":
                if len(ixp_Corrected) >= config["min_symmetricLocalSwing"]:
                    ix_diff = np.diff(ixp_Corrected)
                    if ix_diff.size > 0 and np.sum(
                        convolve(ix_diff < N, h, mode="full")
                        >= config["min_symmetricLocalSwing"] - 1
                    ):
                        pp = 1
                        p = 1
                if len(ixn_Corrected) >= config["min_symmetricLocalSwing"]:
                    ix_diff = np.diff(ixn_Corrected)
                    if ix_diff.size > 0 and np.sum(
                        convolve(ix_diff < N, h, mode="full")
                        >= config["min_symmetricLocalSwing"] - 1
                    ):
                        pn = 1
                        p = 1
                p = max(pp, pn)
                info = f"P:{p} GSw:{GSw_pn}-{GSw_np} LSw:{pp}-{pn} - No Local Symmetricity with Global Swing"
                if plot == True:
                    plot_swing(x, ixp, ixn, ixo, th, info)

    elif config["symmetricGlobalSwing_type"] == "No":
        # Global swing is disabled, check local symmetry only
        if config["symmetricLocalSwing_type"] == "Full-Symmetric":
            if len(ixp_Corrected) >= config["min_symmetricLocalSwing"]:
                ix_diff = np.diff(ixp_Corrected)
                if ix_diff.size > 0 and np.sum(
                    convolve(ix_diff < N, h, mode="full")
                    >= config["min_symmetricLocalSwing"] - 1
                ):
                    pp = 1
            if len(ixn_Corrected) >= config["min_symmetricLocalSwing"]:
                ix_diff = np.diff(ixn_Corrected)
                if ix_diff.size > 0 and np.sum(
                    convolve(ix_diff < N, h, mode="full")
                    >= config["min_symmetricLocalSwing"] - 1
                ):
                    pn = 1
            p = pp * pn
            info = f"P:{p} Local Swing Only (Full): LSw:{pp}-{pn}"

        elif config["symmetricLocalSwing_type"] == "Half-Symmetric":
            if len(ixp_Corrected) >= config["min_symmetricLocalSwing"]:
                ix_diff = np.diff(ixp_Corrected)
                if ix_diff.size > 0 and np.sum(
                    convolve(ix_diff < N, h, mode="full")
                    >= config["min_symmetricLocalSwing"] - 1
                ):
                    pp = 1
            elif len(ixp_Corrected) == config["min_symmetricLocalSwing"] - 1:
                h2 = np.ones(config["min_symmetricLocalSwing"] - 2)
                ix_diff = np.diff(ixp_Corrected)
                if ix_diff.size > 0 and np.sum(
                    convolve(ix_diff < N, h2, mode="full")
                    >= config["min_symmetricLocalSwing"] - 2
                ):
                    pp_nonSym = 1

            if len(ixn_Corrected) >= config["min_symmetricLocalSwing"]:
                ix_diff = np.diff(ixn_Corrected)
                if ix_diff.size > 0 and np.sum(
                    convolve(ix_diff < N, h, mode="full")
                    >= config["min_symmetricLocalSwing"] - 1
                ):
                    pn = 1
            elif len(ixn_Corrected) == config["min_symmetricLocalSwing"] - 1:
                h2 = np.ones(config["min_symmetricLocalSwing"] - 2)
                ix_diff = np.diff(ixn_Corrected)
                if ix_diff.size > 0 and np.sum(
                    convolve(ix_diff < N, h2, mode="full")
                    >= config["min_symmetricLocalSwing"] - 2
                ):
                    pn_nonSym = 1

            p = max([pn * pp, pn * pp_nonSym, pp * pn_nonSym])
            info = f"P:{p} GSw:{GSw_pn}-{GSw_np} LSw:{pp}({pp_nonSym})-{pn}({pn_nonSym}) - Half Local Symmetricity with No Global Swing"

        elif config["symmetricLocalSwing_type"] == "No-Symmetric":
            if len(ixp_Corrected) >= config["min_symmetricLocalSwing"]:
                ix_diff = np.diff(ixp_Corrected)
                if ix_diff.size > 0 and np.sum(
                    convolve((ix_diff < N).astype(int), h, mode="full")
                    >= config["min_symmetricLocalSwing"] - 1
                ):
                    pp = 1
                    p = 1
            if len(ixn_Corrected) >= config["min_symmetricLocalSwing"]:
                ix_diff = np.diff(ixn_Corrected)
                if ix_diff.size > 0 and np.sum(
                    convolve((ix_diff < N).astype(int), h, mode="full")
                    >= config["min_symmetricLocalSwing"] - 1
                ):
                    pn = 1
                    p = 1
            p = max(pp, pn)
            info = f"P:{p} GSw:{GSw_pn}-{GSw_np} LSw:{pp}-{pn} - No Local Symmetricity with No Global Swing"
    return p, info


def zerocross_count(data):
    """
    Computes the zero-crossing number for a 1D array.

    Parameters:
        data (ndarray): 1D array where zero crossings are counted.

    Returns:
        count (int): Number of zero-crossings.
        idx (ndarray): Indices where zero crossings occur.
    """
    x_sign = np.sign(data)
    x_sign = np.where(x_sign == 0, 1, x_sign)  # Set zero elements to 1
    x_sign_diff = np.diff(x_sign)  # Detect change in sign
    idx = np.where(x_sign_diff != 0)[0]
    count = len(idx)

    return count, idx


def temp_variance(data, frame, overlap, var_type=1):
    """
    Computes the temporal variance of a signal using a moving window.

    Parameters:
        data (numpy array): Input signal
        frame (int or numpy array): Window length (if scalar) or window array
        overlap (int): Overlapping samples
        var_type (int): Type of computation (1 = variance, 2 = standard deviation, 3 = mean)

    Returns:
        numpy array: Computed variance, standard deviation, or mean
    """
    if np.isscalar(frame):
        winlength = frame
        window = np.ones(winlength)
    elif np.ndim(frame) == 1:
        window = np.array(frame)
        winlength = len(window)
    else:
        raise ValueError("Frame must be either a scalar or a 1D array.")

    lx = len(data)

    # Compute valid start indices for windows
    start_indices = np.arange(0, lx - winlength + 1, winlength - overlap)

    # Create a list of index slices
    indices = [
        np.arange(start, start + winlength)
        for start in start_indices
        if start + winlength <= lx
    ]

    # If no valid indices are found, return an empty array
    if not indices:
        return np.array([])

    # Convert list of indices to a 2D numpy array
    indices = np.array(indices)

    # Replicate the window to match the number of windows
    window = np.tile(window, (indices.shape[0], 1))

    # Compute the required measure based on var_type
    if var_type == 1:
        y = np.var(window * data[indices], axis=1, ddof=1)
    elif var_type == 2:
        y = np.std(window * data[indices], axis=1, ddof=1)
    elif var_type == 3:
        y = np.mean(window * data[indices], axis=1)
    else:
        raise ValueError(
            "Invalid type. Choose 1 (variance), 2 (std deviation), or 3 (mean)."
        )

    return y


def check_centralized_component(
    data, center_range=1 / 10, side_range=1 / 3, wnd=8, overlap=4, param=3
):
    """
    Checks if an event is centralized based on variance in different regions.

    Parameters:
        data (ndarray): 1D array of filtered event data (e.g., HFO range).
        center_range (float): Fraction of event at the center to evaluate.
        side_range (float): Fraction of event at the sides to evaluate.
        wnd (int): Window size for computing standard deviation.
        overlap (int): Overlap size for computing standard deviation.
        param (float): Multiplier for the threshold.

    Returns:
        accept (bool): Whether the event is accepted as HFO.
        th (float): Threshold computed for the event.
    """
    n = len(data)

    # Compute standard deviation in center and side regions
    sd_center = temp_variance(
        data[int(n * center_range) - 1 : int(n * (1 - center_range)) + 1],
        wnd,
        overlap,
        2,
    )
    sd_side = np.concatenate(
        [
            temp_variance(data[: int(n * side_range)] + 1, wnd, overlap, 2),
            temp_variance(data[int(2 * n * side_range) - 1 :], wnd, overlap, 2),
        ]
    )

    # Compute threshold
    th = param * np.median(sd_side)

    # Check if any value in center exceeds the threshold
    FF_mid = np.where(sd_center > th)[0]
    accept = bool(len(FF_mid) > 0)

    return accept, th


def hfo_amp_detector(x, sD=None, th=None, fs=2000, fc=80, Nc=4):
    """
    HFO Amplitude Detector

    Parameters:
        x (ndarray): Filtered signal in the HFO range.
        sD (float, optional): Standard deviation threshold.
        th (float, optional): Amplitude threshold (default 4 uV).
        fs (int, optional): Sampling rate (default 2000 Hz).
        fc (int, optional): Minimum HFO frequency (default 80 Hz).
        Nc (int, optional): Minimum number of threshold crossings (default 4).

    Returns:
        p (int): 1 if detection criteria are met, otherwise 0.
    """

    # Default threshold setting
    if th is None:
        th = 4  # Default threshold 4 uV

    if sD is not None and sD > 4:
        th = sD  # Override if standard deviation threshold is higher than 4 uV

    N = round(1 / fc * fs)  # Minimum duration between crossings
    h = np.ones(Nc - 1)  # Convolution filter for crossings
    p = 0  # Default detection result

    # Check threshold crossings for (x - th)
    _, ix = zerocross_count(x - th)
    if len(ix) > 0:
        ix_diff = np.diff(ix)
        if len(ix_diff) > 0:
            dx = convolve(ix_diff < N, h, mode="valid")
            if np.sum(dx >= Nc - 1) > 0:
                return 1  # Detection criteria met

    # Check threshold crossings for (x + th)
    _, ix = zerocross_count(x + th)
    if len(ix) > 0:
        ix_diff = np.diff(ix)
        if len(ix_diff) > 0:
            dx = convolve(ix_diff < N, h, mode="valid")
            if np.sum(dx >= Nc - 1) > 0:
                return 1  # Detection criteria met

    return p  # Return 0 if criteria not met
