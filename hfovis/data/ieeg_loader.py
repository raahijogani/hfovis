from .loaders import IEEGDataLoader


def load_ieeg_from_fileinfo(file_info, data_path, annotations_path, baseline):
    """
    Load iEEG data from file information.

    Parameters
    ----------
    file_info : dict
        Dictionary containing file information with keys like 'Filename_baseline', 'subject', 'SOZ', and 'outcome'.
    data_path : str
        Path to the directory containing iEEG data files.
    annotations_path : str
        Path to the directory containing annotations files. A dummy can be provided if
        not used.
    baseline : str
        Baseline identifier, typically 'baseline' or 'postbaseline'.

    Returns
    -------
    tuple
        A tuple containing:
        - data : np.ndarray
            The iEEG data array.
        - fs : int
            The sampling frequency of the iEEG data.
        - channel_names : list
            List of channel names in the iEEG data.
        - subject_info : dict
            Dictionary containing subject information with keys 'subject', 'SOZ', and 'outcome'.
    """
    loader = IEEGDataLoader(annotations_path, data_path)
    montage, patientInfo, data = loader.load_data(file_info[f"Filename_{baseline}"])

    fs = (
        int(montage["SampleRate"])
        if isinstance(montage["SampleRate"], (int, float))
        else int(montage["SampleRate"].item())
    )
    channel_names = montage["ChannelNames"]

    subject_info = {
        "subject": file_info.get("subject"),
        "SOZ": file_info.get("SOZ", []),
        "outcome": file_info.get("outcome"),
    }

    return data, fs, channel_names, subject_info


def load_annotations(data_path, annotations_path):
    """Load annotations from the specified path."""
    loader = IEEGDataLoader(annotations_path, data_path)
    return loader.load_annotations()
