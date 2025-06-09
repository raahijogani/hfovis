from .loaders import IEEGDataLoader


def load_ieeg_from_fileinfo(file_info, data_path, annotations_path, baseline):
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
    loader = IEEGDataLoader(annotations_path, data_path)
    return loader.load_annotations()

