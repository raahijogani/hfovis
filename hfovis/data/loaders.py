import numpy as np
import os
import h5py
import json


class IEEGDataLoader:
    def __init__(self, annotations_path, data_path):
        self.annotations_path = annotations_path
        self.data_path = data_path
        os.chdir(self.data_path)

    def load_annotations(self):
        try:
            with open(self.annotations_path, "r") as f:
                annotations = json.load(f)
        except FileNotFoundError:
            annotations = []
            print("annotations.json not found. Continuing without annotations.")
        return annotations

    def extract_data(self, item):
        """
        Recursively extract data from HDF5 items.
        """
        if isinstance(item, h5py.Group):
            # Handle groups (or MATLAB structs)
            if "MATLAB_class" in item.attrs and item.attrs["MATLAB_class"] == b"cell":
                cell_data = {}
                for key in item.keys():
                    cell_data[key] = self.extract_data(item[key])
                return cell_data
            else:
                group_data = {}
                for key in item.keys():
                    group_data[key] = self.extract_data(item[key])
                return group_data

        elif isinstance(item, h5py.Dataset):
            data = item[()]  # Read the dataset into memory

            # 1) If it's a scalar reference or array of references:
            if isinstance(data, h5py.Reference):
                return self.extract_data(item.file[data])
            elif isinstance(data, np.ndarray) and data.dtype.kind == "O":
                ref_list = []
                for ref in data.flatten():
                    extracted_data = self.extract_data(item.file[ref])
                    if isinstance(extracted_data, list):
                        ref_list.append([str(x) for x in extracted_data])
                    elif isinstance(extracted_data, bytes):
                        ref_list.append(extracted_data.decode("utf-8"))
                    else:
                        ref_list.append(extracted_data)
                return ref_list

            # 1.5) If it's a uint16 array, decode it as a string:
            if data.dtype == np.uint16:
                try:
                    return "".join([chr(c) for c in data.flatten()])
                except Exception as e:
                    return data

            # 2) If it's a byte string, decode it:
            if isinstance(data, bytes):
                return data.decode("utf-8")

            return data

        else:
            return None

    def load_data(self, mat_file_path):
        try:
            with h5py.File(mat_file_path, "r") as mat_file:
                keys = list(mat_file.keys())
                montage = (
                    self.extract_data(mat_file["montage"])
                    if "montage" in keys
                    else None
                )
                patientInfo = (
                    self.extract_data(mat_file["patientInfo"])
                    if "patientInfo" in keys
                    else None
                )
                data = self.extract_data(mat_file["data"]) if "data" in keys else None
                iEEG_data = data["data"].transpose()
                return montage, patientInfo, iEEG_data
        except Exception as e:
            raise e
