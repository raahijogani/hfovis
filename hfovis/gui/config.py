import os
from dataclasses import dataclass, field, fields


def _is_valid_file(file_path: str, extension: str) -> bool:
    """
    Check if the given file path is valid and has the specified extension.

    Parameters
    ----------
    file_path : str
        The file path to validate.
    extension : str
        The required file extension (e.g., '.pkl', '.npy').

    Returns
    -------
    bool
        True if the file path is valid and has the specified extension, False otherwise.
    """
    # Empty string is considered valid
    if not file_path:
        return True

    if not file_path.endswith(extension):
        return False

    # Check if the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        return False

    return True


@dataclass
class GeneralConfig:
    name = "General"

    metadata_filename: str = field(
        default="",
        metadata={
            "label": "Metadata Save Location (.pkl)",
            "description": "Where to save the metadata file. Must be .pkl absolute path",
            "valid": lambda x: _is_valid_file(x, ".pkl"),
            "error_message": "Metadata filename must be a valid .pkl file path",
            "file_dialog": "create file",
            "file_filter": "Pickle files (*.pkl)",
            "file_extension": ".pkl",
        },
    )
    raw_data_filename: str = field(
        default="",
        metadata={
            "label": "Raw Data Save Location (.npy)",
            "description": "Where to save the raw data file. Must be .npy absolute path",
            "valid": lambda x: _is_valid_file(x, ".npy"),
            "error_message": "Raw data filename must be a valid .npy file path",
            "file_dialog": "create file",
            "file_filter": "Numpy files (*.npy)",
            "file_extension": ".npy",
        },
    )
    filtered_data_filename: str = field(
        default="",
        metadata={
            "label": "Filtered Data Save Location (.npy)",
            "description": "Where to save the filtered data file. Must be .npy absolute path",
            "valid": lambda x: _is_valid_file(x, ".npy"),
            "error_message": "Filtered data filename must be a valid .npy file path",
            "file_dialog": "create file",
            "file_filter": "Numpy files (*.npy)",
            "file_extension": ".npy",
        },
    )
    montage_location: str = field(
        default="",
        metadata={
            "label": "Montage Location",
            "description": "Absolute path to channel montage. Each channel should be on a new line.",
            "valid": lambda x: os.path.isfile(x) if x else True,
            "error_message": "Montage location not found",
            "file_dialog": "find file",
            "editable_after_start": False,
        },
    )

    def get_validation_messages(self):
        """
        Collects validation messages for each field in the dataclass.

        Returns
        -------
        dict
            A dictionary where keys are field names and values are error messages
            if the field does not meet its validation criteria.
        """
        messages = {}

        for f in fields(self):
            value = getattr(self, f.name)
            meets_condition = f.metadata.get("valid", lambda x: True)
            label = f.metadata.get("label", f.name)

            if not meets_condition(value):
                messages[f.name] = f.metadata.get(
                    "error_message", f"Invalid value {value} for {label}"
                )

        return messages

    def validate(self):
        """
        Validates the configuration by checking each field against its validation
        criteria.

        Raises
        ------
        ValueError
            If any field does not meet its validation criteria, a ValueError is raised
        """
        messages = self.get_validation_messages()
        if messages:
            raise ValueError(f"Configuration validation failed: {messages}")

    def update(self, **kwargs):
        """
        Update the configuration with new values for specified fields.

        Parameters
        ----------
        **kwargs
            Keyword arguments where keys are field names and values are the new values
            to set for those fields.

        Raises
        ------
        ValueError
            If a key in kwargs does not correspond to a field in the dataclass,
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")
        self.validate()
