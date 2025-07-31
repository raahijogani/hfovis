from dataclasses import dataclass, field, fields


@dataclass
class DetectorConfig:
    name = "Detector Configuration"

    fs: float = field(
        default=2048.0,
        metadata={
            "label": "Sampling Frequency (Hz)",
            "description": "The sampling frequency of the stream in Hz.",
            "valid": lambda x: x > 0,
            "error_message": "Sampling frequency must be a positive number.",
            "editable_after_start": False,
        },
    )
    channels: int = field(
        default=1,
        metadata={
            "label": "Number of Channels",
            "description": "The number of channels in the stream. Ignored if montage is set",
            "valid": lambda x: x > 0,
            "error_message": "Number of channels must be a positive integer.",
            "editable_after_start": False,
        },
    )
    hfo_band: list[int] = field(
        default_factory=lambda: [80, 600],
        metadata={
            "label": "HFO Band (Hz)",
            "description": "Frequency band for HFO detection (in Hz).",
            "valid": lambda x: len(x) == 2 and x[0] < x[1],
            "error_message": "HFO band must be a list of two integers where the first is less than the second.",
            "editable_after_start": False,
        },
    )
    ripple_band: list[int] = field(
        default_factory=lambda: [80, 270],
        metadata={
            "label": "Ripple Band (Hz)",
            "description": "Frequency band of ripple oscillations (in Hz).",
            "valid": lambda x: len(x) == 2 and x[0] < x[1],
            "error_message": "Ripple band must be a list of two integers where the first is less than the second.",
            "editable_after_start": False,
        },
    )
    fast_ripple_band: list[int] = field(
        default_factory=lambda: [230, 600],
        metadata={
            "label": "Fast Ripple Band (Hz)",
            "description": "Frequency band of fast ripple oscillations (in Hz).",
            "valid": lambda x: len(x) == 2 and x[0] < x[1],
            "error_message": "Fast ripple band must be a list of two integers where the first is less than the second.",
            "editable_after_start": False,
        },
    )
    adaptive_threshold_window_size_ms: float = field(
        default=500.0,
        metadata={
            "label": "Adaptive Threshold Window Size (ms)",
            "description": "Window size over which standard deviations should be calculated for adaptive thresholding.",
            "valid": lambda x: x > 0,
            "error_message": "Adaptive threshold window size must be a positive number.",
            "editable_after_start": False,
        },
    )
    adaptive_threshold_overlap_ms: float = field(
        default=200.0,
        metadata={
            "label": "Adaptive Threshold Overlap (ms)",
            "description": "Overlap between consecutive standard deviation windows for adaptive thresholding.",
            "valid": lambda x: x >= 0,
            "error_message": "Adaptive threshold overlap must be a positive number.",
            "editable_after_start": False,
        },
    )
    adaptive_threshold_num_windows: int = field(
        default=100,
        metadata={
            "label": "Number of Adaptive Threshold Windows",
            "description": "Number of standard deviations to calculate median over to get adaptive threshold.",
            "valid": lambda x: x > 0,
            "error_message": "Number of adaptive threshold windows must be a positive integer.",
            "editable_after_start": False,
        },
    )
    adaptive_threshold_num_windows_overlap: int = field(
        default=100,
        metadata={
            "label": "Number of Overlapping Adaptive Threshold Windows",
            "description": "Number of adaptive threshold windows that overlap with each other.",
            "valid": lambda x: x >= 0,
            "error_message": "Number of adaptive threshold windows overlap must be a non-negative integer.",
            "editable_after_start": False,
        },
    )
    min_threshold: float = field(
        default=5.0,
        metadata={
            "label": "Minimum Threshold",
            "description": "Minimum threshold for HFO detection.",
            "valid": lambda x: x > 0,
            "error_message": "Minimum threshold must be a positive number.",
            "editable_after_start": False,
        },
    )
    threshold_multiplier: float = field(
        default=5.0,
        metadata={
            "label": "Threshold Multiplier",
            "description": "Multiplier for the adaptive threshold to determine the detection threshold.",
            "valid": lambda x: x > 0,
            "error_message": "Threshold multiplier must be a positive number.",
            "editable_after_start": False,
        },
    )
    burst_window_size_ms: float = field(
        default=320.0,
        metadata={
            "label": "Burst Window Size (ms)",
            "description": "Window size for burst detection in milliseconds.",
            "valid": lambda x: x > 0,
            "error_message": "Burst window size must be a positive number.",
            "editable_after_start": False,
        },
    )
    burst_window_overlap_ms: float = field(
        default=64.0,
        metadata={
            "label": "Burst Window Overlap (ms)",
            "description": "Overlap between consecutive burst windows in milliseconds.",
            "valid": lambda x: x >= 0,
            "error_message": "Burst window overlap must be a positive number.",
            "editable_after_start": False,
        },
    )
    side_max_crossings: int = field(
        default=4,
        metadata={
            "label": "Max Side Crossings",
            "description": "Maximum number of threshold crossings on either side of the burst window allowed to consider a burst as a candidate HFO.",
            "valid": lambda x: x >= 0,
            "error_message": "Max side crossings must be a non-negative integer.",
            "editable_after_start": False,
        },
    )
    center_min_crossings: int = field(
        default=6,
        metadata={
            "label": "Min Center Crossings",
            "description": "Minimum number of threshold crossings in the center of the burst window to consider it a candidate HFO.",
            "valid": lambda x: x >= 0,
            "error_message": "Min center crossings must be a non-negative integer.",
            "editable_after_start": False,
        },
    )
    visualization_window_size_ms: float = field(
        default=200.0,
        metadata={
            "label": "Visualization Window Size (ms)",
            "description": "Window size for visualizing detected HFOs in milliseconds.",
            "valid": lambda x: x > 0,
            "error_message": "Visualization window size must be a positive number.",
            "editable_after_start": False,
        },
    )
    low_band: float = field(
        default=1.0,
        metadata={
            "label": "Low Band (Hz)",
            "description": "Low band frequency for high-pass filtering to remove DC offset.",
            "valid": lambda x: x >= 0,
            "error_message": "Low band frequency must be a non-negative number.",
            "editable_after_start": False,
        },
    )
    ring_buffer_size_s: float = field(
        default=10.0,
        metadata={
            "label": "Ring Buffer Size (s)",
            "description": "Size of ring buffer used to store raw data for visualization.",
            "valid": lambda x: x > 0,
            "error_message": "Ring buffer size must be a positive number.",
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
