from dataclasses import _MISSING_TYPE, fields

from PyQt6 import QtWidgets


class FilePickerWidget(QtWidgets.QWidget):
    """
    A widget that combines a `QLineEdit` and a `QPushButton` to allow users to select or
    create a file.

    Parameters
    ----------
    parent : QtWidgets.QWidget, optional
        The parent widget for this widget.
    mode : {"find file", "create file"}, default="find file"
        The mode of the file picker. "find file" allows users to select an existing file
        while "create file" allows users to specify a new file to create.
    file_filter : str, default="All Files (*)"
        The filter for the file dialog. This determines which files are shown in the dialog.
    extension : str, default=""
        Used to add extension in case user doesn't provide it when creating a file.

    Attributes
    ----------
    line_edit : QtWidgets.QLineEdit
        The line edit where the selected or created file path is displayed.
    browse_button : QtWidgets.QPushButton
        The button that opens the file dialog for selecting or creating a file.
    mode : str
        The mode of the file picker, either "find file" or "create file".
    file_filter : str
        The filter for the file dialog.
    extension : str
        The file extension to append when creating a file.
    line_edit.setText(value: str)
        Set the text in the line edit to the specified value.

    Methods
    -------
    open_dialog()
        Opens a file dialog based on the mode. If in "find file" mode, it
        allows the user to select an existing file. If in "create file" mode, it
        allows the user to specify a new file to create, appending the specified
        extension if necessary.
    text() -> str
        Returns the text currently in the line edit.
    setText(value: str)
        Sets the text in the line edit to the specified value.
    setToolTip(text: str)
        Sets the tooltip for both the line edit and the browse button.
    setEnabled(enabled: bool)
        Enables or disables the line edit and browse button based on the provided boolean.
    setStyleSheet(style: str)
        Sets the style sheet for the line edit.
    """

    def __init__(
        self,
        parent=None,
        mode: str = "find file",  # "find file" or "create file"
        file_filter: str = "All Files (*)",
        extension: str = "",
    ):
        super().__init__(parent)

        self.mode = mode
        self.file_filter = file_filter
        self.extension = extension

        self.line_edit = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_button)

        self.browse_button.clicked.connect(self.open_dialog)

    def _append_extension(self, path: str):
        if path and self.extension and not path.endswith(self.extension):
            return f"{path}{self.extension}"
        return path

    def open_dialog(self):
        if self.mode == "find file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select File", "", self.file_filter
            )
        elif self.mode == "create file":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Create File", "", self.file_filter
            )
            path = self._append_extension(path)
        else:
            path = ""

        if path:
            self.line_edit.setText(path)

    def text(self):
        return self.line_edit.text()

    def setText(self, value: str):
        self.line_edit.setText(value)

    def setToolTip(self, text: str):
        """
        Sets the tooltip for both the line edit and the browse button.

        Parameters
        ----------
        text : str
            The text to set as the tooltip for both the line edit and the browse button.
        """
        self.line_edit.setToolTip(text)
        self.browse_button.setToolTip(text)

    def setEnabled(self, enabled: bool):
        """
        Enables or disables the line edit and browse button based on the provided
        boolean.

        Parameters
        ----------
        enabled : bool
            If True, enables the line edit and browse button; if False, disables them.
        """
        self.line_edit.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)

    def setStyleSheet(self, style: str):
        """
        Sets the style sheet for the line edit.

        Parameters
        ----------
        style : str
            The style sheet to apply to the line edit.
        """
        self.line_edit.setStyleSheet(style)


class ConfigMenu:
    """
    A configuration menu that displays a list of configurations in a vertical layout,
    each within its own group box. Each configuration is represented by a dataclass,
    and each field in the dataclass is displayed as a label and an input widget.

    Parameters
    ----------
    parent : QtWidgets.QWidget
        The parent widget for this configuration menu.
    configs : list
        A list of dataclass instances representing the configurations to be displayed.

    Attributes
    ----------
    configs : list
    parent : QtWidgets.QWidget
    vertical_layout : QtWidgets.QVBoxLayout
    group_boxes : list[QtWidgets.QGroupBox]
        A list of group boxes, one for each configuration.
    form_layouts : list[QtWidgets.QFormLayout]
        A list of form layouts, one for each group box, containing the configuration
        fields.
    item_line_edits : dict
        A dictionary mapping field names to their corresponding `QLineEdit` or
        `FilePickerWidget` instances for user input.

    Methods
    -------
    display_errors(messages: dict)
        Displays error messages for fields that have validation issues by setting the
        style of the corresponding input widgets to indicate an error and setting their
        tooltips to the error messages.
    apply_changes()
        Applies changes made in the input widgets to the corresponding fields in the
        configurations. Validates the input and displays error messages if any issues
        are found.
    reset_defaults()
        Resets the input widgets to their default values as defined in the dataclass
        fields. Clears any error styles or tooltips.
    disable_editing()
        Disables editing of the input widgets based on the `editable_after_start`
        metadata of each field. If a field is not editable after the start, its input
        widget will be disabled.
    """

    def __init__(self, parent: QtWidgets.QWidget, configs: list):
        self.configs = configs
        self.parent = parent

        self.vertical_layout = QtWidgets.QVBoxLayout(parent)
        self.group_boxes = [QtWidgets.QGroupBox(config.name) for config in configs]
        self.form_layouts = [QtWidgets.QFormLayout() for group_box in self.group_boxes]

        self.item_line_edits = {}

        for config, group_box, form_layout in zip(
            self.configs, self.group_boxes, self.form_layouts
        ):
            self._add_items(config, form_layout)
            group_box.setLayout(form_layout)
            self.vertical_layout.addWidget(group_box)

    def _add_items(self, config, form_layout):
        for field in fields(config):
            name = field.metadata.get("label", field.name)
            label = QtWidgets.QLabel(name)

            file_dialog_mode = field.metadata.get("file_dialog", None)
            file_filter = field.metadata.get("file_filter", "All Files (*)")

            if file_dialog_mode in ("find file", "create file"):
                line_edit = FilePickerWidget(
                    parent=self.parent,
                    mode=file_dialog_mode,
                    file_filter=file_filter,
                    extension=field.metadata.get("file_extension", ""),
                )
            else:
                line_edit = QtWidgets.QLineEdit()

            # Store the line_edit for validation & apply_changes
            assert field.name not in self.item_line_edits, f"Duplicate field name '{
                field.name
            }'"
            self.item_line_edits[field.name] = line_edit

            # Set default value
            if type(field.default_factory) is not _MISSING_TYPE:
                default_list = field.default_factory()
                default_value = ", ".join(map(str, default_list))
            elif field.default is not None:
                default_value = str(field.default)
            else:
                default_value = ""
            line_edit.setText(default_value)

            # Set tooltip
            tooltip = field.metadata.get("description", "")
            if tooltip:
                line_edit.setToolTip(tooltip)
                label.setToolTip(tooltip)

            form_layout.addRow(label, line_edit)

    def display_errors(self, messages):
        """
        Displays error messages for fields that have validation issues by setting the
        style of the corresponding input widgets to indicate an error and setting their
        tooltips to the error messages.

        Parameters
        ----------
        messages : dict
            A dictionary where keys are field names and values are error messages.
        """
        for config in self.configs:
            for field in fields(config):
                if field.name in messages:
                    message = messages[field.name]
                    self.item_line_edits[field.name].setStyleSheet(
                        "border: 1px solid red;"
                    )
                    self.item_line_edits[field.name].setToolTip(message)
                else:
                    self.item_line_edits[field.name].setStyleSheet("")
                    self.item_line_edits[field.name].setToolTip(
                        field.metadata.get("description", "")
                    )

    def apply_changes(self):
        formatting_messages = {}
        validation_messages = {}
        for config in self.configs:
            for field in fields(config):
                if field.name in self.item_line_edits:
                    line_edit = self.item_line_edits[field.name]
                    value = line_edit.text().strip()

                    try:
                        if type(field.default_factory) is not _MISSING_TYPE:
                            attr = list(map(float, value.split(",")))
                        elif field.type is int:
                            attr = int(value)
                        elif field.type is float:
                            attr = float(value)
                        elif field.type is str:
                            attr = value
                    except ValueError:
                        formatting_messages[field.name] = "Invalid formatting"
                        continue

                    setattr(config, field.name, attr)
            validation_messages.update(config.get_validation_messages())

        # Formatting messages take precedence
        validation_messages.update(formatting_messages)
        self.display_errors(validation_messages)

    def reset_defaults(self):
        for config in self.configs:
            for field in fields(config):
                if field.name in self.item_line_edits:
                    if not self.item_line_edits[field.name].isEnabled():
                        continue
                    line_edit = self.item_line_edits[field.name]
                    if type(field.default_factory) is not _MISSING_TYPE:
                        value = field.default_factory()
                        default_value = ", ".join(map(str, value))
                    elif field.default is not None:
                        value = field.default
                        default_value = str(field.default)
                    else:
                        value = None
                        default_value = ""
                    setattr(config, field.name, value)
                    line_edit.setText(default_value)
                    line_edit.setStyleSheet("")
                    line_edit.setToolTip(field.metadata.get("description", ""))

    def disable_editing(self):
        for config in self.configs:
            for field in fields(config):
                if field.name in self.item_line_edits:
                    line_edit = self.item_line_edits[field.name]
                    line_edit.setEnabled(
                        field.metadata.get("editable_after_start", True)
                    )
