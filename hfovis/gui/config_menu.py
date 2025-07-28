from PyQt6 import QtWidgets
from dataclasses import fields, _MISSING_TYPE


class ConfigMenu:
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
            widget = QtWidgets.QLineEdit()

            assert field.name not in self.item_line_edits, f"Duplicate field name '{
                field.name
            }' found in configuration '{config.name}'."
            self.item_line_edits[field.name] = widget

            # Set default value
            if type(field.default_factory) is not _MISSING_TYPE:
                default_list = field.default_factory()
                default_value = ", ".join(map(str, default_list))
            elif field.default is not None:
                default_value = str(field.default)
            else:
                default_value = ""
            widget.setText(default_value)

            # Set tooltip
            tooltip = field.metadata.get("description", "")
            if tooltip:
                widget.setToolTip(tooltip)
                label.setToolTip(tooltip)

            # Add to form
            form_layout.addRow(label, widget)

    def display_errors(self, messages):
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
                        formatting_messages[field.name] = f"Invalid formatting"
                        continue

                    setattr(config, field.name, attr)
            validation_messages.update(config.get_validation_messages())

        # Formatting messages take precedence
        formatting_messages.update(validation_messages)
        self.display_errors(formatting_messages)

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
