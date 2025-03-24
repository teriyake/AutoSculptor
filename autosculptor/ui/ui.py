from PySide2.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QComboBox, QSlider, QHBoxLayout, QGroupBox, QCheckBox, QSpinBox, QTabWidget, QWidget
)
from PySide2.QtCore import Qt
import maya.OpenMayaUI as omui
from shiboken2 import wrapInstance
import maya.cmds as cmds

def get_maya_main_window():
    main_window_ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(main_window_ptr), QDialog)

class SculptingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        
        # Mesh Selection
        mesh_group = QGroupBox("Mesh Selection")
        mesh_layout = QHBoxLayout()
        mesh_layout.addWidget(QLabel("Selected Mesh:"))
        self.mesh_button = QPushButton("Select/Deselect Mesh")
        mesh_layout.addWidget(self.mesh_button)
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)
        
        # Stroke List
        stroke_group = QGroupBox("Stroke History")
        stroke_layout = QVBoxLayout()
        self.stroke_list = QTableWidget()
        self.stroke_list.setColumnCount(6)
        self.stroke_list.setHorizontalHeaderLabels(["ID", "Type", "Mode", "Falloff", "Size", "Strength"])
        self.stroke_list.setMinimumHeight(150)
        stroke_layout.addWidget(self.stroke_list)
        self.delete_stroke_btn = QPushButton("Delete Stroke")
        stroke_layout.addWidget(self.delete_stroke_btn)
        stroke_group.setLayout(stroke_layout)
        layout.addWidget(stroke_group)
        
        # Sample List
        sample_group = QGroupBox("Samples")
        sample_layout = QVBoxLayout()
        self.sample_list = QTableWidget()
        self.sample_list.setColumnCount(5)
        self.sample_list.setHorizontalHeaderLabels(["ID", "Position", "Normal", "Pressure", "Timestamp"])
        self.sample_list.setMinimumHeight(150)
        sample_layout.addWidget(self.sample_list)
        
        sample_buttons_layout = QHBoxLayout()
        self.add_sample_btn = QPushButton("Add Sample")
        self.remove_sample_btn = QPushButton("Remove Sample")
        self.clear_samples_btn = QPushButton("Clear All")
        sample_buttons_layout.addWidget(self.add_sample_btn)
        sample_buttons_layout.addWidget(self.remove_sample_btn)
        sample_buttons_layout.addWidget(self.clear_samples_btn)
        
        sample_layout.addLayout(sample_buttons_layout)
        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)
        
        self.setLayout(layout)

class SuggestionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        
        # Prediction Frame
        prediction_group = QGroupBox("Prediction")
        prediction_layout = QVBoxLayout()
        
        checkbox_layout = QHBoxLayout()
        self.enable_prediction = QCheckBox("Enable Prediction")
        self.preview_prediction = QCheckBox("Preview Selected Prediction")
        checkbox_layout.addWidget(self.enable_prediction)
        checkbox_layout.addWidget(self.preview_prediction)
        prediction_layout.addLayout(checkbox_layout)
        
        stroke_id_layout = QHBoxLayout()
        self.max_prediction_count = QSpinBox()
        self.max_prediction_count.setMinimum(1)
        self.max_prediction_count.setMaximum(100)
        
        stroke_id_layout.addWidget(QLabel("Max Prediction Count"))
        stroke_id_layout.addWidget(self.max_prediction_count)
        
        prediction_layout.addLayout(stroke_id_layout)
        
        prediction_group.setLayout(prediction_layout)
        layout.addWidget(prediction_group)
        
        # Stroke List
        stroke_group = QGroupBox("Stroke Preditions")
        stroke_layout = QVBoxLayout()
        self.stroke_list = QTableWidget()
        self.stroke_list.setColumnCount(6)
        self.stroke_list.setHorizontalHeaderLabels(["ID", "Type", "Mode", "Falloff", "Size", "Strength"])
        self.stroke_list.setMinimumHeight(150)
        stroke_layout.addWidget(self.stroke_list)
        stroke_group.setLayout(stroke_layout)
        layout.addWidget(stroke_group)
        
        # Sample List
        sample_group = QGroupBox("Samples")
        sample_layout = QVBoxLayout()
        self.sample_list = QTableWidget()
        self.sample_list.setColumnCount(4)
        self.sample_list.setHorizontalHeaderLabels(["ID","Position", "Normal", "Pressure", "Timestamp"])
        self.sample_list.setMinimumHeight(150)
        sample_layout.addWidget(self.sample_list)
        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)
        
        # Buttons Row
        button_layout = QHBoxLayout()
        self.accept_sel_btn = QPushButton("Accept Selected")
        self.accept_all_btn = QPushButton("Accept All")
        self.recompute_btn = QPushButton("Recompute")
        button_layout.addWidget(self.accept_sel_btn)
        button_layout.addWidget(self.accept_all_btn)
        button_layout.addWidget(self.recompute_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)

class AutoSculptorToolWindow(QDialog):
    def __init__(self, parent=get_maya_main_window()):
        super().__init__(parent)
        self.setWindowTitle("Sculpting and Suggestion Tool")
        self.setGeometry(100, 100, 600, 700)
        
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        self.sculpting_tab = SculptingPanel()
        self.suggestion_tab = SuggestionPanel()
        
        self.tabs.addTab(self.sculpting_tab, "Sculpting History")
        self.tabs.addTab(self.suggestion_tab, "Sculpting Suggestion")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        
        self.setDockable()
        
    def setDockable(self):
        try:
            if cmds.workspaceControl("AutoSculptor", exists=True):
                cmds.deleteUI("AutoSculptor", control=True)
            self.setObjectName("AutoSculptorToolWindow")
            workspace_control = cmds.workspaceControl("AutoSculptor", dockToMainWindow=("right", 1))
            mixin_ptr = omui.MQtUtil.findControl("AutoSculptor")
            if mixin_ptr:
                mixin_widget = wrapInstance(int(mixin_ptr), QWidget)
                mixin_widget.layout().addWidget(self)
        except Exception as e:
            print("Failed to dock window:", e)

def show_sculpting_tool_window():
    global sculpting_tool_window
    try:
        sculpting_tool_window.close()
    except:
        pass
    sculpting_tool_window = AutoSculptorToolWindow()
    sculpting_tool_window.show()
