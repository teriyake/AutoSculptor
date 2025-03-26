from PySide2.QtWidgets import (
	QDialog,
	QVBoxLayout,
	QLabel,
	QPushButton,
	QTableWidget,
	QTableWidgetItem,
	QComboBox,
	QSlider,
	QHBoxLayout,
	QGroupBox,
	QCheckBox,
	QSpinBox,
	QTabWidget,
	QWidget,
)
from PySide2.QtCore import Qt
import maya.OpenMayaUI as omui
from shiboken2 import wrapInstance
import maya.cmds as cmds
import numpy as np

from autosculptor.core.data_structures import Sample, Stroke, Workflow
from autosculptor.core.brush import Brush, BrushMode
from autosculptor.maya.capture import SculptCapture


def get_maya_main_window():
	main_window_ptr = omui.MQtUtil.mainWindow()
	return wrapInstance(int(main_window_ptr), QDialog)


def generate_random_sample():
	"""
	Generate a random Sample instance with random attributes.
	"""
	position = np.random.uniform(
		-10, 10, 3
	)  # Random 3D position within [-10, 10] range
	normal = np.random.randn(3)  # Random normal vector
	normal /= np.linalg.norm(normal)  # Normalize to unit vector
	size = np.random.uniform(0.1, 5.0)  # Random size between 0.1 and 5.0
	pressure = np.random.uniform(0.1, 1.0)  # Random pressure between 0.1 and 1.0
	timestamp = np.random.uniform(0, 100)  # Random timestamp between 0 and 100
	curvature = np.random.uniform(-1.0, 1.0)  # Random curvature between -1.0 and 1.0

	return Sample(position, normal, size, pressure, timestamp, curvature)


def generate_random_stroke(num_samples):
	"""
	Generate a Stroke with a specified number of random Sample instances.
	"""
	stroke = Stroke()
	stroke.brush_size = np.random.uniform(0.1, 5.0)
	stroke.brush_strength = np.random.uniform(0.1, 1.0)
	stroke.brush_mode = np.random.choice(["ADD", "SUBTRACT", "SMOOTH"])
	stroke.brush_falloff = np.random.choice(["smooth", "linear", "constant"])
	stroke.stroke_type = np.random.choice(["surface", "freeform"])

	for _ in range(num_samples):
		sample = generate_random_sample()
		stroke.add_sample(sample)
	return stroke


def generate_random_workflow(num_strokes):
	"""
	Generate a Workflow with a specified number of Strokes, each containing a specified number of Samples.
	"""
	workflow = Workflow()
	for _ in range(num_strokes):
		stroke = generate_random_stroke(int(np.random.uniform(1.0, 10.0)))
		workflow.add_stroke(stroke)
	return workflow


class SculptingPanel(QWidget):
	def __init__(self, main_window_ref, parent=None):
		super().__init__(parent)
		self.main_window = main_window_ref
		layout = QVBoxLayout()

		# Mesh Selection
		mesh_group = QGroupBox("Mesh Selection")
		mesh_layout = QHBoxLayout()
		self.mesh_name_label = QLabel("Selected Mesh:")
		mesh_layout.addWidget(self.mesh_name_label)
		self.mesh_button = QPushButton("Select/Deselect Mesh")
		self.mesh_button.clicked.connect(self.on_select_mesh)
		mesh_layout.addWidget(self.mesh_button)
		mesh_group.setLayout(mesh_layout)
		layout.addWidget(mesh_group)

		# Stroke List
		stroke_group = QGroupBox("Stroke History")
		stroke_layout = QVBoxLayout()
		self.enable_capture = QCheckBox("Enable History Capture")
		self.enable_capture.stateChanged.connect(self.on_enable_capture_changed)
		stroke_layout.addWidget(self.enable_capture)
		self.stroke_list = QTableWidget()
		self.stroke_list.setColumnCount(5)
		self.stroke_list.setHorizontalHeaderLabels(
			["Type", "Mode", "SmpCount", "Size", "Strength"]
		)
		self.stroke_list.setMinimumHeight(150)
		self.stroke_list.itemSelectionChanged.connect(self.on_stroke_selection_changed)
		stroke_layout.addWidget(self.stroke_list)
		self.delete_stroke_btn = QPushButton("Delete Stroke")
		self.delete_stroke_btn.clicked.connect(self.on_delete_stroke)
		self.delete_stroke_btn.setEnabled(False)
		stroke_layout.addWidget(self.delete_stroke_btn)
		stroke_group.setLayout(stroke_layout)
		layout.addWidget(stroke_group)

		# Sample List
		sample_group = QGroupBox("Samples")
		sample_layout = QVBoxLayout()
		self.sample_list = QTableWidget()
		self.sample_list.setColumnCount(5)
		self.sample_list.setHorizontalHeaderLabels(
			["Position", "Normal", "Size", "Pressure", "Timestamp"]
		)
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

		self.workflow = None
		# self.update(generate_random_workflow(5))

	def update(self, workflow):
		"""
		Update the stroke list with provided workflow.
		Each stroke should be a dictionary with keys: brush_type, brush_mode, brush_falloff, brush_size, brush_strength
		"""
		self.stroke_list.setRowCount(0)  # Clear existing rows
		for stroke in workflow.strokes:
			row_position = self.stroke_list.rowCount()
			self.stroke_list.insertRow(row_position)
			# Columns: ["Type", "Mode", "SmpCount", "Size", "Strength"]
			self.stroke_list.setItem(
				row_position, 0, QTableWidgetItem(stroke.stroke_type)
			)
			self.stroke_list.setItem(
				row_position, 1, QTableWidgetItem(stroke.brush_mode)
			)
			self.stroke_list.setItem(
				row_position, 2, QTableWidgetItem(str(len(stroke.samples)))
			)
			self.stroke_list.setItem(
				row_position, 3, QTableWidgetItem(str(stroke.brush_size))
			)
			self.stroke_list.setItem(
				row_position, 4, QTableWidgetItem(str(stroke.brush_strength))
			)
		self.workflow = workflow

		if len(self.workflow.strokes) > 0:
			self.delete_stroke_btn.setEnabled(True)
		else:
			self.delete_stroke_btn.setEnabled(False)

	def update_sample_list(self, stroke):
		"""
		Update the sample list with provided stroke.
		stroke should be Class Stroke
		"""
		self.sample_list.setRowCount(0)  # Clear existing rows
		for smp in stroke.samples:
			row_position = self.sample_list.rowCount()
			self.sample_list.insertRow(row_position)
			# Columns: ["Position", "Normal", "Size", "Pressure", "Timestamp"]
			self.sample_list.setItem(
				row_position, 0, QTableWidgetItem(str(smp.position))
			)
			self.sample_list.setItem(row_position, 1, QTableWidgetItem(str(smp.normal)))
			self.sample_list.setItem(row_position, 2, QTableWidgetItem(str(smp.size)))
			self.sample_list.setItem(
				row_position, 3, QTableWidgetItem(str(smp.pressure))
			)
			self.sample_list.setItem(
				row_position, 4, QTableWidgetItem(str(smp.timestamp))
			)

	def on_select_mesh(self):
		if self.main_window and self.main_window.sculpt_capture:
			sculpt_capture = self.main_window.sculpt_capture
			mesh_name = sculpt_capture.get_selected_mesh_name()
			if mesh_name:
				if sculpt_capture.mesh_name != mesh_name:
					print(f"SculptingPanel: Mesh selected - {mesh_name}")
					sculpt_capture.mesh_name = mesh_name

					sculpt_capture.previous_positions = {}
					sculpt_capture.previous_positions[
						mesh_name
					] = sculpt_capture.get_world_space_positions(mesh_name)
					# self.main_window.sculpt_capture.current_workflow = Workflow()
					# self.update(Workflow())
					self.mesh_name_label.setText(
						f"Selected Mesh: {mesh_name.split('|')[-1]}"
					)
				else:
					print(f"SculptingPanel: Mesh already selected - {mesh_name}")
					self.mesh_name_label.setText(
						f"Selected Mesh: {mesh_name.split('|')[-1]}"
					)
			else:
				sculpt_capture.mesh_name = None
				self.mesh_name_label.setText("Selected Mesh: None")
				print("SculptingPanel: No valid mesh selected.")
		else:
			try:
				sel = cmds.ls(sl=1, type="mesh", l=True) or cmds.ls(
					sl=1, dag=1, type="transform", l=True
				)
				if sel:
					shapes = cmds.listRelatives(sel[0], s=1, type="mesh", f=1) or (
						[sel[0]] if cmds.nodeType(sel[0]) == "mesh" else []
					)
					if shapes:
						self.mesh_name_label.setText(
							f"Selected Mesh: {shapes[0].split('|')[-1]}"
						)
						print(
							f"SculptingPanel: Mesh {shapes[0]} selected (capture inactive)."
						)
					else:
						self.mesh_name_label.setText("Selected Mesh: None (Invalid)")
						print(
							"SculptingPanel: Selection is not a mesh (capture inactive)."
						)
				else:
					self.mesh_name_label.setText("Selected Mesh: None")
					print("SculptingPanel: Nothing selected (capture inactive).")
			except Exception as e:
				print(f"SculptingPanel: Error checking selection: {e}")
				self.mesh_name_label.setText("Selected Mesh: Error")

	def on_enable_capture_changed(self, state):
		if self.main_window:
			if state == Qt.Checked:
				if not self.main_window.sculpt_capture:
					print("SculptingPanel: Enabling capture...")

					self.main_window.sculpt_capture = SculptCapture(
						update_history_callback=self.update,
						update_suggestion_callback=self.main_window.suggestion_tab.update,
					)

					if not self.main_window.sculpt_capture.mesh_name:
						self.on_select_mesh()
					if self.main_window.sculpt_capture.mesh_name:
						self.main_window.sculpt_capture.start_capture()
						self.mesh_button.setEnabled(False)
					else:
						print("SculptingPanel: Cannot start capture, no mesh selected.")
						self.enable_capture.setChecked(False)
				else:
					print("SculptingPanel: Capture already enabled.")
			else:
				if self.main_window.sculpt_capture:
					print("SculptingPanel: Disabling capture...")
					self.main_window.sculpt_capture.stop_capture()
					# self.update(Workflow())
					# self.main_window.suggestion_tab.update(Workflow())
					self.mesh_button.setEnabled(True)
				else:
					print("SculptingPanel: Capture already disabled.")

	def on_delete_stroke(self):
		selected_indices = self.stroke_list.selectedIndexes()
		if selected_indices:
			print(f"Selected index: {selected_indices[0].row()}")

		original_index = selected_indices[0].row()

		if self.main_window and self.main_window.sculpt_capture:
			if (
				0
				<= original_index
				< len(self.main_window.sculpt_capture.current_workflow.strokes)
			):
				self.main_window.sculpt_capture.delete_stroke(original_index)
			else:
				print(
					f"Error: Cannot delete stroke, invalid original index {original_index}"
				)

	def on_stroke_selection_changed(self):
		"""
		Handle the event when the stroke selection changes.
		Updates the sample list based on the selected stroke.
		"""
		selected_indexes = self.stroke_list.selectedIndexes()
		if selected_indexes:
			print(selected_indexes[0].row())
			self.update_sample_list(self.workflow.strokes[selected_indexes[0].row()])

	def select(self, index):
		pass

	def cleanup(self):
		"""Clean up resources."""
		print("SculptingPanel cleanup: Disconnecting signals.")

		self.mesh_button.clicked.disconnect(self.on_select_mesh)
		self.enable_capture.stateChanged.disconnect(self.on_enable_capture_changed)
		self.stroke_list.itemSelectionChanged.disconnect(
			self.on_stroke_selection_changed
		)
		self.delete_stroke_btn.clicked.disconnect(self.on_delete_stroke)

		# TODO: Make sure to disconnect other signals here if we connect them later


class SuggestionPanel(QWidget):
	def __init__(self, main_window_ref, parent=None):
		super().__init__(parent)
		layout = QVBoxLayout()
		self.main_window = main_window_ref

		# Prediction Frame
		prediction_group = QGroupBox("Prediction")
		prediction_layout = QVBoxLayout()

		checkbox_layout = QHBoxLayout()
		self.enable_prediction = QCheckBox("Enable Prediction")
		self.preview_prediction = QCheckBox("Preview Selected Prediction")
		self.enable_prediction.stateChanged.connect(self.on_enable_prediction_changed)
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
		self.stroke_list.setColumnCount(5)
		self.stroke_list.setHorizontalHeaderLabels(
			["Type", "Mode", "Size", "Strength", "Falloff"]
		)
		self.stroke_list.setMinimumHeight(150)
		self.stroke_list.itemSelectionChanged.connect(self.on_stroke_selection_changed)
		stroke_layout.addWidget(self.stroke_list)
		stroke_group.setLayout(stroke_layout)
		layout.addWidget(stroke_group)

		# Sample List
		sample_group = QGroupBox("Samples")
		sample_layout = QVBoxLayout()
		self.sample_list = QTableWidget()
		self.sample_list.setColumnCount(5)
		self.sample_list.setHorizontalHeaderLabels(
			["Position", "Normal", "Size", "Pressure", "Timestamp"]
		)
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

		self.workflow = None
		# self.update(generate_random_workflow(5))

	def update(self, workflow):
		"""
		Update the stroke list with provided workflow.
		Each stroke should be a dictionary with keys: brush_type, brush_mode, brush_falloff, brush_size, brush_strength
		"""
		self.stroke_list.setRowCount(0)  # Clear existing rows
		for stroke in workflow.strokes:
			row_position = self.stroke_list.rowCount()
			self.stroke_list.insertRow(row_position)
			# Columns: ["Type", "Mode", "SmpCount", "Size", "Strength"]
			self.stroke_list.setItem(
				row_position, 0, QTableWidgetItem(stroke.stroke_type)
			)
			self.stroke_list.setItem(
				row_position, 1, QTableWidgetItem(stroke.brush_mode)
			)
			self.stroke_list.setItem(
				row_position, 2, QTableWidgetItem(str(len(stroke.samples)))
			)
			self.stroke_list.setItem(
				row_position, 3, QTableWidgetItem(str(stroke.brush_size))
			)
			self.stroke_list.setItem(
				row_position, 4, QTableWidgetItem(str(stroke.brush_strength))
			)
		self.workflow = workflow

	def update_sample_list(self, stroke):
		"""
		Update the sample list with provided stroke.
		stroke should be Class Stroke
		"""
		self.sample_list.setRowCount(0)  # Clear existing rows
		for smp in stroke.samples:
			row_position = self.sample_list.rowCount()
			self.sample_list.insertRow(row_position)
			# Columns: ["Position", "Normal", "Size", "Pressure", "Timestamp"]
			self.sample_list.setItem(
				row_position, 0, QTableWidgetItem(str(smp.position))
			)
			self.sample_list.setItem(row_position, 1, QTableWidgetItem(str(smp.normal)))
			self.sample_list.setItem(row_position, 2, QTableWidgetItem(str(smp.size)))
			self.sample_list.setItem(
				row_position, 3, QTableWidgetItem(str(smp.pressure))
			)
			self.sample_list.setItem(
				row_position, 4, QTableWidgetItem(str(smp.timestamp))
			)

	def on_stroke_selection_changed(self):
		"""
		Handle the event when the stroke selection changes.
		Updates the sample list based on the selected stroke.
		"""
		selected_indexes = self.stroke_list.selectedIndexes()
		if selected_indexes:
			print(selected_indexes[0].row())
			self.update_sample_list(self.workflow.strokes[selected_indexes[0].row()])

	def on_enable_prediction_changed(self, state):
		is_enabled = state == Qt.Checked
		print(f"SuggestionPanel: Enable Prediction changed to: {is_enabled}")
		self.recompute_btn.setEnabled(is_enabled)

		if self.main_window and self.main_window.sculpt_capture:
			self.main_window.sculpt_capture.set_suggestions_enabled(is_enabled)
		elif is_enabled:
			print(
				"SuggestionPanel: Capture instance doesn't exist yet. Suggestions will generate once capture starts."
			)

		self.on_stroke_selection_changed()

	def on_recompute_clicked(self):
		print("SuggestionPanel: Force Recompute clicked.")
		if self.main_window and self.main_window.sculpt_capture:
			sc = self.main_window.sculpt_capture
			if sc.is_capturing and sc.suggestions_enabled:
				if len(sc.current_workflow.strokes) > 1:
					sc.generate_suggestions()
				else:
					om2.MGlobal.displayWarning(
						"Cannot recompute: No stroke history captured yet."
					)
					sc.clear_suggestions()
			elif not sc.is_capturing:
				om2.MGlobal.displayWarning(
					"Cannot recompute: History capture is not enabled."
				)
			else:
				om2.MGlobal.displayWarning(
					"Cannot recompute: Suggestion generation is disabled."
				)
				sc.clear_suggestions()
		else:
			om2.MGlobal.displayWarning(
				"Cannot recompute: Capture system not initialized."
			)

	def cleanup(self):
		"""Clean up resources."""
		print("SuggestionPanel cleanup: Disconnecting signals.")

		self.enable_prediction.stateChanged.disconnect(
			self.on_enable_prediction_changed
		)
		self.stroke_list.itemSelectionChanged.disconnect(
			self.on_stroke_selection_changed
		)
		self.recompute_btn.clicked.disconnect(self.on_recompute_clicked)

		# TODO: Make sure to disconnect other signals here if we connect them later


class AutoSculptorToolWindow(QDialog):
	def __init__(self, parent=get_maya_main_window()):
		super().__init__(parent)
		self.sculpt_capture = None

		self.setWindowTitle("Sculpting and Suggestion Tool")
		self.setGeometry(100, 100, 600, 700)

		layout = QVBoxLayout()
		self.tabs = QTabWidget()

		self.sculpting_tab = SculptingPanel(self)
		self.suggestion_tab = SuggestionPanel(self)

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
			workspace_control = cmds.workspaceControl(
				"AutoSculptor", dockToMainWindow=("right", 1)
			)
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
	return sculpting_tool_window
