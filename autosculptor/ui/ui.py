from PySide6.QtWidgets import (  # type: ignore
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
	QMessageBox,
	QAbstractItemView,
)
from PySide6.QtCore import Qt  # type: ignore
from PySide6.QtGui import QColor, QShortcut, QKeySequence  # type: ignore
import maya.OpenMayaUI as omui  # type: ignore
import maya.api.OpenMaya as om2  # type: ignore
from shiboken6 import wrapInstance  # type: ignore
import maya.cmds as cmds  # type: ignore
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin  # type: ignore
import numpy as np  # type: ignore

from autosculptor.core.data_structures import Sample, Stroke, Workflow
from autosculptor.core.brush import Brush, BrushMode
from autosculptor.core.surface_brush import SurfaceBrush
from autosculptor.core.freeform_brush import FreeformBrush
from autosculptor.maya.capture import SculptCapture
from autosculptor.suggestions.visualization import StrokeVisualizer
from autosculptor.core.mesh_interface import MeshInterface
from autosculptor.maya.utils import get_active_camera_lookat_vector

from typing import Optional, List


def get_maya_main_window():
	main_window_ptr = omui.MQtUtil.mainWindow()
	return wrapInstance(int(main_window_ptr), QWidget)


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


def approx(num_or_list):

	if isinstance(num_or_list, np.ndarray):
		res_text = ""
		for num in num_or_list:
			res_text += str(round(num, 2)) + ","
		return res_text[: len(res_text) - 1]
	return str(round(num_or_list, 2))


class SculptingPanel(QWidget):
	HISTORY_VIZ_COLOR = (0.2, 0.5, 1.0)
	HISTORY_VIZ_TRANSPARENCY = (0.6, 0.6, 0.6)

	CLONE_VIZ_COLOR = (0.0, 1.0, 0.5)
	CLONE_VIZ_TRANSPARENCY = (0.6, 0.6, 0.6)

	CONTEXT_COLOR = QColor(200, 220, 255)

	VIZ_LAYER_NAME = "AutoSculptor_VizLayer"

	def __init__(self, parent=None):
		super().__init__(parent)
		# self.main_window = main_window_ref # Removed
		self.viz_display_layer = None
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
		self.stroke_list.setSelectionBehavior(QAbstractItemView.SelectRows)
		self.stroke_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
		self.stroke_list.setHorizontalHeaderLabels(
			["Type", "Mode", "SmpCount", "Size", "Strength"]
		)
		self.stroke_list.setMinimumHeight(150)
		self.stroke_list.itemSelectionChanged.connect(self.on_stroke_selection_changed)
		self.stroke_list.horizontalHeader().setStretchLastSection(True)
		stroke_layout.addWidget(self.stroke_list)

		hist_btns_layout = QHBoxLayout()
		self.set_context_btn = QPushButton("Set Context")
		self.set_context_btn.setToolTip(
			"Set the selected strokes as the active context for suggestions."
		)
		self.set_context_btn.clicked.connect(self.on_set_context)
		self.set_context_btn.setEnabled(False)
		hist_btns_layout.addWidget(self.set_context_btn)

		self.clear_context_btn = QPushButton("Clear Context")
		self.clear_context_btn.setToolTip(
			"Use the full stroke history for suggestions."
		)
		self.clear_context_btn.clicked.connect(self.on_clear_context)
		self.clear_context_btn.setEnabled(False)
		hist_btns_layout.addWidget(self.clear_context_btn)

		self.delete_stroke_btn = QPushButton("Delete Stroke")
		self.delete_stroke_btn.clicked.connect(self.on_delete_stroke)
		self.delete_stroke_btn.setEnabled(False)
		hist_btns_layout.addWidget(self.delete_stroke_btn)

		self.clear_all_btn = QPushButton("Clear All")
		self.clear_all_btn.clicked.connect(self.on_clear_all_clicked)
		self.clear_all_btn.setEnabled(False)
		hist_btns_layout.addWidget(self.clear_all_btn)
		stroke_layout.addLayout(hist_btns_layout)

		clone_btns_layout = QHBoxLayout()
		self.clone_stroke_btn = QPushButton("Clone Stroke")
		self.clone_stroke_btn.clicked.connect(self.on_clone_stroke_clicked)
		self.clone_stroke_btn.setEnabled(False)
		clone_btns_layout.addWidget(self.clone_stroke_btn)

		self.cancel_clone_btn = QPushButton("Cancel Clone")
		self.cancel_clone_btn.clicked.connect(self.cancel_clone_target_selection)
		self.cancel_clone_btn.setVisible(False)
		clone_btns_layout.addWidget(self.cancel_clone_btn)

		stroke_layout.addLayout(clone_btns_layout)

		stroke_group.setLayout(stroke_layout)
		layout.addWidget(stroke_group)

		# Sample List
		sample_group = QGroupBox("Samples")
		sample_layout = QVBoxLayout()
		self.sample_list = QTableWidget()
		self.sample_list.setColumnCount(5)
		self.sample_list.setSelectionBehavior(QAbstractItemView.SelectRows)
		self.sample_list.setHorizontalHeaderLabels(
			["Position", "Normal", "Size", "Pressure", "Timestamp"]
		)
		self.sample_list.setMinimumHeight(150)
		sample_layout.addWidget(self.sample_list)

		sample_buttons_layout = QHBoxLayout()
		self.add_sample_btn = QPushButton("Add Sample")
		self.remove_sample_btn = QPushButton("Remove Sample")
		sample_buttons_layout.addWidget(self.add_sample_btn)
		sample_buttons_layout.addWidget(self.remove_sample_btn)

		sample_layout.addLayout(sample_buttons_layout)
		sample_group.setLayout(sample_layout)
		layout.addWidget(sample_group)

		self.setLayout(layout)

		self.workflow: Optional[Workflow] = None
		self._active_context_indices: set = set()

		self.history_visualizer: Optional[StrokeVisualizer] = None

		self.is_waiting_for_clone_target = False
		self.source_stroke_for_clone_index: Optional[int] = None
		self.target_selection_scriptJob: Optional[int] = None
		self.clone_visualizer: Optional[StrokeVisualizer] = None
		self.pending_cloned_stroke: Optional[Stroke] = None

	def update(self, workflow):
		"""
		Update the stroke list with provided workflow.
		Each stroke should be a dictionary with keys: brush_type, brush_mode, brush_falloff, brush_size, brush_strength
		"""
		self.stroke_list.setRowCount(0)  # Clear existing rows
		self.workflow = workflow
		self.clear_history_visualization()

		try:  # temporarily disconnect selection changed signal
			self.stroke_list.itemSelectionChanged.disconnect(
				self.on_stroke_selection_changed
			)
		except (TypeError, RuntimeError):
			pass

		active_context_indices = self._active_context_indices
		for i, stroke in enumerate(self.workflow.strokes):
			row_position = self.stroke_list.rowCount()
			self.stroke_list.insertRow(row_position)

			# Columns: ["Type", "Mode", "SmpCount", "Size", "Strength"]
			item_type = QTableWidgetItem(stroke.stroke_type or "N/A")
			item_mode = QTableWidgetItem(stroke.brush_mode or "N/A")
			item_count = QTableWidgetItem(str(len(stroke.samples)))
			item_size = QTableWidgetItem(
				approx(stroke.brush_size) if stroke.brush_size is not None else "N/A"
			)
			item_strength = QTableWidgetItem(
				approx(stroke.brush_strength)
				if stroke.brush_strength is not None
				else "N/A"
			)

			items = [item_type, item_mode, item_count, item_size, item_strength]

			for item in items:
				item.setFlags(item.flags() & ~Qt.ItemIsEditable)

			self.stroke_list.setItem(row_position, 0, item_type)
			self.stroke_list.setItem(row_position, 1, item_mode)
			self.stroke_list.setItem(row_position, 2, item_count)
			self.stroke_list.setItem(row_position, 3, item_size)
			self.stroke_list.setItem(row_position, 4, item_strength)

		self.stroke_list.itemSelectionChanged.connect(self.on_stroke_selection_changed)

		if len(self.workflow.strokes) > 0:
			self.delete_stroke_btn.setEnabled(True)
			self.clear_all_btn.setEnabled(True)
			self.clone_stroke_btn.setEnabled(True)
		else:
			self.delete_stroke_btn.setEnabled(False)
			self.clear_all_btn.setEnabled(False)
			self.clone_stroke_btn.setEnabled(False)

		if active_context_indices:
			print("ACTIVE CONTEXT FOUND")
			self.clear_context_btn.setEnabled(True)
		else:
			print("*****NO ACTIVE CONTEXT FOUND*****")
			self.clear_context_btn.setEnabled(False)

		self._apply_context_highlighting(active_context_indices)

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
				row_position, 0, QTableWidgetItem(approx(smp.position))
			)
			self.sample_list.setItem(
				row_position, 1, QTableWidgetItem(approx(smp.normal))
			)
			self.sample_list.setItem(
				row_position, 2, QTableWidgetItem(approx(smp.size))
			)
			self.sample_list.setItem(
				row_position, 3, QTableWidgetItem(approx(smp.pressure))
			)
			self.sample_list.setItem(
				row_position, 4, QTableWidgetItem(approx(smp.timestamp))
			)

	def _apply_context_highlighting(self, active_context_indices):
		"""Applies background highlighting to rows corresponding to active context strokes."""
		default_color = self.palette().color(self.backgroundRole())
		for row in range(self.stroke_list.rowCount()):
			is_context = row in active_context_indices
			for col in range(self.stroke_list.columnCount()):
				item = self.stroke_list.item(row, col)
				if item:
					if is_context:
						item.setBackground(self.CONTEXT_COLOR)
					else:
						item.setBackground(default_color)

	def clear_history_visualization(self):
		if self.history_visualizer:
			try:
				self.history_visualizer.clear()
			except Exception as e:
				print(f"SculptingPanel: Error clearing history visualizer: {e}")
			finally:
				self.history_visualizer = None

	def clear_clone_preview_visualization(self):
		if self.clone_visualizer:
			try:
				self.clone_visualizer.clear()
			except Exception as e:
				print(f"SculptingPanel: Error clearing clone preview visualizer: {e}")
			finally:
				self.clone_visualizer = None

		if self.pending_cloned_stroke:
			self.pending_cloned_stroke = None

		self.clone_stroke_btn.setText("Clone Stroke")
		self.clone_stroke_btn.setEnabled(len(self.stroke_list.selectedItems()) > 0)
		self.cancel_clone_btn.setVisible(False)

	def on_select_mesh(self):
		main_window = self.parent().parent().parent()
		if (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			sculpt_capture = main_window.sculpt_capture
			mesh_name = sculpt_capture.get_selected_mesh_name()
			if mesh_name:
				if sculpt_capture.mesh_name != mesh_name:
					sculpt_capture.mesh_name = mesh_name

					sculpt_capture.previous_positions = {}
					sculpt_capture.previous_positions[
						mesh_name
					] = sculpt_capture.get_world_space_positions(mesh_name)
					print(f"SculptingPanel: Mesh selected - {mesh_name}")
					# main_window.sculpt_capture.current_workflow = Workflow()
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
		main_window = self.parent().parent().parent()

		if (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			if state == Qt.CheckState.Checked.value:
				if not main_window.sculpt_capture.mesh_name:
					self.on_select_mesh()

				if main_window.sculpt_capture.mesh_name:
					main_window.sculpt_capture.start_capture()
					self.mesh_button.setEnabled(False)
					print("SculptingPanel: Capture enabled.")
				else:
					om2.MGlobal.displayWarning(
						"Please select a mesh before enabling capture."
					)
					self.enable_capture.setChecked(False)
					self.mesh_button.setEnabled(True)
					print("SculptingPanel: Cannot start capture, no mesh selected.")
			else:
				print("SculptingPanel: Disabling capture...")
				main_window.sculpt_capture.stop_capture()
				self.mesh_button.setEnabled(True)
				print("SculptingPanel: Capture disabled.")
		else:
			print("SculptingPanel: Main window or sculpt_capture not available.")

		is_capture_actually_on = (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
			and main_window.sculpt_capture.is_capturing
		)

		can_clone = is_capture_actually_on and len(self.stroke_list.selectedItems()) > 0
		self.clone_stroke_btn.setEnabled(can_clone)

		if not is_capture_actually_on:
			self.clear_clone_preview_visualization()

	def on_delete_stroke(self):
		self.clear_clone_preview_visualization()

		selected_indices = self.stroke_list.selectedIndexes()
		if selected_indices:
			print(f"Selected index: {selected_indices[0].row()}")

		original_index = selected_indices[0].row()

		main_window = self.parent().parent().parent()
		if (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			if (
				0
				<= original_index
				< len(main_window.sculpt_capture.current_workflow.strokes)
			):
				self.clear_history_visualization()
				main_window.sculpt_capture.delete_stroke(original_index)
			else:
				print(
					f"Error: Cannot delete stroke, invalid original index {original_index}"
				)

	def on_stroke_selection_changed(self):
		"""
		Handle the event when the stroke selection changes.
		Updates the sample list based on the selected stroke.
		"""
		self.clear_history_visualization()  # Clear previous viz first
		self.clear_clone_preview_visualization()

		selected_indexes = self.stroke_list.selectedIndexes()
		selected_rows = sorted(list(set(index.row() for index in selected_indexes)))

		if selected_rows:
			if len(selected_rows) == 1:
				selected_row = selected_rows[0]
				if self.workflow and 0 <= selected_row < len(self.workflow.strokes):
					selected_stroke = self.workflow.strokes[selected_row]
					self.update_sample_list(selected_stroke)
				else:
					self.sample_list.setRowCount(0)
			else:
				self.sample_list.setRowCount(0)

			if self.workflow:
				selected_strokes_workflow = Workflow()
				for row in selected_rows:
					if self.workflow and 0 <= row < len(self.workflow.strokes):
						selected_strokes_workflow.add_stroke(self.workflow.strokes[row])

				if selected_strokes_workflow.strokes:
					try:
						self.history_visualizer = StrokeVisualizer(
							selected_strokes_workflow,
							color=self.HISTORY_VIZ_COLOR,
							transparency=self.HISTORY_VIZ_TRANSPARENCY,
						)
						avg_radius = 0.2
						all_samples = [
							smp
							for stroke in selected_strokes_workflow.strokes
							for smp in stroke.samples
						]
						if all_samples:
							avg_radius = (
								np.mean([smp.size for smp in all_samples]) * 0.5
							)
							avg_radius = max(0.2, avg_radius)

						hist_viz_tubes = self.history_visualizer.visualize(
							avg_radius, 8
						)

						if hist_viz_tubes:
							if not self.viz_display_layer or not cmds.objExists(
								self.viz_display_layer
							):
								self.viz_display_layer = cmds.createDisplayLayer(
									name=self.VIZ_LAYER_NAME, number=1, empty=True
								)
								cmds.setAttr(f"{self.viz_display_layer}.displayType", 2)
								print(
									f"SculptingPanel: Created display layer: {self.viz_display_layer}"
								)

							cmds.editDisplayLayerMembers(
								self.viz_display_layer, hist_viz_tubes, noRecurse=True
							)
							print(
								f"SculptingPanel: Added {len(hist_viz_tubes)} history visualization tubes to layer {self.viz_display_layer}"
							)
						else:
							print(
								f"SculptingPanel: Failed to create history visualization for selected strokes"
							)
					except ValueError as ve:
						print(
							f"SculptingPanel: Cannot visualize selected strokes: {ve}"
						)
						self.history_visualizer = None
					except Exception as e:
						print(f"SculptingPanel: Error visualizing history strokes: {e}")
						import traceback

						traceback.print_exc()
						self.history_visualizer = None
				else:
					self.sample_list.setRowCount(0)
		else:
			self.delete_stroke_btn.setEnabled(False)
			self.set_context_btn.setEnabled(False)
			self.clone_stroke_btn.setEnabled(False)
			self.sample_list.setRowCount(0)

		has_selection = len(selected_rows) > 0
		is_capture_active = (
			self.parent().parent().parent()
			and isinstance(self.parent().parent().parent(), AutoSculptorToolWindow)
			and hasattr(self.parent().parent().parent(), "sculpt_capture")
			and self.parent().parent().parent().sculpt_capture
			and self.parent().parent().parent().sculpt_capture.is_capturing
		)

		self.delete_stroke_btn.setEnabled(has_selection)
		self.set_context_btn.setEnabled(has_selection)

		can_start_clone = has_selection and len(selected_rows) > 0 and is_capture_active
		if (
			not self.is_waiting_for_clone_target
			and self.clone_stroke_btn.text() == "Clone Stroke"
		):
			self.clone_stroke_btn.setEnabled(can_start_clone)

	def on_set_context(self):
		"""Sets the currently selected strokes as the active context."""
		selected_rows = sorted(
			list(set(index.row() for index in self.stroke_list.selectedIndexes()))
		)
		if not selected_rows:
			om2.MGlobal.displayWarning("No strokes selected to set as context.")
			return

		print(f"SculptingPanel: Setting context to selected rows: {selected_rows}")
		main_window = self.parent().parent().parent()
		if (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			main_window.sculpt_capture.set_active_context(selected_rows)
			self._active_context_indices = set(selected_rows)
			self.update(main_window.sculpt_capture.current_workflow)
		else:
			print("SculptingPanel: Capture system not available to set context.")

	def on_clear_context(self):
		"""Clears the active context, using the full history."""
		print("SculptingPanel: Clearing active context.")
		main_window = self.parent().parent().parent()
		if (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			main_window.sculpt_capture.clear_active_context()
			self._active_context_indices = set()
			self.update(main_window.sculpt_capture.current_workflow)
		else:
			print("SculptingPanel: Capture system not available to clear context.")

	def on_clone_stroke_clicked(self):
		if self.clone_stroke_btn.text() == "Clone Stroke":
			self._start_clone_preview_workflow()
		elif self.clone_stroke_btn.text() == "Preview Clone":
			pass
		elif self.clone_stroke_btn.text() == "Accept Clone":
			self._accept_cloned_stroke()

	def _start_clone_preview_workflow(self):
		if self.is_waiting_for_clone_target:
			print(
				"SculptingPanel: Already waiting for target selection. Cancelling previous request."
			)
			self.cancel_clone_target_selection()

		selected_indexes = self.stroke_list.selectedIndexes()
		if not selected_indexes:
			om2.MGlobal.displayWarning(
				"Please select one or more history strokes to clone."
			)
			return

		self.source_stroke_for_clone_index = sorted(
			list(set(index.row() for index in selected_indexes))
		)
		print(
			f"SculptingPanel: Source stroke indices for clone: {self.source_stroke_for_clone_index}"
		)

		if not cmds.selectMode(q=True, component=True):
			cmds.selectMode(component=True)
			cmds.selectType(
				vertex=True,
				allComponents=False,
			)
			cmds.select(clear=True)

		main_window = self.parent().parent().parent()
		if not (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
			and main_window.sculpt_capture.is_capturing
		):
			om2.MGlobal.displayWarning("Please enable history capture first.")
			return

		self.is_waiting_for_clone_target = True
		self.clone_stroke_btn.setText("Preview Clone")
		self.clone_stroke_btn.setEnabled(False)
		self.cancel_clone_btn.setVisible(True)
		self.clear_clone_preview_visualization()
		self.pending_cloned_stroke = None

		cmds.inViewMessage(
			amg="<hl>Select Target Vertex:</hl> Please select a single vertex on the mesh to anchor the clone.",
			pos="midCenter",
			fade=True,
			fts=12,
			fot=100,
		)

		self.target_selection_scriptJob = cmds.scriptJob(
			event=["SelectionChanged", self._handle_target_vertex_selection],
			runOnce=False,
			# protected=True
		)
		print(
			f"SculptingPanel: Waiting for target vertex selection (Job ID: {self.target_selection_scriptJob})..."
		)

	def _accept_cloned_stroke(self):
		if not self.pending_cloned_stroke or not self.pending_cloned_stroke.strokes:
			om2.MGlobal.displayWarning("No clone preview available to accept.")
			return

		main_window = self.parent().parent().parent()
		if not (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
			and main_window.sculpt_capture.is_capturing
		):
			om2.MGlobal.displayWarning("Cannot accept clone: Capture is not active.")
			return

		print("SculptingPanel: Accepting cloned stroke...")

		sc = main_window.sculpt_capture
		try:
			sc.apply_cloned_stroke(self.pending_cloned_stroke)

			self.clear_clone_preview_visualization()
			self.source_stroke_for_clone_index = None

		except Exception as e:
			print(f"SculptingPanel: Error accepting clone: {e}")
			om2.MGlobal.displayError("Failed to apply cloned stroke.")
			self.clear_clone_preview_visualization()
			self.source_stroke_for_clone_index = None

	def _kill_target_selection_scriptJob(self):
		if self.target_selection_scriptJob:
			job_id = self.target_selection_scriptJob
			self.target_selection_scriptJob = None
			try:
				if cmds.scriptJob(exists=job_id):
					cmds.scriptJob(kill=job_id, force=True)
					print(
						f"SculptingPanel: Killed target selection script job {job_id}"
					)
			except Exception as e:
				print(f"SculptingPanel: Error killing script job {job_id}: {e}")

	def cancel_clone_target_selection(self):
		self._kill_target_selection_scriptJob()
		cmds.selectMode(object=True)
		self.is_waiting_for_clone_target = False
		self.source_stroke_for_clone_index = None
		self.clear_clone_preview_visualization()
		cmds.inViewMessage(clear="midCenter")

	def _handle_target_vertex_selection(self):
		if not self.is_waiting_for_clone_target:
			return

		selection = cmds.ls(selection=True, flatten=True)
		if not selection or len(selection) != 1 or ".vtx[" not in selection[0]:
			return

		target_vertex_str = selection[0]
		print(f"SculptingPanel: Target vertex selected: {target_vertex_str}")

		self._kill_target_selection_scriptJob()
		self.is_waiting_for_clone_target = False
		cmds.selectMode(object=True)
		self.clone_stroke_btn.setText("Preview Clone")
		self.clone_stroke_btn.setEnabled(True)

		main_window = self.parent().parent().parent()
		if not (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			print("SculptingPanel: Capture system not available.")
			return
		if (
			self.source_stroke_for_clone_index is None
			or not self.source_stroke_for_clone_index
		):
			print("SculptingPanel: Error - Source stroke indices lost or empty.")
			return

		sc = main_window.sculpt_capture
		if not sc.mesh_name:
			om2.MGlobal.displayError(
				"SculptingPanel: Sculpting mesh not set in capture system."
			)
			return

		mesh_data = MeshInterface.get_mesh_data(sc.mesh_name)
		if not mesh_data:
			om2.MGlobal.displayError(
				"SculptingPanel: Failed to get mesh data for cloning."
			)
			return
		sc._ensure_synthesizer_mesh(mesh_data)

		if not sc.synthesizer:
			om2.MGlobal.displayError(
				"SculptingPanel: Synthesizer failed to initialize."
			)
			return

		if not target_vertex_str.startswith(sc.mesh_name.split("|")[-1] + "."):
			if not target_vertex_str.startswith(
				cmds.listRelatives(sc.mesh_name, p=True, f=True)[0].split("|")[-1] + "."
			):
				om2.MGlobal.displayError(
					f"Please select a vertex on the target mesh: {sc.mesh_name}"
				)
				self.source_stroke_for_clone_index = None
				return

		try:
			target_pos_list = cmds.pointPosition(target_vertex_str, world=True)
			target_pos = np.array(target_pos_list, dtype=np.float64)

			sel_list = om2.MSelectionList()
			sel_list.add(target_vertex_str)
			comp_obj = sel_list.getComponent(0)
			mesh_path = comp_obj[0]
			vertex_comp = comp_obj[1]

			if (
				not vertex_comp.isNull()
				and vertex_comp.apiType() == om2.MFn.kMeshVertComponent
			):
				mesh_fn = om2.MFnMesh(mesh_path)
				vert_iter = om2.MItMeshVertex(mesh_path, vertex_comp)
				target_normal_vec = vert_iter.getNormal(om2.MSpace.kWorld)
				target_normal = np.array(
					[target_normal_vec.x, target_normal_vec.y, target_normal_vec.z],
					dtype=np.float64,
				)
				norm_mag = np.linalg.norm(target_normal)
				if norm_mag > 1e-6:
					target_normal /= norm_mag
				else:
					target_normal = np.array([0.0, 1.0, 0.0])
			else:
				om2.MGlobal.displayError("Failed to get target vertex normal.")
				self.source_stroke_for_clone_index = None
				return

			source_strokes = [
				sc.current_workflow.strokes[i]
				for i in self.source_stroke_for_clone_index
				if 0 <= i < len(sc.current_workflow.strokes)
			]

			if not source_strokes:
				om2.MGlobal.displayError(
					"Selected source strokes are empty or invalid."
				)
				self.source_stroke_for_clone_index = None
				return

			first_source_stroke = source_strokes[0]
			if not first_source_stroke.samples:
				om2.MGlobal.displayError("First selected source stroke is empty.")
				self.source_stroke_for_clone_index = None
				return

			source_anchor_pos = first_source_stroke.samples[0].position
			source_anchor_normal = first_source_stroke.samples[0].normal

			cloned_workflow = sc.synthesizer.clone_workflow(
				source_strokes=source_strokes,
				source_anchor_pos=source_anchor_pos,
				source_anchor_normal=source_anchor_normal,
				target_anchor_pos=target_pos,
				target_anchor_normal=target_normal,
				target_mesh_data=MeshInterface.get_mesh_data(sc.mesh_name),
				scale_factor=1.0,
			)

			if cloned_workflow and cloned_workflow.strokes:
				self.pending_cloned_stroke = cloned_workflow
				print(
					f"SculptingPanel: Clone preview generated with {len(self.pending_cloned_stroke.strokes)} strokes."
				)
				for i, stroke in enumerate(self.pending_cloned_stroke.strokes):
					print(f"  Stroke {i}: {len(stroke.samples)} samples")

				try:
					print("SculptingPanel: Attempting to visualize cloned strokes...")
					self.clone_visualizer = StrokeVisualizer(
						self.pending_cloned_stroke,
						color=self.CLONE_VIZ_COLOR,
						transparency=self.CLONE_VIZ_TRANSPARENCY,
					)
					avg_radius = 0.2
					if (
						self.pending_cloned_stroke.strokes
						and self.pending_cloned_stroke.strokes[0].samples
					):
						avg_radius = (
							np.mean(
								[
									s.size
									for s in self.pending_cloned_stroke.strokes[
										0
									].samples
								]
							)
							* 0.5
						)
						avg_radius = max(0.2, avg_radius)

					clone_viz_tubes = self.clone_visualizer.visualize(avg_radius, 8)
					print(
						f"SculptingPanel: visualize returned {len(clone_viz_tubes) if clone_viz_tubes else 0} tubes."
					)

					if clone_viz_tubes:
						if not self.viz_display_layer or not cmds.objExists(
							self.viz_display_layer
						):
							self.viz_display_layer = cmds.createDisplayLayer(
								name=self.VIZ_LAYER_NAME, number=1, empty=True
							)
							cmds.setAttr(f"{self.viz_display_layer}.displayType", 2)
							print(
								f"SculptingPanel: Created display layer: {self.viz_display_layer}"
							)

						cmds.editDisplayLayerMembers(
							self.viz_display_layer, clone_viz_tubes, noRecurse=True
						)
						print(
							f"SculptingPanel: Added {len(clone_viz_tubes)} clone visualization tubes to layer {self.viz_display_layer}"
						)
					else:
						print(f"SculptingPanel: Failed to create clone visualization")

				except Exception as viz_e:
					print(f"SculptingPanel: Error visualizing cloned strokes: {viz_e}")
					import traceback

					traceback.print_exc()
					self.clone_visualizer = None

				self.clone_stroke_btn.setText("Accept Clone")
				self.clone_stroke_btn.setEnabled(True)
				self.cancel_clone_btn.setVisible(True)
			else:
				om2.MGlobal.displayError("Failed to generate clone preview.")
				self.clear_clone_preview_visualization()

		except Exception as e:
			print(f"SculptingPanel: Error during target selection handling: {e}")
			import traceback

			traceback.print_exc()
			om2.MGlobal.displayError("Error processing target selection.")
			self.cancel_clone_target_selection()
		finally:
			pass

	def on_accept_clone_clicked(self):
		if not self.pending_cloned_stroke or not self.pending_cloned_stroke.strokes:
			om2.MGlobal.displayWarning("No clone preview available to accept.")
			return

		main_window = self.parent().parent().parent()
		if not (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
			and main_window.sculpt_capture.is_capturing
		):
			om2.MGlobal.displayWarning("Cannot accept clone: Capture is not active.")
			return

		print("SculptingPanel: Accepting cloned stroke...")

		sc = main_window.sculpt_capture
		try:
			sc.apply_cloned_stroke(self.pending_cloned_stroke)

			self.clear_clone_preview_visualization()
			self.pending_cloned_stroke = None
			if hasattr(self, "accept_clone_btn") and isinstance(
				self.accept_clone_btn, QPushButton
			):
				self.accept_clone_btn.setEnabled(False)
			self.source_stroke_for_clone_index = None

		except Exception as e:
			print(f"SculptingPanel: Error accepting clone: {e}")
			om2.MGlobal.displayError("Failed to apply cloned stroke.")
			self.clear_clone_preview_visualization()
			self.pending_cloned_stroke = None
			if hasattr(self, "accept_clone_btn") and isinstance(
				self.accept_clone_btn, QPushButton
			):
				self.accept_clone_btn.setEnabled(False)
			self.source_stroke_for_clone_index = None

	def on_clear_all_clicked(self):
		"""Clears the entire stroke history."""
		print("SculptingPanel: Clearing all history...")
		main_window = self.parent().parent().parent()
		if (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			main_window.sculpt_capture.clear_history()
			main_window.sculpt_capture.current_workflow = Workflow()
			self.update(main_window.sculpt_capture.current_workflow)
			self.sample_list.setRowCount(0)
			self.clear_history_visualization()
			self.clear_clone_preview_visualization()
			self.delete_stroke_btn.setEnabled(False)
			self.clone_stroke_btn.setEnabled(False)
			om2.MGlobal.displayInfo("Stroke history cleared.")  # type: ignore
		else:
			self.update(Workflow())
			self.sample_list.setRowCount(0)
			self.clear_history_visualization()
			self.clear_clone_preview_visualization()
			self.delete_stroke_btn.setEnabled(False)
			self.clone_stroke_btn.setEnabled(False)
			print("SculptingPanel: Cleared UI history (capture inactive).")

	def select(self, index):
		pass

	def cleanup(self):
		"""Clean up resources."""
		print("SculptingPanel cleanup: Disconnecting signals.")

		self.clear_history_visualization()
		self.clear_clone_preview_visualization()
		self._kill_target_selection_scriptJob()

		if self.viz_display_layer and cmds.objExists(self.viz_display_layer):
			try:
				print(
					f"SculptingPanel cleanup: Deleting display layer {self.viz_display_layer}"
				)
				cmds.delete(self.viz_display_layer)
				self.viz_display_layer = None
			except Exception as e:
				print(f"SculptingPanel cleanup: Error deleting display layer: {e}")

		cmds.selectMode(object=True)
		self.mesh_button.clicked.disconnect(self.on_select_mesh)
		self.enable_capture.stateChanged.disconnect(self.on_enable_capture_changed)
		self.stroke_list.itemSelectionChanged.disconnect(
			self.on_stroke_selection_changed
		)
		self.delete_stroke_btn.clicked.disconnect(self.on_delete_stroke)
		self.clone_stroke_btn.clicked.disconnect(self.on_clone_stroke_clicked)
		self.clear_all_btn.clicked.disconnect(self.on_clear_all_clicked)
		self.set_context_btn.clicked.disconnect(self.on_set_context)
		self.clear_context_btn.clicked.disconnect(self.on_clear_context)

		# TODO: Make sure to disconnect other signals here if we connect them later


class SuggestionPanel(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		layout = QVBoxLayout()
		# self.main_window = main_window_ref # Removed

		# Prediction Frame
		prediction_group = QGroupBox("Prediction")
		prediction_layout = QVBoxLayout()

		checkbox_layout = QHBoxLayout()
		self.enable_prediction = QCheckBox("Enable Prediction")
		self.preview_prediction = QCheckBox("Preview Selected Prediction")
		self.enable_prediction.stateChanged.connect(self.on_enable_prediction_changed)

		self.enable_auto_camera = QCheckBox("Enable Auto Camera")
		self.enable_auto_camera.setToolTip(
			"Automatically move camera to view suggestions"
		)
		self.enable_auto_camera.stateChanged.connect(self.on_enable_auto_camera_changed)

		checkbox_layout.addWidget(self.enable_prediction)
		checkbox_layout.addWidget(self.preview_prediction)
		checkbox_layout.addWidget(self.enable_auto_camera)

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
		self.stroke_list.setSelectionBehavior(QAbstractItemView.SelectRows)
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
		self.sample_list.setSelectionBehavior(QAbstractItemView.SelectRows)
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
		self.reject_sel_btn = QPushButton("Rejected Selected")
		self.accept_sel_btn.clicked.connect(self.on_accept_sel_clicked)
		self.accept_all_btn.clicked.connect(self.on_accept_all_clicked)
		self.reject_sel_btn.clicked.connect(self.on_reject_sel_clicked)
		button_layout.addWidget(self.accept_sel_btn)
		button_layout.addWidget(self.accept_all_btn)
		button_layout.addWidget(self.reject_sel_btn)

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
				row_position, 3, QTableWidgetItem(approx(stroke.brush_size))
			)
			self.stroke_list.setItem(
				row_position, 4, QTableWidgetItem(approx(stroke.brush_strength))
			)
		self.workflow = workflow

		if self.stroke_list.rowCount() > 0:
			self.stroke_list.selectRow(0)

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
				row_position, 0, QTableWidgetItem(approx(smp.position))
			)
			self.sample_list.setItem(
				row_position, 1, QTableWidgetItem(approx(smp.normal))
			)
			self.sample_list.setItem(
				row_position, 2, QTableWidgetItem(approx(smp.size))
			)
			self.sample_list.setItem(
				row_position, 3, QTableWidgetItem(approx(smp.pressure))
			)
			self.sample_list.setItem(
				row_position, 4, QTableWidgetItem(approx(smp.timestamp))
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
		is_enabled = state == Qt.CheckState.Checked.value
		# print(f"SuggestionPanel: Enable Prediction changed to: {is_enabled}")
		self.reject_sel_btn.setEnabled(is_enabled)
		self.accept_sel_btn.setEnabled(is_enabled)
		self.accept_all_btn.setEnabled(is_enabled)
		self.enable_auto_camera.setEnabled(is_enabled)

		main_window = self.parent().parent().parent()
		if (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			main_window.sculpt_capture.set_suggestions_enabled(is_enabled)
		elif is_enabled:
			print(
				"SuggestionPanel: Capture instance doesn't exist yet. Suggestions will generate once capture starts."
			)

		self.on_stroke_selection_changed()

	def on_enable_auto_camera_changed(self, state):
		is_enabled = state == Qt.CheckState.Checked.value
		print(f"SuggestionPanel: Enable Auto Camera changed to: {is_enabled}")
		main_window = self.parent().parent().parent()
		if (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			main_window.sculpt_capture.auto_camera_enabled = is_enabled
			if is_enabled and main_window.sculpt_capture.current_suggestions:
				main_window.sculpt_capture._update_auto_camera()
			if not is_enabled:
				main_window.sculpt_capture.restore_previous_camera()
		else:
			print("SuggestionPanel: Capture instance not available for camera control.")

	def on_accept_sel_clicked(self):
		selection_index = self.stroke_list.selectedIndexes()[0].row()
		main_window = self.parent().parent().parent()
		if (
			selection_index != -1
			and main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			print(f"SuggestionPanel: Accepting suggestion index {selection_index}")
			main_window.sculpt_capture.accept_selected_suggestion(selection_index)
		else:
			print("SuggestionPanel: No suggestion selected or capture inactive.")

	def on_accept_all_clicked(self):
		main_window = self.parent().parent().parent()
		if (
			self.workflow
			and len(self.workflow.strokes) > 0
			and main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
		):
			print(
				f"SuggestionPanel: Accepting all {len(self.workflow.strokes)} suggestions."
			)
			main_window.sculpt_capture.accept_all_suggestions()
		else:
			print("SuggestionPanel: No suggestions to accept or capture inactive.")

	def on_reject_sel_clicked(self):
		selected_indexes = self.stroke_list.selectedIndexes()
		if not selected_indexes:
			om2.MGlobal.displayWarning("Please select a suggestion to reject.")  # type: ignore
			return

		selection_index = selected_indexes[0].row()

		main_window = self.parent().parent().parent()
		if (
			main_window
			and isinstance(main_window, AutoSculptorToolWindow)
			and hasattr(main_window, "sculpt_capture")
			and main_window.sculpt_capture
			and main_window.sculpt_capture.suggestions_enabled
		):
			main_window.sculpt_capture.reject_suggestion(selection_index)
			if self.workflow and 0 <= selection_index < len(self.workflow.strokes):
				del self.workflow.strokes[selection_index]
				self.update(self.workflow)
				self.sample_list.setRowCount(0)
			else:
				print(
					f"SuggestionPanel: Error rejecting suggestion, invalid index {selection_index} or workflow missing."
				)
		else:
			om2.MGlobal.displayWarning("Cannot reject: Capture inactive or suggestions disabled.")  # type: ignore

	def cleanup(self):
		"""Clean up resources."""
		print("SuggestionPanel cleanup: Disconnecting signals.")

		self.enable_prediction.stateChanged.disconnect(
			self.on_enable_prediction_changed
		)
		self.stroke_list.itemSelectionChanged.disconnect(
			self.on_stroke_selection_changed
		)
		self.enable_auto_camera.stateChanged.disconnect(
			self.on_enable_auto_camera_changed
		)
		self.accept_sel_btn.clicked.disconnect(self.on_accept_sel_clicked)
		self.accept_all_btn.clicked.disconnect(self.on_accept_all_clicked)
		self.reject_sel_btn.clicked.disconnect(self.on_reject_sel_clicked)

		# TODO: Make sure to disconnect other signals here if we connect them later


class AutoSculptorToolWindow(MayaQWidgetDockableMixin, QDialog):
	def __init__(self, parent=None):
		if cmds.workspaceControl("AutoSculptor", exists=True):
			cmds.deleteUI("AutoSculptor", control=True)
		super().__init__()
		# self.sculpt_capture = None # Moved creation to after tabs are initialized

		self.setWindowTitle("AutoSculptor")
		self.setGeometry(100, 100, 600, 700)

		layout = QVBoxLayout()
		self.tabs = QTabWidget()

		self.sculpting_tab = SculptingPanel(self)
		self.suggestion_tab = SuggestionPanel(self)

		self.tabs.addTab(self.sculpting_tab, "Sculpting History")
		self.tabs.addTab(self.suggestion_tab, "Sculpting Suggestion")

		layout.addWidget(self.tabs)
		self.setLayout(layout)

		# Create sculpt_capture here after tabs are initialized
		self.sculpt_capture = SculptCapture(
			update_history_callback=self.sculpting_tab.update,
			update_suggestion_callback=self.suggestion_tab.update,
		)

		self._setup_shortcuts()

	def _setup_shortcuts(self):
		"""Sets up PySide6 shortcuts for the tool window."""
		main_window = get_maya_main_window()

		self.shortcut_toggle_capture = QShortcut(
			QKeySequence(Qt.ControlModifier | Qt.AltModifier | Qt.Key_X), self
		)
		self.shortcut_toggle_capture.setContext(Qt.ApplicationShortcut)
		self.shortcut_toggle_capture.activated.connect(
			lambda: cmds.autoSculptorAction(toggleCapture=True)
		)

		self.shortcut_toggle_suggestions = QShortcut(
			QKeySequence(Qt.ControlModifier | Qt.AltModifier | Qt.Key_D), self
		)
		self.shortcut_toggle_suggestions.setContext(Qt.ApplicationShortcut)
		self.shortcut_toggle_suggestions.activated.connect(
			lambda: cmds.autoSculptorAction(toggleSuggestions=True)
		)

		self.shortcut_accept_selected = QShortcut(
			QKeySequence(Qt.ControlModifier | Qt.AltModifier | Qt.Key_A), self
		)
		self.shortcut_accept_selected.setContext(Qt.ApplicationShortcut)
		self.shortcut_accept_selected.activated.connect(
			self.suggestion_tab.on_accept_sel_clicked
		)

		self.shortcut_accept_all = QShortcut(
			QKeySequence(
				Qt.ControlModifier | Qt.AltModifier | Qt.ShiftModifier | Qt.Key_A
			),
			self,
		)
		self.shortcut_accept_all.setContext(Qt.ApplicationShortcut)
		self.shortcut_accept_all.activated.connect(
			self.suggestion_tab.on_accept_all_clicked
		)

		self.shortcut_reject_selected = QShortcut(
			QKeySequence(Qt.ControlModifier | Qt.AltModifier | Qt.Key_Z), self
		)
		self.shortcut_reject_selected.setContext(Qt.ApplicationShortcut)
		self.shortcut_reject_selected.activated.connect(
			self.suggestion_tab.on_reject_sel_clicked
		)

		print("AutoSculptorToolWindow: PySide6 shortcuts setup.")

	def closeEvent(self, event):
		"""Override close event for cleanup."""
		if hasattr(self, "sculpting_tab") and self.sculpting_tab:
			self.sculpting_tab.cleanup()
		if hasattr(self, "suggestion_tab") and self.suggestion_tab:
			self.suggestion_tab.cleanup()

		if hasattr(self, "sculpt_capture") and self.sculpt_capture:
			self.sculpt_capture.cleanup()
			self.sculpt_capture = None

		if hasattr(self, "shortcut_toggle_capture") and self.shortcut_toggle_capture:
			self.shortcut_toggle_capture.deleteLater()
		if (
			hasattr(self, "shortcut_toggle_suggestions")
			and self.shortcut_toggle_suggestions
		):
			self.shortcut_toggle_suggestions.deleteLater()
		if hasattr(self, "shortcut_accept_selected") and self.shortcut_accept_selected:
			self.shortcut_accept_selected.deleteLater()
		if hasattr(self, "shortcut_accept_all") and self.shortcut_accept_all:
			self.shortcut_accept_all.deleteLater()
		if hasattr(self, "shortcut_reject_selected") and self.shortcut_reject_selected:
			self.shortcut_reject_selected.deleteLater()

		super().closeEvent(event)
		print("AutoSculptorToolWindow cleaned up.")


def show_sculpting_tool_window():
	global sculpting_tool_window
	try:
		sculpting_tool_window.close()
	except:
		pass
	sculpting_tool_window = AutoSculptorToolWindow()
	sculpting_tool_window.show(dockable=True)
	return sculpting_tool_window
