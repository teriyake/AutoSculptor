import maya.cmds as cmds  # type: ignore
import maya.api.OpenMaya as om2  # type: ignore
import numpy as np
import time
from typing import Optional

from autosculptor.core.data_structures import Sample, Stroke, Workflow
from autosculptor.core.mesh_interface import MeshInterface
from autosculptor.analysis.synthesis import StrokeSynthesizer
from autosculptor.analysis.parameterization import StrokeParameterizer
from autosculptor.suggestions.visualization import StrokeVisualizer
from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator
from autosculptor.maya.utils import get_active_camera_lookat_vector


class SculptCapture:
	"""
	Class to manage the sculpting capture process.
	"""

	INTERPOLATED_SAMPLES_PER_EVENT = 5
	STROKE_END_TIMEOUT = 0.25

	def __init__(self, update_history_callback=None, update_suggestion_callback=None):
		self.previous_positions = {}
		self.current_workflow = Workflow()
		self.script_job_number = None
		self.mesh_name = None
		self.synthesizer = None
		self.current_suggestions = []
		self.suggestion_visualizers = []
		self.update_history_callback = update_history_callback
		self.update_suggestion_callback = update_suggestion_callback
		self.is_capturing = False
		self.suggestions_enabled = False

		self.is_user_actively_sculpting = False
		self.active_stroke_in_progress: Optional[Stroke] = None
		self.last_change_time: float = 0.0

	def get_world_space_positions(self, mesh_name):
		"""Gets world-space vertex positions for a mesh by name."""
		try:
			selection_list = om2.MSelectionList()
			selection_list.add(mesh_name)
			dag_path = selection_list.getDagPath(0)
			mesh_fn = om2.MFnMesh(dag_path)
			points = mesh_fn.getPoints(om2.MSpace.kWorld)
			return np.array([[p.x, p.y, p.z] for p in points])
		except (RuntimeError, TypeError, ValueError) as e:
			return None

	def get_world_space_normals(self, mesh_name):
		"""Gets world-space normals for a mesh by name."""
		try:
			selection_list = om2.MSelectionList()
			selection_list.add(mesh_name)
			dag_path = selection_list.getDagPath(0)
			mesh_fn = om2.MFnMesh(dag_path)
			normals = mesh_fn.getNormals(om2.MSpace.kWorld)
			return np.array([[n.x, n.y, n.z] for n in normals])
		except (RuntimeError, TypeError, ValueError) as e:
			return None

	def get_active_sculpt_tool(self):
		"""
		Gets the name of the active sculpting tool context.
		Returns the context name (string) or None.
		"""
		context = cmds.currentCtx()

		if "sculptMesh" in context:
			return context
		return None

	def get_brush_size_and_pressure(self, tool_name):
		"""Gets the brush size and pressure using sculptMeshCacheCtx."""
		if not tool_name:
			return None

		try:
			size = cmds.sculptMeshCacheCtx(tool_name, q=True, sz=True)
			strength = cmds.sculptMeshCacheCtx(tool_name, q=True, st=True)

			size = size if size is not None else 1.0
			strength = strength if strength is not None else 1.0

			strength /= 100
			return size, strength

		except Exception as e:
			print(f"Failed to get value from tool: {e}")
			return 1.0, 1.0

	def process_mesh_changes(self):
		"""Processes mesh changes after a potential sculpt operation."""

		if not self.mesh_name:
			return
		if not self.synthesizer is None:
			self.synthesizer.parameterizer.mesh_data = MeshInterface.get_mesh_data(
				self.mesh_name
			)

		tool_name = self.get_active_sculpt_tool()
		if not tool_name:
			return

		brush_params = self.get_brush_size_and_pressure(tool_name)
		if brush_params is None:
			return
		brush_size, brush_pressure = brush_params

		try:
			current_points = self.get_world_space_positions(self.mesh_name)
			if current_points is None:
				return

			if self.mesh_name not in self.previous_positions:
				self.previous_positions[self.mesh_name] = current_points
				return

			previous_points = self.previous_positions[self.mesh_name]
			if len(previous_points) != len(current_points):
				self.previous_positions[self.mesh_name] = current_points
				return

			current_stroke = Stroke()
			current_stroke.stroke_type = "surface"
			current_time = cmds.currentTime(query=True)

			normals = self.get_world_space_normals(self.mesh_name)
			if normals is None:
				return

			# TODO: We may want to do something else here (i.e., do not use moved vertices as samples points of the captured stroke)
			moved_indices = [
				i
				for i in range(len(current_points))
				if np.linalg.norm(current_points[i] - previous_points[i]) > 1e-5
			]
			if not moved_indices:
				return
			print(f"SculptCapture: Detected {len(moved_indices)} moved vertices.")

			"""
			moved_points_current = current_points[moved_indices]
			centroid = np.mean(moved_points_current, axis=0)

			mesh_data = MeshInterface.get_mesh_data(self.mesh_name)
			closest_point_data = MeshInterface.find_closest_point(mesh_data, centroid)

			if closest_point_data and closest_point_data[0] is not None:
				sample_position = np.array(closest_point_data[0])
				sample_normal = np.array(closest_point_data[1])
				norm_mag = np.linalg.norm(sample_normal)
				if norm_mag > 1e-6:
					sample_normal /= norm_mag
				else:
					sample_normal = np.array([0.0, 1.0, 0.0])

				current_time = time.time()

				representative_sample = Sample(
					position=sample_position,
					normal=sample_normal,
					size=brush_size,
					pressure=brush_pressure,
					timestamp=current_time,
				)

				current_stroke = Stroke()
				current_stroke.stroke_type = "surface"
				current_stroke.brush_size = brush_size
				current_stroke.brush_strength = brush_pressure
				# TODO: Determine brush mode (Add/Subtract/Smooth) if possible from context
				current_stroke.brush_mode = "ADD"
				current_stroke.brush_falloff = "smooth"

				current_stroke.add_sample(representative_sample)
				print(
					f"SculptCapture: Created representative sample at {sample_position}"
				)

			else:
				print(
					"SculptCapture: Could not find closest point for centroid. Skipping stroke."
				)
				self.previous_positions[self.mesh_name] = current_points
				if self.synthesizer and self.synthesizer.parameterizer:
					self.synthesizer.parameterizer.mesh_data = mesh_data
					self.synthesizer.parameterizer.geo_calc = CachedGeodesicCalculator(
						mesh_data.vertices, mesh_data.faces
					)
				return

			"""

			samples = [
				Sample(
					position=current_points[i],
					normal=normals[i],
					size=brush_size,
					pressure=brush_pressure,
					timestamp=current_time,
				)
				for i in moved_indices
			]
			for s in samples:
				current_stroke.add_sample(s)

			if len(current_stroke) > 0:
				mesh_data = MeshInterface.get_mesh_data(self.mesh_name)
				if not mesh_data:
					self.previous_positions[self.mesh_name] = current_points
					return

				if not self.synthesizer:
					self.synthesizer = StrokeSynthesizer(mesh_data)
				elif self.synthesizer.parameterizer:
					self.synthesizer.parameterizer.mesh_data = mesh_data
					self.synthesizer.parameterizer.geo_calc = CachedGeodesicCalculator(
						mesh_data.vertices, mesh_data.faces
					)
				else:
					self.synthesizer.parameterizer = StrokeParameterizer(mesh_data)

				camera_lookat = get_active_camera_lookat_vector()

				self.synthesizer.parameterizer.parameterize_stroke(
					current_stroke, camera_lookat
				)
				self.current_workflow.add_stroke(current_stroke)

				if self.update_history_callback:
					self.update_history_callback(self.copy_workflow())
				else:
					print("SculptCapture: update_history_callback does not exist!!!")

				if self.suggestions_enabled and len(self.current_workflow.strokes) > 1:
					print("SculptCapture: Generating suggestions...")
					self.generate_suggestions()
				elif self.suggestions_enabled:
					print(
						"SculptCapture: Suggestions enabled, but not enough history yet."
					)
				else:
					self.clear_suggestions()

			self.previous_positions[self.mesh_name] = current_points

			for viz in self.suggestion_visualizers:
				viz.clear()
			self.suggestion_visualizers.clear()

			if self.suggestions_enabled and self.current_suggestions:
				print(f"Visualizing {len(self.current_suggestions)} suggestions.")
				for suggestion_stroke in self.current_suggestions:
					if suggestion_stroke and len(suggestion_stroke.samples) > 0:
						try:
							visualizer = StrokeVisualizer(suggestion_stroke)
							viz_radius = (
								suggestion_stroke.samples[0].size
								if suggestion_stroke.samples
								else 0.1
							)
							visualizer.visualize(0.2, 8)
							self.suggestion_visualizers.append(visualizer)
						except Exception as viz_e:
							print(
								f"SculptCapture: Error visualizing suggestion: {viz_e}"
							)

		except Exception as e:
			print(f"Error in process_mesh_changes: {e}")
			import traceback

			traceback.print_exc()
			if self.mesh_name in self.previous_positions:
				self.previous_positions[
					self.mesh_name
				] = self.get_world_space_positions(self.mesh_name)

	def get_selected_mesh_name(self):
		"""Gets the full path of the selected mesh shape node."""
		selected_objects = cmds.ls(selection=True, long=True)
		if not selected_objects:
			om2.MGlobal.displayError("No object selected")
			return None

		try:
			shapes = cmds.listRelatives(selected_objects[0], shapes=True, fullPath=True)
			if not shapes:
				om2.MGlobal.displayError("Selected object has no shape nodes")
				return None

			if cmds.objectType(shapes[0]) == "mesh":
				return shapes[0]
			else:
				om2.MGlobal.displayError("Selected object is not a mesh")
				return None
		except Exception as e:
			om2.MGlobal.displayError(f"Error while selecting a mesh: {e}")
			return None

	def generate_suggestions(self):
		"""Generate autocomplete suggestions using StrokeSynthesizer"""
		if not self.mesh_name or not cmds.objExists(self.mesh_name):
			print("SculptCapture: Mesh no longer exists, clearing reference")
			self.mesh_name = None
			return

		try:
			mesh_data = MeshInterface.get_mesh_data(self.mesh_name)
			if not mesh_data:
				return

			if not self.synthesizer:
				self.synthesizer = StrokeSynthesizer(mesh_data)

			suggestion = self.synthesizer.synthesize_next_stroke(
				self.current_workflow,
				MeshInterface.get_mesh_data(self.mesh_name),
				5,
				100.0,
			)
			if suggestion:
				self.current_suggestions = [suggestion]
				print(
					f"SculptCapture: Suggestion generated: {len(suggestion.samples)} samples"
				)
			else:
				self.current_suggestions = []
				print("SculptCapture: No suitable suggestion found.")

			if self.update_suggestion_callback:
				suggestion_workflow = Workflow()
				for s in self.current_suggestions:
					suggestion_workflow.add_stroke(s)
				self.update_suggestion_callback(suggestion_workflow)

		except Exception as e:
			print(f"SculptCapture: Error in generate_suggestions: {str(e)}")
			import traceback

			traceback.print_exc()
			self.mesh_name = None

	def copy_workflow(self):
		new_workflow = Workflow()
		new_workflow.strokes = list(self.current_workflow.strokes)
		new_workflow.region = self.current_workflow.region

		return new_workflow

	def start_capture(self):
		if self.script_job_number is not None:
			return

		if not self.mesh_name:
			print("Cannot start capture: No mesh selected.")

			selected = self.get_selected_mesh_name()
			if not selected:
				om2.MGlobal.displayError(
					"Please select a mesh before starting capture."
				)
				return
			self.mesh_name = selected

		self.previous_positions[self.mesh_name] = self.get_world_space_positions(
			self.mesh_name
		)

		self.script_job_number = cmds.scriptJob(
			event=["idle", self.process_mesh_changes],
			killWithScene=True,
		)
		self.is_capturing = True
		print(
			f"SculptCapture: Capture started. Registered script job: {self.script_job_number} for mesh: {self.mesh_name}"
		)

	def stop_capture(self):
		if self.script_job_number is not None:
			try:
				cmds.scriptJob(kill=self.script_job_number, force=True)
				print(f"Killed script job {self.script_job_number}")
			except Exception as e:
				print(f"Error killing script job {self.script_job_number}: {e}")
			finally:
				self.script_job_number = None
				self.is_capturing = False

				if self.mesh_name in self.previous_positions:
					del self.previous_positions[self.mesh_name]
				self.clear_suggestions()
				print("Capture stopped.")
		else:
			print("Capture not running.")

	def register_script_job(self):
		"""Registers a script job to monitor for changes after sculpting."""
		if self.script_job_number is None:
			self.mesh_name = self.get_selected_mesh_name()
			if self.mesh_name:
				self.script_job_number = cmds.scriptJob(
					event=["idle", self.process_mesh_changes],
					killWithScene=True,
				)
				print(
					f"Registered script job: {self.script_job_number} for mesh: {self.mesh_name}"
				)
			else:
				print("No mesh selected.")
		else:
			print("Script job already registered.")

	def unregister_script_job(self):
		"""Unregisters the script job."""
		if self.script_job_number is not None:
			cmds.scriptJob(kill=self.script_job_number, force=True)
			self.script_job_number = None
			self.mesh_name = None
			print("Unregistered script job")
		else:
			print("No script job to unregister.")

	def clear_suggestions(self):
		"""Clears the current suggestion strokes and updates the UI."""
		if self.current_suggestions:
			print("SculptCapture: Clearing suggestions.")
		self.current_suggestions = []

		if self.update_suggestion_callback:
			self.update_suggestion_callback(Workflow())

		for viz in self.suggestion_visualizers:
			viz.clear()
		self.suggestion_visualizers.clear()

	def delete_stroke(self, stroke_index):
		if 0 <= stroke_index < len(self.current_workflow):
			del self.current_workflow.strokes[stroke_index]
			print(f"SculptCapture: Deleted stroke at index {stroke_index}")

			if self.update_history_callback:
				self.update_history_callback(self.copy_workflow())

			if self.suggestions_enabled:
				print("SculptCapture: Regenerating suggestions after deletion...")
				self.generate_suggestions()
			else:
				self.clear_suggestions()
		else:
			print(f"WARNING: Invalid stroke index for deletion: {stroke_index}")

	def set_suggestions_enabled(self, enabled: bool):
		self.suggestions_enabled = enabled
		if not enabled:
			self.clear_suggestions()

			for viz in self.suggestion_visualizers:
				viz.clear()
			self.suggestion_visualizers.clear()
		elif enabled and self.is_capturing and len(self.current_workflow.strokes) > 0:
			self.generate_suggestions()

	def cleanup(self):
		self.stop_capture()
		self.clear_suggestions()
		print("SculptCapture cleaned up.")
