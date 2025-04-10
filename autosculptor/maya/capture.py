import maya.cmds as cmds  # type: ignore
import maya.api.OpenMaya as om2  # type: ignore
import numpy as np
import time
from typing import Optional, Tuple

from autosculptor.core.data_structures import Sample, Stroke, Workflow
from autosculptor.core.mesh_interface import MeshData, MeshInterface
from autosculptor.analysis.synthesis import StrokeSynthesizer
from autosculptor.analysis.parameterization import StrokeParameterizer
from autosculptor.suggestions.visualization import StrokeVisualizer
from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator
from autosculptor.maya.utils import (
	get_active_camera_details,
	get_active_camera_lookat_vector,
)
from autosculptor.core.brush import BrushMode
from autosculptor.core.surface_brush import SurfaceBrush
from autosculptor.core.freeform_brush import FreeformBrush


class SculptCapture:
	"""
	Class to manage the sculpting capture process.
	"""

	INTERPOLATED_SAMPLES_PER_EVENT = 5
	STROKE_END_TIMEOUT = 0.25
	DEFAULT_CAMERA_DISTANCE_FACTOR = 5.0

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
		self.preview_enabled = False
		self.previewing_suggestion_index: Optional[int] = None

		self.is_user_actively_sculpting = False
		self.active_stroke_in_progress: Optional[Stroke] = None
		self.last_change_time: float = 0.0

		self.clone_preview_visualizer: Optional[StrokeVisualizer] = None

		self.auto_camera_enabled: bool = False
		self.previous_camera_state: Optional[
			Tuple[om2.MPoint, om2.MVector, om2.MVector]
		] = None  # pos, lookat, up

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

	def _ensure_synthesizer_mesh(self, mesh_data: MeshData):
		"""
		Ensures synthesizer exists and its parameterizer has current mesh data.
		"""
		if not self.synthesizer:
			try:
				self.synthesizer = StrokeSynthesizer(mesh_data)
				print("SculptCapture: Initialized StrokeSynthesizer.")
			except Exception as e:
				print(f"SculptCapture: Error initializing synthesizer: {e}")
				self.synthesizer = None
		elif self.synthesizer.parameterizer:
			if not np.array_equal(
				self.synthesizer.parameterizer.mesh_data.vertices, mesh_data.vertices
			):
				self.synthesizer.parameterizer.mesh_data = mesh_data
				try:
					self.synthesizer.parameterizer.geo_calc = CachedGeodesicCalculator(
						mesh_data.vertices, mesh_data.faces
					)
				except Exception as e:
					print(f"SculptCapture: Error updating geo_calc: {e}")
					self.synthesizer.parameterizer.geo_calc = None
		else:
			try:
				self.synthesizer.parameterizer = StrokeParameterizer(mesh_data)
				print(
					"SculptCapture: Initialized missing parameterizer in synthesizer."
				)
			except Exception as e:
				print(f"SculptCapture: Error initializing missing parameterizer: {e}")

	def process_mesh_changes(self):
		"""Processes mesh changes after a potential sculpt operation."""
		if not self.mesh_name:
			return

		current_time = time.time()

		if (
			self.active_stroke_in_progress
			and current_time - self.last_change_time > self.STROKE_END_TIMEOUT
		):

			print(
				f"SculptCapture: Timeout ({current_time - self.last_change_time:.2f}s > {self.STROKE_END_TIMEOUT}s) detected, finalizing previous stroke."
			)

			camera_lookat = get_active_camera_lookat_vector()
			if self.synthesizer and self.synthesizer.parameterizer:
				try:
					self.synthesizer.parameterizer.parameterize_stroke(
						self.active_stroke_in_progress, camera_lookat
					)
				except Exception as param_e:
					print(f"  Warning: Failed to parameterize stroke: {param_e}")

			self.current_workflow.add_stroke(self.active_stroke_in_progress)
			print(
				f"SculptCapture: Finalized stroke with {len(self.active_stroke_in_progress.samples)} samples"
			)

			if self.update_history_callback:
				self.update_history_callback(self.copy_workflow())

			self.active_stroke_in_progress = None

			if self.suggestions_enabled and len(self.current_workflow.strokes) > 2:
				self.generate_suggestions()

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

			moved_indices = []
			displacements = []
			displacement_magnitudes = []

			for i in range(len(current_points)):
				displacement = current_points[i] - previous_points[i]
				magnitude = np.linalg.norm(displacement)
				if magnitude > 1e-5:
					moved_indices.append(i)
					displacements.append(displacement)
					displacement_magnitudes.append(magnitude)

			if not moved_indices:
				return

			print(f"SculptCapture: Detected {len(moved_indices)} moved vertices.")
			self.last_change_time = current_time

			mesh_data = MeshInterface.get_mesh_data(self.mesh_name)
			if not mesh_data:
				self.previous_positions[self.mesh_name] = current_points
				return

			if not self.synthesizer:
				self.synthesizer = StrokeSynthesizer(mesh_data)
				print("SculptCapture: Initialized StrokeSynthesizer.")
			self._ensure_synthesizer_mesh(mesh_data)

			normals = self.get_world_space_normals(self.mesh_name)
			if normals is None:
				return

			current_time = time.time()
			stroke_continued = False

			if self.active_stroke_in_progress:
				current_stroke = self.active_stroke_in_progress
				stroke_continued = True
				print("SculptCapture: Continuing active stroke")
			else:
				tool_name = self.get_active_sculpt_tool()
				if not tool_name:
					return

				brush_params = self.get_brush_size_and_pressure(tool_name)
				if brush_params is None:
					return
				brush_size, brush_pressure = brush_params

				current_stroke = Stroke()
				current_stroke.stroke_type = (
					"surface"  # TODO: Get actual stroke type from tool ctx
				)
				current_stroke.brush_size = brush_size
				current_stroke.brush_strength = brush_pressure
				current_stroke.brush_mode = (
					"ADD"  # TODO: Get actual brush mode from tool ctx
				)
				current_stroke.brush_falloff = "smooth"  # TODO:
				self.active_stroke_in_progress = current_stroke
				stroke_continued = False
				print("SculptCapture: Created new stroke")

			moved_points = current_points[moved_indices]
			min_coords = np.min(moved_points, axis=0)
			max_coords = np.max(moved_points, axis=0)

			bbox_size = max_coords - min_coords
			bbox_diagonal = np.linalg.norm(bbox_size)

			divisions = max(1, int(bbox_diagonal / (brush_size * 0.5)))
			divisions = min(divisions, 10)

			if divisions <= 1:
				centroid = np.mean(moved_points, axis=0)
				closest_point_data = MeshInterface.find_closest_point(
					mesh_data, centroid
				)

				if closest_point_data and closest_point_data[0] is not None:
					sample_position = np.array(closest_point_data[0])
					sample_normal = np.array(closest_point_data[1])

					norm_mag = np.linalg.norm(sample_normal)
					if norm_mag > 1e-6:
						sample_normal /= norm_mag
					else:
						sample_normal = np.array([0.0, 1.0, 0.0])

					sample = Sample(
						position=sample_position,
						normal=sample_normal,
						size=brush_size,
						pressure=brush_pressure,
						timestamp=current_time,
					)
					current_stroke.add_sample(sample)
					print(f"SculptCapture: Added centroid at {sample_position}")
			else:
				from sklearn.decomposition import PCA

				if len(moved_points) >= 2:
					pca = PCA(n_components=1)
					pca.fit(moved_points)

					principal_direction = pca.components_[0]

					projections = np.dot(
						moved_points - np.mean(moved_points, axis=0),
						principal_direction,
					)

					sorted_indices = np.argsort(projections)

					num_samples = min(divisions, len(sorted_indices))

					sample_indices = []
					if num_samples > 1:
						for i in range(num_samples):
							idx = int(
								(i / (num_samples - 1)) * (len(sorted_indices) - 1)
							)
							sample_indices.append(sorted_indices[idx])
					else:
						sample_indices = [sorted_indices[len(sorted_indices) // 2]]

					for idx in sample_indices:
						original_idx = moved_indices[idx]
						point = current_points[original_idx]
						normal = normals[original_idx]

						norm_mag = np.linalg.norm(normal)
						if norm_mag > 1e-6:
							normal = normal / norm_mag

						time_offset = 0.01 * sample_indices.index(idx)
						sample = Sample(
							position=point,
							normal=normal,
							size=brush_size,
							pressure=brush_pressure,
							timestamp=current_time + time_offset,
						)

						is_distinct = True
						for existing_sample in current_stroke.samples:
							distance = np.linalg.norm(existing_sample.position - point)
							if distance < brush_size * 0.2:
								is_distinct = False
								break

						if is_distinct:
							current_stroke.add_sample(sample)
							print(f"SculptCapture: Added sample at {point}")
				else:
					centroid = np.mean(moved_points, axis=0)
					closest_point_data = MeshInterface.find_closest_point(
						mesh_data, centroid
					)

					if closest_point_data and closest_point_data[0] is not None:
						sample_position = np.array(closest_point_data[0])
						sample_normal = np.array(closest_point_data[1])

						norm_mag = np.linalg.norm(sample_normal)
						if norm_mag > 1e-6:
							sample_normal /= norm_mag
						else:
							sample_normal = np.array([0.0, 1.0, 0.0])

						sample = Sample(
							position=sample_position,
							normal=sample_normal,
							size=brush_size,
							pressure=brush_pressure,
							timestamp=current_time,
						)
						current_stroke.add_sample(sample)
						print(
							f"SculptCapture: Added centroid sample at {sample_position}"
						)

			if stroke_continued:
				print(
					f"SculptCapture: Continued active stroke, now {len(current_stroke.samples)} samples"
				)
			else:
				print(
					f"SculptCapture: Started new active stroke with {len(current_stroke.samples)} samples"
				)

			if len(current_stroke.samples) > 50:
				camera_lookat = get_active_camera_lookat_vector()

				self.synthesizer.parameterizer.parameterize_stroke(
					current_stroke, camera_lookat
				)
				self.current_workflow.add_stroke(current_stroke)

				print(
					f"SculptCapture: Finalized stroke with {len(current_stroke.samples)} samples"
				)

				if self.update_history_callback:
					self.update_history_callback(self.copy_workflow())

				self.active_stroke_in_progress = None

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
								suggestion_stroke.samples[0].size * 0.5
								if suggestion_stroke.samples
								else 0.2
							)
							suggestion_viz_tube = visualizer.visualize(viz_radius, 8)
							cmds.select(suggestion_viz_tube)
							disp_layer = cmds.createDisplayLayer()
							cmds.setAttr(f"{disp_layer}.displayType", 2)
							cmds.select(None)

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
		self.previous_camera_state = None
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
				None,
				0.2,
			)
			if suggestion:
				self.current_suggestions = [suggestion]
				self._update_auto_camera()
				print(
					f"SculptCapture: Suggestion generated: {len(suggestion.samples)} samples"
				)
			else:
				self.current_suggestions = []
				self.previous_camera_state = None
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
			self.clear_suggestions()
			self.previous_camera_state = None
			self.mesh_name = None

	def copy_workflow(self):
		new_workflow = Workflow()
		new_workflow.strokes = list(self.current_workflow.strokes)
		new_workflow.region = self.current_workflow.region

		return new_workflow

	def _store_current_camera_state(self):
		"""
		Stores the current active camera's position, target, and up vector.
		"""
		cam_details = get_active_camera_details()
		if not cam_details:
			self.previous_camera_state = None
			return

		cam_transform_name, cam_dag_path = cam_details
		try:
			cam_fn = om2.MFnCamera(cam_dag_path)
			eye_point = cam_fn.eyePoint(om2.MSpace.kWorld)
			view_dir = cam_fn.viewDirection(om2.MSpace.kWorld)
			center_of_interest = eye_point + view_dir * cam_fn.centerOfInterest
			up_dir = cam_fn.upDirection(om2.MSpace.kWorld)

			self.previous_camera_state = (eye_point, center_of_interest, up_dir)
			# print(f"Stored camera state: Pos={eye_point}, Target={center_of_interest}, Up={up_dir}")
		except Exception as e:
			print(f"Error storing camera state: {e}")
			self.previous_camera_state = None

	def _update_auto_camera(self):
		"""Moves the active camera to view the first suggestion."""
		if not self.auto_camera_enabled:
			return
		if not self.current_suggestions:
			return

		suggestion_stroke = self.current_suggestions[0]
		if not suggestion_stroke or not suggestion_stroke.samples:
			print("First suggestion is invalid, cannot move camera.")
			return

		cam_details = get_active_camera_details()
		if not cam_details:
			print("Cannot move camera: No active camera found.")
			return
		cam_transform_name, cam_dag_path = cam_details

		try:
			sample_positions = np.array([s.position for s in suggestion_stroke.samples])
			sample_normals = np.array([s.normal for s in suggestion_stroke.samples])
			sample_sizes = np.array([s.size for s in suggestion_stroke.samples])

			target_point = np.mean(sample_positions, axis=0)
			avg_normal = np.mean(sample_normals, axis=0)
			avg_size = np.mean(sample_sizes) if len(sample_sizes) > 0 else 1.0

			norm_mag = np.linalg.norm(avg_normal)
			if norm_mag < 1e-6:
				closest_pt_data = MeshInterface.find_closest_point(
					MeshInterface.get_mesh_data(self.mesh_name), target_point
				)
				if closest_pt_data and closest_pt_data[1] is not None:
					avg_normal = np.array(closest_pt_data[1])
				else:
					avg_normal = np.array([0.0, 1.0, 0.0])
			else:
				avg_normal /= norm_mag

			distance = max(avg_size, 1.0) * self.DEFAULT_CAMERA_DISTANCE_FACTOR
			cam_pos = target_point + avg_normal * distance

			world_up = om2.MVector(0.0, 1.0, 0.0)

			if self.previous_camera_state is None:
				self._store_current_camera_state()

			print(f"Auto-moving camera to view target {target_point} from ~{cam_pos}")
			cmds.viewLookAt(cam_dag_path, pos=list(target_point))
			cmds.refresh(force=True)

		except Exception as e:
			print(f"Error during automatic camera update: {e}")
			import traceback

			traceback.print_exc()

	def restore_previous_camera(self):
		"""Restores the camera to the state before the last automatic move."""
		if self.previous_camera_state is None:
			print("No previous camera state stored to restore.")
			return

		cam_details = get_active_camera_details()
		if not cam_details:
			print("Cannot restore camera: No active camera found.")
			return
		cam_transform_name, cam_dag_path = cam_details

		try:
			print("Restoring previous camera view...")
			prev_pos, prev_target, prev_up = self.previous_camera_state

			cmds.viewLookAt(cam_dag_path, pos=list(prev_pos))
			cmds.refresh(force=True)

			self.previous_camera_state = None
		except Exception as e:
			print(f"Error restoring previous camera state: {e}")

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
				self.previous_camera_state = None
				self.clear_clone_preview_visualization()
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

	def clear_suggestion_visualizers(self):
		if self.suggestion_visualizers:
			for viz in self.suggestion_visualizers:
				viz.clear()
			self.suggestion_visualizers.clear()

	def clear_clone_preview_visualization(self):
		if self.clone_preview_visualizer:
			self.clone_preview_visualizer.clear()

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

	def visualize_suggestions(self):
		"""Creates visualizers for the current suggestions."""
		self.clear_suggestion_visualizers()
		if not self.mesh_name or not cmds.objExists(self.mesh_name):
			print("Cannot visualize suggestions, mesh not valid.")
			return

		if self.current_suggestions:
			print(
				f"Attempting to visualize {len(self.current_suggestions)} suggestions..."
			)
			for i, stroke in enumerate(self.current_suggestions):
				try:
					viz = StrokeVisualizer(self.mesh_name, f"suggestion_viz_{i}")
					viz.set_stroke(stroke)
					viz.show()
					self.suggestion_visualizers.append(viz)
					print(f"Visualizer created for suggestion {i}")
				except Exception as e:
					print(f"Error creating visualizer for suggestion {i}: {e}")
					import traceback

					traceback.print_exc()

	def set_suggestions_enabled(self, enabled: bool):
		self.suggestions_enabled = enabled
		if not enabled:
			self.clear_suggestions()
			self.auto_camera_enabled = False
		elif enabled and self.is_capturing and len(self.current_workflow.strokes) > 1:
			self.generate_suggestions()
			self.visualize_suggestions()

	def accept_selected_suggestion(self, suggestion_index: int):
		"""Applies the selected suggestion stroke to the mesh."""
		if not self.is_capturing:
			om2.MGlobal.displayWarning(
				"Cannot accept suggestion: Capture is not active."
			)
			return
		if not self.mesh_name or not cmds.objExists(self.mesh_name):
			om2.MGlobal.displayError("Cannot accept suggestion: Target mesh not found.")
			return
		if suggestion_index < 0 or suggestion_index >= len(self.current_suggestions):
			om2.MGlobal.displayWarning(f"Invalid suggestion index: {suggestion_index}")
			return

		accepted_stroke = self.current_suggestions[suggestion_index]
		if not accepted_stroke or not accepted_stroke.samples:
			om2.MGlobal.displayWarning("Selected suggestion is empty or invalid.")
			return

		# print(f"SculptCapture: Accepting suggestion index {suggestion_index}...")

		try:
			mesh_data = MeshInterface.get_mesh_data(self.mesh_name)
			if not mesh_data:
				om2.MGlobal.displayError("Failed to get mesh data for applying stroke.")
				return

			cmds.undoInfo(
				openChunk=True, chunkName="AcceptSelectedAutoSculptorSuggestion"
			)
			undo_chunk_is_open = True

			brush_class = None
			if accepted_stroke.stroke_type == "surface":
				brush_class = SurfaceBrush
			elif accepted_stroke.stroke_type == "freeform":
				brush_class = FreeformBrush
			else:
				if undo_chunk_is_open:
					cmds.undoInfo(closeChunk=True)
					undo_chunk_is_open = False
				raise ValueError(
					f"Unknown stroke type for accepted suggestion: {accepted_stroke.stroke_type}"
				)

			brush = brush_class(
				size=accepted_stroke.brush_size or 1.0,
				strength=accepted_stroke.brush_strength or 0.5,
				mode=BrushMode[accepted_stroke.brush_mode]
				if accepted_stroke.brush_mode
				else BrushMode.ADD,
				falloff="linear",
			)
			print(
				f"  Applying using Brush: {brush_class.__name__}, Size: {brush.size:.2f}, Strength: {brush.strength:.2f}, Mode: {brush.mode.name}"
			)

			apply_success = True
			for i, sample in enumerate(accepted_stroke.samples):
				print(
					f"  Applying sample {i+1}/{len(accepted_stroke.samples)} at {sample.position}"
				)
				try:
					brush.apply_to_mesh(mesh_data, sample)
				except Exception as apply_err:
					print(f"  Error applying sample {i}: {apply_err}")
					apply_success = False
					break

			if not apply_success:
				cmds.undoInfo(closeChunk=True)
				cmds.undo()
				om2.MGlobal.displayError(
					"Failed to apply accepted stroke. Operation cancelled and undone."
				)
				return

			print("  Stroke application complete.")

			final_mesh_data = MeshInterface.get_mesh_data(self.mesh_name)
			if final_mesh_data and self.synthesizer and self.synthesizer.parameterizer:
				self._ensure_synthesizer_mesh(final_mesh_data)
				camera_lookat = get_active_camera_lookat_vector()
				print("  Parameterizing accepted stroke...")
				self.synthesizer.parameterizer.parameterize_stroke(
					accepted_stroke, camera_lookat
				)
			else:
				print("  Warning: Could not re-parameterize accepted stroke.")

			self.current_workflow.add_stroke(accepted_stroke)
			self.previous_positions[self.mesh_name] = (
				final_mesh_data.vertices if final_mesh_data else mesh_data.vertices
			)

			self.current_suggestions.pop(suggestion_index)
			self.clear_suggestion_visualizers()

			if self.update_history_callback:
				self.update_history_callback(self.copy_workflow())
			if self.update_suggestion_callback:
				suggestion_workflow = Workflow()
				suggestion_workflow.strokes = list(self.current_suggestions)
				self.update_suggestion_callback(suggestion_workflow)

			# self.generate_suggestions()

			if undo_chunk_is_open:
				cmds.undoInfo(closeChunk=True)
				undo_chunk_is_open = False

		except Exception as e:
			print(f"Critical Error during accept_suggestion: {e}")
			import traceback

			traceback.print_exc()
			if undo_chunk_is_open:
				try:
					cmds.undoInfo(closeChunk=True)
					undo_chunk_is_open = False
					print("Attempting undo due to critical error...")
					if not cmds.undoInfo(q=True, undoQueueEmpty=True):
						cmds.undo()
				except Exception as final_close_err:
					print(f"Error closing undo chunk: {final_close_err}")

	def accept_all_suggestions(self):
		if not self.is_capturing:
			om2.MGlobal.displayWarning(
				"Cannot accept suggestions: Capture is not active."
			)
			return
		if not self.current_suggestions:
			om2.MGlobal.displayInfo("No suggestions available to accept.")
			return

		num_to_accept = len(self.current_suggestions)
		print(f"SculptCapture: Accepting all {num_to_accept} suggestions...")

		cmds.undoInfo(openChunk=True, chunkName="AcceptAllAutoSculptorSuggestions")
		try:
			accepted_count = 0
			original_indices = list(range(num_to_accept))

			for i in range(num_to_accept):
				self.accept_suggestion(0)
				if not self.mesh_name or not cmds.objExists(self.mesh_name):
					print("  Mesh became invalid during 'Accept All'. Aborting.")
					break
				accepted_count += 1

			print(f"SculptCapture: Finished accepting {accepted_count} suggestions.")

		except Exception as e:
			print(f"Error during accept_all_suggestions: {e}")
			import traceback

			traceback.print_exc()
		finally:
			if cmds.undoInfo(q=True, open=True):
				cmds.undoInfo(closeChunk=True)

	def apply_cloned_stroke(self, cloned_stroke: Stroke):
		"""
		Applies a validated cloned stroke to the mesh and adds it to history.
		"""
		if not self.is_capturing:
			raise RuntimeError("Cannot apply clone: Capture is not active.")
		if not self.mesh_name or not cmds.objExists(self.mesh_name):
			raise RuntimeError("Cannot apply clone: Target mesh not found.")
		if not cloned_stroke or not cloned_stroke.samples:
			raise ValueError("Cannot apply clone: Cloned stroke is invalid or empty.")

		print(
			f"SculptCapture: Applying cloned stroke with {len(cloned_stroke.samples)} samples..."
		)

		try:
			mesh_data = MeshInterface.get_mesh_data(self.mesh_name)
			if not mesh_data:
				raise RuntimeError(
					"Failed to get mesh data for applying cloned stroke."
				)

			cmds.undoInfo(openChunk=True, chunkName="ApplyAutoSculptorClone")
			undo_chunk_is_open = True

			brush_class = None
			if cloned_stroke.stroke_type == "surface":
				brush_class = SurfaceBrush
			elif cloned_stroke.stroke_type == "freeform":
				brush_class = FreeformBrush
			else:
				if undo_chunk_is_open:
					cmds.undoInfo(closeChunk=True)
				raise ValueError(
					f"Unknown stroke type for cloned stroke: {cloned_stroke.stroke_type}"
				)

			brush = brush_class(
				size=cloned_stroke.brush_size or 1.0,
				strength=cloned_stroke.brush_strength or 0.5,
				mode=BrushMode[cloned_stroke.brush_mode]
				if cloned_stroke.brush_mode
				else BrushMode.ADD,
				falloff=cloned_stroke.brush_falloff or "smooth",
			)
			# print(
			# f"  Applying using Brush: {brush_class.__name__}, Size: {brush.size:.2f}, Strength: {brush.strength:.2f}, Mode: {brush.mode.name}, Falloff: {brush.falloff}"
			# )

			apply_success = True
			num_applied = 0
			for i, sample in enumerate(cloned_stroke.samples):
				# print(f"  Applying sample {i+1}/{len(cloned_stroke.samples)} at {sample.position}") # Verbose
				try:
					brush.apply_to_mesh(mesh_data, sample)
					num_applied += 1
				except Exception as apply_err:
					print(f"  Error applying cloned sample {i}: {apply_err}")
					# apply_success = False
					# break
					print("   Skipping sample and attempting to continue...")

			print(f"  Applied {num_applied} samples from cloned stroke.")

			final_mesh_data = MeshInterface.get_mesh_data(self.mesh_name)
			if final_mesh_data and self.synthesizer and self.synthesizer.parameterizer:
				self._ensure_synthesizer_mesh(final_mesh_data)
				camera_lookat = get_active_camera_lookat_vector()
				# print("  Parameterizing applied clone stroke...")
				try:
					self.synthesizer.parameterizer.parameterize_stroke(
						cloned_stroke, camera_lookat
					)
				except Exception as param_err:
					print(
						f"  Warning: Failed to parameterize applied clone stroke: {param_err}"
					)
			else:
				print(
					"  Warning: Could not re-parameterize applied clone stroke (missing synthesizer/parameterizer)."
				)

			self.current_workflow.add_stroke(cloned_stroke)
			self.previous_positions[self.mesh_name] = (
				final_mesh_data.vertices if final_mesh_data else mesh_data.vertices
			)

			if self.update_history_callback:
				self.update_history_callback(self.copy_workflow())

			if self.suggestions_enabled:
				# print("  Regenerating suggestions after clone application...")
				self.generate_suggestions()

			if undo_chunk_is_open:
				cmds.undoInfo(closeChunk=True)
				undo_chunk_is_open = False
			print("SculptCapture: Clone application successful.")

		except Exception as e:
			print(f"SculptCapture: Critical Error during apply_cloned_stroke: {e}")
			import traceback

			traceback.print_exc()
			if undo_chunk_is_open:
				try:
					cmds.undoInfo(closeChunk=True)
					print("Attempting undo due to critical error...")
					if not cmds.undoInfo(q=True, undoQueueEmpty=True):
						cmds.undo()
				except Exception as final_close_err:
					print(f"Error closing undo chunk: {final_close_err}")
			raise

	def cleanup(self):
		self.stop_capture()
		self.clear_suggestions()
		self.clear_clone_preview_visualization()
		self.synthesizer = None
		self.previous_camera_state = None
		print("SculptCapture cleaned up.")
