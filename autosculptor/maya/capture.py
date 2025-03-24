import maya.cmds as cmds
import maya.api.OpenMaya as om2
import numpy as np

from autosculptor.core.data_structures import Sample, Stroke, Workflow
from autosculptor.core.mesh_interface import MeshInterface
from autosculptor.analysis.synthesis import StrokeSynthesizer
from autosculptor.analysis.parameterization import StrokeParameterizer
from autosculptor.maya.viewport_drawer import ViewportDataCache


class SculptCapture:
	"""
	Class to manage the sculpting capture process.
	"""

	def __init__(self):
		self.previous_positions = {}  # {str: np.ndarray}
		self.current_workflow = Workflow()
		self.script_job_number = None
		self.mesh_name = None
		self.synthesizer = None
		self.current_suggestions = []
		print("SculptCapture initialized.")

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
			# size = cmds.artAttrCtx(tool_name, q=True, r=True)
			# strength = cmds.artAttrCtx(tool_name, q=True, st=True)
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

			moved_indices = [
				i
				for i in range(len(current_points))
				if np.linalg.norm(current_points[i] - previous_points[i]) > 1e-5
			]
			if not moved_indices:
				return

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
					return

				parameterizer = StrokeParameterizer(mesh_data)
				parameterizer.parameterize_stroke(current_stroke, [0, 0, 1])
				self.current_workflow.add_stroke(current_stroke)

			self.previous_positions[self.mesh_name] = current_points

			if len(self.current_workflow.strokes) > 1:
				self.generate_suggestions()

		except Exception as e:
			print(f"Error in process_mesh_changes: {e}")
			import traceback

			traceback.print_exc()

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
			print("Mesh no longer exists, clearing reference")
			self.mesh_name = None
			return

		try:
			mesh_data = MeshInterface.get_mesh_data(self.mesh_name)
			if not mesh_data:
				return

			selection_list = om2.MSelectionList()
			selection_list.add(self.mesh_name)
			dag_path = selection_list.getDagPath(0)

			transform_mobj = dag_path.transform()
			transform_fn = om2.MFnTransform(transform_mobj)

			matrix = transform_fn.transformation().asMatrix()
			ViewportDataCache().set_mesh_transform(matrix)

			if not self.synthesizer:
				self.synthesizer = StrokeSynthesizer(mesh_data)

			self.current_suggestions = self.synthesizer.synthesize_next_stroke(
				self.current_workflow, 1.0, 1.0, 10
			)

			ViewportDataCache().update_suggestions(self.current_suggestions)

			from maya.api.OpenMayaUI import M3dView

			view = M3dView.active3dView()
			view.refresh(True, True)
			# cmds.refresh(currentView=True, force=False)

		except Exception as e:
			print(f"Error in generate_suggestions: {str(e)}")
			import traceback

			traceback.print_exc()
			self.mesh_name = None

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
