from autosculptor.core.data_structures import Sample, Stroke, Workflow
import numpy as np
import maya.cmds as cmds  # type: ignore
import maya.api.OpenMaya as om  # type: ignore
import maya.utils
import math


def numpy_to_mvector(np_array):
	return om.MVector(np_array[0], np_array[1], np_array[2])


def compute_tangent(p0, p1):
	return (p1 - p0).normalize()


def compute_frame(prev_normal, tangent):
	# Compute a normal using parallel transport
	binormal = tangent ^ prev_normal  # Cross product
	if binormal.length() < 1e-5:
		binormal = tangent ^ om.MVector(0, 1, 0)
	binormal.normalize()
	normal = binormal ^ tangent
	return normal.normalize(), binormal


class StrokeVisualizer:
	DEFAULT_COLOR = (1.0, 1.0, 0.0)
	DEFAULT_TRANSPARENCY = (0.8, 0.8, 0.8)

	def __init__(self, stroke_or_workflow, color=None, transparency=None):
		"""
		Initialize the StrokeVisualizer with a Stroke object or a Workflow object.

		Args:
			stroke_or_workflow (Union[Stroke, Workflow]): The stroke or workflow to visualize.
			color (tuple, optional): RGB color tuple (e.g., (1.0, 0.0, 0.0) for red). Defaults to yellow.
			transparency (tuple, optional): RGB transparency tuple (e.g., (0.5, 0.5, 0.5)). Defaults to semi-transparent.
		"""
		self.stroke_or_workflow = stroke_or_workflow
		# Track all nodes created by this instance so that we can free up resources later
		self.created_nodes = []

		viz_color = color if color is not None else self.DEFAULT_COLOR
		viz_transparency = (
			transparency if transparency is not None else self.DEFAULT_TRANSPARENCY
		)
		self.shader = None
		self.shading_group = self.create_shader(
			"vizShader", viz_color, viz_transparency
		)

	def create_shader(
		self,
		name="Autosculptor_VizTubeShader",
		color=(1, 0, 0),
		transparency=(0.5, 0.5, 0.5),
	):
		# Create shader
		self.shader = cmds.shadingNode("blinn", asShader=True, name=name)
		self.created_nodes.append(self.shader)
		cmds.setAttr(
			self.shader + ".color", color[0], color[1], color[2], type="double3"
		)
		cmds.setAttr(
			self.shader + ".transparency",
			transparency[0],
			transparency[1],
			transparency[2],
			type="double3",
		)

		# Create shading group
		shading_group = cmds.sets(
			renderable=True, noSurfaceShader=True, empty=True, name=self.shader + "SG"
		)
		self.created_nodes.append(shading_group)
		cmds.connectAttr(
			self.shader + ".outColor", shading_group + ".surfaceShader", force=True
		)

		return shading_group

	def create_tube(self, samples, radius=0.1, segments=8):
		"""Creates the tube mesh geometry and returns its transform node name."""
		verts = []
		poly_counts = []
		poly_connects = []

		normal = om.MVector(0, 1, 0)  # Initial normal guess

		circles = []
		for i in range(len(samples)):
			p = samples[i]
			if i < len(samples) - 1:
				tangent = compute_tangent(p, samples[i + 1])
			else:
				tangent = compute_tangent(samples[i - 1], p)

			normal, binormal = compute_frame(normal, tangent)

			circle = []
			for j in range(segments):
				angle = 2 * math.pi * j / segments
				dir_vec = math.cos(angle) * normal + math.sin(angle) * binormal
				circle.append(p + dir_vec * radius)
				verts.append(p + dir_vec * radius)
			circles.append(circle)

		# Create quads between circles
		for i in range(len(circles) - 1):
			for j in range(segments):
				next_j = (j + 1) % segments
				idx0 = i * segments + j
				idx1 = i * segments + next_j
				idx2 = (i + 1) * segments + next_j
				idx3 = (i + 1) * segments + j

				poly_counts.append(4)
				poly_connects.extend([idx0, idx1, idx2, idx3])

		# Convert to MPointArray
		mpoints = om.MPointArray([om.MPoint(v) for v in verts])

		# Create mesh
		mesh_fn = om.MFnMesh()
		mesh_fn.create(mpoints, poly_counts, poly_connects)

		# Assign shader to the mesh
		mesh_obj = mesh_fn.object()
		dag_path = om.MFnDagNode(mesh_obj).fullPathName()
		self.assign_shader(dag_path)
		self.created_nodes.append(dag_path)

		return dag_path

	def assign_shader(self, mesh_name):
		"""
		Assign a transparent yellow material to the specified geometry.

		Args:
			mesh_name (str): The name of the geometry to assign the material to.
		"""
		# Assign the shader to the geometry
		cmds.sets(mesh_name, edit=True, forceElement=self.shading_group)

	def visualize(self, radius=0.1, sections=8):
		"""
		Visualize the stroke(s) as tube(s) in Maya.

		Args:
			radius (float, optional): The radius of the tube(s). Defaults to 0.1.
			sections (int, optional): The number of sections for the circle profile. Defaults to 8.

		Returns:
			List[str]: A list of names of the created tubular geometries.
		"""
		created_tubes = []
		strokes_to_visualize = []

		if isinstance(self.stroke_or_workflow, Stroke):
			strokes_to_visualize = [self.stroke_or_workflow]
		elif isinstance(self.stroke_or_workflow, Workflow):
			strokes_to_visualize = self.stroke_or_workflow.strokes
		else:
			print("StrokeVisualizer: Invalid input type for visualization.")
			return []

		for stroke in strokes_to_visualize:
			if not stroke or not stroke.samples or len(stroke.samples) < 2:
				print(
					"StrokeVisualizer: Skipping visualization for empty or invalid stroke."
				)
				continue

			positions = [numpy_to_mvector(sample.position) for sample in stroke.samples]

			try:
				tube = self.create_tube(positions, radius, sections)
				if tube:
					created_tubes.append(tube)
			except Exception as e:
				print(f"StrokeVisualizer: Error creating tube for stroke: {e}")
				import traceback

				traceback.print_exc()

		return created_tubes

	def clear(self):
		"""
		Deletes the geometry and shaders created by this visualizer instance.
		"""
		print(f"StrokeVisualizer: Clearing nodes for instance {id(self)}...")
		nodes_to_delete = []
		for node_name in self.created_nodes:
			if node_name and cmds.objExists(node_name):
				nodes_to_delete.append(node_name)

		if nodes_to_delete:
			try:
				cmds.undoInfo(state=False)  # disable undo here
				cmds.delete(nodes_to_delete)
				cmds.undoInfo(state=True)
			except Exception as e:
				print(f"StrokeVisualizer: Error deleting nodes: {e}")
				cmds.undoInfo(state=True)  # re-enable undo if some error occurs

		self.created_nodes = []
		print(f"StrokeVisualizer: Clearing complete for instance {id(self)}.")
