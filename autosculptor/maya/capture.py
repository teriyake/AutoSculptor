from autosculptor.core.data_structures import Sample, Stroke, Workflow
import numpy as np
import maya.cmds as cmds  # type: ignore
import maya.api.OpenMaya as om  # type: ignore
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

	def __init__(self, stroke, color=None, transparency=None):
		"""
		Initialize the StrokeVisualizer with a Stroke object.

		Args:
			stroke (Stroke): The stroke to visualize.
			color (tuple, optional): RGB color tuple (e.g., (1.0, 0.0, 0.0) for red). Defaults to yellow.
			transparency (tuple, optional): RGB transparency tuple (e.g., (0.5, 0.5, 0.5)). Defaults to semi-transparent.
		"""
		self.stroke = stroke
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
		self, name="myShader", color=(1, 0, 0), transparency=(0.5, 0.5, 0.5)
	):
		# Create shader
		self.shader = cmds.shadingNode("blinn", asShader=True, name=name)
		self.created_nodes.append(self.shader)  # Track shader node
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
		self.created_nodes.append(shading_group)  # Track shading group
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
		self.created_nodes.append(dag_path)  # Track mesh transform node

		return dag_path  # Return the name (DAG path)

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
		Visualize the stroke as a tube in Maya.

		Args:
			radius (float, optional): The radius of the tube. Defaults to 0.1.
			sections (int, optional): The number of sections for the circle profile. Defaults to 8.

		Returns:
			str: The name of the created tubular geometry.
		"""
		positions = [
			numpy_to_mvector(sample.position) for sample in self.stroke.samples
		]

		if len(positions) < 2:
			return None

		# Store the original selection
		original_selection = cmds.ls(selection=True, long=True) or []

		tube = self.create_tube(positions, radius, sections)

		# Try to restore the selection
		try:
			# Make sure the orignal selection still exists
			valid_original_selection = [
				item for item in original_selection if cmds.objExists(item)
			]
			if valid_original_selection:
				cmds.select(valid_original_selection, replace=True)
			else:
				cmds.select(clear=True)
				if original_selection:
					print(
						"StrokeVisualizer: Original selection no longer exists, clearing selection."
					)
		except Exception as e:
			print(f"StrokeVisualizer: Warning! Failed to restore selection: {e}")
			cmds.select(clear=True)

		return tube  # tube is now the DAG path string

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
				cmds.delete(nodes_to_delete)
			except Exception as e:
				print(f"StrokeVisualizer: Error deleting nodes: {e}")

		self.created_nodes = []
		print(f"StrokeVisualizer: Clearing complete for instance {id(self)}.")
