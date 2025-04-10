from autosculptor.core.data_structures import Sample, Stroke, Workflow
import numpy as np
import maya.cmds as cmds  # type: ignore


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

		# Create a new shader for visualization
		self.shader = cmds.shadingNode(
			"lambert", asShader=True, name="transparentYellowShader"
		)
		self.shading_group = cmds.sets(
			renderable=True, noSurfaceShader=True, empty=True, name=self.shader + "SG"
		)
		# SG needs to be cleaned up
		self.created_nodes.append(self.shading_group)
		cmds.connectAttr(
			self.shader + ".outColor", self.shading_group + ".surfaceShader"
		)

		# Set the color to yellow
		cmds.setAttr(
			self.shader + ".color",
			viz_color[0],
			viz_color[1],
			viz_color[2],
			type="double3",
		)

		# Set transparency to make the material semi-transparent
		cmds.setAttr(
			self.shader + ".transparency",
			viz_transparency[0],
			viz_transparency[1],
			viz_transparency[2],
			type="double3",
		)

	def create_curve_from_stroke(self):
		"""
		Create a NURBS curve in Maya from the stroke samples.

		Returns:
			str: The name of the created NURBS curve.
		"""
		# Extract positions from stroke samples
		positions = [sample.position for sample in self.stroke.samples]

		# Flatten the list of positions
		curve_points = [
			tuple(map(float, sample.position)) for sample in self.stroke.samples
		]

		# Create the NURBS curve
		if len(curve_points) < 4:
			curve = cmds.curve(p=curve_points, d=1)  # Linear curve for fewer points
		else:
			curve = cmds.curve(p=curve_points, d=3)  # Cubic curve
		# cmds.fitBspline(ch=0, tol=0.01)

		return curve

	def create_tube_from_curve(self, curve, radius=0.1, sections=8):
		"""
		Create a tubular geometry by extruding a circle along the given curve.

		Args:
			curve (str): The name of the curve to extrude along.
			radius (float, optional): The radius of the tube. Defaults to 0.1.
			sections (int, optional): The number of sections for the circle profile. Defaults to 8.

		Returns:
			str: The name of the created tubular geometry.
		"""
		# Create a circle to serve as the profile for extrusion
		circle = cmds.circle(radius=radius, sections=sections, normal=(0, 1, 0))[0]

		# Extrude the circle along the curve to create the tube
		tube = cmds.extrude(
			circle, curve, et=2, fixedPath=True, useComponentPivot=1, polygon=1, ch=True
		)[0]

		# Make sure the extruded tube will be deleted later
		self.created_nodes.append(tube)

		# Cap the ends of the tube
		cmds.polyCloseBorder(tube)

		# Delete the original circle and curve
		cmds.delete(circle, curve)

		# Shading
		self.assign_transparent_yellow_material(tube)

		return tube

	def assign_transparent_yellow_material(self, geometry):
		"""
		Assign a transparent yellow material to the specified geometry.

		Args:
			geometry (str): The name of the geometry to assign the material to.
		"""
		# Assign the shader to the geometry
		cmds.sets(geometry, edit=True, forceElement=self.shading_group)

	def visualize(self, radius=0.1, sections=8):
		"""
		Visualize the stroke as a tube in Maya.

		Args:
			radius (float, optional): The radius of the tube. Defaults to 0.1.
			sections (int, optional): The number of sections for the circle profile. Defaults to 8.

		Returns:
			str: The name of the created tubular geometry.
		"""
		# Store the original selection
		original_selection = cmds.ls(selection=True, long=True) or []

		curve = self.create_curve_from_stroke()
		tube = self.create_tube_from_curve(curve, radius, sections)

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

		return tube

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


def test_stroke_visualizer():
	"""Test case for StrokeVisualizer"""
	# Clear the Maya scene
	cmds.file(new=True, force=True)

	# Create a Stroke instance
	stroke = Stroke()

	# Add sample points to the stroke
	sample_positions = [[0, 0, 0], [1, 1, 0], [2, 1.5, 0], [3, 2, 0], [4, 2.5, 0]]

	for i, pos in enumerate(sample_positions):
		sample = Sample(position=pos, size=1.0, pressure=1.0, timestamp=i)
		stroke.add_sample(sample)

	# Create the visualizer
	visualizer = StrokeVisualizer(stroke)

	# Generate the tube
	tube = visualizer.visualize(radius=0.2, sections=12)

	# Check if the tube exists in the Maya scene
	assert cmds.objExists(tube), f"Test Failed: Tube '{tube}' was not created."

	print(f"Test Passed: Tube '{tube}' successfully created.")
