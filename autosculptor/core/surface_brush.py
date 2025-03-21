"""
Surface brush
"""

import numpy as np
from .brush import Brush, BrushMode
from .data_structures import Sample
from .mesh_interface import MeshInterface


class SurfaceBrush(Brush):
	def __init__(self, size=1.0, strength=1.0, mode=BrushMode.ADD, falloff="smooth"):
		"""
		Initialize a new SurfaceBrush.
		Args:
		    size (float): Size of the brush
		    strength (float): Strength of the brush effect
		    mode (BrushMode): Operation mode (ADD, SUBTRACT, SMOOTH)
		    falloff (str): Falloff type ('smooth', 'linear', 'constant')
		"""
		super().__init__(size, strength, mode, falloff)

	def add_sample(self, position, normal=None, pressure=1.0, timestamp=0):
		"""
		Add a sample to the current stroke.
		Args:
		    position (tuple or list): 3D position on the mesh surface
		    normal (tuple or list): Surface normal at the position
		    pressure (float, optional): Pressure applied at this sample
		    timestamp (float, optional): Time when this sample was created
		Returns:
		    Sample: The created and added sample
		"""
		if self.current_stroke is None:
			self.begin_stroke()
		if normal is None:
			normal = [0, 1, 0]
		sample = Sample(position, normal, self.size, pressure, timestamp)
		self.current_stroke.add_sample(sample)
		self.current_stroke.stroke_type = "surface"
		return sample

	def apply_to_mesh(self, mesh_data, sample):
		"""
		Apply the surface brush effect to a mesh at a given sample.
		For surface brush, we move vertices along their normals.
		"""
		if not mesh_data.maya_mesh_fn:
			print("Warning: Maya mesh function not available.")
			return

		print(f"Type of mesh_data: {type(mesh_data)}")
		print(f"Type of sample: {type(sample)}")

		brush_position = sample.position
		brush_size = self.size * sample.size
		brush_strength = self.strength * sample.pressure
		affected_vertices_indices = []
		affected_vertices_local_positions = []

		print(f"Sample size: {sample.size}")

		print("Starting to check vertex distance... ")
		for i, vertex in enumerate(mesh_data.vertices):
			vertex_pos = np.array(vertex)
			distance = np.linalg.norm(vertex_pos - brush_position)
			if distance < brush_size:
				affected_vertices_indices.append(i)
				affected_vertices_local_positions.append(vertex_pos)

		if len(affected_vertices_indices) == 0:
			print("No vertices affected.")
			return

		displacements = []
		for i, vertex_index in enumerate(affected_vertices_indices):
			vertex_pos = affected_vertices_local_positions[i]
			distance = np.linalg.norm(vertex_pos - brush_position)
			falloff = self.calculate_falloff(distance)

			vertex_normal = MeshInterface.get_vertex_normals_for_indices(
				mesh_data, [vertex_index]
			)[0]

			displacement_direction = self.get_displacement_vector(vertex_normal)
			if displacement_direction is None:
				continue

			displacement = displacement_direction * falloff * brush_strength
			if self.mode == BrushMode.SUBTRACT:
				print(f"normal: {vertex_normal}\tdisplacement: {displacement}")

			displacements.append(displacement)

		new_vertices = mesh_data.vertices[:]
		for i, vertex_index in enumerate(affected_vertices_indices):
			new_vertices[vertex_index] = (
				mesh_data.vertices[vertex_index] + displacements[i]
			)

		MeshInterface.update_mesh_vertices(mesh_data, new_vertices)

	def calculate_falloff(self, distance):
		"""
		Calculate the falloff value based on the distance from brush center.
		Args:
		    distance (float): Distance from the brush center
		Returns:
		    float: Falloff value between 0 and 1
		"""
		normalized_dist = min(distance / self.size, 1.0)
		if self.falloff == "constant":
			return float(np.where(normalized_dist < 1.0, 1.0, 0.0))
		elif self.falloff == "linear":
			return max(0.0, 1.0 - normalized_dist)
		else:  # "smooth"
			return max(0.0, 1.0 - (normalized_dist * normalized_dist))
