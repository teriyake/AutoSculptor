"""
Surface brush
"""

import numpy as np
from .brush import Brush, BrushMode
from .data_structures import Sample
from .mesh_interface import MeshInterface
import maya.api.OpenMaya as om2


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

		brush_position = np.array(sample.position, dtype=np.float64)
		brush_radius = sample.size if sample.size > 1e-6 else self.size
		brush_strength_factor = self.strength * sample.pressure * 100 * 100

		affected_indices = []
		distances = []
		vertices_np = np.array(mesh_data.vertices, dtype=np.float64)
		vertex_distances = np.linalg.norm(vertices_np - brush_position, axis=1)

		affected_mask = vertex_distances < brush_radius
		affected_indices = np.where(affected_mask)[0]
		distances = vertex_distances[affected_mask]

		if len(affected_indices) == 0:
			return

		new_vertices = vertices_np.copy()
		mesh_fn = mesh_data.maya_mesh_fn
		dag_path = mesh_data.maya_dag_path

		if self.mode == BrushMode.SMOOTH:
			original_affected_positions = new_vertices[affected_indices].copy()
			smoothed_positions = np.zeros_like(original_affected_positions)

			vert_iter = om2.MItMeshVertex(dag_path)

			for i, vtx_idx in enumerate(affected_indices):
				try:
					vert_iter.setIndex(int(vtx_idx))
					connected_face_indices = vert_iter.getConnectedFaces()

					neighbor_indices = set()
					for face_idx in connected_face_indices:
						poly_verts = mesh_fn.getPolygonVertices(face_idx)
						for poly_vtx_idx in poly_verts:
							neighbor_indices.add(poly_vtx_idx)

					neighbor_indices.discard(vtx_idx)
					connected_vertices = list(neighbor_indices)

					if not connected_vertices:
						smoothed_positions[i] = original_affected_positions[i]
						continue

					neighbor_positions = vertices_np[connected_vertices]
					average_pos = np.mean(neighbor_positions, axis=0)
					smoothed_positions[i] = average_pos
				except Exception as e:
					print(
						f"  Warning: Error processing neighbors via faces for vertex {vtx_idx}: {e}"
					)
					import traceback

					traceback.print_exc()
					smoothed_positions[i] = original_affected_positions[i]

			for i, vtx_idx in enumerate(affected_indices):
				falloff = self.calculate_falloff(distances[i], brush_radius)
				smoothing_amount = falloff * brush_strength_factor
				smoothing_amount = np.clip(smoothing_amount, 0.0, 1.0)
				new_vertices[vtx_idx] = (
					1.0 - smoothing_amount
				) * original_affected_positions[
					i
				] + smoothing_amount * smoothed_positions[
					i
				]

		else:
			vertex_normals = MeshInterface.get_vertex_normals_for_indices(
				mesh_data, affected_indices
			)

			for i, vtx_idx in enumerate(affected_indices):
				falloff = self.calculate_falloff(distances[i], brush_radius)
				vertex_normal = vertex_normals[i]

				displacement_magnitude = falloff * brush_strength_factor * 0.1
				if self.mode == BrushMode.ADD:
					displacement = vertex_normal * displacement_magnitude
				elif self.mode == BrushMode.SUBTRACT:
					displacement = -vertex_normal * displacement_magnitude
				else:
					continue

				new_vertices[vtx_idx] += displacement

		MeshInterface.update_mesh_vertices(mesh_data, new_vertices)
		mesh_data.vertices = new_vertices

	def calculate_falloff(self, distance, brush_radius):
		"""
		Calculate the falloff value based on the distance from brush center.
		Args:
			distance (float): Distance from the brush center
			brush radius (float): Radius of the brush
		Returns:
			float: Falloff value between 0 and 1
		"""
		if brush_radius <= 1e-6:
			return 0.0

		normalized_dist = min(distance / brush_radius, 1.0)
		if self.falloff == "constant":
			return float(np.where(normalized_dist < 1.0, 1.0, 0.0))
		elif self.falloff == "linear":
			return max(0.0, 1.0 - normalized_dist)
		else:  # "smooth"
			return max(0.0, 1.0 - (normalized_dist * normalized_dist))
