import numpy as np
import potpourri3d as pp3d
from typing import List, Optional, Dict, Tuple, Union
import time


class CachedGeodesicCalculator:
	"""
	A class for efficient geodesic distance calculation on meshes with caching.

	This calculator uses the heat method for fast approximation of geodesic distances
	and implements caching to speed up repeated queries from the same source points.
	"""

	def __init__(self, vertices: np.ndarray, faces: np.ndarray, cache_size: int = 100):
		"""
		Initialize the geodesic calculator with mesh data.

		Args:
		    vertices (np.ndarray): Vertex positions of shape (N, 3)
		    faces (np.ndarray): Face indices of shape (M, 3)
		    cache_size (int): Maximum number of source vertices to cache distances for
		"""
		self.vertices = vertices
		self.faces = faces
		self.cache_size = cache_size
		self.cache: Dict[int, np.ndarray] = {}
		self.cache_order: List[int] = []

		try:
			self.solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
		except Exception as e:
			raise RuntimeError(
				f"Failed to initialize heat method solver: {e}. Check if your mesh is valid and manifold."
			)

		try:
			from scipy.spatial import cKDTree

			self.kdtree = cKDTree(vertices)
		except ImportError:
			self.kdtree = None
			print("Warning: scipy not available, vertex lookup will be slower.")

	def find_closest_vertex(self, position: np.ndarray) -> int:
		"""
		Find the vertex index closest to a given 3D position.

		Args:
		    position (np.ndarray): 3D position to find closest vertex for

		Returns:
		    int: Index of the closest vertex
		"""
		if self.kdtree is not None:
			distance, index = self.kdtree.query(position, k=1)
			return index
		else:
			distances = np.linalg.norm(self.vertices - position, axis=1)
			return np.argmin(distances)

	def find_closest_vertices(self, positions: np.ndarray) -> np.ndarray:
		"""
		Find the vertex indices closest to a batch of 3D positions.

		Args:
		    positions (np.ndarray): Array of 3D positions of shape (N, 3)

		Returns:
		    np.ndarray: Array of indices of the closest vertices
		"""
		if positions.size == 0:
			return np.array([], dtype=np.int32)

		if self.kdtree is not None:
			distances, indices = self.kdtree.query(positions, k=1)
			return indices
		else:
			indices = np.zeros(len(positions), dtype=np.int32)
			for i, pos in enumerate(positions):
				distances = np.linalg.norm(self.vertices - pos, axis=1)
				indices[i] = np.argmin(distances)
			return indices

	def _update_cache(self, vertex_idx: int, distances: np.ndarray) -> None:
		"""
		Update the cache with new distances.

		Args:
		    vertex_idx (int): Source vertex index
		    distances (np.ndarray): Array of distances from source to all vertices
		"""
		if vertex_idx in self.cache:
			self.cache_order.remove(vertex_idx)
			self.cache_order.append(vertex_idx)
			self.cache[vertex_idx] = distances
			return

		self.cache[vertex_idx] = distances
		self.cache_order.append(vertex_idx)

		if len(self.cache_order) > self.cache_size:
			oldest_idx = self.cache_order.pop(0)
			del self.cache[oldest_idx]

	def compute_distance(
		self, source_position: np.ndarray, target_positions: Optional[np.ndarray] = None
	) -> np.ndarray:
		"""
		Compute geodesic distances from a source point to target points.

		Args:
		    source_position (np.ndarray): Source position in 3D
		    target_positions (np.ndarray, optional): Target positions in 3D of shape (N, 3)
		        If None, returns distances to all vertices.

		Returns:
		    np.ndarray: Array of geodesic distances
		"""
		source_idx = self.find_closest_vertex(source_position)

		if source_idx in self.cache:
			self.cache_order.remove(source_idx)
			self.cache_order.append(source_idx)

			all_distances = self.cache[source_idx]
		else:
			try:
				all_distances = self.solver.compute_distance(source_idx)
				self._update_cache(source_idx, all_distances)
			except Exception as e:
				raise RuntimeError(f"Error computing geodesic distances: {e}")

		if target_positions is not None:
			target_indices = self.find_closest_vertices(target_positions)
			return all_distances[target_indices]

		return all_distances

	def compute_many_to_many(
		self, source_positions: np.ndarray, target_positions: np.ndarray
	) -> np.ndarray:
		"""
		Compute geodesic distances from multiple source points to multiple target points.

		Args:
		    source_positions (np.ndarray): Source positions in 3D of shape (M, 3)
		    target_positions (np.ndarray): Target positions in 3D of shape (N, 3)

		Returns:
		    np.ndarray: Matrix of geodesic distances of shape (M, N)
		"""
		if source_positions.size == 0 or target_positions.size == 0:
			return None

		num_sources = len(source_positions)
		num_targets = len(target_positions)
		result = np.zeros((num_sources, num_targets))

		for i, source_pos in enumerate(source_positions):
			result[i, :] = self.compute_distance(source_pos, target_positions)

		return result

	def clear_cache(self) -> None:
		"""Clear the distance cache."""
		self.cache.clear()
		self.cache_order.clear()

	def get_mesh_area(self) -> float:
		"""
		Calculate the total surface area of the mesh.

		Returns:
		    float: Total surface area
		"""
		total_area = 0.0

		for face in self.faces:
			v1, v2, v3 = self.vertices[face]
			edge1 = v2 - v1
			edge2 = v3 - v1
			face_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
			total_area += face_area

		return total_area
