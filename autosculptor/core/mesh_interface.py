"""
Mesh interface

Provides an abstraction layer for accessing and modifying mesh data in Maya.
"""

import numpy as np

try:
	import maya.cmds as cmds  # type: ignore
	import maya.api.OpenMaya as om2  # type: ignore

	MAYA_AVAILABLE = True
except ImportError:
	MAYA_AVAILABLE = False
	print("Maya modules not available. Running in standalone mode.")


class MeshData:
	"""
	Class to store and manipulate mesh data.
	"""

	def __init__(self):
		self.vertices = np.array([], dtype=np.float64)
		self.normals = np.array([], dtype=np.float64)
		self.faces = np.array([], dtype=np.int32)
		self.maya_dag_path = None
		self.maya_mesh_fn = None
		self.original_faces = []  # for normal calculation after triangulation

	def __repr__(self):
		vert_count = (
			self.vertices.shape[0]
			if isinstance(self.vertices, np.ndarray)
			else len(self.vertices)
		)
		face_count = (
			self.faces.shape[0]
			if isinstance(self.faces, np.ndarray)
			else len(self.faces)
		)
		return f"MeshData(vertices={vert_count}, faces={face_count})"

	def triangulate(self):
		"""Triangulates the mesh data.

		This modifies the self.faces and self.normals arrays in-place.
		It uses the Maya API for triangulation and correct normal handling.
		"""
		if not self.maya_mesh_fn:
			raise RuntimeError("Cannot triangulate without Maya mesh function.")

		try:
			_, triangle_vertices_int_array = self.maya_mesh_fn.getTriangles()
			self.faces = np.array(triangle_vertices_int_array, dtype=np.int32).reshape(
				-1, 3
			)

		except Exception as e:
			print(f"Error during triangulation: {e}")
			self.faces = np.array([], dtype=np.int32)


class MeshInterface:
	"""
	Interface for accessing and modifying mesh data in Maya or standalone mode.
	"""

	@staticmethod
	def get_mesh_data(mesh_name):
		"""
		Get mesh data from a Maya mesh.

		Args:
			mesh_name (str): Name of the mesh in Maya

		Returns:
			MeshData: Object containing mesh data
		"""
		mesh_data = MeshData()

		if not MAYA_AVAILABLE:
			print("Maya is not available. Cannot get mesh data.")
			return mesh_data

		selection_list = om2.MSelectionList()
		try:
			selection_list.add(mesh_name)
		except:
			raise ValueError(f"Mesh {mesh_name} not found in scene")

		try:
			dag_path = selection_list.getDagPath(0)
			if not dag_path.hasFn(om2.MFn.kMesh):
				dag_path.extendToShape()
				if not dag_path.hasFn(om2.MFn.kMesh):
					raise TypeError(f"Object '{mesh_name}' is not a mesh.")

			mesh_fn = om2.MFnMesh(dag_path)

			mesh_data.maya_dag_path = dag_path
			mesh_data.maya_mesh_fn = mesh_fn

			points = mesh_fn.getPoints(om2.MSpace.kWorld)
			mesh_data.vertices = np.array(
				[[p.x, p.y, p.z] for p in points], dtype=np.float64
			)

			normals = mesh_fn.getVertexNormals(False, om2.MSpace.kWorld)
			mesh_data.normals = np.array(
				[[n.x, n.y, n.z] for n in normals], dtype=np.float64
			)
			norms = np.linalg.norm(mesh_data.normals, axis=1, keepdims=True)
			norms[norms == 0] = 1.0
			mesh_data.normals /= norms

			num_polygons = mesh_fn.numPolygons
			mesh_data.original_faces = []
			for i in range(num_polygons):
				vertices = mesh_fn.getPolygonVertices(i)
				mesh_data.original_faces.append(list(vertices))

			mesh_data.triangulate()

		except Exception as e:
			print(f"Error getting mesh data for '{mesh_name}': {e}")
			import traceback

			traceback.print_exc()
			return MeshData()

		return mesh_data

	@staticmethod
	def update_mesh_vertices(mesh_data, new_vertices):
		"""
		Update the vertices of a Maya mesh.

		Args:
			mesh_data (MeshData): Mesh data object with Maya references
			new_vertices (list): New vertex positions
		"""
		if not MAYA_AVAILABLE:
			print("Maya is not available. Cannot update mesh vertices.")
			return

		try:
			if isinstance(new_vertices, np.ndarray):
				num_verts = new_vertices.shape[0]
				if num_verts != mesh_data.maya_mesh_fn.numVertices:
					print(
						f"Error: Vertex count mismatch ({num_verts} vs {mesh_data.maya_mesh_fn.numVertices}). Cannot update."
					)
					return

				points = om2.MPointArray(new_vertices.tolist())

			else:
				num_verts = len(new_vertices)
				if num_verts != mesh_data.maya_mesh_fn.numVertices:
					print(
						f"Error: Vertex count mismatch ({num_verts} vs {mesh_data.maya_mesh_fn.numVertices}). Cannot update."
					)
					return
				points = om2.MPointArray(new_vertices)

			mesh_data.maya_mesh_fn.setPoints(points, om2.MSpace.kWorld)
			mesh_data.maya_mesh_fn.updateSurface()  # Important!
			mesh_data.vertices = (
				np.array(new_vertices, dtype=np.float64)
				if not isinstance(new_vertices, np.ndarray)
				else new_vertices
			)

		except Exception as e:
			print(f"Error updating mesh vertices: {e}")
			import traceback

			traceback.print_exc()

	@staticmethod
	def find_closest_point(mesh_data, point):
		if not MAYA_AVAILABLE:
			print("Maya is not available. Cannot find closest point.")
			return None, None, None

		try:
			if not hasattr(point, "__getitem__") or len(point) < 3:
				raise ValueError(
					f"Invalid point format passed to find_closest_point: {point}"
				)

			maya_point = om2.MPoint(point[0], point[1], point[2])

			closest_data = mesh_data.maya_mesh_fn.getClosestPointAndNormal(
				maya_point, om2.MSpace.kWorld
			)

			closest_point_om = closest_data[0]
			closest_normal_om = closest_data[1]
			face_id_value = closest_data[2]

			closest_point_list = [
				closest_point_om.x,
				closest_point_om.y,
				closest_point_om.z,
			]

			norm_mag = closest_normal_om.length()
			if norm_mag > 1e-9:
				closest_normal_list = [
					closest_normal_om.x / norm_mag,
					closest_normal_om.y / norm_mag,
					closest_normal_om.z / norm_mag,
				]
			else:
				closest_normal_list = [0.0, 1.0, 0.0]

			# print(f"Closest point: {closest_point_list}, Normal: {closest_normal_list}, Face: {face_id_value}")
			return closest_point_list, closest_normal_list, face_id_value

		except Exception as e:
			print(f"Error in find_closest_point: {e}")
			# print(f"  Input point: {point}") # Debug input point
			import traceback

			traceback.print_exc()

	@staticmethod
	def get_vertex_normals_for_indices(mesh_data, vertex_indices):
		try:
			valid_indices = [
				idx for idx in vertex_indices if 0 <= idx < mesh_data.normals.shape[0]
			]
			if len(valid_indices) != len(vertex_indices):
				return (
					mesh_data.normals[valid_indices]
					if valid_indices
					else np.zeros((0, 3))
				)

			return mesh_data.normals[vertex_indices]
		except IndexError as e:
			print(f"Error accessing vertex normals by index: {e}")
			return np.zeros((len(vertex_indices), 3))
		except Exception as e:
			print(f"Unexpected error in get_vertex_normals_for_indices: {e}")
			return np.zeros((len(vertex_indices), 3))
