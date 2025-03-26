"""
Mesh interface

Provides an abstraction layer for accessing and modifying mesh data in Maya.
"""

import numpy as np

try:
	import maya.cmds as cmds
	import maya.OpenMaya as om

	MAYA_AVAILABLE = True
except ImportError:
	MAYA_AVAILABLE = False
	print("Maya modules not available. Running in standalone mode.")


class MeshData:
	"""
	Class to store and manipulate mesh data.
	"""

	def __init__(self):
		self.vertices = []
		self.normals = []
		self.faces = []
		self.maya_dag_path = None
		self.maya_mesh_fn = None
		self.original_faces = []  # for normal calculation after triangulation

	def __repr__(self):
		return f"MeshData(vertices={len(self.vertices)}, faces={len(self.faces)})"

	def triangulate(self):
		"""Triangulates the mesh data.

		This modifies the self.faces and self.normals arrays in-place.
		It uses the Maya API for triangulation and correct normal handling.
		"""
		if not self.maya_mesh_fn:
			raise RuntimeError("Cannot triangulate without Maya mesh function.")

		triangles = om.MIntArray()
		triangle_vertices = om.MIntArray()
		self.maya_mesh_fn.getTriangles(triangles, triangle_vertices)

		self.faces = np.array(triangle_vertices).reshape(-1, 3)

		new_normals = []
		# util = om.MScriptUtil()

		for i in range(self.faces.shape[0]):
			v1_idx, v2_idx, v3_idx = self.faces[i]

			v1 = self.vertices[v1_idx]
			v2 = self.vertices[v2_idx]
			v3 = self.vertices[v3_idx]

			normal = np.cross(v2 - v1, v3 - v1)
			normal = normal / (np.linalg.norm(normal) + 1e-8)
			new_normals.extend([normal, normal, normal])

		self.normals = np.array(new_normals)

		if len(self.normals) != len(triangle_vertices):
			raise RuntimeError(
				f"Normal count ({len(self.normals)}) does not match vertex count after triangulation ({len(triangle_vertices)})"
			)


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

		selection_list = om.MSelectionList()
		try:
			selection_list.add(mesh_name)
		except:
			raise ValueError(f"Mesh {mesh_name} not found in scene")

		dag_path = om.MDagPath()
		selection_list.getDagPath(0, dag_path)

		mesh_fn = om.MFnMesh(dag_path)

		mesh_data.maya_dag_path = dag_path
		mesh_data.maya_mesh_fn = mesh_fn

		points = om.MPointArray()
		mesh_fn.getPoints(points)

		mesh_data.vertices = np.array(
			[[points[i].x, points[i].y, points[i].z] for i in range(points.length())],
			dtype=np.float64,
		)

		normals = om.MFloatVectorArray()
		mesh_fn.getNormals(normals)

		mesh_data.normals = np.array(
			[
				[normals[i].x, normals[i].y, normals[i].z]
				for i in range(normals.length())
			],
			dtype=np.float64,
		)

		polygon_counts = om.MIntArray()
		polygon_connects = om.MIntArray()
		mesh_fn.getVertices(polygon_counts, polygon_connects)

		face_index = 0
		for i in range(polygon_counts.length()):
			count = polygon_counts[i]
			face = []
			for j in range(count):
				face.append(polygon_connects[face_index])
				face_index += 1
			mesh_data.original_faces.append(face)

		mesh_data.triangulate()

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

		if isinstance(new_vertices, np.ndarray):
			points = om.MPointArray()
			for i in range(new_vertices.shape[0]):
				points.append(
					om.MPoint(
						new_vertices[i, 0], new_vertices[i, 1], new_vertices[i, 2]
					)
				)
		else:
			points = om.MPointArray()
			for i in range(len(new_vertices)):
				points.append(
					om.MPoint(
						new_vertices[i][0], new_vertices[i][1], new_vertices[i][2]
					)
				)

		mesh_data.maya_mesh_fn.setPoints(points)
		mesh_data.vertices = new_vertices

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
			maya_point = om.MPoint(point[0], point[1], point[2], 1.0)
		except Exception as e:
			print(f"Error creating MPoint from: {point}")
			raise e

		closest_point = om.MPoint()
		closest_normal = om.MVector()

		util = om.MScriptUtil()
		util.createFromInt(0)
		face_id_ptr = util.asIntPtr()

		mesh_data.maya_mesh_fn.getClosestPoint(
			maya_point, closest_point, om.MSpace.kWorld, face_id_ptr
		)

		face_id_value = util.getInt(face_id_ptr)

		try:
			if (
				face_id_value >= 0
				and face_id_value < mesh_data.maya_mesh_fn.numPolygons()
			):
				mesh_data.maya_mesh_fn.getClosestNormal(
					maya_point, closest_normal, om.MSpace.kWorld
				)
			else:
				print(
					f"Warning: Invalid face ID {face_id_value} returned by getClosestPoint. Using default normal."
				)
				closest_normal = om.MVector(0, 1, 0)

		except Exception as e:
			print(f"Error getting closest normal for face {face_id_value}: {e}")
			closest_normal = om.MVector(0, 1, 0)

		closest_point_list = [closest_point.x, closest_point.y, closest_point.z]
		closest_normal_list = [closest_normal.x, closest_normal.y, closest_normal.z]

		return closest_point_list, closest_normal_list, face_id_value

	@staticmethod
	def get_vertex_normals_for_indices(mesh_data, vertex_indices):

		if not MAYA_AVAILABLE:
			print("Maya is not available. Cannot get vertex normals.")
			return np.zeros((len(vertex_indices), 3))

		return mesh_data.normals[vertex_indices]
