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

	def __repr__(self):
		return f"MeshData(vertices={len(self.vertices)}, faces={len(self.faces)})"


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

		for i in range(points.length()):
			mesh_data.vertices.append([points[i].x, points[i].y, points[i].z])

		normals = om.MFloatVectorArray()
		mesh_fn.getNormals(normals)

		for i in range(normals.length()):
			mesh_data.normals.append([normals[i].x, normals[i].y, normals[i].z])

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
			mesh_data.faces.append(face)

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
		points = om.MPointArray()
		for i in range(len(new_vertices)):
			points.append(
				om.MPoint(new_vertices[i][0], new_vertices[i][1], new_vertices[i][2])
			)

		mesh_data.maya_mesh_fn.setPoints(points)

		mesh_data.vertices = new_vertices

	@staticmethod
	def find_closest_point(mesh_data, point):
		"""
		Find the closest point on the mesh to a given point.

		Args:
		    mesh_data (MeshData): Mesh data object
		    point (list or tuple): 3D point to find closest point to

		Returns:
		    tuple: (closest_point, normal, face_id)
		"""
		if not MAYA_AVAILABLE:
			print("Maya is not available. Cannot find closest point.")
			return None, None, None

		maya_point = om.MPoint([point[0], point[1], point[2]])

		closest_point = om.MPoint()
		closest_normal = om.MVector()
		face_id = om.MScriptUtil().asIntPtr()

		mesh_data.maya_mesh_fn.getClosestPoint(
			maya_point, closest_point, om.MSpace.kWorld, face_id
		)

		face_id_value = om.MScriptUtil.getInt(face_id)
		mesh_data.maya_mesh_fn.getFaceVertexNormal(face_id_value, 0, closest_normal)

		closest_point_list = [closest_point.x, closest_point.y, closest_point.z]
		closest_normal_list = [closest_normal.x, closest_normal.y, closest_normal.z]

		return closest_point_list, closest_normal_list, face_id_value

	@staticmethod
	def get_vertex_normals_for_indices(mesh_data, vertex_indices):
		"""
		Get normals for specific vertices of a Maya mesh.
		Args:
		    mesh_data (MeshData): Mesh data object with Maya references
		    vertex_indices (list): List of vertex indices
		Returns:
		    np.ndarray: Array of vertex normals (each normal is a np.array of size 3)
		"""
		if not MAYA_AVAILABLE:
			print("Maya is not available. Cannot get vertex normals.")
			return np.zeros((len(vertex_indices), 3))

		mesh_fn = mesh_data.maya_mesh_fn
		vertex_normals = []

		for index in vertex_indices:
			normal = om.MVector()
			mesh_fn.getVertexNormal(index, False, normal, om.MSpace.kWorld)
			vertex_normals.append(
				np.array([normal.x, normal.y, normal.z], dtype=np.float32)
			)

		return np.array(vertex_normals)
