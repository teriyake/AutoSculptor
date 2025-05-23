import maya.cmds as cmds  # type: ignore
import maya.api.OpenMaya as om2  # type: ignore
import maya.utils
import numpy as np
import traceback
from typing import Optional, Tuple


def get_active_camera_lookat_vector(
	default_vector=np.array([0.0, 0.0, 1.0], dtype=np.float64)
):
	"""
	Calculates the normalized look-at vector of the camera in the
	currently active Maya model panel.

	Args:
		default_vector (np.ndarray): The vector to return if the lookat
									 vector cannot be determined.
									 Defaults to [0, 0, 1].

	Returns:
		np.ndarray: A normalized numpy array representing the camera's
					world-space look-at vector.
	"""
	try:
		panel = cmds.getPanel(withFocus=True)

		if not panel or cmds.getPanel(typeOf=panel) != "modelPanel":
			visible_panels = cmds.getPanel(visiblePanels=True)
			model_panels = [
				p for p in visible_panels if cmds.getPanel(typeOf=p) == "modelPanel"
			]
			if not model_panels:
				print(
					"get_active_camera_lookat_vector: No active or visible model panel found."
				)
				return default_vector
			panel = model_panels[0]
			# print(f"get_active_camera_lookat_vector: Using first visible model panel: {panel}")

		cam_transform = cmds.modelEditor(panel, query=True, camera=True)

		if not cam_transform or not cmds.objExists(cam_transform):
			print(
				f"get_active_camera_lookat_vector: Could not find camera for panel '{panel}'."
			)
			return default_vector

		sel = om2.MSelectionList()
		sel.add(cam_transform)
		cam_dag_path = sel.getDagPath(0)
		cam_matrix = cam_dag_path.inclusiveMatrix()

		local_z_axis_world = np.array(
			[
				cam_matrix[2],
				cam_matrix[6],
				cam_matrix[10],
			],
			dtype=np.float64,
		)

		camera_lookat = -local_z_axis_world

		norm = np.linalg.norm(camera_lookat)
		if norm > 1e-9:
			return camera_lookat / norm
		else:
			print(
				f"get_active_camera_lookat_vector: Warning - Camera '{cam_transform}' has near-zero lookat vector length."
			)
			return default_vector

	except Exception as e:
		print(
			f"get_active_camera_lookat_vector: Error getting camera lookat vector: {e}"
		)
		# traceback.print_exc()
		return default_vector


def get_active_camera_details() -> Optional[Tuple[str, om2.MDagPath]]:
	"""
	Gets the transform name and DAG path of the active model panel camera.
	"""
	try:
		panel = cmds.getPanel(withFocus=True)
		if not panel or cmds.getPanel(typeOf=panel) != "modelPanel":
			visible_panels = cmds.getPanel(visiblePanels=True)
			model_panels = [
				p for p in visible_panels if cmds.getPanel(typeOf=p) == "modelPanel"
			]
			if not model_panels:
				return None
			panel = model_panels[0]

		cam_transform = cmds.modelEditor(panel, query=True, camera=True)
		if not cam_transform or not cmds.objExists(cam_transform):
			return None

		sel = om2.MSelectionList()
		sel.add(cam_transform)
		cam_dag_path = sel.getDagPath(0)
		return cam_transform, cam_dag_path
	except Exception as e:
		print(f"Error getting active camera: {e}")
		return None


def get_tool_name_deferred(callback_function):
	"""
	Attempts to get the sculpt tool display name using contextInfo,
	but defers the execution to avoid timing issues in plugins.

	Args:
		callback_function: A function that will be called with the
						   result (the tool name string or None).
	"""
	try:
		ctx_name = cmds.currentCtx()

		def deferred_query():
			print(f"Executing deferred query for context: {ctx_name}")
			result = None
			try:
				result = cmds.contextInfo(ctx_name, title=True)
				print(f"Deferred query result: {result}")
			except Exception as e:
				print(
					f"Error during deferred cmds.contextInfo('{ctx_name}', title=True): {e}"
				)
			finally:
				callback_function(result)

		maya.utils.executeDeferred(deferred_query)

	except Exception as e:
		print(f"Error setting up deferred execution: {e}")
		callback_function(None)


def get_tool_name_main_thread():
	"""
	Attempts to get the sculpt tool display name using contextInfo,
	ensuring the call happens on the main Maya thread (otherwise Maya complains).

	Returns:
		The tool name string or None if an error occurs.
	"""
	try:
		ctx_name = cmds.currentCtx()

		def main_thread_action():
			print(f"Executing main_thread_action for context: {ctx_name}")
			try:
				result = cmds.contextInfo(ctx_name, title=True)
				print(f"Main thread query result: {result}")
				return result
			except Exception as e:
				print(
					f"Error during main_thread_action cmds.contextInfo('{ctx_name}', title=True): {e}"
				)
				return None

		tool_name = maya.utils.executeInMainThreadWithResult(main_thread_action)
		return tool_name

	except Exception as e:
		print(f"Error setting up main thread execution: {e}")
		return None
