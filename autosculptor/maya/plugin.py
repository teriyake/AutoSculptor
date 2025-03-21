import sys
import os
import time

try:
	import maya.cmds as cmds
	import maya.OpenMayaMPx as OpenMayaMPx
	import maya.OpenMaya as om
	import maya.OpenMayaUI as omui
except ImportError:
	print("Maya modules not available. Running in standalone mode.")

import numpy as np
from autosculptor.core.data_structures import Workflow, Sample
from autosculptor.core.surface_brush import SurfaceBrush
from autosculptor.core.freeform_brush import FreeformBrush
from autosculptor.core.brush import BrushMode
from autosculptor.core.mesh_interface import MeshInterface
from autosculptor.analysis.parameterization import StrokeParameterizer
from autosculptor.utils.utils import Utils

current_workflow = Workflow()

CMD_AUTO_SCULPTOR_TEST = om.MTypeId(0x8723)
CONTEXT_AUTO_SCULPTOR = om.MTypeId(0x8725)


class AutoSculptorContext(OpenMayaMPx.MPxContext):
	def __init__(self):
		OpenMayaMPx.MPxContext.__init__(self)
		self.camera_lookat = None
		self.tool_cmd = AutoSculptorToolCmd()
		self.tool_cmd.m_pContext = self
		self.setImage("autosculptor_bug.svg", OpenMayaMPx.MPxContext.kImage1)

	def toolOnSetup(self, event):
		print("AutoSculptorContext.toolOnSetup: CALLED")
		self.update_camera_lookat(event)
		return True

	def toolOffCleanup(self):
		pass

	def doPress(self, event, list, matrix):
		om.MGlobal.displayInfo("AutoSculptorContext: doPress called")
		self.update_camera_lookat(event)
		event.setModifiers(event.modifiers() & ~om.MEvent.shiftModifier)
		if self.tool_cmd:
			self.tool_cmd.doPress(event)
		event.clear()
		return True

	def doDrag(self, event, list, matrix):
		om.MGlobal.displayInfo("AutoSculptorContext: doDrag called")
		self.update_camera_lookat(event)
		self.tool_cmd.doDrag(event)
		event.clear()
		return True

	def doRelease(self, event, list, matrix):
		om.MGlobal.displayInfo("AutoSculptorContext: doRelease called")
		self.update_camera_lookat(event)
		self.tool_cmd.doRelease(event)
		event.clear()
		return True

	def doHold(self, event, list, matrix):
		if self.tool_cmd:
			self.tool_cmd.doHold(event)
		return True

	def update_camera_lookat(self, event):
		"""Updates the stored camera look-at direction."""
		try:
			view = omui.M3dView.active3dView()
		except RuntimeError as e:
			om.MGlobal.displayWarning("No active 3D view found.")
			self.camera_lookat = np.array([0, 0, 1], dtype=np.float32)
			return

		camera_path = om.MDagPath()
		try:
			view.getCamera(camera_path)
		except RuntimeError as e:
			om.MGlobal.displayError("Failed to get camera")
			self.camera_lookat = np.array([0, 0, 1], dtype=np.float32)
			return

		try:
			camera = om.MFnCamera(camera_path)
			lookat = camera.viewDirection(om.MSpace.kWorld)
			self.camera_lookat = np.array(
				[lookat.x, lookat.y, lookat.z], dtype=np.float32
			)
		except RuntimeError as e:
			om.MGlobal.displayError(f"Error getting camera lookat: {e}")
			self.camera_lookat = np.array([0, 0, 1], dtype=np.float32)


class AutoSculptorContextCmd(OpenMayaMPx.MPxContextCommand):
	kContextCmdName = "autoSculptorContext"

	def __init__(self):
		OpenMayaMPx.MPxContextCommand.__init__(self)
		self.m_pContext = None

	def makeObj(self):
		om.MGlobal.displayInfo("AutoSculptorContextCmd: makeObj called")
		self.m_pContext = AutoSculptorContext()
		return self.m_pContext

	def doIt(self, args):
		om.MGlobal.displayInfo("AutoSculptorContextCmd: doIt called")

	@staticmethod
	def creator():
		return AutoSculptorContextCmd()


class AutoSculptorToolCmd(OpenMayaMPx.MPxToolCommand):
	kPluginCmdName = "autoSculptor"
	kBrushTypeFlag = "-bt"
	kBrushTypeLongFlag = "-brushType"
	kBrushSizeFlag = "-bs"
	kBrushSizeLongFlag = "-brushSize"
	kBrushStrengthFlag = "-bst"
	kBrushStrengthLongFlag = "-brushStrength"
	kBrushModeFlag = "-bm"
	kBrushModeLongFlag = "-brushMode"
	kBrushFalloffFlag = "-bf"
	kBrushFalloffLongFlag = "-brushFalloff"
	kContinuousUpdateFlag = "-cu"
	kContinuousUpdateLongFlag = "-continuousUpdate"
	kPlaybackFlag = "-pb"
	kPlaybackLongFlag = "-playback"

	def __init__(self):
		OpenMayaMPx.MPxToolCommand.__init__(self)
		om.MGlobal.displayInfo("AutoSculptorToolCmd: __init__ called")
		self.m_pContext = None

		self.mesh_data = None
		self.brush = None
		self.brush_type = "surface"
		self.brush_size = 1.0
		self.brush_strength = 0.5
		self.brush_mode = BrushMode.ADD
		self.brush_falloff = "smooth"
		self.continuous_update = False
		self.last_point = None
		self.last_normal = None
		self.last_maya_point = None
		self.draw_manager = None
		self.view = None
		self.playback_mode = False
		self.playback_stroke_index = 0
		self.playback_timer_id = None

	def doIt(self, args):
		om.MGlobal.displayInfo("AutoSculptorToolCmd: doIt called")
		try:
			arg_data = om.MArgDatabase(self.syntax(), args)
			self.parse_arguments(arg_data)

			# cmds.setToolTo("autoSculptorContext1")

		except Exception as e:
			om.MGlobal.displayError(f"Error in doIt: {e}")
			om.MGlobal.displayInfo(Utils.full_stack())

	def doPress(self, event):
		"""Handles the initial mouse press event."""
		om.MGlobal.displayInfo("AutoSculptorToolCmd: doPress event")
		self.m_pContext.update_camera_lookat(event)

		if self.playback_mode:
			return

		cmds.undoInfo(openChunk=True)
		self.get_mesh_under_cursor(event)

		if self.mesh_data is None:
			om.MGlobal.displayWarning("No mesh found under the cursor.")
			return

		self.brush = self.create_brush()
		self.brush.begin_stroke()

		self.view = omui.M3dView.active3dView()
		self.draw_manager = om.MUIDrawManager.getDrawManager(self.view, True)
		self.draw_manager.beginDrawable()

		self.setHelpString("Click and drag on a mesh to sculpt.")
		self.sample_point(event)

	def doDrag(self, event):
		"""Handles mouse drag events (continuous sampling)."""
		om.MGlobal.displayInfo("AutoSculptorToolCmd: doDrag event")
		self.m_pContext.update_camera_lookat(event)

		if self.playback_mode:
			return

		if self.mesh_data is None or self.brush is None:
			return

		self.sample_point(event)

	def doRelease(self, event):
		"""Handles mouse release events (finalize the stroke)."""
		om.MGlobal.displayInfo("AutoSculptorToolCmd: doRelease event")
		self.m_pContext.update_camera_lookat(event)

		if self.playback_mode:
			return

		if self.mesh_data is None or self.brush is None:
			return

		if not self.continuous_update:
			self.apply_all_samples()
		stroke = self.brush.end_stroke()
		if stroke:
			global current_workflow

			parameterizer = StrokeParameterizer(self.mesh_data)
			parameterizer.parameterize_stroke(stroke, self.m_pContext.camera_lookat)

			current_workflow.add_stroke(stroke)

			om.MGlobal.displayInfo(
				f"Added stroke to workflow. Total strokes: {len(current_workflow)}"
			)

		self.draw_manager.endDrawable()
		cmds.undoInfo(closeChunk=True)

	def doHold(self, event):
		"""Handles the mouse being held down (for continuous updates)."""
		self.doDrag(event)

	def sample_point(self, event):
		"""Samples a point and adds it to the current stroke."""

		if self.brush_type == "freeform":
			pos = om.MPoint()
			dir_vec = om.MVector()
			self.view.viewToWorld(event.x(), event.y(), pos, dir_vec)
			normal_np = np.array([dir_vec.x, dir_vec.y, dir_vec.z])
			position_np = np.array([pos.x, pos.y, pos.z])
		else:
			try:
				hit_point, hit_normal, _ = MeshInterface.find_closest_point(
					self.mesh_data,
					[self.last_point.x, self.last_point.y, self.last_point.z],
				)
				position_np = np.array(hit_point)
				normal_np = np.array(hit_normal)
			except Exception as e:
				om.MGlobal.displayError(f"Error during find_closest_point: {e}")
				return

		if (
			self.last_point is not None
			and np.linalg.norm(position_np - self.last_point) < 0.01 * self.brush_size
		):
			return

		timestamp = time.time()
		pressure = event.pressure()

		om.MGlobal.displayInfo(
			f"Sampling at: {position_np}, Normal: {normal_np}, Pressure: {pressure}, Time: {timestamp}"
		)
		sample = self.brush.add_sample(
			position=position_np,
			normal=normal_np,
			pressure=pressure,
			timestamp=timestamp,
		)
		self.last_point = position_np
		self.last_normal = normal_np
		if self.continuous_update:
			self.brush.apply_to_mesh(self.mesh_data, sample)

		self.draw_feedback(event)

	def create_brush(self):
		"""Creates a brush instance based on current settings."""
		if self.brush_type == "surface":
			return SurfaceBrush(
				size=self.brush_size,
				strength=self.brush_strength,
				mode=self.brush_mode,
				falloff=self.brush_falloff,
			)
		elif self.brush_type == "freeform":
			return FreeformBrush(
				size=self.brush_size,
				strength=self.brush_strength,
				mode=self.brush_mode,
				falloff=self.brush_falloff,
			)
		else:
			raise ValueError(f"Invalid brush type: {self.brush_type}")

	def get_mesh_under_cursor(self, event):
		"""Performs hit testing at cursor position"""
		self.mesh_data = None

		x = event.x()
		y = event.y()

		selection_list = om.MSelectionList()
		pos = om.MPoint()
		view = omui.M3dView.active3dView()

		try:
			view.viewToWorld(x, y, pos, om.MVector())
			hit_point = om.MPoint()

			self.draw_manager.beginDrawable()
			self.draw_manager.setColor(om.MColor([1, 0, 0]))
			self.draw_manager.sphere(om.MPoint(hit_point), 0.5)
			self.draw_manager.endDrawable()
			view.refresh(True, True)

			om.MGlobal.selectFromScreen(
				x,
				y,
				om.MGlobal.MeshSelectAction.kSelectMesh,
				om.MGlobal.ListAdjustment.kAddToList,
			)
			om.MGlobal.getActiveSelectionList(selection_list)

			if selection_list.isEmpty():
				return

			dag_path = om.MDagPath()
			selection_list.getDagPath(0, dag_path)

			if dag_path.apiType() != om.MFn.kMesh:
				return

			mesh_name = dag_path.partialPathName()
			self.mesh_data = MeshInterface.get_mesh_data(mesh_name)

			mesh_fn = om.MFnMesh(dag_path)
			hit_point = om.MPoint()
			mesh_fn.getClosestPoint(pos, hit_point, om.MSpace.kWorld)
			self.last_point = np.array([hit_point.x, hit_point.y, hit_point.z])

			om.MGlobal.displayInfo(f"Selected mesh: {mesh_name}")

		except Exception as e:
			om.MGlobal.displayError(f"Hit test failed: {str(e)}")

		finally:
			om.MGlobal.clearSelectionList()

	def get_cursor_position(self, event):
		"""Gets the 3D cursor position and the MPoint for closest point calculations."""
		x = event.x()
		y = event.y()
		self.last_maya_point = om.MPoint()
		vec = om.MVector()
		om.MGlobal.viewFrame(event.view(), self.last_maya_point, vec)
		self.last_point = self.last_maya_point

		return x, y, self.last_maya_point

	def draw_feedback(self, event):
		"""Draws visual feedback for the brush."""

		if self.draw_manager is None:
			return

		self.draw_manager.clear()

		x, y, _ = self.get_cursor_position(event)
		world_pt = om.MPoint()
		world_vec = om.MVector()
		self.view.viewToWorld(x, y, world_pt, world_vec)
		self.draw_manager.setColor(om.MColor([1, 1, 0]))
		self.draw_manager.circle(world_pt, world_vec, self.brush_size, True)

		if self.playback_mode:
			if self.playback_stroke_index < len(current_workflow.strokes):
				stroke = current_workflow.strokes[self.playback_stroke_index]
				self.draw_manager.setColor(om.MColor([0, 0, 1]))
				points = om.MPointArray()
				for sample in stroke.samples:
					points.append(om.MPoint(sample.position))
				self.draw_manager.lineStrip(points, True)

		elif self.brush and self.brush.current_stroke:
			self.draw_manager.setColor(om.MColor([0, 1, 0]))
			points = om.MPointArray()
			for sample in self.brush.current_stroke.samples:
				points.append(om.MPoint(sample.position))
			self.draw_manager.lineStrip(points, True)

		self.draw_manager.endDrawable()

	def apply_all_samples(self):
		"""Applies all samples in the current stroke to the mesh."""
		if not self.brush or not self.brush.current_stroke or not self.mesh_data:
			return

		for sample in self.brush.current_stroke.samples:
			self.brush.apply_to_mesh(self.mesh_data, sample)

	@staticmethod
	def cmdCreator():
		return AutoSculptorToolCmd()

	@staticmethod
	def syntaxCreator():
		syntax = om.MSyntax()
		syntax.addFlag(
			AutoSculptorToolCmd.kBrushTypeFlag,
			AutoSculptorToolCmd.kBrushTypeLongFlag,
			om.MSyntax.kString,
		)
		syntax.addFlag(
			AutoSculptorToolCmd.kBrushSizeFlag,
			AutoSculptorToolCmd.kBrushSizeLongFlag,
			om.MSyntax.kDouble,
		)
		syntax.addFlag(
			AutoSculptorToolCmd.kBrushStrengthFlag,
			AutoSculptorToolCmd.kBrushStrengthLongFlag,
			om.MSyntax.kDouble,
		)
		syntax.addFlag(
			AutoSculptorToolCmd.kBrushModeFlag,
			AutoSculptorToolCmd.kBrushModeLongFlag,
			om.MSyntax.kString,
		)
		syntax.addFlag(
			AutoSculptorToolCmd.kBrushFalloffFlag,
			AutoSculptorToolCmd.kBrushFalloffLongFlag,
			om.MSyntax.kString,
		)
		syntax.addFlag(
			AutoSculptorToolCmd.kContinuousUpdateFlag,
			AutoSculptorToolCmd.kContinuousUpdateLongFlag,
			om.MSyntax.kBoolean,
		)
		syntax.addFlag(
			AutoSculptorToolCmd.kPlaybackFlag,
			AutoSculptorToolCmd.kPlaybackLongFlag,
			om.MSyntax.kBoolean,
		)

		return syntax

	def parse_arguments(self, arg_data):
		"""Parses command arguments from the Maya command line or UI."""

		if arg_data.isFlagSet(self.kBrushTypeFlag):
			self.brush_type = arg_data.flagArgumentString(self.kBrushTypeFlag, 0)
		if arg_data.isFlagSet(self.kBrushSizeFlag):
			self.brush_size = arg_data.flagArgumentDouble(self.kBrushSizeFlag, 0)
		if arg_data.isFlagSet(self.kBrushStrengthFlag):
			self.brush_strength = arg_data.flagArgumentDouble(
				self.kBrushStrengthFlag, 0
			)
		if arg_data.isFlagSet(self.kBrushModeFlag):
			brush_mode_str = arg_data.flagArgumentString(self.kBrushModeFlag, 0)
			try:
				self.brush_mode = BrushMode[brush_mode_str.upper()]
			except KeyError:
				om.MGlobal.displayError(f"Invalid brush mode: {brush_mode_str}")
		if arg_data.isFlagSet(self.kBrushFalloffFlag):
			self.brush_falloff = arg_data.flagArgumentString(self.kBrushFalloffFlag, 0)

		if arg_data.isFlagSet(self.kContinuousUpdateFlag):
			self.continuous_update = arg_data.flagArgumentBool(
				self.kContinuousUpdateFlag, 0
			)

		if arg_data.isFlagSet(self.kPlaybackFlag):
			self.playback_mode = arg_data.flagArgumentBool(self.kPlaybackFlag, 0)
			if self.playback_mode:
				self.start_playback()
			else:
				self.stop_playback()

	def start_playback(self):
		"""Starts the playback of recorded strokes."""

		if not current_workflow.strokes:
			om.MGlobal.displayWarning("No strokes recorded to playback.")
			self.playback_mode = False
			return

		om.MGlobal.displayInfo("Starting playback...")
		self.playback_mode = True
		self.playback_stroke_index = 0
		self.playback_timer_id = cmds.scriptJob(
			event=["idle", self.advance_playback], killWithScene=True
		)

		self.draw_feedback(om.MEvent())

	def advance_playback(self):
		"""Advances the playback to the next stroke."""

		if not self.playback_mode:
			return

		self.playback_stroke_index += 1
		if self.playback_stroke_index >= len(current_workflow.strokes):
			self.stop_playback()
			return

		self.draw_feedback(om.MEvent())

	def stop_playback(self):
		"""Stops the playback."""
		om.MGlobal.displayInfo("Stopping playback...")
		self.playback_mode = False
		if self.playback_timer_id is not None:
			try:
				cmds.scriptJob(kill=self.playback_timer_id)
			except RuntimeError:
				pass
		self.playback_timer_id = None
		self.playback_stroke_index = 0
		self.draw_feedback(om.MEvent())


class AutoSculptorTestCmd(OpenMayaMPx.MPxCommand):
	kPluginCmdName = "autoSculptorTest"

	kBrushTypeFlag = "-bt"
	kBrushTypeLongFlag = "-brushType"
	kBrushSizeFlag = "-bs"
	kBrushSizeLongFlag = "-brushSize"
	kBrushStrengthFlag = "-bst"
	kBrushStrengthLongFlag = "-brushStrength"
	kBrushModeFlag = "-bm"
	kBrushModeLongFlag = "-brushMode"
	kBrushFalloffFlag = "-bf"
	kBrushFalloffLongFlag = "-brushFalloff"
	kSamplePositionFlag = "-sp"
	kSamplePositionLongFlag = "-samplePosition"
	kSampleNormalFlag = "-sn"
	kSampleNormalLongFlag = "-sampleNormal"
	kSamplePressureFlag = "-spr"
	kSamplePressureLongFlag = "-samplePressure"

	def __init__(self):
		OpenMayaMPx.MPxCommand.__init__(self)

	@staticmethod
	def cmdCreator():
		return AutoSculptorTestCmd()

	@staticmethod
	def syntaxCreator():
		syntax = om.MSyntax()
		syntax.addFlag(
			AutoSculptorTestCmd.kBrushTypeFlag,
			AutoSculptorTestCmd.kBrushTypeLongFlag,
			om.MSyntax.kString,
		)
		syntax.addFlag(
			AutoSculptorTestCmd.kBrushSizeFlag,
			AutoSculptorTestCmd.kBrushSizeLongFlag,
			om.MSyntax.kDouble,
		)
		syntax.addFlag(
			AutoSculptorTestCmd.kBrushStrengthFlag,
			AutoSculptorTestCmd.kBrushStrengthLongFlag,
			om.MSyntax.kDouble,
		)
		syntax.addFlag(
			AutoSculptorTestCmd.kBrushModeFlag,
			AutoSculptorTestCmd.kBrushModeLongFlag,
			om.MSyntax.kString,
		)
		syntax.addFlag(
			AutoSculptorTestCmd.kBrushFalloffFlag,
			AutoSculptorTestCmd.kBrushFalloffLongFlag,
			om.MSyntax.kString,
		)
		syntax.addFlag(
			AutoSculptorTestCmd.kSamplePositionFlag,
			AutoSculptorTestCmd.kSamplePositionLongFlag,
			om.MSyntax.kDouble,
			om.MSyntax.kDouble,
			om.MSyntax.kDouble,
		)
		syntax.addFlag(
			AutoSculptorTestCmd.kSampleNormalFlag,
			AutoSculptorTestCmd.kSampleNormalLongFlag,
			om.MSyntax.kDouble,
			om.MSyntax.kDouble,
			om.MSyntax.kDouble,
		)
		syntax.addFlag(
			AutoSculptorTestCmd.kSamplePressureFlag,
			AutoSculptorTestCmd.kSamplePressureLongFlag,
			om.MSyntax.kDouble,
		)

		syntax.useSelectionAsDefault(True)
		syntax.setObjectType(om.MSyntax.kSelectionList, 1, 1)
		return syntax

	def doIt(self, args):
		cmds.undoInfo(openChunk=True)
		try:
			arg_data = om.MArgDatabase(self.syntax(), args)

			selection = om.MSelectionList()
			arg_data.getObjects(selection)
			if selection.isEmpty():
				om.MGlobal.displayWarning("Please select a mesh first")
				return

			dag_path = om.MDagPath()
			selection.getDagPath(0, dag_path)

			if dag_path.apiType() != om.MFn.kMesh:
				om.MGlobal.displayWarning("Selected object is not a mesh")
				return
			mesh_name = dag_path.partialPathName()

			brush_type = (
				arg_data.flagArgumentString(self.kBrushTypeFlag, 0)
				if arg_data.isFlagSet(self.kBrushTypeFlag)
				else "surface"
			)
			brush_size = (
				arg_data.flagArgumentDouble(self.kBrushSizeFlag, 0)
				if arg_data.isFlagSet(self.kBrushSizeFlag)
				else 1.0
			)
			brush_strength = (
				arg_data.flagArgumentDouble(self.kBrushStrengthFlag, 0)
				if arg_data.isFlagSet(self.kBrushStrengthFlag)
				else 0.5
			)
			brush_mode_str = (
				arg_data.flagArgumentString(self.kBrushModeFlag, 0)
				if arg_data.isFlagSet(self.kBrushModeFlag)
				else "add"
			)
			brush_falloff = (
				arg_data.flagArgumentString(self.kBrushFalloffFlag, 0)
				if arg_data.isFlagSet(self.kBrushFalloffFlag)
				else "smooth"
			)

			try:
				brush_mode = BrushMode[brush_mode_str.upper()]
			except KeyError:
				om.MGlobal.displayError(f"Invalid brush mode: {brush_mode_str}")
				return

			if arg_data.isFlagSet(self.kSamplePositionFlag):
				sample_pos = [
					arg_data.flagArgumentDouble(self.kSamplePositionFlag, i)
					for i in range(3)
				]
			else:
				sample_pos = [0, 0, 0]

			if arg_data.isFlagSet(self.kSampleNormalFlag):
				sample_normal = [
					arg_data.flagArgumentDouble(self.kSampleNormalFlag, i)
					for i in range(3)
				]
			else:
				sample_normal = [0, 1, 0]

			sample_pressure = (
				arg_data.flagArgumentDouble(self.kSamplePressureFlag, 0)
				if arg_data.isFlagSet(self.kSamplePressureFlag)
				else 1.0
			)

			if brush_type == "surface":
				brush = SurfaceBrush(
					size=brush_size,
					strength=brush_strength,
					mode=brush_mode,
					falloff=brush_falloff,
				)
			elif brush_type == "freeform":
				brush = FreeformBrush(
					size=brush_size,
					strength=brush_strength,
					mode=brush_mode,
					falloff=brush_falloff,
				)
			else:
				om.MGlobal.displayError(f"Invalid brush type: {brush_type}")
				return

			mesh_data = MeshInterface.get_mesh_data(mesh_name)

			sample = brush.add_sample(sample_pos, sample_normal, sample_pressure, 0.0)

			brush.apply_to_mesh(mesh_data, sample)
			stroke = brush.end_stroke()
			om.MGlobal.displayInfo(f"Created stroke: {stroke}")

		except Exception as e:
			om.MGlobal.displayError(f"Error: {str(e)}")
			om.MGlobal.displayInfo(Utils.full_stack())
			return
		finally:
			cmds.undoInfo(closeChunk=True)

		om.MGlobal.displayInfo("AutoSculptor test completed successfully!")


def initializePlugin(mobject):
	mplugin = OpenMayaMPx.MFnPlugin(mobject, "AutoSculptor", "1.0", "Any")
	try:
		mplugin.registerCommand(
			AutoSculptorTestCmd.kPluginCmdName,
			AutoSculptorTestCmd.cmdCreator,
			AutoSculptorTestCmd.syntaxCreator,
		)
		om.MGlobal.displayInfo("AutoSculptorTestCmd loaded successfully!")

		mplugin.registerContextCommand(
			AutoSculptorContextCmd.kContextCmdName, AutoSculptorContextCmd.creator
		)

		mplugin.registerCommand(
			AutoSculptorToolCmd.kPluginCmdName,
			AutoSculptorToolCmd.cmdCreator,
			AutoSculptorToolCmd.syntaxCreator,
		)

		om.MGlobal.displayInfo("AutoSculptorTool loaded successfully!")

		context_name = cmds.autoSculptorContext()
		cmds.setToolTo(context_name)
		om.MGlobal.displayInfo(
			f"AutoSculptor plugin loaded successfully! Tool context: {context_name}"
		)
		om.MGlobal.displayInfo(
			f"To activate the tool, run: cmds.setToolTo('{context_name}')"
		)

	except Exception as e:
		sys.stderr.write(f"Failed to register command: autoSculptorTest: {e}\n")


def uninitializePlugin(mobject):
	mplugin = OpenMayaMPx.MFnPlugin(mobject)
	try:
		mplugin.deregisterCommand(AutoSculptorTestCmd.kPluginCmdName)
		om.MGlobal.displayInfo("AutoSculptorTestCmd unloaded successfully!")
		mplugin.deregisterCommand(AutoSculptorToolCmd.kPluginCmdName)
		mplugin.deregisterContextCommand(AutoSculptorContextCmd.kContextCmdName)
		om.MGlobal.displayInfo("AutoSculptorTool unloaded successfully!")
	except Exception as e:
		sys.stderr.write(f"Failed to unregister command: autoSculptorTest: {e}\n")
