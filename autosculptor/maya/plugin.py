import sys
import os

try:
	import maya.cmds as cmds  # type: ignore
	import maya.OpenMayaMPx as OpenMayaMPx  # type: ignore
	import maya.OpenMaya as om  # type: ignore
	import maya.api.OpenMayaRender as omr  # type: ignore
except ImportError:
	print("Maya modules not available. Running in standalone mode.")

from autosculptor.core.surface_brush import SurfaceBrush
from autosculptor.core.freeform_brush import FreeformBrush
from autosculptor.core.brush import BrushMode
from autosculptor.core.mesh_interface import MeshInterface
from autosculptor.utils.utils import Utils
from autosculptor.maya.test_ui import TestUI
from autosculptor.maya.viewport_drawer import SuggestionDrawer

CMD_AUTO_SCULPTOR_TEST = om.MTypeId(0x8723)

COMMAND_CATEGORY = "AutoSculptor"
TOGGLE_CAPTURE_CMD_NAME = "autoSculptorToggleCaptureCmd"
TOGGLE_SUGGESTIONS_CMD_NAME = "autoSculptorToggleSuggestionsCmd"
ACCEPT_SELECTED_CMD_NAME = "autoSculptorAcceptSelectedCmd"
ACCEPT_ALL_CMD_NAME = "autoSculptorAcceptAllCmd"

NAMED_COMMANDS = [
	TOGGLE_CAPTURE_CMD_NAME,
	TOGGLE_SUGGESTIONS_CMD_NAME,
	ACCEPT_SELECTED_CMD_NAME,
	ACCEPT_ALL_CMD_NAME,
]


class AutoSculptorTestCmd(OpenMayaMPx.MPxCommand):
	"""
	Command to test Brush.
	"""

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
		"""
		Create the syntax object for the command.
		"""
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
			print(f"Created stroke: {stroke}")

		except Exception as e:
			om.MGlobal.displayError(f"Error: {str(e)}")
			print(Utils.full_stack())
			return
		finally:
			cmds.undoInfo(closeChunk=True)

		om.MGlobal.displayInfo("AutoSculptor test completed successfully!")


class AutoSculptorDrawNode(OpenMayaMPx.MPxLocatorNode):
	"""Dummy node to host the draw override."""

	id = om.MTypeId(0x8724)

	def __init__(self):
		OpenMayaMPx.MPxLocatorNode.__init__(self)

	@classmethod
	def creator(cls):
		return OpenMayaMPx.asMPxPtr(cls())

	@classmethod
	def initialize(cls):
		pass


def show_sculpting_tool_window_action():
	window_name = "AutoSculptorToolWindow"
	if cmds.window(window_name, q=True, exists=True):
		cmds.deleteUI(window_name, window=True)

	import autosculptor.ui.ui as ui

	ui.show_sculpting_tool_window()


def create_test_sphere_action():
	cmds.polySphere(
		name="autoScuptorTestSphere", radius=10, subdivisionsX=50, subdivisionsY=50
	)


def create_menu():
	menu_name = "AutoSculptorMenu"
	if cmds.menu(menu_name, exists=True):
		cmds.deleteUI(menu_name)
	cmds.menu(menu_name, parent="MayaWindow", tearOff=False, label="AutoSculptor")

	cmds.menuItem(
		parent=menu_name,
		label="Open UI Panel",
		command=lambda *args: show_sculpting_tool_window_action(),
	)

	cmds.menuItem(
		parent=menu_name,
		label="Create Test Sphere",
		command=lambda *args: create_test_sphere_action(),
	)


def initializePlugin(mobject):
	mplugin = OpenMayaMPx.MFnPlugin(mobject, "AutoSculptor", "1.0", "Any")
	try:
		mplugin.registerCommand(
			AutoSculptorTestCmd.kPluginCmdName,
			AutoSculptorTestCmd.cmdCreator,
			AutoSculptorTestCmd.syntaxCreator,
		)

		mplugin.registerNode(
			"autoSculptorDrawOverride",
			AutoSculptorDrawNode.id,
			AutoSculptorDrawNode.creator,
			AutoSculptorDrawNode.initialize,
			OpenMayaMPx.MPxNode.kLocatorNode,
		)
		omr.MDrawRegistry.registerDrawOverrideCreator(
			SuggestionDrawer.drawDBClassification,
			"autoSculptorDrawOverride",
			SuggestionDrawer.creator,
		)

		create_menu()

		om.MGlobal.displayInfo(
			"AutoSculptor plugin loaded. UI, commands, and hotkeys registered!"
		)
	except Exception as e:
		sys.stderr.write(f"Failed to initialize AutoSculptor: {e}\n")
		import traceback

		traceback.print_exc()
		# raise


def uninitializePlugin(mobject):
	mplugin = OpenMayaMPx.MFnPlugin(mobject)
	try:
		mplugin.deregisterCommand(AutoSculptorTestCmd.kPluginCmdName)
		omr.MDrawRegistry.deregisterDrawOverrideCreator(
			SuggestionDrawer.drawDBClassification, SuggestionDrawer.drawRegistrantId
		)
		om.MGlobal.displayInfo(
			"AutoSculptor plugin unloaded, command and draw override deregistered!"
		)
		cmds.deleteUI("AutoSculptorMenu", menu=True)

		om.MGlobal.displayInfo("AutoSculptor plugin unloaded.")

	except Exception as e:
		sys.stderr.write(f"Failed to clean up AutoSculptor: {e}\n")
		import traceback

		traceback.print_exc()
		# raise
