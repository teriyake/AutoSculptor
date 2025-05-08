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

CMD_AUTO_SCULPTOR_TEST = om.MTypeId(0x8723)


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


class AutoSculptorActionCmd(OpenMayaMPx.MPxCommand):
	"""
	Command to perform various AutoSculptor actions like toggling capture/suggestions,
	and accepting/rejecting suggestions.
	"""

	kPluginCmdName = "autoSculptorAction"

	kToggleCaptureFlag = "-tc"
	kToggleCaptureLongFlag = "-toggleCapture"
	kToggleSuggestionsFlag = "-ts"
	kToggleSuggestionsLongFlag = "-toggleSuggestions"
	kAcceptSelectedFlag = "-as"
	kAcceptSelectedLongFlag = "-acceptSelectedSuggestion"
	kAcceptAllFlag = "-aa"
	kAcceptAllLongFlag = "-acceptAllSuggestions"
	kRejectSelectedFlag = "-rs"
	kRejectSelectedLongFlag = "-rejectSelectedSuggestion"

	def __init__(self):
		OpenMayaMPx.MPxCommand.__init__(self)

	@staticmethod
	def cmdCreator():
		print("DEBUG: AutoSculptorActionCmd cmdCreator called")
		try:
			cmd_instance = AutoSculptorActionCmd()
			print("DEBUG: AutoSculptorActionCmd instance created successfully")
			return cmd_instance
		except Exception as e:
			print(f"DEBUG: Exception in AutoSculptorActionCmd cmdCreator: {e}")
			import traceback

			traceback.print_exc()
			raise

	@staticmethod
	def syntaxCreator():
		"""
		Create the syntax object for the command.
		"""
		syntax = om.MSyntax()
		syntax.addFlag(
			AutoSculptorActionCmd.kToggleCaptureFlag,
			AutoSculptorActionCmd.kToggleCaptureLongFlag,
			om.MSyntax.kBoolean,
		)
		syntax.addFlag(
			AutoSculptorActionCmd.kToggleSuggestionsFlag,
			AutoSculptorActionCmd.kToggleSuggestionsLongFlag,
			om.MSyntax.kBoolean,
		)
		syntax.addFlag(
			AutoSculptorActionCmd.kAcceptSelectedFlag,
			AutoSculptorActionCmd.kAcceptSelectedLongFlag,
			om.MSyntax.kBoolean,
		)
		syntax.addFlag(
			AutoSculptorActionCmd.kAcceptAllFlag,
			AutoSculptorActionCmd.kAcceptAllLongFlag,
			om.MSyntax.kBoolean,
		)
		syntax.addFlag(
			AutoSculptorActionCmd.kRejectSelectedFlag,
			AutoSculptorActionCmd.kRejectSelectedLongFlag,
			om.MSyntax.kBoolean,
		)

		return syntax

	def doIt(self, args):
		try:
			arg_data = om.MArgDatabase(self.syntax(), args)

			global sculpting_tool_window
			if (
				"sculpting_tool_window" not in globals()
				or not sculpting_tool_window
				or not hasattr(sculpting_tool_window, "sculpt_capture")
			):
				om.MGlobal.displayWarning(
					"AutoSculptor UI window is not open or not fully initialized."
				)
				return

			sculpt_capture = sculpting_tool_window.sculpt_capture

			if arg_data.isFlagSet(self.kToggleCaptureFlag):
				if sculpt_capture.is_capturing:
					sculpt_capture.stop_capture()
					om.MGlobal.displayInfo("AutoSculptor Capture Disabled.")
					sculpting_tool_window.sculpting_tab.enable_capture.setChecked(False)
				else:
					if not sculpt_capture.mesh_name:
						sculpting_tool_window.sculpting_tab.on_select_mesh()

					if sculpt_capture.mesh_name:
						sculpt_capture.start_capture()
						om.MGlobal.displayInfo("AutoSculptor Capture Enabled.")
						sculpting_tool_window.sculpting_tab.enable_capture.setChecked(
							True
						)
					else:
						om.MGlobal.displayWarning(
							"Cannot enable capture: No mesh selected."
						)
						sculpting_tool_window.sculpting_tab.enable_capture.setChecked(
							False
						)

			if arg_data.isFlagSet(self.kToggleSuggestionsFlag):
				is_enabled = sculpt_capture.suggestions_enabled
				sculpt_capture.set_suggestions_enabled(not is_enabled)
				om.MGlobal.displayInfo(
					f"AutoSculptor Suggestions {'Enabled' if not is_enabled else 'Disabled'}."
				)
				sculpting_tool_window.suggestion_tab.enable_prediction.setChecked(
					not is_enabled
				)

			if arg_data.isFlagSet(self.kAcceptSelectedFlag):
				selected_indexes = (
					sculpting_tool_window.suggestion_tab.stroke_list.selectedIndexes()
				)
				if selected_indexes:
					selection_index = selected_indexes[0].row()
					if sculpt_capture.suggestions_enabled:
						sculpt_capture.accept_selected_suggestion(selection_index)
						om.MGlobal.displayInfo(
							f"Accepted suggestion index {selection_index}."
						)
					else:
						om.MGlobal.displayWarning(
							"Suggestions are disabled. Cannot accept."
						)
				else:
					om.MGlobal.displayWarning("No suggestion selected to accept.")

			if arg_data.isFlagSet(self.kAcceptAllFlag):
				if sculpt_capture.suggestions_enabled:
					sculpt_capture.accept_all_suggestions()
					om.MGlobal.displayInfo("Accepted all suggestions.")
				else:
					om.MGlobal.displayWarning(
						"Suggestions are disabled. Cannot accept all."
					)

			if arg_data.isFlagSet(self.kRejectSelectedFlag):
				selected_indexes = (
					sculpting_tool_window.suggestion_tab.stroke_list.selectedIndexes()
				)
				if not selected_indexes:
					om.MGlobal.displayWarning("Please select a suggestion to reject.")
					return

				selection_index = selected_indexes[0].row()

				if sculpt_capture.suggestions_enabled:
					sculpt_capture.reject_suggestion(selection_index)
					om.MGlobal.displayInfo(
						f"Rejected suggestion index {selection_index}."
					)
					if (
						sculpting_tool_window.suggestion_tab.workflow
						and 0
						<= selection_index
						< len(sculpting_tool_window.suggestion_tab.workflow.strokes)
					):
						del sculpting_tool_window.suggestion_tab.workflow.strokes[
							selection_index
						]
						sculpting_tool_window.suggestion_tab.update(
							sculpting_tool_window.suggestion_tab.workflow
						)
						sculpting_tool_window.suggestion_tab.sample_list.setRowCount(0)
				else:
					om.MGlobal.displayWarning(
						"Suggestions are disabled. Cannot reject."
					)

		except Exception as e:
			om.MGlobal.displayError(f"Error executing AutoSculptor action: {str(e)}")
			import traceback

			traceback.print_exc()


def show_sculpting_tool_window_action():
	window_name = "AutoSculptorToolWindow"
	if cmds.window(window_name, q=True, exists=True):
		cmds.deleteUI(window_name, window=True)

	import autosculptor.ui.ui as ui

	global sculpting_tool_window
	sculpting_tool_window = ui.show_sculpting_tool_window()


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

	cmds.menuItem(
		parent=menu_name,
		label="Toggle Capture",
		command=lambda *args: cmds.autoSculptorAction(toggleCapture=True),
	)
	cmds.menuItem(
		parent=menu_name,
		label="Toggle Suggestions",
		command=lambda *args: cmds.autoSculptorAction(toggleSuggestions=True),
	)
	cmds.menuItem(
		parent=menu_name,
		label="Accept Selected Suggestion",
		command=lambda *args: cmds.autoSculptorAction(acceptSelectedSuggestion=True),
	)
	cmds.menuItem(
		parent=menu_name,
		label="Accept All Suggestions",
		command=lambda *args: cmds.autoSculptorAction(acceptAllSuggestions=True),
	)
	cmds.menuItem(
		parent=menu_name,
		label="Reject Selected Suggestion",
		command=lambda *args: cmds.autoSculptorAction(rejectSelectedSuggestion=True),
	)


def initializePlugin(mobject):
	mplugin = OpenMayaMPx.MFnPlugin(mobject, "AutoSculptor", "1.0", "Any")
	try:
		mplugin.registerCommand(
			AutoSculptorTestCmd.kPluginCmdName,
			AutoSculptorTestCmd.cmdCreator,
			AutoSculptorTestCmd.syntaxCreator,
		)
		mplugin.registerCommand(
			AutoSculptorActionCmd.kPluginCmdName,
			AutoSculptorActionCmd.cmdCreator,
			AutoSculptorActionCmd.syntaxCreator,
		)

		create_menu()
		cmds.nameCommand(
			"AutoSculptorToggleCaptureNamedCmd",
			annotation="Toggle Capture",
			command=f"cmds.{AutoSculptorActionCmd.kPluginCmdName}({AutoSculptorActionCmd.kToggleCaptureLongFlag}=True)",
		)
		cmds.nameCommand(
			"AutoSculptorToggleSuggestionNamedCmd",
			annotation="Toggle Suggestion",
			command=f"cmds.{AutoSculptorActionCmd.kPluginCmdName}({AutoSculptorActionCmd.kToggleSuggestionsLongFlag}=True)",
		)
		cmds.nameCommand(
			"AutoSculptorAcceptSelectedNamedCmd",
			annotation="Accept Selected Suggestion(s)",
			command=f"cmds.{AutoSculptorActionCmd.kPluginCmdName}({AutoSculptorActionCmd.kAcceptSelectedLongFlag}=True)",
		)
		cmds.nameCommand(
			"AutoSculptorAcceptAllNamedCmd",
			annotation="Accept All Suggestions",
			command=f"cmds.{AutoSculptorActionCmd.kPluginCmdName}({AutoSculptorActionCmd.kAcceptAllLongFlag}=True)",
		)
		cmds.nameCommand(
			"AutoSculptorRejectSelectedNamedCmd",
			annotation="Reject Selected Suggestions",
			command=f"cmds.{AutoSculptorActionCmd.kPluginCmdName}({AutoSculptorActionCmd.kRejectSelectedLongFlag}=True)",
		)

		om.MGlobal.displayInfo(
			"AutoSculptor plugin loaded. UI and commands registered!"
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
		mplugin.deregisterCommand(AutoSculptorActionCmd.kPluginCmdName)

		cmds.deleteUI("AutoSculptorMenu", menu=True)

		om.MGlobal.displayInfo("AutoSculptor plugin unloaded.")

	except Exception as e:
		sys.stderr.write(f"Failed to clean up AutoSculptor: {e}\n")
		import traceback

		traceback.print_exc()
		# raise
