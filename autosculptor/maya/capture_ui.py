import maya.cmds as cmds
import maya.api.OpenMaya as om2
import sys
import os

from autosculptor.core.data_structures import Sample, Stroke, Workflow
from autosculptor.maya.viewport_drawer import ViewportDataCache
from functools import partial


class SculptCaptureUI:
	WINDOW_NAME = "SculptCaptureWindow"

	def __init__(self, sculpt_capture_instance=None):
		if sculpt_capture_instance:
			self.capture = sculpt_capture_instance
		else:
			from autosculptor.maya.capture import SculptCapture

			self.capture = SculptCapture()

		self.is_capturing = False
		self.create_ui()
		self.viewport_cache = ViewportDataCache()
		self.drawer_node = None

	def create_ui(self):
		if cmds.window(self.WINDOW_NAME, exists=True):
			cmds.deleteUI(self.WINDOW_NAME)

		cmds.window(
			self.WINDOW_NAME,
			title="AutoSculptor Capture",
			width=400,
			height=300,
			closeCommand=self.on_close,
		)

		main_layout = cmds.columnLayout(adjustableColumn=True, columnAttach=("both", 5))

		cmds.text(label="AutoSculptor Stroke Capture", font="boldLabelFont", height=30)
		cmds.text(label="Record sculpting strokes for suggestion synthesis", height=25)
		cmds.separator(height=10, style="none")

		cmds.frameLayout(label="Mesh Selection", collapsable=True, collapse=False)
		cmds.columnLayout(adjustableColumn=True, columnAttach=("both", 5))

		cmds.rowLayout(numberOfColumns=2, columnWidth2=(300, 80), adjustableColumn=1)
		self.mesh_text = cmds.textField(
			editable=False, placeholderText="No mesh selected"
		)
		cmds.button(label="Select Mesh", command=self.select_mesh)
		cmds.setParent("..")

		cmds.separator(height=10)
		cmds.setParent("..")
		cmds.setParent("..")

		cmds.frameLayout(label="Capture Controls", collapsable=True, collapse=False)
		cmds.columnLayout(adjustableColumn=True, columnAttach=("both", 5))

		cmds.rowLayout(numberOfColumns=2, columnWidth2=(190, 190), adjustableColumn=1)
		self.start_button = cmds.button(
			label="Start Capturing",
			command=self.start_capture,
			backgroundColor=[0.2, 0.7, 0.2],
		)
		self.stop_button = cmds.button(
			label="Stop Capturing",
			command=self.stop_capture,
			backgroundColor=[0.7, 0.2, 0.2],
			enable=False,
		)
		cmds.setParent("..")

		cmds.separator(height=10)
		cmds.button(
			label="Create Test Sphere",
			command=self.create_test_object,
			annotation="Creates a polygon sphere for testing",
		)

		cmds.separator(height=15)
		self.status_text = cmds.text(
			label="Ready to capture", font="boldLabelFont", height=30
		)
		cmds.setParent("..")
		cmds.setParent("..")

		cmds.frameLayout(label="Captured Data", collapsable=True, collapse=False)
		cmds.columnLayout(adjustableColumn=True, columnAttach=("both", 5))

		cmds.rowLayout(
			numberOfColumns=3, columnWidth3=(130, 130, 130), adjustableColumn=1
		)
		cmds.button(label="Print Workflow", command=self.print_workflow)
		cmds.button(label="Print Last Stroke", command=self.print_last_stroke)
		cmds.button(label="Clear Data", command=self.clear_data)
		cmds.setParent("..")

		cmds.separator(height=10)
		self.stats_text = cmds.text(label="No data captured yet", height=25)

		cmds.setParent("..")
		cmds.setParent("..")

		cmds.showWindow(self.WINDOW_NAME)
		self.update_stats()

		# cmds.scriptJob(event=["SelectionChanged", self.update_viewport_suggestions])

	def select_mesh(self, *args):
		mesh_name = self.capture.get_selected_mesh_name()
		if mesh_name:
			short_name = mesh_name.split("|")[-1]
			cmds.textField(self.mesh_text, edit=True, text=short_name)
			self.capture.mesh_name = mesh_name
			cmds.text(self.status_text, edit=True, label=f"Selected mesh: {short_name}")
		else:
			cmds.textField(self.mesh_text, edit=True, text="")
			cmds.text(self.status_text, edit=True, label="Failed to select mesh")

	def start_capture(self, *args):
		if not self.capture.mesh_name:
			self.select_mesh()
			if not self.capture.mesh_name:
				om2.MGlobal.displayError("Please select a mesh first")
				return

		self.setup_viewport_drawing()
		self.capture.register_script_job()

		self.is_capturing = True
		cmds.button(self.start_button, edit=True, enable=False)
		cmds.button(self.stop_button, edit=True, enable=True)
		cmds.text(self.status_text, edit=True, label="✓ Capturing strokes...")

		self.update_viewport_suggestions()

	def stop_capture(self, *args):
		self.capture.unregister_script_job()

		self.is_capturing = False
		cmds.button(self.start_button, edit=True, enable=True)
		cmds.button(self.stop_button, edit=True, enable=False)
		cmds.text(self.status_text, edit=True, label="Capture stopped")
		self.update_stats()

	def setup_viewport_drawing(self):
		"""Create and configure viewport overlay"""
		if not cmds.objExists("AutoSculptorOverlay"):
			self.drawer_node = cmds.createNode("transform", name="AutoSculptorOverlay")
			cmds.setAttr(self.drawer_node + ".overrideEnabled", 1)
			# cmds.setAttr(self.drawer_node + ".overrideDisplayType", 2)
			cmds.setAttr(self.drawer_node + ".overrideDisplayType", 0)

		if not cmds.objExists("AutoSculptorDrawNode"):
			draw_node = cmds.createNode(
				"autoSculptorDrawOverride",
				name="AutoSculptorDrawNode",
				parent="AutoSculptorOverlay",
			)
			# cmds.parent(draw_node, "AutoSculptorOverlay")
			cmds.setAttr(draw_node + ".overrideEnabled", 1)

		cmds.setAttr("AutoSculptorOverlay.overrideEnabled", 1)
		cmds.setAttr("AutoSculptorOverlay.overrideVisibility", 0)
		cmds.setAttr("AutoSculptorOverlay.overrideDisplayType", 0)

	def update_viewport_suggestions(self, *args):
		"""Main update method for viewport suggestions"""
		if not self.capture.mesh_name:
			return

		try:
			self.viewport_cache.update_suggestions(self.capture.current_suggestions)
			self.refresh_viewport()

		except Exception as e:
			print(f"Suggestion update failed: {str(e)}")

	def create_test_object(self, *args):
		sphere_name = "autoSculptorTestSphere"
		count = 1

		while cmds.objExists(f"{sphere_name}{count}"):
			count += 1

		sphere_name = f"{sphere_name}{count}"

		cmds.polySphere(name=sphere_name, radius=5, subdivisionsX=40, subdivisionsY=40)
		cmds.select(sphere_name)

		self.select_mesh()
		cmds.text(
			self.status_text, edit=True, label=f"Created test sphere: {sphere_name}"
		)

	def print_workflow(self, *args):
		workflow = self.capture.current_workflow
		if not workflow or len(workflow.strokes) == 0:
			print("No workflow data captured yet.")
			return

		print("\n==== WORKFLOW DETAILS ====")
		print(f"Total strokes: {len(workflow.strokes)}")
		total_samples = sum(len(stroke) for stroke in workflow.strokes)
		print(f"Total samples: {total_samples}")

		for i, stroke in enumerate(workflow.strokes):
			print(f"\nStroke {i+1}:")
			print(f"  Type: {stroke.stroke_type}")
			print(f"  Samples: {len(stroke.samples)}")
			if len(stroke.samples) > 0:
				first_sample = stroke.samples[0]
				print(f"  First sample position: {first_sample.position}")
				print(f"  First sample normal: {first_sample.normal}")
				print(f"  Brush size: {first_sample.size}")
				print(f"  Brush pressure: {first_sample.pressure}")

	def print_last_stroke(self, *args):
		workflow = self.capture.current_workflow
		if not workflow or len(workflow.strokes) == 0:
			print("No strokes captured yet.")
			return

		last_stroke = workflow.strokes[-1]
		print("\n==== LAST STROKE DETAILS ====")
		print(f"Stroke type: {last_stroke.stroke_type}")
		print(f"Number of samples: {len(last_stroke.samples)}")

		for i, sample in enumerate(
			last_stroke.samples[: min(10, len(last_stroke.samples))]
		):
			print(f"\nSample {i+1}:")
			print(f"  Position: {sample.position}")
			print(f"  Normal: {sample.normal}")
			print(f"  Size: {sample.size}")
			print(f"  Pressure: {sample.pressure}")
			print(f"  Timestamp: {sample.timestamp}")

		if len(last_stroke.samples) > 10:
			print(f"\n... and {len(last_stroke.samples) - 10} more samples")

	def clear_data(self, *args):
		self.capture.current_workflow = Workflow()
		self.capture.previous_positions = {}
		cmds.text(self.status_text, edit=True, label="Data cleared")
		self.update_stats()

	def update_stats(self, *args):
		workflow = self.capture.current_workflow
		if workflow and len(workflow.strokes) > 0:
			total_samples = sum(len(stroke) for stroke in workflow.strokes)
			cmds.text(
				self.stats_text,
				edit=True,
				label=f"Strokes: {len(workflow.strokes)} | Samples: {total_samples}",
			)
		else:
			cmds.text(self.stats_text, edit=True, label="No data captured yet")

	def refresh_viewport(self):
		cmds.refresh(currentView=True, force=False)

	def on_close(self, *args):
		if self.is_capturing:
			self.stop_capture()

		if self.drawer_node and cmds.objExists(self.drawer_node):
			cmds.delete(self.drawer_node)

		self.viewport_cache.update_suggestions([])
		self.refresh_viewport()

	@staticmethod
	def launch(sculpt_capture_instance=None):
		ui = SculptCaptureUI(sculpt_capture_instance)
		return ui


def main():
	ui = SculptCaptureUI.launch()
	return ui


if __name__ == "__main__":
	ui = main()
