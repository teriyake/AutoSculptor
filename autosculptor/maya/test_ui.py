import sys
import os
from functools import partial
import uuid

try:
    import maya.cmds as cmds
    import maya.OpenMaya as om
except ImportError:
    print("Maya modules not available. Running in standalone mode.")

from autosculptor.core.brush import BrushMode
from autosculptor.core.surface_brush import SurfaceBrush
from autosculptor.core.freeform_brush import FreeformBrush
from autosculptor.core.mesh_interface import MeshInterface
from autosculptor.core.data_structures import Sample, Stroke


class TestUI:
    """
    UI for testing the AutoSculptor functionality.
    Creates a window with controls for all brush parameters and samples management.
    """

    WINDOW_NAME = "autoSculptorTestUI"
    WINDOW_TITLE = "AutoSculptor Test Tool"

    def __init__(self):
        self.samples = []
        self.active_brush = None
        self.active_mesh_data = None
        self.active_mesh_name = None
        self.current_stroke_id = None
        self.create_ui()

    def create_ui(self):
        """Create the main UI window and controls."""
        if cmds.window(self.WINDOW_NAME, exists=True):
            cmds.deleteUI(self.WINDOW_NAME)

        cmds.window(self.WINDOW_NAME, title=self.WINDOW_TITLE, widthHeight=(400, 650))

        main_layout = cmds.columnLayout(adjustableColumn=True, columnAttach=("both", 5))

        cmds.frameLayout(
            label="Mesh Selection", collapsable=True, marginHeight=5, marginWidth=5
        )
        cmds.button(label="Use Selected Mesh", command=self.use_selected_mesh)
        self.mesh_text = cmds.textField(editable=False, text="No mesh selected")
        cmds.setParent("..")

        cmds.frameLayout(
            label="Brush Parameters", collapsable=True, marginHeight=5, marginWidth=5
        )

        cmds.text(label="Brush Type:")
        self.brush_type = cmds.optionMenu()
        cmds.menuItem(label="surface")
        cmds.menuItem(label="freeform")

        cmds.text(label="Brush Size:")
        self.brush_size = cmds.floatSliderGrp(
            field=True, minValue=0.1, maxValue=10.0, value=1.0, step=0.1
        )

        cmds.text(label="Brush Strength:")
        self.brush_strength = cmds.floatSliderGrp(
            field=True, minValue=0.01, maxValue=1.0, value=0.5, step=0.01
        )

        cmds.text(label="Brush Mode:")
        self.brush_mode = cmds.optionMenu()
        cmds.menuItem(label="ADD")
        cmds.menuItem(label="SUBTRACT")
        cmds.menuItem(label="SMOOTH")

        cmds.text(label="Brush Falloff:")
        self.brush_falloff = cmds.optionMenu()
        cmds.menuItem(label="smooth")
        cmds.menuItem(label="linear")
        cmds.menuItem(label="constant")
        cmds.setParent("..")

        cmds.frameLayout(
            label="Current Stroke", collapsable=True, marginHeight=5, marginWidth=5
        )
        self.stroke_status = cmds.textField(editable=False, text="No active stroke")

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(180, 180), adjustableColumn=2)
        cmds.button(
            label="Begin New Stroke",
            command=self.begin_new_stroke,
            backgroundColor=[0.2, 0.5, 0.2],
        )
        cmds.button(
            label="End Current Stroke",
            command=self.end_current_stroke,
            backgroundColor=[0.5, 0.2, 0.2],
        )
        cmds.setParent("..")
        cmds.setParent("..")

        self.samples_frame = cmds.frameLayout(
            label="Samples", collapsable=True, marginHeight=5, marginWidth=5
        )

        self.samples_layout = cmds.columnLayout(adjustableColumn=True)

        self.sample_count_text = cmds.text(label="0 samples added")

        self.sample_list = cmds.textScrollList(
            numberOfRows=5, allowMultiSelection=True, selectCommand=self.sample_selected
        )

        cmds.separator(height=10)
        cmds.text(label="Sample Position (XYZ):")
        self.sample_pos = cmds.floatFieldGrp(
            numberOfFields=3, value1=0.0, value2=0.0, value3=0.0
        )

        cmds.text(label="Sample Normal (XYZ):")
        self.sample_normal = cmds.floatFieldGrp(
            numberOfFields=3, value1=0.0, value2=0.0, value3=0.0
        )

        cmds.text(label="Sample Pressure:")
        self.sample_pressure = cmds.floatSliderGrp(
            field=True, minValue=0.0, maxValue=1.0, value=1.0, step=0.01
        )

        cmds.text(label="Sample Time:")
        self.sample_time = cmds.floatField(value=0.0, step=0.1)

        cmds.rowLayout(
            numberOfColumns=3, columnWidth3=(120, 120, 120), adjustableColumn=3
        )
        cmds.button(label="Add Sample", command=self.add_sample)
        cmds.button(label="Remove Selected", command=self.remove_selected_samples)
        cmds.button(label="Clear All", command=self.clear_samples)
        cmds.setParent("..")

        cmds.separator(height=10)
        cmds.rowLayout(numberOfColumns=2, columnWidth2=(180, 180), adjustableColumn=2)
        cmds.button(label="Add From Selection", command=self.add_from_selection)
        cmds.button(label="Auto Time Spacing", command=self.auto_time_spacing)
        cmds.setParent("..")

        cmds.setParent("..")
        cmds.setParent("..")

        cmds.separator(height=10)
        cmds.button(
            label="Apply All Samples",
            command=self.apply_all_samples,
            height=50,
            backgroundColor=[0.2, 0.6, 0.2],
        )

        cmds.showWindow(self.WINDOW_NAME)

    def use_selected_mesh(self, *args):
        """Get the currently selected mesh and update the UI."""
        selection = om.MSelectionList()
        om.MGlobal.getActiveSelectionList(selection)

        if selection.isEmpty():
            cmds.textField(self.mesh_text, edit=True, text="No mesh selected")
            return

        dag_path = om.MDagPath()
        selection.getDagPath(0, dag_path)

        if dag_path.apiType() != om.MFn.kMesh:
            cmds.textField(
                self.mesh_text, edit=True, text="Selected object is not a mesh"
            )
            return

        mesh_name = dag_path.partialPathName()
        cmds.textField(self.mesh_text, edit=True, text=mesh_name)
        self.active_mesh_name = mesh_name
        self.active_mesh_data = MeshInterface.get_mesh_data(mesh_name)

    def begin_new_stroke(self, *args):
        """Begin a new brush stroke."""
        if not self.active_mesh_name:
            cmds.warning("Please select a mesh first")
            return

        brush_type = cmds.optionMenu(self.brush_type, query=True, value=True)
        brush_size = cmds.floatSliderGrp(self.brush_size, query=True, value=True)
        brush_strength = cmds.floatSliderGrp(
            self.brush_strength, query=True, value=True
        )
        brush_mode_str = cmds.optionMenu(self.brush_mode, query=True, value=True)
        brush_falloff = cmds.optionMenu(self.brush_falloff, query=True, value=True)

        try:
            brush_mode = BrushMode[brush_mode_str]
        except KeyError:
            cmds.warning(f"Invalid brush mode: {brush_mode_str}")
            return

        if brush_type == "surface":
            self.active_brush = SurfaceBrush(
                size=brush_size,
                strength=brush_strength,
                mode=brush_mode,
                falloff=brush_falloff,
            )
        elif brush_type == "freeform":
            self.active_brush = FreeformBrush(
                size=brush_size,
                strength=brush_strength,
                mode=brush_mode,
                falloff=brush_falloff,
            )
        else:
            cmds.warning(f"Invalid brush type: {brush_type}")
            return

        self.active_brush.begin_stroke()
        print(f"current active stroke: {self.active_brush.current_stroke}")

        self.current_stroke_id = str(uuid.uuid4())

        cmds.textField(
            self.stroke_status,
            edit=True,
            text=f"Active stroke: {self.current_stroke_id[:8]}... (0 samples)",
        )

        self.clear_samples()

        cmds.inViewMessage(message="New stroke started!", pos="midCenter", fade=True)

    def end_current_stroke(self, *args):
        """End the current brush stroke."""
        if not self.active_brush or not self.current_stroke_id:
            cmds.warning("No active stroke to end")
            return

        stroke = self.active_brush.end_stroke()

        cmds.textField(self.stroke_status, edit=True, text="No active stroke")

        sample_count = len(self.samples)
        cmds.inViewMessage(
            message=f"Stroke ended with {sample_count} samples",
            pos="midCenter",
            fade=True,
        )

        self.current_stroke_id = None

    def add_sample(self, *args):
        """Add a sample with the current UI values to the samples list."""
        if not self.active_brush or not self.current_stroke_id:
            cmds.warning("Please begin a stroke first")
            return

        position = cmds.floatFieldGrp(self.sample_pos, query=True, value=True)
        normal = cmds.floatFieldGrp(self.sample_normal, query=True, value=True)
        pressure = cmds.floatSliderGrp(self.sample_pressure, query=True, value=True)
        timestamp = cmds.floatField(self.sample_time, query=True, value=True)

        sample = self.active_brush.add_sample(
            position=position, normal=normal, pressure=pressure, timestamp=timestamp
        )

        self.samples.append(
            {
                "id": len(self.samples),
                "position": position,
                "normal": normal,
                "pressure": pressure,
                "time": timestamp,
                "sample_obj": sample,
            }
        )

        if self.active_mesh_data:
            self.active_brush.apply_to_mesh(self.active_mesh_data, sample)

        self.update_sample_list()

        cmds.textField(
            self.stroke_status,
            edit=True,
            text=f"Active stroke: {self.current_stroke_id[:8]}... ({len(self.samples)} samples)",
        )

        cmds.text(self.sample_count_text, edit=True, backgroundColor=[0.2, 0.6, 0.2])
        cmds.evalDeferred(
            partial(
                cmds.text,
                self.sample_count_text,
                edit=True,
                backgroundColor=[0.27, 0.27, 0.27],
            )
        )

    def update_sample_list(self):
        """Update the UI list of samples."""
        cmds.textScrollList(self.sample_list, edit=True, removeAll=True)

        for i, sample in enumerate(self.samples):
            pos = sample["position"]
            pressure = sample["pressure"]
            time = sample["time"]

            display_text = f"Sample {i}: Pos({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | P: {pressure:.2f} | T: {time:.2f}"
            cmds.textScrollList(self.sample_list, edit=True, append=display_text)

        self.update_sample_count()

    def sample_selected(self):
        """Handler for when a sample is selected in the list."""
        selected_indices = cmds.textScrollList(
            self.sample_list, query=True, selectIndexedItem=True
        )
        if not selected_indices:
            return

        sample_idx = selected_indices[0] - 1
        if 0 <= sample_idx < len(self.samples):
            sample = self.samples[sample_idx]

            cmds.floatFieldGrp(
                self.sample_pos,
                edit=True,
                value1=sample["position"][0],
                value2=sample["position"][1],
                value3=sample["position"][2],
            )
            cmds.floatFieldGrp(
                self.sample_normal,
                edit=True,
                value1=sample["normal"][0],
                value2=sample["normal"][1],
                value3=sample["normal"][2],
            )
            cmds.floatSliderGrp(
                self.sample_pressure, edit=True, value=sample["pressure"]
            )
            cmds.floatField(self.sample_time, edit=True, value=sample["time"])

    def remove_selected_samples(self, *args):
        """Remove the selected samples from the list."""
        if not self.active_brush or not self.active_brush.current_stroke:
            return

        selected_indices = cmds.textScrollList(
            self.sample_list, query=True, selectIndexedItem=True
        )
        if not selected_indices:
            return

        indices_to_remove = [idx - 1 for idx in selected_indices]
        indices_to_remove.sort(reverse=True)

        for idx in indices_to_remove:
            if 0 <= idx < len(self.samples):
                self.samples.pop(idx)

        self.recreate_stroke()

        self.update_sample_list()

        if self.current_stroke_id:
            cmds.textField(
                self.stroke_status,
                edit=True,
                text=f"Active stroke: {self.current_stroke_id[:8]}... ({len(self.samples)} samples)",
            )

    def recreate_stroke(self):
        """Recreate the current stroke with the current samples (after removing some)."""
        if not self.active_brush:
            return

        self.active_brush.end_stroke()

        self.active_brush.begin_stroke()

        for sample in self.samples:
            new_sample = self.active_brush.add_sample(
                position=sample["position"],
                normal=sample["normal"],
                pressure=sample["pressure"],
                timestamp=sample["time"],
            )

            sample["sample_obj"] = new_sample

            if self.active_mesh_data:
                self.active_brush.apply_to_mesh(self.active_mesh_data, new_sample)

    def auto_time_spacing(self, *args):
        """Automatically space out the time values for all samples."""
        if not self.samples:
            return

        for i, sample in enumerate(self.samples):
            sample["time"] = float(i)

        self.recreate_stroke()

        self.update_sample_list()

        cmds.inViewMessage(
            message="Auto time spacing applied!", pos="midCenter", fade=True
        )

    def clear_samples(self, *args):
        """Clear all samples from the list."""
        if self.active_brush and self.active_brush.current_stroke:
            self.active_brush.end_stroke()
            self.active_brush.begin_stroke()

        self.samples = []
        self.update_sample_list()

        if self.current_stroke_id:
            cmds.textField(
                self.stroke_status,
                edit=True,
                text=f"Active stroke: {self.current_stroke_id[:8]}... (0 samples)",
            )

    def update_sample_count(self):
        """Update the sample count display."""
        cmds.text(
            self.sample_count_text,
            edit=True,
            label=f"{len(self.samples)} sample{'s' if len(self.samples) != 1 else ''} added",
        )

    def add_from_selection(self, *args):
        """Add a sample based on the currently selected vertex or point on mesh."""
        if not self.active_brush or not self.current_stroke_id:
            cmds.warning("Please begin a stroke first")
            return

        selection = om.MSelectionList()
        position = [0, 0, 0]
        normal = [0, 1, 0]

        om.MGlobal.getActiveSelectionList(selection)
        if selection.isEmpty():
            cmds.warning(
                "Nothing selected. Select a vertex or use Ctrl+click on mesh surface."
            )
            return

        try:
            if self.is_vertex_selected():
                vertex_position = cmds.pointPosition(
                    cmds.ls(selection=True, flatten=True)[0]
                )
                position = list(vertex_position)

                try:
                    mesh_name = self.active_mesh_name
                    if mesh_name:
                        vertex_id = self.get_selected_vertex_id()
                        if vertex_id is not None:
                            normal_result = cmds.polyNormalPerVertex(
                                f"{mesh_name}.vtx[{vertex_id}]",
                                query=True,
                                normalXYZ=True,
                            )
                            if normal_result and len(normal_result) >= 3:
                                normal = [
                                    normal_result[0],
                                    normal_result[1],
                                    normal_result[2],
                                ]
                except Exception as e:
                    print(f"Could not get vertex normal: {str(e)}")
            else:
                selection_items = cmds.ls(selection=True)
                if selection_items:
                    position_info = cmds.xform(
                        selection_items[0],
                        query=True,
                        worldSpace=True,
                        translation=True,
                    )
                    if position_info:
                        position = position_info

            print(f"selected vertex pos: {position}")
            print(f"selected vertex nor: {normal}")
            cmds.floatFieldGrp(
                self.sample_pos,
                edit=True,
                value1=position[0],
                value2=position[1],
                value3=position[2],
            )
            cmds.floatFieldGrp(
                self.sample_normal,
                edit=True,
                value1=normal[0],
                value2=normal[1],
                value3=normal[2],
            )

            if self.samples:
                last_time = self.samples[-1]["time"]
                cmds.floatField(self.sample_time, edit=True, value=last_time + 1.0)

            self.add_sample()

        except Exception as e:
            cmds.warning(f"Error adding sample from selection: {str(e)}")

    def is_vertex_selected(self):
        """Check if a vertex is currently selected."""
        selection = cmds.ls(selection=True, flatten=True)
        if not selection:
            return False

        for item in selection:
            if ".vtx[" in item:
                return True
        return False

    def get_selected_vertex_id(self):
        """Get the ID of the selected vertex."""
        selection = cmds.ls(selection=True, flatten=True)
        if not selection:
            return None

        for item in selection:
            if ".vtx[" in item:
                start_idx = item.find("[") + 1
                end_idx = item.find("]")
                if start_idx > 0 and end_idx > start_idx:
                    try:
                        return int(item[start_idx:end_idx])
                    except ValueError:
                        pass
        return None

    def apply_all_samples(self, *args):
        """Apply all samples to the mesh."""
        if not self.active_brush or not self.active_brush.current_stroke:
            cmds.warning("Please begin a stroke first")
            return

        if not self.active_mesh_data:
            mesh_name = cmds.textField(self.mesh_text, query=True, text=True)
            if (
                not mesh_name
                or mesh_name == "No mesh selected"
                or mesh_name == "Selected object is not a mesh"
            ):
                cmds.warning("Please select a valid mesh first")
                return

            self.active_mesh_data = MeshInterface.get_mesh_data(mesh_name)

        if not self.samples:
            cmds.warning("No samples added. Please add at least one sample.")
            return

        cmds.undoInfo(openChunk=True)

        try:
            for sample in self.samples:
                self.active_brush.apply_to_mesh(
                    self.active_mesh_data, sample["sample_obj"]
                )

            cmds.inViewMessage(
                message="All samples applied!", pos="midCenter", fade=True
            )
        except Exception as e:
            cmds.error(f"Error applying samples: {str(e)}")
        finally:
            cmds.undoInfo(closeChunk=True)


def show_auto_sculptor_ui():
    """
    Create and display the AutoSculptor UI.
    This function can be called from Maya's script editor.
    """
    ui = TestUI()
    return ui


if __name__ == "__main__" or "sphinx" in sys.modules:
    show_auto_sculptor_ui()
