from autosculptor.core.data_structures import Sample, Stroke, Workflow
import numpy as np
import maya.cmds as cmds

class StrokeVisualizer:
    def __init__(self, stroke):
        """
        Initialize the StrokeVisualizer with a Stroke object.

        Args:
            stroke (Stroke): The stroke to visualize.
        """
        self.stroke = stroke
        
        # Create a new shader for visualization
        self.shader = cmds.shadingNode('lambert', asShader=True, name='transparentYellowShader')
        self.shading_group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=self.shader + 'SG')
        cmds.connectAttr(self.shader + '.outColor', self.shading_group + '.surfaceShader')

        # Set the color to yellow
        cmds.setAttr(self.shader + '.color', 1.0, 1.0, 0.0, type='double3')

        # Set transparency to make the material semi-transparent
        cmds.setAttr(self.shader + '.transparency', 0.8, 0.8, 0.8, type='double3')

    def create_curve_from_stroke(self):
        """
        Create a NURBS curve in Maya from the stroke samples.

        Returns:
            str: The name of the created NURBS curve.
        """
        # Extract positions from stroke samples
        positions = [sample.position for sample in self.stroke.samples]

        # Flatten the list of positions
        curve_points = [tuple(map(float, sample.position)) for sample in self.stroke.samples]

        # Create the NURBS curve
        if len(curve_points) < 4:
            curve = cmds.curve(p=curve_points, d=1)  # Linear curve for fewer points
        else:
            curve = cmds.curve(p=curve_points, d=3)  # Cubic curve
            
        return curve

    def create_tube_from_curve(self, curve, radius=0.1, sections=8):
        """
        Create a tubular geometry by extruding a circle along the given curve.

        Args:
            curve (str): The name of the curve to extrude along.
            radius (float, optional): The radius of the tube. Defaults to 0.1.
            sections (int, optional): The number of sections for the circle profile. Defaults to 8.

        Returns:
            str: The name of the created tubular geometry.
        """
        # Create a circle to serve as the profile for extrusion
        circle = cmds.circle(radius=radius, sections=sections, normal=(0, 1, 0))[0]

        # Extrude the circle along the curve to create the tube
        tube = cmds.extrude(circle, curve, et=2, fixedPath=True, useComponentPivot=1, polygon=1, ch=True)[0]

        # Cap the ends of the tube
        cmds.polyCloseBorder(tube)

        # Delete the original circle and curve
        cmds.delete(circle, curve)
        
        #Shading
        self.assign_transparent_yellow_material(tube)
        
        return tube
        
    def assign_transparent_yellow_material(self, geometry):
        """
        Assign a transparent yellow material to the specified geometry.

        Args:
            geometry (str): The name of the geometry to assign the material to.
        """
        # Assign the shader to the geometry
        cmds.sets(geometry, edit=True, forceElement=self.shading_group)

    def visualize(self, radius=0.1, sections=8):
        """
        Visualize the stroke as a tube in Maya.

        Args:
            radius (float, optional): The radius of the tube. Defaults to 0.1.
            sections (int, optional): The number of sections for the circle profile. Defaults to 8.

        Returns:
            str: The name of the created tubular geometry.
        """
        curve = self.create_curve_from_stroke()
        tube = self.create_tube_from_curve(curve, radius, sections)
        return tube

def test_stroke_visualizer():
    """Test case for StrokeVisualizer"""
    # Clear the Maya scene
    cmds.file(new=True, force=True)

    # Create a Stroke instance
    stroke = Stroke()

    # Add sample points to the stroke
    sample_positions = [
        [0, 0, 0],  [1, 1, 0],  [2, 1.5, 0],  [3, 2, 0],  [4, 2.5, 0]
    ]

    for i, pos in enumerate(sample_positions):
        sample = Sample(position=pos, size=1.0, pressure=1.0, timestamp=i)
        stroke.add_sample(sample)

    # Create the visualizer
    visualizer = StrokeVisualizer(stroke)

    # Generate the tube
    tube = visualizer.visualize(radius=0.2, sections=12)

    # Check if the tube exists in the Maya scene
    assert cmds.objExists(tube), f"Test Failed: Tube '{tube}' was not created."

    print(f"Test Passed: Tube '{tube}' successfully created.")

