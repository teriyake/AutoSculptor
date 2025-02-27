"""
Tests for Brush
"""

import unittest
import numpy as np
from autosculptor.core.surface_brush import SurfaceBrush
from autosculptor.core.freeform_brush import FreeformBrush
from autosculptor.core.mesh_interface import SimpleMesh


class TestBrushes(unittest.TestCase):

    def setUp(self):
        self.surface_brush = SurfaceBrush(size=2.0, strength=0.5)
        self.freeform_brush = FreeformBrush(size=1.5, strength=0.7)
        self.mesh = SimpleMesh().create_cube(size=10.0)

    def test_surface_brush_creation(self):
        self.assertEqual(self.surface_brush.size, 2.0)
        self.assertEqual(self.surface_brush.strength, 0.5)
        self.assertEqual(self.surface_brush.falloff, "smooth")

    def test_freeform_brush_creation(self):
        self.assertEqual(self.freeform_brush.size, 1.5)
        self.assertEqual(self.freeform_brush.strength, 0.7)
        self.assertEqual(self.freeform_brush.falloff, "smooth")

    def test_stroke_creation(self):
        self.surface_brush.begin_stroke()

        sample1 = self.surface_brush.add_sample((0, 0, 0), (0, 1, 0), 1.0, 0.0)
        sample2 = self.surface_brush.add_sample((1, 0, 0), (0, 1, 0), 0.8, 0.1)

        stroke = self.surface_brush.end_stroke()

        self.assertEqual(len(stroke), 2)
        self.assertEqual(stroke.stroke_type, "surface")
        self.assertEqual(stroke.start_time, 0.0)
        self.assertEqual(stroke.end_time, 0.1)

    def test_mesh_interaction(self):
        self.assertEqual(len(self.mesh.vertices), 8)
        self.assertEqual(len(self.mesh.faces), 12)

        self.assertTrue(
            np.array_equal(self.mesh.vertices[0], np.array([-5.0, -5.0, -5.0]))
        )
        self.assertTrue(
            np.array_equal(self.mesh.vertices[6], np.array([5.0, 5.0, 5.0]))
        )

        self.mesh.displace_vertices([0, 1, 2, 3], [np.array([0, 0, -1])] * 4)

        self.assertTrue(
            np.array_equal(self.mesh.vertices[0], np.array([-5.0, -5.0, -6.0]))
        )


if __name__ == "__main__":
    unittest.main()
