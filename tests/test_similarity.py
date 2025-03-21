import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autosculptor.core.data_structures import Sample, Stroke, Workflow
from autosculptor.analysis.parameterization import StrokeParameterizer
from autosculptor.analysis.similarity import (
	calculate_sample_differential,
	calculate_stroke_neighborhood_distance,
)


class MockMeshData:
	def __init__(self):
		self.vertices, self.faces = self.create_cube_mesh()
		self.normals = np.zeros_like(self.vertices)

	def create_cube_mesh(self):
		"""Creates vertices and faces for a simple cube."""
		vertices = np.array(
			[
				[-0.5, -0.5, -0.5],
				[0.5, -0.5, -0.5],
				[0.5, 0.5, -0.5],
				[-0.5, 0.5, -0.5],
				[-0.5, -0.5, 0.5],
				[0.5, -0.5, 0.5],
				[0.5, 0.5, 0.5],
				[-0.5, 0.5, 0.5],
			],
			dtype=np.float32,
		)

		faces = np.array(
			[
				[0, 1, 2],
				[0, 2, 3],
				[1, 5, 6],
				[1, 6, 2],
				[5, 4, 7],
				[5, 7, 6],
				[4, 0, 3],
				[4, 3, 7],
				[3, 2, 6],
				[3, 6, 7],
				[4, 5, 1],
				[4, 1, 0],
			],
			dtype=np.int32,
		)
		return vertices, faces


def create_test_stroke(
	stroke_type, positions, normals=None, sizes=None, timestamps=None
):
	"""Helper function to create test strokes."""
	stroke = Stroke()
	stroke.stroke_type = stroke_type
	if normals is None:
		normals = [[0, 0, 1]] * len(positions)
	if sizes is None:
		sizes = [1.0] * len(positions)
	if timestamps is None:
		timestamps = range(len(positions))

	for i, pos in enumerate(positions):
		sample = Sample(
			position=np.array(pos),
			normal=np.array(normals[i]),
			size=sizes[i],
			timestamp=timestamps[i],
		)
		stroke.add_sample(sample)
	return stroke


class TestSimilarityFunctions(unittest.TestCase):
	def setUp(self):
		self.mesh_data = MockMeshData()
		self.parameterizer = StrokeParameterizer(self.mesh_data)
		self.camera_lookat = np.array([0, 0, 1])

	def test_calculate_sample_differential_surface_position(self):
		sample1 = Sample(position=[0, 0, 0])
		sample2 = Sample(position=[0, 0, 0])
		sample1.ts = 0.5
		sample1.ds = 0.2
		sample2.ts = 0.6
		sample2.ds = 0.3

		diff = calculate_sample_differential(
			sample1, sample2, "surface", wp=1.0, wa=0.0, wt=0.0
		)
		self.assertTrue(np.allclose(diff[0], [0.5 - 0.6, 0.2 - 0.3]))
		self.assertTrue(np.allclose(diff[1], [0.0, 0.0]))
		self.assertTrue(np.allclose(diff[2], [0.0]))

	def test_calculate_sample_differential_freeform_position(self):
		sample1 = Sample(position=[0, 0, 0])
		sample2 = Sample(position=[0, 0, 0])
		sample1.xs = 0.1
		sample1.ys = 0.2
		sample1.zs = 0.3
		sample2.xs = 0.2
		sample2.ys = 0.3
		sample2.zs = 0.4

		diff = calculate_sample_differential(
			sample1, sample2, "freeform", wp=1.0, wa=0.0, wt=0.0
		)
		self.assertTrue(np.allclose(diff[0], [0.1 - 0.2, 0.2 - 0.3, 0.3 - 0.4]))
		self.assertTrue(np.allclose(diff[1], [0.0, 0.0]))
		self.assertTrue(np.allclose(diff[2], [0.0]))

	def test_calculate_sample_differential_appearance_normal(self):
		normal1 = np.array([0, 0, 1])
		normal2 = np.array([0, 1, 0])
		sample1 = Sample(position=[0, 0, 0], normal=normal1)
		sample2 = Sample(position=[0, 0, 0], normal=normal2)
		diff = calculate_sample_differential(
			sample1, sample2, "surface", wp=0.0, wa=1.0, wt=0.0, wn=1.0, wc=0.0
		)
		self.assertTrue(np.allclose(diff[0], [0.0, 0.0]))
		self.assertAlmostEqual(diff[1][0], 1.0 - np.dot(normal1, normal2))
		self.assertTrue(np.allclose(diff[1][1], 0.0))
		self.assertTrue(np.allclose(diff[2], [0.0]))

	def test_calculate_sample_differential_appearance_curvature(self):
		sample1 = Sample(position=[0, 0, 0], curvature=0.5)
		sample2 = Sample(position=[0, 0, 0], curvature=0.8)
		diff = calculate_sample_differential(
			sample1, sample2, "surface", wp=0.0, wa=1.0, wt=0.0, wn=0.0, wc=1.0
		)
		self.assertTrue(np.allclose(diff[0], [0.0, 0.0]))
		self.assertTrue(np.allclose(diff[1][0], 0.0))
		self.assertAlmostEqual(diff[1][1], abs(0.5 - 0.8))
		self.assertTrue(np.allclose(diff[2], [0.0]))

	def test_calculate_sample_differential_temporal(self):
		sample1 = Sample(position=[0, 0, 0], timestamp=1.0)
		sample2 = Sample(position=[0, 0, 0], timestamp=2.5)
		diff = calculate_sample_differential(
			sample1, sample2, "surface", wp=0.0, wa=0.0, wt=1.0
		)
		self.assertTrue(np.allclose(diff[0], [0.0, 0.0]))
		self.assertTrue(np.allclose(diff[1], [0.0, 0.0]))
		self.assertTrue(np.allclose(diff[2], [1.0 - 2.5]))

	def test_calculate_stroke_neighborhood_distance_similar_strokes(self):
		stroke1 = create_test_stroke("surface", [[0, 0, 0], [0.5, 0, 0]])
		stroke2 = create_test_stroke("surface", [[0, 0.1, 0], [0.5, 0.1, 0]])
		distance = calculate_stroke_neighborhood_distance(stroke1, stroke2)
		self.assertLess(distance, 0.5)

	def test_calculate_stroke_neighborhood_distance_dissimilar_strokes(self):
		stroke1 = create_test_stroke("surface", [[0, 0, 0], [0.5, 0, 0]])
		stroke2 = create_test_stroke("surface", [[1, 1, 0], [1.5, 1, 0]])
		self.parameterizer.parameterize_stroke(stroke1, self.camera_lookat)
		self.parameterizer.parameterize_stroke(stroke2, self.camera_lookat)
		distance = calculate_stroke_neighborhood_distance(stroke1, stroke2)
		print(
			calculate_sample_differential(
				stroke1.samples[0],
				stroke2.samples[0],
				"surface",
				wp=1.0,
				wa=0.0,
				wt=1.0,
			)
		)
		self.assertGreater(distance, 0.5)

	def test_calculate_stroke_neighborhood_distance_empty_strokes(self):
		stroke1 = Stroke()
		stroke2 = Stroke()
		distance = calculate_stroke_neighborhood_distance(stroke1, stroke2)
		self.assertEqual(distance, 0.0)

	def test_calculate_stroke_neighborhood_distance_stroke_vs_empty(self):
		stroke1 = create_test_stroke("surface", [[0, 0, 0], [0.5, 0, 0]])
		stroke2 = Stroke()
		distance = calculate_stroke_neighborhood_distance(stroke1, stroke2)
		self.assertEqual(distance, 0.0)


if __name__ == "__main__":
	unittest.main()
