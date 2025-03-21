import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autosculptor.core.data_structures import Sample, Stroke, Workflow
from autosculptor.core.mesh_interface import MeshData
from autosculptor.analysis.synthesis import StrokeSynthesizer
from autosculptor.analysis.parameterization import StrokeParameterizer
from autosculptor.analysis.geodesic_calculator import (
	CachedGeodesicCalculator,
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


class TestStrokeSynthesizer(unittest.TestCase):
	def setUp(self):
		self.mesh_data = MockMeshData()
		self.synthesizer = StrokeSynthesizer(self.mesh_data)
		self.parameterizer = StrokeParameterizer(self.mesh_data)
		self.max_spatial_distance = 1.0
		self.max_temporal_difference = 1.0
		self.camera_lookat = np.array([0, 0, 1])

	def create_parameterized_test_stroke(
		self, stroke_type, positions, normals=None, sizes=None, timestamps=None
	):
		stroke = create_test_stroke(stroke_type, positions, normals, sizes, timestamps)
		self.parameterizer.parameterize_stroke(stroke, self.camera_lookat)
		return stroke

	def test_initialize_suggestions_empty_workflow(self):
		workflow = Workflow()
		suggestions = self.synthesizer.initialize_suggestions(workflow)
		self.assertEqual(len(suggestions), 0)

	def test_initialize_suggestions_single_stroke(self):
		workflow = Workflow()
		stroke1 = create_test_stroke("surface", [[0, 0, 0], [0.5, 0, 0]])
		workflow.add_stroke(stroke1)
		suggestions = self.synthesizer.initialize_suggestions(workflow)
		self.assertEqual(len(suggestions), 0)

	def test_initialize_suggestions_multiple_strokes(
		self,
	):
		workflow = Workflow()
		stroke1 = self.create_parameterized_test_stroke(
			"surface", [[0, 0, 0], [0.5, 0, 0]], timestamps=[0, 1]
		)
		stroke2 = self.create_parameterized_test_stroke(
			"surface", [[0, 0.1, 0], [0.5, 0.1, 0]], timestamps=[2, 3]
		)
		stroke3 = self.create_parameterized_test_stroke(
			"surface", [[0.1, 0, 0], [0.6, 0, 0]], timestamps=[4, 5]
		)
		workflow.add_stroke(stroke1)
		workflow.add_stroke(stroke2)
		workflow.add_stroke(stroke3)

		suggestions = self.synthesizer.initialize_suggestions(workflow)
		self.assertEqual(len(suggestions), 2)

		self.assertEqual(suggestions[0].stroke_type, "surface")
		self.assertEqual(len(suggestions[0].samples), 2)
		self.assertAlmostEqual(suggestions[0].samples[0].position[0], 0.0, delta=0.15)
		self.assertAlmostEqual(suggestions[0].samples[0].position[1], 0.1, delta=0.1)

		self.assertEqual(suggestions[1].stroke_type, "surface")
		self.assertEqual(len(suggestions[1].samples), 2)
		self.assertAlmostEqual(suggestions[1].samples[0].position[0], 0.1, delta=0.1)
		self.assertAlmostEqual(suggestions[1].samples[0].position[1], 0.0, delta=0.1)

	def test_initialize_suggestions_indexing_past_longer(
		self,
	):
		"""Tests correct indexing when past_stroke is longer than next_stroke_to_copy."""
		workflow = Workflow()
		stroke1 = self.create_parameterized_test_stroke(
			"surface", [[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]], timestamps=[0, 1, 2]
		)
		stroke2 = self.create_parameterized_test_stroke(
			"surface", [[0, 0.1, 0], [0.5, 0.1, 0]], timestamps=[3, 4]
		)
		stroke3 = self.create_parameterized_test_stroke(
			"surface", [[0.1, 0, 0], [0.6, 0, 0], [1.1, 0, 0]], timestamps=[5, 6, 7]
		)
		workflow.add_stroke(stroke1)
		workflow.add_stroke(stroke2)
		workflow.add_stroke(stroke3)

		suggestions = self.synthesizer.initialize_suggestions(workflow)
		self.assertEqual(len(suggestions), 2)

	def test_initialize_suggestions_indexing_last_longer(
		self,
	):
		"""Tests correct indexing when last_stroke is longer than past_stroke."""
		workflow = Workflow()
		stroke1 = create_test_stroke(
			"surface", [[0, 0, 0], [0.5, 0, 0]], timestamps=[0, 1]
		)
		stroke2 = create_test_stroke(
			"surface", [[0, 0.1, 0], [0.5, 0.1, 0]], timestamps=[2, 3]
		)
		stroke3 = create_test_stroke(
			"surface", [[0.1, 0, 0], [0.6, 0, 0], [1.1, 0, 0]], timestamps=[4, 5, 6]
		)
		workflow.add_stroke(stroke1)
		workflow.add_stroke(stroke2)
		workflow.add_stroke(stroke3)

		suggestions = self.synthesizer.initialize_suggestions(workflow)
		self.assertEqual(len(suggestions), 2)

	def test_calculate_energy_empty_workflow(self):
		workflow = Workflow()
		candidate_stroke = create_test_stroke("surface", [[0, 0, 0], [0.5, 0, 0]])
		energy = self.synthesizer._calculate_energy(
			candidate_stroke,
			workflow,
			self.max_spatial_distance,
			self.max_temporal_difference,
		)
		self.assertEqual(energy, 0.0)

	def test_calculate_energy_similar_strokes(self):
		workflow = Workflow()
		stroke1 = create_test_stroke("surface", [[0, 0, 0], [0.5, 0, 0]])
		stroke2 = create_test_stroke("surface", [[0, 0.1, 0], [0.5, 0.1, 0]])
		workflow.add_stroke(stroke1)
		workflow.add_stroke(stroke2)
		candidate_stroke = create_test_stroke("surface", [[0.2, 0, 0], [0.7, 0, 0]])

		energy = self.synthesizer._calculate_energy(
			candidate_stroke,
			workflow,
			self.max_spatial_distance,
			self.max_temporal_difference,
		)
		self.assertLess(energy, 1.0)

	def test_calculate_energy_dissimilar_strokes(self):
		workflow = Workflow()
		stroke1 = self.create_parameterized_test_stroke(
			"surface", [[0, 0, 0], [0.5, 0, 0]]
		)
		stroke2 = self.create_parameterized_test_stroke(
			"surface", [[0, 0.1, 0], [0.5, 0.1, 0]]
		)
		workflow.add_stroke(stroke1)
		workflow.add_stroke(stroke2)
		candidate_stroke = self.create_parameterized_test_stroke(
			"surface", [[1, 1, 0], [1.5, 1, 0]]
		)

		energy = self.synthesizer._calculate_energy(
			candidate_stroke,
			workflow,
			self.max_spatial_distance,
			self.max_temporal_difference,
		)
		self.assertGreater(energy, 1.0)

	def test_synthesize_next_stroke_no_suggestions(
		self,
	):
		workflow = Workflow()
		stroke1 = create_test_stroke("surface", [[0, 0, 0], [0.5, 0, 0]])
		workflow.add_stroke(stroke1)
		next_stroke = self.synthesizer.synthesize_next_stroke(
			workflow, self.max_spatial_distance, self.max_temporal_difference
		)
		self.assertIsNone(next_stroke)

	def test_synthesize_next_stroke_no_good_candidates(self):
		workflow = Workflow()
		stroke1 = self.create_parameterized_test_stroke(
			"surface", [[0, 0, 0], [0.5, 0, 0]]
		)
		stroke2 = self.create_parameterized_test_stroke(
			"surface", [[1, 1, 0], [1.5, 1, 0]]
		)
		workflow.add_stroke(stroke1)
		workflow.add_stroke(stroke2)
		next_stroke = self.synthesizer.synthesize_next_stroke(
			workflow,
			self.max_spatial_distance,
			self.max_temporal_difference,
			energy_threshold=0.01,
		)
		self.assertIsNone(next_stroke)

	def test_synthesize_next_stroke_multiple_past_strokes(
		self,
	):
		workflow = Workflow()
		stroke1 = self.create_parameterized_test_stroke(
			"surface", [[0, 0, 0], [0.5, 0, 0]], timestamps=[0, 1]
		)
		stroke2 = self.create_parameterized_test_stroke(
			"surface", [[0, 0.1, 0], [0.5, 0.1, 0]], timestamps=[2, 3]
		)
		stroke3 = self.create_parameterized_test_stroke(
			"surface", [[0.1, 0, 0], [0.6, 0, 0]], timestamps=[4, 5]
		)
		workflow.add_stroke(stroke1)
		workflow.add_stroke(stroke2)
		workflow.add_stroke(stroke3)

		next_stroke = self.synthesizer.synthesize_next_stroke(
			workflow, self.max_spatial_distance, self.max_temporal_difference
		)
		self.assertIsNotNone(next_stroke)
		self.assertEqual(next_stroke.stroke_type, "surface")
		self.assertEqual(len(next_stroke.samples), 2)
		self.assertAlmostEqual(next_stroke.samples[0].position[0], 0.0, delta=0.15)
		self.assertAlmostEqual(next_stroke.samples[0].position[1], 0.1, delta=0.1)


if __name__ == "__main__":
	unittest.main()
