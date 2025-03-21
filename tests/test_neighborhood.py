import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autosculptor.core.data_structures import (
	Sample,
	Stroke,
	Workflow,
)
from autosculptor.core.mesh_interface import MeshData
from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator
from autosculptor.analysis.neighborhood import (
	calculate_sample_neighborhood,
	calculate_stroke_neighborhoods,
	calculate_stroke_similarity,
)


class MockMeshData:
	"""Mock mesh data for testing"""

	def __init__(self):
		x, y = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4))
		z = np.zeros_like(x)

		vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

		faces = []
		for i in range(3):
			for j in range(3):
				v0 = i * 4 + j
				v1 = i * 4 + (j + 1)
				v2 = (i + 1) * 4 + j
				v3 = (i + 1) * 4 + (j + 1)

				faces.append([v0, v1, v3])
				faces.append([v0, v3, v2])

		self.vertices = vertices
		self.faces = np.array(faces)


class TestNeighborhoodAnalysis(unittest.TestCase):
	"""Tests for neighborhood analysis functionality"""

	def setUp(self):
		"""Set up test fixtures"""
		self.mesh_data = MockMeshData()
		self.geo_calc = CachedGeodesicCalculator(
			self.mesh_data.vertices, self.mesh_data.faces
		)

		self.workflow = Workflow()
		self.stroke1 = Stroke()
		for i in range(5):
			position = np.array([i * 0.3, 1.0, 0.0])
			self.stroke1.add_sample(
				Sample(
					position=position,
					normal=np.array([0, 0, 1]),
					timestamp=i * 0.1,
					pressure=1.0,
				)
			)
		self.workflow.add_stroke(self.stroke1)

		self.stroke2 = Stroke()
		for i in range(5):
			position = np.array([i * 0.3, 1.2, 0.0])
			self.stroke2.add_sample(
				Sample(
					position=position,
					normal=np.array([0, 0, 1]),
					timestamp=0.5 + i * 0.1,
					pressure=1.0,
				)
			)
		self.workflow.add_stroke(self.stroke2)

	def test_calculate_sample_neighborhood_empty_workflow(self):
		"""Test with an empty workflow."""
		empty_workflow = Workflow()
		sample = self.stroke1.samples[2]
		neighborhood = calculate_sample_neighborhood(
			sample,
			self.stroke1,
			empty_workflow,
			self.mesh_data,
			self.geo_calc,
			max_spatial_distance=1.0,
			max_temporal_difference=0.3,
		)
		self.assertEqual(len(neighborhood), 0)

	def test_calculate_sample_neighborhood_single_stroke(self):
		"""Test with a single stroke in the workflow."""
		single_stroke_workflow = Workflow()
		single_stroke_workflow.add_stroke(self.stroke1)

		sample = self.stroke1.samples[2]
		neighborhood = calculate_sample_neighborhood(
			sample,
			self.stroke1,
			single_stroke_workflow,
			self.mesh_data,
			self.geo_calc,
			max_spatial_distance=1.0,
			max_temporal_difference=0.3,
		)
		self.assertEqual(len(neighborhood), 2)
		self.assertIn(self.stroke1.samples[0], neighborhood)
		self.assertIn(self.stroke1.samples[1], neighborhood)
		self.assertNotIn(self.stroke1.samples[2], neighborhood)
		self.assertNotIn(self.stroke1.samples[3], neighborhood)
		self.assertNotIn(self.stroke1.samples[4], neighborhood)

	def test_calculate_sample_neighborhood_multiple_strokes(self):
		"""Test with multiple strokes, checking for neighbors in previous strokes."""
		sample = self.stroke2.samples[2]
		neighborhood = calculate_sample_neighborhood(
			sample,
			self.stroke2,
			self.workflow,
			self.mesh_data,
			self.geo_calc,
			max_spatial_distance=1.0,
			max_temporal_difference=0.3,
		)
		self.assertGreater(len(neighborhood), 0)
		self.assertIn(self.stroke1.samples[4], neighborhood)
		self.assertIn(self.stroke2.samples[0], neighborhood)
		self.assertIn(self.stroke2.samples[1], neighborhood)
		self.assertNotIn(self.stroke2.samples[2], neighborhood)
		self.assertNotIn(self.stroke2.samples[3], neighborhood)

	def test_calculate_sample_neighborhood_temporal_limit(self):
		"""Test the max_temporal_difference parameter."""
		sample = self.stroke2.samples[2]
		neighborhood = calculate_sample_neighborhood(
			sample,
			self.stroke2,
			self.workflow,
			self.mesh_data,
			self.geo_calc,
			max_spatial_distance=1.0,
			max_temporal_difference=0.1,
		)
		self.assertIn(self.stroke2.samples[1], neighborhood)
		self.assertNotIn(self.stroke2.samples[0], neighborhood)
		self.assertNotIn(self.stroke1.samples[4], neighborhood)

	def test_calculate_sample_neighborhood_spatial_limit(self):
		"""Test the max_spatial_distance parameter."""
		sample = self.stroke2.samples[2]
		neighborhood = calculate_sample_neighborhood(
			sample,
			self.stroke2,
			self.workflow,
			self.mesh_data,
			self.geo_calc,
			max_spatial_distance=0.2,
			max_temporal_difference=0.5,
		)
		self.assertNotIn(self.stroke2.samples[1], neighborhood)
		self.assertIn(self.stroke1.samples[4], neighborhood)

	def test_calculate_stroke_neighborhoods_with_workflow(self):
		"""Test calculate_stroke_neighborhoods with the workflow."""
		neighborhoods = calculate_stroke_neighborhoods(
			self.stroke2,
			self.workflow,
			self.mesh_data,
			max_spatial_distance=1.0,
			max_temporal_difference=0.3,
		)
		self.assertEqual(len(neighborhoods), len(self.stroke2.samples))
		self.assertGreater(len(neighborhoods[2]), 0)

	def test_calculate_stroke_similarity_with_workflow(self):
		"""Test stroke similarity with the workflow context."""
		similarity = calculate_stroke_similarity(
			self.stroke1,
			self.stroke2,
			self.mesh_data,
			max_spatial_distance=1.0,
			max_temporal_difference=0.5,
		)
		self.assertGreater(similarity, 0.0)

		different_stroke = Stroke()
		for i in range(5):
			position = np.array([2.0, i * 0.5, 0.0])
			different_stroke.add_sample(
				Sample(
					position=position,
					normal=np.array([0, 0, 1]),
					timestamp=i * 0.1,
				)
			)
		self.workflow.add_stroke(different_stroke)

		diff_similarity = calculate_stroke_similarity(
			self.stroke1,
			different_stroke,
			self.mesh_data,
			max_spatial_distance=1.0,
			max_temporal_difference=0.5,
		)
		self.assertLess(diff_similarity, similarity)

	def test_empty_strokes(self):
		"""Test behavior with empty strokes (should not crash)."""
		empty_stroke = Stroke()
		empty_workflow = Workflow()
		empty_workflow.add_stroke(empty_stroke)

		neighborhoods = calculate_stroke_neighborhoods(
			empty_stroke,
			empty_workflow,
			self.mesh_data,
			max_spatial_distance=1.0,
			max_temporal_difference=0.3,
		)
		self.assertEqual(len(neighborhoods), 0)

		similarity = calculate_stroke_similarity(
			self.stroke1,
			empty_stroke,
			self.mesh_data,
			max_spatial_distance=1.0,
			max_temporal_difference=0.5,
		)
		self.assertEqual(similarity, 0.0)


if __name__ == "__main__":
	unittest.main()
