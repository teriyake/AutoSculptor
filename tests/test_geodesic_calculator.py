import unittest
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator


class TestGeodesicCalculator(unittest.TestCase):
	"""Tests for the CachedGeodesicCalculator class"""

	def setUp(self):
		"""Set up test fixtures"""
		self.vertices = np.array(
			[
				[0, 0, 0],
				[1, 0, 0],
				[1, 1, 0],
				[0, 1, 0],
				[0, 0, 1],
				[1, 0, 1],
				[1, 1, 1],
				[0, 1, 1],
			],
			dtype=np.float64,
		)

		self.faces = np.array(
			[
				[0, 1, 2],
				[0, 2, 3],
				[4, 5, 6],
				[4, 6, 7],
				[0, 1, 5],
				[0, 5, 4],
				[3, 2, 6],
				[3, 6, 7],
				[0, 3, 7],
				[0, 7, 4],
				[1, 2, 6],
				[1, 6, 5],
			],
			dtype=np.int32,
		)

		self.calculator = CachedGeodesicCalculator(
			self.vertices, self.faces, cache_size=5
		)

	def test_initialization(self):
		"""Test that the calculator initializes correctly"""
		self.assertIsNotNone(self.calculator)
		self.assertEqual(self.calculator.cache_size, 5)
		self.assertEqual(len(self.calculator.cache), 0)

	def test_find_closest_vertex(self):
		"""Test finding the closest vertex to a position"""
		idx = self.calculator.find_closest_vertex(np.array([0, 0, 0]))
		self.assertEqual(idx, 0)

		idx = self.calculator.find_closest_vertex(np.array([0.1, 0.1, 0.1]))
		self.assertEqual(idx, 0)

		idx = self.calculator.find_closest_vertex(np.array([0.5, 0, 0]))
		self.assertIn(idx, [0, 1])

	def test_find_closest_vertices(self):
		"""Test finding the closest vertices to multiple positions"""
		positions = np.array([[0, 0, 0], [1, 1, 1], [0.1, 0.1, 0.1]])

		indices = self.calculator.find_closest_vertices(positions)
		self.assertEqual(indices[0], 0)
		self.assertEqual(indices[1], 6)
		self.assertEqual(indices[2], 0)

	def test_compute_distance(self):
		"""Test computing geodesic distances"""
		distances = self.calculator.compute_distance(np.array([0, 0, 0]))
		self.assertAlmostEqual(distances[0], 0, delta=1e-5)

		self.assertAlmostEqual(distances[1], 1.0, delta=0.2)
		self.assertAlmostEqual(distances[3], 1.0, delta=0.2)
		self.assertAlmostEqual(distances[4], 1.0, delta=0.2)

		self.assertAlmostEqual(distances[2], np.sqrt(2), delta=0.2)

		self.assertAlmostEqual(distances[6], np.sqrt(3), delta=0.3)

	def test_compute_specific_distances(self):
		"""Test computing distances to specific target points"""
		source = np.array([0, 0, 0])
		targets = np.array([[1, 0, 0], [1, 1, 1]])

		distances = self.calculator.compute_distance(source, targets)
		self.assertEqual(len(distances), 2)
		self.assertAlmostEqual(distances[0], 1.0, delta=0.2)
		self.assertAlmostEqual(distances[1], np.sqrt(3), delta=0.3)

	def test_cache_functionality(self):
		"""Test that the cache is working correctly"""
		source = np.array([0, 0, 0])

		start_time = time.time()
		self.calculator.compute_distance(source)
		first_call_time = time.time() - start_time

		start_time = time.time()
		self.calculator.compute_distance(source)
		second_call_time = time.time() - start_time

		self.assertGreater(len(self.calculator.cache), 0)

		self.assertLessEqual(second_call_time, first_call_time * 1.5)

	def test_compute_many_to_many(self):
		"""Test computing distances between multiple sources and targets"""
		sources = np.array([[0, 0, 0], [1, 1, 1]])

		targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

		distances = self.calculator.compute_many_to_many(sources, targets)
		self.assertEqual(distances.shape, (2, 3))

		self.assertAlmostEqual(distances[0, 0], 1.0, delta=0.2)
		self.assertAlmostEqual(distances[0, 1], 1.0, delta=0.2)
		self.assertAlmostEqual(distances[0, 2], 1.0, delta=0.2)

		self.assertAlmostEqual(distances[1, 0], 2.0, delta=0.5)
		self.assertAlmostEqual(distances[1, 1], 2.0, delta=0.5)
		self.assertAlmostEqual(distances[1, 2], 2.0, delta=0.5)

	def test_get_mesh_area(self):
		"""Test calculating the total mesh area"""
		area = self.calculator.get_mesh_area()
		self.assertAlmostEqual(area, 6.0, delta=0.001)

	def test_cache_eviction(self):
		"""Test that old entries are evicted when the cache is full"""
		for i in range(8):
			self.calculator.compute_distance(self.vertices[i])

		self.assertLessEqual(len(self.calculator.cache), self.calculator.cache_size)

		self.assertNotIn(0, self.calculator.cache)

	def test_clear_cache(self):
		"""Test clearing the cache"""
		self.calculator.compute_distance(np.array([0, 0, 0]))
		self.calculator.compute_distance(np.array([1, 1, 1]))

		self.assertGreater(len(self.calculator.cache), 0)

		self.calculator.clear_cache()

		self.assertEqual(len(self.calculator.cache), 0)
		self.assertEqual(len(self.calculator.cache_order), 0)


if __name__ == "__main__":
	unittest.main()
