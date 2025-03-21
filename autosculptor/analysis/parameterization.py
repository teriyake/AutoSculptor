import numpy as np
from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator
from autosculptor.core.data_structures import Sample, Stroke
from typing import List, Tuple


class StrokeParameterizer:
	"""
	Class for parameterizing brush strokes.
	"""

	def __init__(self, mesh_data):
		"""
		Initialize the StrokeParameterizer.

		Args:
		    mesh_data: Mesh Data
		"""
		self.mesh_data = mesh_data
		self.geo_calc = CachedGeodesicCalculator(mesh_data.vertices, mesh_data.faces)

	def normalize_stroke_parameters(self, stroke: Stroke):
		"""Normalizes the ts and ds parameters of all samples in a stroke.
		 For Surface Brush:
		     ts (stroke arclength): Normalized to [0, 1]
		     ds (geodesic distance): Normalized to [-1, 1] based on brush radius.
		 For Freeform Brush:
		     xs, ys (cross-product-based distances): Normalized to [-1, 1]
		     zs (stroke arclength): Normalized to [0,1].

		Args:
		     stroke (Stroke): The stroke to normalize.
		"""

		if stroke.stroke_type == "surface":
			if not stroke.samples:
				return

			total_arc_length = 0.0
			for i in range(1, len(stroke.samples)):
				total_arc_length += np.linalg.norm(
					stroke.samples[i].position - stroke.samples[i - 1].position
				)
			print(f"Total arc length: {total_arc_length}")

			cumulative_arc_length = 0.0
			for i, sample in enumerate(stroke.samples):
				if i > 0:
					cumulative_arc_length += np.linalg.norm(
						sample.position - stroke.samples[i - 1].position
					)
				sample.ts = (
					cumulative_arc_length / total_arc_length
					if total_arc_length > 0
					else 0.0
				)
				print(
					f"Sample {i}: cumulative_arc_length={cumulative_arc_length}, ts={sample.ts}"
				)

				min_dist = float("inf")
				closest_stroke_sample_index = 0

				for j, stroke_sample in enumerate(stroke.samples):
					dist = np.linalg.norm(sample.position - stroke_sample.position)
					if sample == stroke_sample:
						continue

					if dist < min_dist:
						min_dist = dist
						closest_stroke_sample_index = j

				closest_sample = stroke.samples[closest_stroke_sample_index]
				geodesic_dist = self.geo_calc.compute_distance(
					closest_sample.position, np.array([sample.position])
				)[0]

				print(
					f"Sample {i}: closest_sample_pos={closest_sample.position}, sample_pos={sample.position}, geodesic_dist={geodesic_dist}"
				)

				test_pos_plus = sample.position + sample.normal * 0.001
				test_pos_minus = sample.position - sample.normal * 0.001

				geodesic_dist_plus = self.geo_calc.compute_distance(
					closest_sample.position, np.array([test_pos_plus])
				)[0]
				geodesic_dist_minus = self.geo_calc.compute_distance(
					closest_sample.position, np.array([test_pos_minus])
				)[0]

				if geodesic_dist_plus < geodesic_dist_minus:
					signed_geodesic_dist = geodesic_dist
				else:
					signed_geodesic_dist = -geodesic_dist

				sample.ds = np.clip(signed_geodesic_dist / sample.size, -1.0, 1.0)

		elif stroke.stroke_type == "freeform":
			if not stroke.samples:
				return

			total_arc_length = 0.0
			for i in range(1, len(stroke.samples)):
				total_arc_length += np.linalg.norm(
					stroke.samples[i].position - stroke.samples[i - 1].position
				)

			cumulative_arc_length = 0.0
			for i, sample in enumerate(stroke.samples):
				if i > 0:
					cumulative_arc_length += np.linalg.norm(
						sample.position - stroke.samples[i - 1].position
					)
				sample.zs = (
					cumulative_arc_length / total_arc_length
					if total_arc_length > 0
					else 0.0
				)

				if i > 0:
					stroke_direction = sample.position - stroke.samples[i - 1].position
					stroke_direction_norm = np.linalg.norm(stroke_direction)
					if stroke_direction_norm > 1e-6:
						stroke_direction = stroke_direction / stroke_direction_norm
					# else:
					# stroke_direction = np.array([0, 0, 0])

				else:
					stroke_direction = sample.normal
					stroke_direction_norm = np.linalg.norm(stroke_direction)
					if stroke_direction_norm > 1e-6:
						stroke_direction = stroke_direction / stroke_direction_norm
					# else:
					# stroke_direction = np.array([0, 0, 0])

				y_direction = np.cross(stroke_direction, sample.camera_lookat)
				y_direction_norm = np.linalg.norm(y_direction)

				if y_direction_norm > 1e-6:
					y_direction = y_direction / y_direction_norm
				else:
					y_direction = np.array([0, 0, 0])

				x_direction = np.cross(y_direction, stroke_direction)
				x_direction_norm = np.linalg.norm(x_direction)
				if x_direction_norm > 1e-6:
					x_direction = x_direction / x_direction_norm
				else:
					x_direction = np.array([0, 0, 0])

				sample.xs = np.dot(sample.position, x_direction) / (sample.size + 1e-8)
				sample.ys = np.dot(sample.position, y_direction) / (sample.size + 1e-8)

	def parameterize_stroke(self, stroke: Stroke, camera_lookat: np.ndarray):
		"""Adds the local parameterization to the samples in a stroke.

		Args:
		    stroke (Stroke): The stroke to parameterize.
		    camera_lookat (np.ndarray):  Camera lookat direction (for freeform strokes)

		"""
		for sample in stroke.samples:
			sample.camera_lookat = camera_lookat

		self.normalize_stroke_parameters(stroke)

	def inverse_parameterize_surface(
		self, stroke: Stroke, ts: float, ds: float, original_sample: Sample
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Converts surface stroke parameters (ts, ds) back to world space,
		using geodesic distance calculation for accurate offset.

		Args:
		    stroke: The original stroke.
		    ts: Normalized arc length along the stroke.
		    ds: Normalized geodesic distance from the stroke path.
		    original_sample: the original sample

		Returns:
		    A tuple (position, normal) in world space.
		"""
		if not stroke.samples:
			return np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

		position_on_stroke, normal_on_stroke = self._get_position_normal_on_stroke_path(
			stroke, ts
		)
		geodesic_offset = ds * original_sample.size
		final_position = position_on_stroke + normal_on_stroke * geodesic_offset
		return final_position, normal_on_stroke

		# NOTE: don't do this
		closest_vertex_index = self.geo_calc.find_closest_vertex(final_position)
		final_position = self.mesh_data.vertices[closest_vertex_index]
		final_normal = self.mesh_data.normals[closest_vertex_index]
		final_normal = final_normal / (np.linalg.norm(final_normal) + 1e-8)

		return final_position, final_normal

	def _get_position_normal_on_stroke_path(self, stroke, ts):
		"""Helper function to get position and normal at a given ts on the stroke path."""
		total_arc_length = 0.0
		for i in range(1, len(stroke.samples)):
			total_arc_length += np.linalg.norm(
				stroke.samples[i].position - stroke.samples[i - 1].position
			)

		target_arc_length = ts * total_arc_length
		cumulative_arc_length = 0.0
		position_on_stroke = None
		normal_on_stroke = None

		for i in range(1, len(stroke.samples)):
			segment_length = np.linalg.norm(
				stroke.samples[i].position - stroke.samples[i - 1].position
			)
			if cumulative_arc_length + segment_length >= target_arc_length:
				alpha = (target_arc_length - cumulative_arc_length) / segment_length
				position_on_stroke = stroke.samples[i - 1].position + alpha * (
					stroke.samples[i].position - stroke.samples[i - 1].position
				)
				normal_on_stroke = stroke.samples[i - 1].normal + alpha * (
					stroke.samples[i].normal - stroke.samples[i - 1].normal
				)
				normal_on_stroke = normal_on_stroke / np.linalg.norm(normal_on_stroke)
				break
			cumulative_arc_length += segment_length
		else:
			position_on_stroke = stroke.samples[-1].position
			normal_on_stroke = stroke.samples[-1].normal
		return position_on_stroke, normal_on_stroke

	def inverse_parameterize_freeform(
		self, stroke: Stroke, xs: float, ys: float, zs: float, original_sample: Sample
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Converts freeform stroke parameters (xs, ys, zs) back to world space.

		Args:
		    stroke: The original stroke.
		    xs: Normalized x offset in the local frame.
		    ys: Normalized y offset in the local frame.
		    zs: Normalized arc length along the stroke.

		Returns:
		    A tuple (position, normal) in world space.
		"""
		if not stroke.samples:
			return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])

		total_arc_length = 0.0
		for i in range(1, len(stroke.samples)):
			total_arc_length += np.linalg.norm(
				stroke.samples[i].position - stroke.samples[i - 1].position
			)

		target_arc_length = zs * total_arc_length
		cumulative_arc_length = 0.0

		for i in range(1, len(stroke.samples)):
			segment_length = np.linalg.norm(
				stroke.samples[i].position - stroke.samples[i - 1].position
			)
			if cumulative_arc_length + segment_length >= target_arc_length:
				alpha = (target_arc_length - cumulative_arc_length) / segment_length
				position_on_stroke = stroke.samples[i - 1].position + alpha * (
					stroke.samples[i].position - stroke.samples[i - 1].position
				)

				if i > 0:
					stroke_direction = (
						stroke.samples[i].position - stroke.samples[i - 1].position
					)
					stroke_direction_norm = np.linalg.norm(stroke_direction)
					if stroke_direction_norm > 1e-6:
						stroke_direction = stroke_direction / stroke_direction_norm
					else:
						stroke_direction = np.array([0, 0, 0])
				else:
					stroke_direction = stroke.samples[0].normal
					if np.linalg.norm(stroke_direction) < 1e-6:
						stroke_direction = np.array([0, 0, 0])

				y_direction = np.cross(stroke_direction, original_sample.camera_lookat)
				y_direction_norm = np.linalg.norm(y_direction)

				if y_direction_norm > 1e-6:
					y_direction = y_direction / y_direction_norm
				else:
					y_direction = np.array([0, 0, 0])

				x_direction = np.cross(y_direction, stroke_direction)
				x_direction_norm = np.linalg.norm(x_direction)
				if x_direction_norm > 1e-6:
					x_direction = x_direction / x_direction_norm
				else:
					x_direction = np.array([0, 0, 0])

				final_position = (
					position_on_stroke
					+ xs * original_sample.size * x_direction
					+ ys * original_sample.size * y_direction
				)
				final_normal = stroke_direction
				break

			cumulative_arc_length += segment_length

		else:
			position_on_stroke = stroke.samples[-1].position
			final_position = position_on_stroke
			final_normal = stroke.samples[-1].normal

		return final_position, final_normal

	def _params_to_world(
		self, sample: Sample, original_stroke: Stroke, stroke_type: str
	) -> Tuple[np.ndarray, np.ndarray]:
		if stroke_type == "surface":
			print(f"ts: {sample.ts}, ds: {sample.ds}")
			return self.inverse_parameterize_surface(
				original_stroke, sample.ts, sample.ds, sample
			)
		elif stroke_type == "freeform":
			return self.inverse_parameterize_freeform(
				original_stroke, sample.xs, sample.ys, sample.zs, sample
			)
		else:
			raise ValueError(f"Invalid stroke type: {stroke_type}")

	def _find_closest_point_on_stroke_path(
		self, stroke: Stroke, position: np.ndarray
	) -> Tuple[float, np.ndarray, int]:
		min_dist = float("inf")
		closest_pos = None
		closest_segment = 0
		closest_t = 0.0

		for i in range(len(stroke.samples) - 1):
			start = stroke.samples[i].position
			end = stroke.samples[i + 1].position
			segment_vec = end - start
			t = np.dot(position - start, segment_vec) / np.dot(segment_vec, segment_vec)
			t = np.clip(t, 0.0, 1.0)
			point_on_segment = start + t * segment_vec
			dist = np.linalg.norm(position - point_on_segment)

			if dist < min_dist:
				min_dist = dist
				closest_pos = point_on_segment
				closest_segment = i
				closest_t = t

		return closest_t, closest_pos, closest_segment
