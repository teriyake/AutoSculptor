import numpy as np
from typing import List, Dict, Tuple, Optional
from autosculptor.core.data_structures import Stroke, Sample, Workflow
from autosculptor.analysis.neighborhood import find_neighboring_strokes
from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator
from scipy.optimize import linear_sum_assignment


def calculate_sample_differential(
	sample1: Sample,
	sample2: Sample,
	stroke_type: str,
	wp: float,
	wa: float,
	wt: float,
	wn: float = 0.8,
	wc: float = 0.3,
) -> np.ndarray:
	"""
	Calculates the differential between two samples (Equation 4).
	Uses normalized arc length (ts/zs) as the temporal component to prevent energy blowing up due to long pauses between strokes.

	Args:
		sample1: The first sample.
		sample2: The second sample.
		stroke_type: "surface" or "freeform"
		wp: Weight for position difference.
		wa: Weight for appearance/attribute difference (currently not used).
		wt: Weight for temporal difference (currently simplified).
		wn: Weight for normal difference (part of appearance).
		wc: Weight for curvature difference (part of appearance).

	Returns:
		A numpy array representing the weighted differential [p, a, t].
	"""

	if stroke_type == "surface":
		# print("_______calculate_sample_differential_______")
		# print(
		# f"  Sample 1: ts={sample1.ts:.3f}, ds={sample1.ds:.3f}, Sample 2: ts={sample2.ts:.3f}, ds={sample2.ds:.3f}"
		# )
		p_diff = np.array([sample1.ts - sample2.ts, sample1.ds - sample2.ds])
		t_prog_diff = sample1.ts - sample2.ts
	elif stroke_type == "freeform":
		p_diff = np.array(
			[sample1.xs - sample2.xs, sample1.ys - sample2.ys, sample1.zs - sample2.zs]
		)
		t_prog_diff = sample1.zs - sample2.zs
	else:
		raise ValueError("Invalid stroke type")

	normal_diff = 1.0 - np.dot(sample1.normal, sample2.normal)
	curvature_diff = abs(sample1.curvature - sample2.curvature)
	a_diff = np.array([wn * normal_diff, wc * curvature_diff])
	# t_diff = np.array([sample1.timestamp - sample2.timestamp])
	t_diff = np.array([t_prog_diff])
	diff = np.array([wp * p_diff, wa * a_diff, wt * t_diff], dtype=object)

	# print(f"Sample Differential - p_diff: {p_diff}, a_diff: {a_diff}, t_diff: {t_diff}")

	return diff


def calculate_stroke_sample_match_distance(
	stroke1: Stroke,
	stroke2: Stroke,
	wp: float = 0.5,
	wa: float = 1.0,
	wt: float = 0.5,
	wn: float = 0.8,
	wc: float = 0.5,
) -> float:
	"""
	Calculates the distance between two strokes based on the optimal matching
	   of their samples.

	Args:
		stroke1: The first stroke.
		stroke2: The second stroke.
		wp: Weight for position difference.
		wa: Weight for appearance/attribute difference.
		wt: Weight for temporal difference.
		wn: Weight for normal difference.
		wc: Weight for curvature difference.

	Returns:
		The distance between the two input strokes (||û(b', b)||^2)
	"""
	if not stroke1.samples or not stroke2.samples:
		return float("inf")

	cost_matrix = np.zeros((len(stroke1.samples), len(stroke2.samples)))
	for i, sample1 in enumerate(stroke1.samples):
		for j, sample2 in enumerate(stroke2.samples):
			diff = calculate_sample_differential(
				sample1, sample2, stroke1.stroke_type, wp, wa, wt, wn, wc
			)
			flat_diff = np.concatenate([np.atleast_1d(x).flatten() for x in diff])
			cost_matrix[i, j] = np.linalg.norm(flat_diff) ** 2
			# print(f"Cost Matrix[{i},{j}]: {cost_matrix[i,j]}, Flat Diff: {flat_diff}")
	try:
		row_ind, col_ind = linear_sum_assignment(cost_matrix)
		matched = cost_matrix[row_ind, col_ind]

		valid_matched = matched[np.isfinite(matched)]

		if len(valid_matched) == 0:
			return float("inf")
		total_distance_sq = np.sum(valid_matched)
		# normalized_distance_sq = total_distance_sq / len(valid_matched)

		return total_distance_sq

	except ValueError as e:
		print(f"Error in linear_sum_assignment: {e}")
		print(f"Cost matrix sample:\n{cost_matrix[:5,:5]}")
		return float("inf")


def calculate_average_stroke_differential(
	stroke1: Stroke,
	stroke2: Stroke,
	wp: float = 0.5,
	wa: float = 1.0,
	wt: float = 0.5,
	wn: float = 0.8,
	wc: float = 0.5,
) -> Optional[np.ndarray]:
	"""
	Calculates the average differential vector û(stroke1, stroke2) between two strokes
	based on optimal sample matching.

	Returns:
		The average flattened differential vector, or None if matching fails.
	"""
	if not stroke1.samples or not stroke2.samples:
		return None
	if stroke1.stroke_type != stroke2.stroke_type:
		return None

	sample1_test = stroke1.samples[0]
	sample2_test = stroke2.samples[0]
	try:
		test_diff = calculate_sample_differential(
			sample1_test, sample2_test, stroke1.stroke_type, wp, wa, wt, wn, wc
		)
		flat_test_diff = np.concatenate([np.atleast_1d(x).flatten() for x in test_diff])
		diff_vector_size = flat_test_diff.shape[0]
	except Exception as e:
		print(f"Error determining diff vector size for average: {e}")
		return None

	cost_matrix = np.zeros((len(stroke1.samples), len(stroke2.samples)))
	for i, sample1 in enumerate(stroke1.samples):
		for j, sample2 in enumerate(stroke2.samples):
			try:
				diff = calculate_sample_differential(
					sample1, sample2, stroke1.stroke_type, wp, wa, wt, wn, wc
				)
				flat_diff = np.concatenate([np.atleast_1d(x).flatten() for x in diff])
				if flat_diff.shape[0] != diff_vector_size:
					cost_matrix[i, j] = float("inf")
				else:
					cost_matrix[i, j] = np.linalg.norm(flat_diff) ** 2
			except Exception:
				cost_matrix[i, j] = float("inf")

	try:
		row_ind, col_ind = linear_sum_assignment(cost_matrix)

		valid_match_indices = [
			(r, c) for r, c in zip(row_ind, col_ind) if np.isfinite(cost_matrix[r, c])
		]
		if not valid_match_indices:
			# print(f"Warning: No finite matches found for avg diff between strokes {id(stroke1)} and {id(stroke2)}.")
			return None

		sum_diff_vector = np.zeros(diff_vector_size)
		num_matches = 0

		for r, c in valid_match_indices:
			sample1 = stroke1.samples[r]
			sample2 = stroke2.samples[c]
			try:
				diff = calculate_sample_differential(
					sample1, sample2, stroke1.stroke_type, wp, wa, wt, wn, wc
				)
				flat_diff = np.concatenate([np.atleast_1d(x).flatten() for x in diff])
				if flat_diff.shape[0] == diff_vector_size:
					sum_diff_vector += flat_diff
					num_matches += 1
				# else: print(f"Warning: inconsistent diff size during avg sum at ({r},{c})")
			except Exception as e:
				# print(f"Error calculating sample diff for avg sum at ({r},{c}): {e}")
				continue

		if num_matches > 0:
			return sum_diff_vector / num_matches
		else:
			# print(f"Warning: No valid differentials summed for avg diff between strokes {id(stroke1)} and {id(stroke2)}.")
			return None

	except ValueError as e:
		# print(f"Error in linear_sum_assignment for avg diff strokes {id(stroke1)}/{id(stroke2)}: {e}")
		return None


def calculate_stroke_match_cost(
	bo: Stroke,
	bi: Stroke,
	workflow: Workflow,
	geo_calc: Optional[CachedGeodesicCalculator],
	neighbor_params: dict = None,
	wp: float = 0.5,
	wa: float = 1.0,
	wt: float = 0.5,
	wn: float = 0.8,
	wc: float = 0.5,
) -> float:
	"""
	Calculates the full neighborhood distance ||n(bo) - n(bi)||² according to Eq 7.

	Args:
		bo: The candidate stroke.
		bi: The historical stroke i.
		workflow: The full historical workflow.
		geo_calc: Geodesic calculator instance.
		neighbor_params: Params for find_neighboring_strokes.
		wp: Weight for position difference.
		wa: Weight for appearance/attribute difference.
		wt: Weight for temporal difference.
		wn: Weight for normal difference.
		wc: Weight for curvature difference.

	Returns:
		The distance between the stroke neighborhoods.
	"""
	if neighbor_params is None:
		neighbor_params = {
			"num_temporal": 2,
			"spatial_radius_factor": 1.5,
			"temporal_window": 10.0,
		}

	central_distance_sq = calculate_stroke_sample_match_distance(
		bo, bi, wp, wa, wt, wn, wc
	)
	if central_distance_sq == float("inf"):
		return float("inf")

	neighbors0 = find_neighboring_strokes(
		target_stroke=bo,
		workflow=workflow,
		num_temporal=2,
		spatial_radius_factor=1.5,
		temporal_window=10.0,
		geo_calc=geo_calc,
	)
	neighborsi = find_neighboring_strokes(
		target_stroke=bi,
		workflow=workflow,
		num_temporal=2,
		spatial_radius_factor=1.5,
		temporal_window=10.0,
		geo_calc=geo_calc,
	)

	neighbor_distance_sum_sq = 0.0

	if neighbors0 and neighborsi:
		num_neighbors0 = len(neighbors0)
		num_neighborsi = len(neighborsi)

		neighbor_cost_matrix = np.full((num_neighbors0, num_neighborsi), float("inf"))

		for j in range(num_neighbors0):
			for k in range(num_neighborsi):
				dist_sq = calculate_stroke_sample_match_distance(
					neighbors0[j], neighborsi[k], wp, wa, wt, wn, wc
				)
				neighbor_cost_matrix[j, k] = dist_sq

		try:
			row_ind, col_ind = linear_sum_assignment(neighbor_cost_matrix)

			valid_matches_costs = neighbor_cost_matrix[row_ind, col_ind]
			neighbor_distance_sum_sq = np.sum(
				valid_matches_costs[np.isfinite(valid_matches_costs)]
			)

		except ValueError as e:
			print(f"Error in neighbor matching linear_sum_assignment: {e}")
			neighbor_distance_sum_sq = 0.0

	# Eq. 7
	total_distance_sq = central_distance_sq + neighbor_distance_sum_sq
	# print("_______calculate_stroke_match_cost_______")
	# print(
	# f"central_distance_sq: {central_distance_sq} neighbor_distance_sum_sq: {neighbor_distance_sum_sq} total_distance_sq: {total_distance_sq}"
	# )

	return total_distance_sq


def calculate_neighborhood_distance(
	bo: Stroke,
	bi: Stroke,
	workflow: Workflow,
	geo_calc: Optional[CachedGeodesicCalculator],
	neighbor_params: dict = None,
	wp: float = 0.5,
	wa: float = 1.0,
	wt: float = 0.5,
	wn: float = 0.8,
	wc: float = 0.5,
) -> float:
	"""
	Calculates the full neighborhood distance ||n(bo) - n(bi)||^2

	Args:
		bo: The candidate stroke (or current stroke in optimization).
		bi: The historical stroke i.
		workflow: The full historical workflow.
		geo_calc: Geodesic calculator instance.
		neighbor_params: Params for find_neighboring_strokes.
		wp, wa, wt, wn, wc: Weights for differential calculation.

	Returns:
		The distance between the stroke neighborhoods, or float('inf').
	"""
	if neighbor_params is None:
		neighbor_params = {
			"num_temporal": 2,
			"spatial_radius_factor": 1.5,
			"temporal_window": 10.0,
		}

	central_distance_sq = calculate_stroke_match_cost(
		bo, bi, workflow, geo_calc, wp, wa, wt, wn, wc
	)
	if not np.isfinite(central_distance_sq):
		return float("inf")

	neighbors0 = find_neighboring_strokes(
		target_stroke=bo,
		workflow=workflow,
		num_temporal=2,
		spatial_radius_factor=1.5,
		temporal_window=10.0,
		geo_calc=geo_calc,
	)
	neighborsi = find_neighboring_strokes(
		target_stroke=bi,
		workflow=workflow,
		num_temporal=2,
		spatial_radius_factor=1.5,
		temporal_window=10.0,
		geo_calc=geo_calc,
	)

	neighbor_term_sum_sq = 0.0

	if neighbors0 and neighborsi:
		num_neighbors0 = len(neighbors0)
		num_neighborsi = len(neighborsi)

		neighbor_match_cost_matrix = np.full(
			(num_neighbors0, num_neighborsi), float("inf")
		)

		for j, n0 in enumerate(neighbors0):
			for k, ni in enumerate(neighborsi):
				cost = calculate_stroke_match_cost(
					n0, ni, workflow, geo_calc, wp, wa, wt, wn, wc
				)
				neighbor_match_cost_matrix[j, k] = (
					cost if np.isfinite(cost) else float("inf")
				)

		try:
			row_ind, col_ind = linear_sum_assignment(neighbor_match_cost_matrix)

			for r, c in zip(row_ind, col_ind):
				if not np.isfinite(neighbor_match_cost_matrix[r, c]):
					continue

				neighbor0 = neighbors0[r]
				neighbori = neighborsi[c]

				diff_bo = calculate_average_stroke_differential(
					neighbor0, bo, wp, wa, wt, wn, wc
				)

				diff_bi = calculate_average_stroke_differential(
					neighbori, bi, wp, wa, wt, wn, wc
				)

				if diff_bo is not None and diff_bi is not None:
					if diff_bo.shape == diff_bi.shape:
						term_sq_norm = np.linalg.norm(diff_bo - diff_bi) ** 2
						neighbor_term_sum_sq += term_sq_norm
					# else: print(f"Warning: Mismatched differential shapes for neighbors {id(neighbor0)}/{id(neighbori)} vs centers {id(bo)}/{id(bi)}")
				# else: print(f"Warning: Could not calculate avg diff for neighbor pair {id(neighbor0)}/{id(neighbori)} vs centers {id(bo)}/{id(bi)}")

		except ValueError as e:
			print(
				f"Error in neighbor matching linear_sum_assignment for neighborhood dist: {e}"
			)
			neighbor_term_sum_sq = 0.0

	total_distance_sq = central_distance_sq + neighbor_term_sum_sq

	# print("_______calculate_neighborhood_distance_______")
	# print(
	# f"Neighborhood Distance ({id(bo)} vs {id(bi)}): Central= {central_distance_sq:.4f}, NeighborTerm= {neighbor_term_sum_sq:.4f}, Total= {total_distance_sq:.4f}"
	# )

	return total_distance_sq if np.isfinite(total_distance_sq) else float("inf")


def calculate_raw_sample_differential(
	sample_target: Sample, sample_base: Sample, stroke_type: str
) -> dict:
	"""
	Calculates the raw differential components û(sample_target, sample_base).
	Represents the transformation needed to get from base to target.

	Args:
		sample_target: The "later" sample (e.g., s_{i+1}).
		sample_base: The "earlier" sample (e.g., s_i).
		stroke_type: "surface" or "freeform".

	Returns:
		A dictionary containing differential components:
		'd_ts', 'd_ds' (for surface)
		'd_xs', 'd_ys', 'd_zs' (for freeform)
		'd_normal' (axis-angle rotation vector),
		'd_curvature', 'd_size', 'd_pressure', 'd_time'
	"""
	diff = {}

	if stroke_type == "surface":
		diff["d_ts"] = sample_target.ts - sample_base.ts
		diff["d_ds"] = sample_target.ds - sample_base.ds
		diff["d_xs"] = 0.0
		diff["d_ys"] = 0.0
		diff["d_zs"] = 0.0
	elif stroke_type == "freeform":
		diff["d_xs"] = sample_target.xs - sample_base.xs
		diff["d_ys"] = sample_target.ys - sample_base.ys
		diff["d_zs"] = sample_target.zs - sample_base.zs
		diff["d_ts"] = 0.0
		diff["d_ds"] = 0.0
	else:
		raise ValueError("Invalid stroke type for differential")

	n_base = sample_base.normal
	n_target = sample_target.normal
	norm_base = np.linalg.norm(n_base)
	norm_target = np.linalg.norm(n_target)
	if norm_base > 1e-6 and norm_target > 1e-6:
		n_base /= norm_base
		n_target /= norm_target
		axis = np.cross(n_base, n_target)
		axis_norm = np.linalg.norm(axis)
		if axis_norm > 1e-6:
			axis /= axis_norm
			dot_product = np.dot(n_base, n_target)
			angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
			diff["d_normal_rot"] = axis * angle
		else:
			if np.dot(n_base, n_target) < 0:
				perp_axis = (
					np.array([1.0, 0.0, 0.0])
					if abs(n_base[0]) < 0.9
					else np.array([0.0, 1.0, 0.0])
				)
				axis = np.cross(n_base, perp_axis)
				axis /= np.linalg.norm(axis)
				diff["d_normal_rot"] = axis * np.pi
			else:
				diff["d_normal_rot"] = np.zeros(3)
	else:
		diff["d_normal_rot"] = np.zeros(3)

	diff["d_curvature"] = sample_target.curvature - sample_base.curvature
	diff["d_size"] = sample_target.size - sample_base.size
	diff["d_pressure"] = sample_target.pressure - sample_base.pressure
	diff["d_time"] = sample_target.timestamp - sample_base.timestamp

	# print("_______calculate_raw_sample_differential_______")
	# print(f"  RawDiff({id(sample_target)}, {id(sample_base)}): {diff}")

	return diff


def calculate_differential_distance_sq(
	bo: Stroke,
	b_prime: Stroke,
	bi: Stroke,
	bi_plus_1: Stroke,
	wp: float = 0.5,
	wa: float = 1.0,
	wt: float = 0.5,
	wn: float = 0.8,
	wc: float = 0.5,
) -> float:
	"""
	Calculates the squared norm of the difference between two stroke differentials.
	"""
	if (
		bo.stroke_type != b_prime.stroke_type
		or bi.stroke_type != bi_plus_1.stroke_type
		or bo.stroke_type != bi.stroke_type
	):
		print("Warning: Stroke type mismatch in differential distance calc.")
		return float("inf")

	diff_bo_prime = calculate_average_stroke_differential(
		bo, b_prime, wp, wa, wt, wn, wc
	)

	diff_bi_plus_1_bi = calculate_average_stroke_differential(
		bi_plus_1, bi, wp, wa, wt, wn, wc
	)

	if diff_bo_prime is None or diff_bi_plus_1_bi is None:
		return float("inf")

	if diff_bo_prime.shape != diff_bi_plus_1_bi.shape:
		return float("inf")

	try:
		distance_sq = np.linalg.norm(diff_bo_prime - diff_bi_plus_1_bi) ** 2
		# print(f"  DiffDist: bo/b'={diff_bo_prime[:3]}, bi+1/bi={diff_bi_plus_1_bi[:3]}, Dist^2={distance_sq}")
		return distance_sq if np.isfinite(distance_sq) else float("inf")
	except Exception as e:
		print(f"Error calculating norm of differential difference: {e}")
		return float("inf")
