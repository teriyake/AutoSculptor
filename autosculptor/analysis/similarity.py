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
	wn: float = 0.5,
	wc: float = 0.3,
) -> np.ndarray:
	"""
	Calculates the differential between two samples (Equation 4).

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
		# print(
		# f"  Sample 1: ts={sample1.ts:.3f}, ds={sample1.ds:.3f}, Sample 2: ts={sample2.ts:.3f}, ds={sample2.ds:.3f}"
		# )
		p_diff = np.array([sample1.ts - sample2.ts, sample1.ds - sample2.ds])
	elif stroke_type == "freeform":
		p_diff = np.array(
			[sample1.xs - sample2.xs, sample1.ys - sample2.ys, sample1.zs - sample2.zs]
		)
	else:
		raise ValueError("Invalid stroke type")

	normal_diff = 1.0 - np.dot(sample1.normal, sample2.normal)
	curvature_diff = abs(sample1.curvature - sample2.curvature)
	a_diff = np.array([wn * normal_diff, wc * curvature_diff])
	t_diff = np.array([sample1.timestamp - sample2.timestamp])
	diff = np.array([wp * p_diff, wa * a_diff, wt * t_diff], dtype=object)

	# print(f"Sample Differential - p_diff: {p_diff}, a_diff: {a_diff}, t_diff: {t_diff}")

	return diff


def calculate_stroke_sample_match_distance(
	stroke1: Stroke,
	stroke2: Stroke,
	wp: float = 1.0,
	wa: float = 0.1,
	wt: float = 0.1,
	wn: float = 0.5,
	wc: float = 0.3,
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
		total_distance_sq = cost_matrix[row_ind, col_ind].sum()

		# TODO: Normalize?
		# num_matches = len(row_ind)
		# normalized_distance_sq = total_distance_sq / num_matches if num_matches > 0 else 0.0
		# return normalized_distance_sq
		return total_distance_sq
	except ValueError as e:
		print(f"Error in linear_sum_assignment: {e}")
		print(f"Cost matrix sample:\n{cost_matrix[:5,:5]}")
		return float("inf")


def calculate_neighborhood_distance(
	bo: Stroke,
	bi: Stroke,
	workflow: Workflow,
	geo_calc: Optional[CachedGeodesicCalculator],
	neighbor_params: dict = None,
	wp: float = 1.0,
	wa: float = 0.1,
	wt: float = 0.1,
	wn: float = 0.5,
	wc: float = 0.3,
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
		bo, workflow, geo_calc=geo_calc, **neighbor_params
	)
	neighborsi = find_neighboring_strokes(
		bi, workflow, geo_calc=geo_calc, **neighbor_params
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

	return total_distance_sq
