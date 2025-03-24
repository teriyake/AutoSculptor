import numpy as np
from typing import List, Dict, Tuple, Optional
from autosculptor.core.data_structures import Stroke, Sample
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


def calculate_stroke_neighborhood_distance(
	stroke1: Stroke,
	stroke2: Stroke,
	wp: float = 1.0,
	wa: float = 0.1,
	wt: float = 0.1,
	wn: float = 0.5,
	wc: float = 0.3,
) -> float:
	"""
	Calculates the distance between two stroke neighborhoods (Equation 7).

	Args:
	    stroke1: The first stroke.
	    stroke2: The second stroke.
	    wp: Weight for position difference.
	    wa: Weight for appearance/attribute difference.
	    wt: Weight for temporal difference.
	    wn: Weight for normal difference.
	    wc: Weight for curvature difference.

	Returns:
	    The distance between the stroke neighborhoods.
	"""
	cost_matrix = np.zeros((len(stroke1.samples), len(stroke2.samples)))
	for i, sample1 in enumerate(stroke1.samples):
		for j, sample2 in enumerate(stroke2.samples):
			diff = calculate_sample_differential(
				sample1, sample2, stroke1.stroke_type, wp, wa, wt, wn, wc
			)
			flat_diff = np.concatenate([x.flatten() for x in diff])
			cost_matrix[i, j] = np.linalg.norm(flat_diff)
			# print(f"Cost Matrix[{i},{j}]: {cost_matrix[i,j]}, Flat Diff: {flat_diff}")

	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	total_distance = cost_matrix[row_ind, col_ind].sum()

	# NOTE: Don't normalize!!!
	# if stroke1.samples and stroke2.samples:
	# total_distance /= max(len(stroke1.samples), len(stroke2.samples))

	return total_distance


def calculate_stroke_similarity(stroke1: Stroke, stroke2: Stroke) -> float:
	"""
	Calculate the similarity between two strokes based on neighborhood analysis.

	Args:
	    stroke1 (Stroke): First stroke to compare
	    stroke2 (Stroke): Second stroke to compare

	Returns:
	    float: Similarity score between 0.0 (not similar) and 1.0 (identical)
	"""
	distance = calculate_stroke_neighborhood_distance(stroke1, stroke2)

	similarity = max(0.0, 1.0 - distance)
	return similarity
