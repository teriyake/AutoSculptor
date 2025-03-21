import numpy as np
from typing import List, Optional, Tuple
from autosculptor.core.data_structures import Sample, Stroke, Workflow
from autosculptor.core.mesh_interface import MeshData
from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator


def calculate_sample_neighborhood(
	sample: Sample,
	current_stroke: Stroke,
	workflow: Workflow,
	mesh_data: MeshData,
	geo_calc: CachedGeodesicCalculator,
	max_spatial_distance: float,
	max_temporal_difference: float,
) -> List[Sample]:
	"""
	Calculate the neighborhood of a sample, considering samples from previous strokes in the workflow.

	Args:
	    sample: The sample for which to calculate the neighborhood.
	    current_stroke: The stroke the sample belongs to.
	    workflow: The entire workflow, including previous strokes.
	    mesh_data: Mesh data.
	    geo_calc: Geodesic calculator.
	    max_spatial_distance: Maximum geodesic distance.
	    max_temporal_difference: Maximum temporal difference.

	Returns:
	    A list of neighboring samples.
	"""
	neighborhood_samples: List[Sample] = []

	for stroke in workflow.strokes:
		is_current_stroke = stroke == current_stroke

		for other_sample in stroke.samples:
			if other_sample == sample and is_current_stroke:
				continue

			if is_current_stroke:
				if other_sample.timestamp >= sample.timestamp:
					continue

			if other_sample.timestamp > sample.timestamp + max_temporal_difference:
				continue

			temporal_diff = sample.timestamp - other_sample.timestamp
			if 0 < temporal_diff <= max_temporal_difference:
				geodesic_distance = geo_calc.compute_distance(
					sample.position, np.array([other_sample.position])
				)[0]

				if geodesic_distance <= max_spatial_distance:
					neighborhood_samples.append(other_sample)

	return neighborhood_samples


def calculate_stroke_neighborhoods(
	stroke: Stroke,
	workflow: Workflow,
	mesh_data: MeshData,
	max_spatial_distance: float,
	max_temporal_difference: float,
) -> List[List[Sample]]:
	"""
	Calculate neighborhoods for all samples in a stroke.

	Args:
	    stroke: The stroke to analyze.
	    workflow: The entire workflow.
	    mesh_data: Mesh data.
	    max_spatial_distance: Maximum geodesic distance.
	    max_temporal_difference: Maximum temporal difference.

	Returns:
	    A list of neighborhood samples for each sample in the stroke.
	"""
	geo_calc = CachedGeodesicCalculator(mesh_data.vertices, mesh_data.faces)

	neighborhoods = []
	for sample in stroke.samples:
		neighborhood = calculate_sample_neighborhood(
			sample,
			stroke,
			workflow,
			mesh_data,
			geo_calc,
			max_spatial_distance,
			max_temporal_difference,
		)
		neighborhoods.append(neighborhood)

	return neighborhoods


def calculate_stroke_similarity(
	stroke1: Stroke,
	stroke2: Stroke,
	mesh_data: MeshData,
	max_spatial_distance: float,
	max_temporal_difference: float,
) -> float:
	"""
	Calculate the similarity between two strokes based on neighborhood analysis.

	Args:
	    stroke1 (Stroke): First stroke to compare
	    stroke2 (Stroke): Second stroke to compare
	    mesh_data (MeshData): Mesh data containing vertices and faces
	    max_spatial_distance (float): Maximum geodesic distance threshold
	    max_temporal_difference (float): Maximum temporal difference threshold

	Returns:
	    float: Similarity score between 0.0 (not similar) and 1.0 (identical)
	"""
	geo_calc = CachedGeodesicCalculator(mesh_data.vertices, mesh_data.faces)

	positions1 = np.array([s.position for s in stroke1.samples])
	positions2 = np.array([s.position for s in stroke2.samples])

	distances = geo_calc.compute_many_to_many(positions1, positions2)

	if distances is None:
		return 0.0

	min_distances1 = np.min(distances, axis=1)
	min_distances2 = np.min(distances, axis=0)

	avg_min_distance1 = np.mean(min_distances1)
	avg_min_distance2 = np.mean(min_distances2)

	symmetric_distance = (avg_min_distance1 + avg_min_distance2) / 2.0

	mesh_diagonal = np.linalg.norm(
		np.max(mesh_data.vertices, axis=0) - np.min(mesh_data.vertices, axis=0)
	)
	normalization_factor = 0.1 * mesh_diagonal

	similarity = max(0.0, 1.0 - symmetric_distance / normalization_factor)

	return similarity