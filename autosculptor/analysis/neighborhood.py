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

	mesh_diagonal = np.linalg.norm(
		np.max(mesh_data.vertices, axis=0) - np.min(mesh_data.vertices, axis=0)
	)
	max_spatial_distance = mesh_diagonal * 0.15

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


def find_neighboring_strokes(
	target_stroke: Stroke,
	workflow: Workflow,
	num_temporal: int = 2,
	spatial_radius_factor: float = 1.5,
	temporal_window: float = 10.0,
	geo_calc: Optional[CachedGeodesicCalculator] = None,
) -> List[Stroke]:
	"""
	Finds neighboring strokes for a given target_stroke within a workflow.
	Includes temporal predecessors and spatially close strokes within a time window.
	"""
	neighbors = []
	target_stroke_index = -1
	try:
		target_stroke_index = workflow.strokes.index(target_stroke)
	except ValueError:
		pass

	if target_stroke_index != -1:
		start_index = max(0, target_stroke_index - num_temporal)
		neighbors.extend(workflow.strokes[start_index:target_stroke_index])
	else:
		if workflow.strokes:
			last_workflow_time = (
				workflow.strokes[-1].end_time if workflow.strokes[-1].end_time else 0
			)

			temporal_candidates = sorted(
				[
					s
					for s in workflow.strokes
					if s.end_time is not None and s.end_time < last_workflow_time
				],
				key=lambda s: s.end_time,
				reverse=True,
			)
			neighbors.extend(temporal_candidates[:num_temporal])

	if not target_stroke.samples:
		return neighbors

	avg_target_size = (
		target_stroke.brush_size
		if target_stroke.brush_size
		else np.mean([s.size for s in target_stroke.samples])
	)
	spatial_threshold = avg_target_size * spatial_radius_factor

	target_time = (
		target_stroke.start_time
		if target_stroke.start_time is not None
		else (workflow.strokes[-1].end_time if workflow.strokes else 0)
	)

	unique_neighbors = set(neighbors)

	for potential_neighbor in workflow.strokes:
		if potential_neighbor == target_stroke:
			continue
		if potential_neighbor in unique_neighbors:
			continue
		if not potential_neighbor.samples:
			continue

		neighbor_time = (
			potential_neighbor.start_time
			if potential_neighbor.start_time is not None
			else 0
		)
		if abs(target_time - neighbor_time) > temporal_window:
			continue

		spatially_close = False
		if geo_calc:
			target_positions = np.array([s.position for s in target_stroke.samples])
			neighbor_positions = np.array(
				[s.position for s in potential_neighbor.samples]
			)

			# TODO: Check bounding boxes first
			min_dist = float("inf")
			# dist = geo_calc.compute_distance(target_positions[0], neighbor_positions[0:1])[0]
			# min_dist = dist

			try:
				dists_target_to_neighbor = np.min(
					geo_calc.compute_many_to_many(target_positions, neighbor_positions),
					axis=1,
				)
				dists_neighbor_to_target = np.min(
					geo_calc.compute_many_to_many(neighbor_positions, target_positions),
					axis=1,
				)

				min_dist = min(
					np.min(dists_target_to_neighbor), np.min(dists_neighbor_to_target)
				)

			except Exception as e:
				print(
					f"Warning: Geodesic distance calculation failed during neighbor finding: {e}"
				)
				continue

			if min_dist < spatial_threshold:
				spatially_close = True
		else:
			pass

		if spatially_close:
			unique_neighbors.add(potential_neighbor)

	return list(unique_neighbors)


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
