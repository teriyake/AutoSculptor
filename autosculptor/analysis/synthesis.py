import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from autosculptor.core.data_structures import Stroke, Sample, Workflow
from autosculptor.core.mesh_interface import MeshData
from autosculptor.analysis.parameterization import StrokeParameterizer
from autosculptor.analysis.similarity import (
	calculate_sample_differential,
	calculate_stroke_neighborhood_distance,
	calculate_stroke_similarity,
)
from autosculptor.analysis.neighborhood import calculate_stroke_neighborhoods


class StrokeSynthesizer:
	def __init__(self, mesh_data: MeshData):
		self.mesh_data = mesh_data
		self.parameterizer = StrokeParameterizer(mesh_data)

	def initialize_suggestions(self, current_workflow: Workflow) -> List[Stroke]:
		"""
		Initializes potential suggestion strokes based on ALL relevant past strokes.
		"""
		suggestions: List[Stroke] = []

		if not current_workflow.strokes or len(current_workflow.strokes) < 2:
			return suggestions

		last_stroke = current_workflow.strokes[-1]
		self.parameterizer.normalize_stroke_parameters(last_stroke)

		for i in range(len(current_workflow.strokes) - 1):
			past_stroke = current_workflow.strokes[i]
			self.parameterizer.normalize_stroke_parameters(past_stroke)
			next_stroke_to_copy = current_workflow.strokes[i + 1]

			similarity_distance = calculate_stroke_neighborhood_distance(
				last_stroke, past_stroke
			)

			if similarity_distance is not None:

				new_suggestion = Stroke()
				new_suggestion.stroke_type = next_stroke_to_copy.stroke_type

				cost_matrix = np.zeros(
					(len(last_stroke.samples), len(past_stroke.samples))
				)
				for m, last_sample in enumerate(last_stroke.samples):
					for n, past_sample in enumerate(past_stroke.samples):
						diff = calculate_sample_differential(
							last_sample,
							past_sample,
							last_stroke.stroke_type,
							1.0,
							0.5,
							0.3,
							0.3,
							0.2,
						)
						cost_matrix[m, n] = np.linalg.norm(
							np.concatenate([x.flatten() for x in diff])
						)
				row_ind, col_ind = linear_sum_assignment(cost_matrix)

				for m, n in zip(row_ind, col_ind):
					last_sample = last_stroke.samples[m]
					past_sample = past_stroke.samples[n]
					next_sample_index = min(n, len(next_stroke_to_copy.samples) - 1)
					next_sample = next_stroke_to_copy.samples[next_sample_index]

					if new_suggestion.stroke_type == "surface":
						delta_ts = next_sample.ts - past_sample.ts
						delta_ds = next_sample.ds - past_sample.ds
						new_ts = last_sample.ts + delta_ts
						new_ds = last_sample.ds + delta_ds

						new_sample = Sample(
							position=last_sample.position.copy(),
							normal=np.array([0, 1, 0]),
							size=next_sample.size,
							pressure=next_sample.pressure,
							timestamp=next_sample.timestamp,
							curvature=next_sample.curvature,
						)
						new_sample.ts = new_ts
						new_sample.ds = new_ds
						new_position, new_normal = self.parameterizer._params_to_world(
							new_sample, last_stroke, "surface"
						)

					elif new_suggestion.stroke_type == "freeform":
						delta_xs = next_sample.xs - past_sample.xs
						delta_ys = next_sample.ys - past_sample.ys
						delta_zs = next_sample.zs - past_sample.zs
						new_xs = last_sample.xs + delta_xs
						new_ys = last_sample.ys + delta_ys
						new_zs = last_sample.zs + delta_zs

						new_sample = Sample(
							position=np.array([0, 0, 0]),
							normal=np.array([0, 1, 0]),
							size=next_sample.size,
							pressure=next_sample.pressure,
							timestamp=next_sample.timestamp,
							curvature=next_sample.curvature,
						)
						new_sample.xs = new_xs
						new_sample.ys = new_ys
						new_sample.zs = new_zs
						new_position, new_normal = self.parameterizer._params_to_world(
							new_sample, last_stroke, "freeform"
						)
					else:
						raise ValueError("Unknown stroke type")

					new_sample.position = new_position
					new_sample.normal = new_normal
					new_suggestion.add_sample(new_sample)

				suggestions.append(new_suggestion)

		return suggestions

	def _calculate_energy(
		self,
		candidate_stroke: Stroke,
		current_workflow: Workflow,
		max_spatial_distance: float,
		max_temporal_difference: float,
	) -> float:
		"""Calculates the energy of a candidate stroke (Equation 11)."""

		if not current_workflow.strokes:
			return 0.0

		min_energy = float("inf")

		for i in range(len(current_workflow.strokes) - 1):
			past_stroke = current_workflow.strokes[i]

			neighborhood_distance = calculate_stroke_neighborhood_distance(
				candidate_stroke,
				past_stroke,
				wp=1.0,
				wa=0.5,
				wt=0.3,
				wn=0.3,
				wc=0.2,
			)
			print(
				f"Candidate Stroke Energy - Past Stroke Index: {i}, Neighborhood Distance: {neighborhood_distance}"
			)

			#  TODO: Θ(bo) constrains... set to zero for now.
			application_constraint = 0.0

			energy = neighborhood_distance + application_constraint

			min_energy = min(min_energy, energy)

		print(f"Calculated Energy for Candidate: {energy}")
		return min_energy

	def synthesize_next_stroke(
		self,
		current_workflow: Workflow,
		max_spatial_distance: float,
		max_temporal_difference: float,
		energy_threshold: float = 10.0,
	) -> Optional[Stroke]:
		"""
		Synthesizes the next stroke based on the energy function (Equation 11).

		Args:
		    current_workflow: The current workflow.

		Returns:
		    The synthesized stroke (or None if no suitable stroke is found).
		"""
		candidate_strokes = self.initialize_suggestions(current_workflow)

		if not candidate_strokes:
			return None

		best_stroke = None
		min_energy = float("inf")

		for candidate_stroke in candidate_strokes:
			energy = self._calculate_energy(
				candidate_stroke,
				current_workflow,
				max_spatial_distance,
				max_temporal_difference,
			)
			if energy < min_energy:
				min_energy = energy
				best_stroke = candidate_stroke

		if min_energy > energy_threshold:
			return None

		return best_stroke
