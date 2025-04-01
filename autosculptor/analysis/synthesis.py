import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from autosculptor.core.data_structures import Stroke, Sample, Workflow
from autosculptor.core.mesh_interface import MeshData
from autosculptor.analysis.parameterization import StrokeParameterizer
from autosculptor.analysis.similarity import (
	calculate_sample_differential,
	calculate_stroke_sample_match_distance,
	calculate_neighborhood_distance,
)
from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator
from autosculptor.analysis.neighborhood import calculate_stroke_neighborhoods
from autosculptor.maya.utils import get_active_camera_lookat_vector


class StrokeSynthesizer:
	def __init__(self, mesh_data: MeshData):
		self.mesh_data = mesh_data
		self.parameterizer = None

		if mesh_data and hasattr(mesh_data, "vertices"):
			self.parameterizer = StrokeParameterizer(mesh_data)
		else:
			print("Warning: StrokeSynthesizer created without valid mesh data")

	def _update_parameterizer_mesh(self, mesh_data: MeshData):
		"""Updates the mesh data used by the internal parameterizer."""
		if self.parameterizer:
			self.parameterizer.mesh_data = mesh_data

			self.parameterizer.geo_calc = CachedGeodesicCalculator(
				mesh_data.vertices, mesh_data.faces
			)
			print("StrokeSynthesizer: Updated parameterizer mesh data.")
		elif (
			mesh_data and hasattr(mesh_data, "vertices") and len(mesh_data.vertices) > 0
		):
			try:
				self.parameterizer = StrokeParameterizer(mesh_data)
				print(
					"StrokeSynthesizer: Initialized parameterizer with new mesh data."
				)
			except Exception as e:
				print(f"Error initializing StrokeParameterizer on update: {e}")
				self.parameterizer = None

	def initialize_suggestions(self, current_workflow: Workflow) -> List[Stroke]:
		"""
		Initializes potential suggestion strokes based on relevant past strokes based on neighborhood similarity.
		"""
		suggestions: List[Stroke] = []

		if not current_workflow.strokes or len(current_workflow.strokes) < 2:
			return suggestions

		last_stroke = current_workflow.strokes[-1]
		camera_lookat = get_active_camera_lookat_vector()
		self.parameterizer.parameterize_stroke(last_stroke, camera_lookat)
		self.parameterizer.normalize_stroke_parameters(last_stroke)
		geo_calc = self.parameterizer.geo_calc

		candidate_indices = []
		for i in range(len(current_workflow.strokes) - 1):
			past_stroke_bi = current_workflow.strokes[i]
			self.parameterizer.parameterize_stroke(past_stroke_bi, camera_lookat)
			self.parameterizer.normalize_stroke_parameters(past_stroke_bi)
			next_stroke_to_copy = current_workflow.strokes[i + 1]

			# TODO: Pass a truncated workflow ending before past_stroke
			neighbor_dist_sq = calculate_neighborhood_distance(
				last_stroke, past_stroke_bi, current_workflow, geo_calc
			)
			if neighbor_dist_sq != float("inf"):
				next_stroke_index = i + 1
				if next_stroke_index < len(current_workflow.strokes):
					candidate_indices.append({"index": i, "dist_sq": neighbor_dist_sq})

		candidate_indices.sort(key=lambda x: x["dist_sq"])
		num_candidates_to_use = min(3, len(candidate_indices))
		print(
			f"Found {len(candidate_indices)} potential matches. Using top {num_candidates_to_use}."
		)

		for candidate in candidate_indices[:num_candidates_to_use]:
			i = candidate["index"]
			past_stroke_bi = current_workflow.strokes[i]
			next_stroke_to_copy_bi_plus_1 = current_workflow.strokes[i + 1]

			print(
				f"\nGenerating candidate from past stroke {i} (Dist^2: {candidate['dist_sq']:.4f})"
			)

			if not next_stroke_to_copy_bi_plus_1.samples:
				print(f"  Skipping: Next stroke {i+1} has no samples.")
				continue

			match_map_prime0_i = self._match_samples(last_stroke, past_stroke_bi)
			if match_map_prime0_i is None:
				print(
					f"  Skipping: Failed to match samples between last stroke and past stroke {i}."
				)
				continue

			new_suggestion = Stroke()
			new_suggestion.stroke_type = next_stroke_to_copy_bi_plus_1.stroke_type

			num_generated_samples = 0
			for idx_prime0, idx_i in match_map_prime0_i.items():
				last_sample = last_stroke.samples[idx_prime0]
				past_sample_bi = past_stroke_bi.samples[idx_i]

				next_sample_index = idx_i
				if next_sample_index >= len(next_stroke_to_copy_bi_plus_1.samples):
					print(
						f"  Warning: Index {next_sample_index} out of bounds for next_stroke {i+1}. Clamping."
					)
					next_sample_index = len(next_stroke_to_copy_bi_plus_1.samples) - 1
					if next_sample_index < 0:
						continue

				next_sample_bi_plus_1 = next_stroke_to_copy_bi_plus_1.samples[
					next_sample_index
				]
				new_sample = Sample(
					position=np.array([0.0, 0.0, 0.0]),
					normal=np.array([0.0, 1.0, 0.0]),
					size=next_sample_bi_plus_1.size,
					pressure=next_sample_bi_plus_1.pressure,
					timestamp=last_sample.timestamp,
					curvature=next_sample_bi_plus_1.curvature,
				)
				new_sample.camera_lookat = camera_lookat

				if new_suggestion.stroke_type == "surface":
					delta_ts = next_sample_bi_plus_1.ts - past_sample_bi.ts
					delta_ds = next_sample_bi_plus_1.ds - past_sample_bi.ds
					new_ts = last_sample.ts + delta_ts
					new_ds = last_sample.ds + delta_ds
					new_sample.ts = np.clip(new_ts, 0.0, 1.0)
					new_sample.ds = np.clip(new_ds, -1.0, 1.0)

					new_sample = Sample(
						position=last_sample.position.copy(),
						normal=np.array([0, 1, 0]),
						size=next_sample_bi_plus_1.size,
						pressure=next_sample_bi_plus_1.pressure,
						timestamp=next_sample_bi_plus_1.timestamp,
						curvature=next_sample_bi_plus_1.curvature,
					)

					print(f"--- Sample Mapping ---")
					print(
						f"    Match: last({idx_prime0})<->past({idx_i}) -> next({next_sample_index})"
					)
					print(
						f"      last_p: ts={last_sample.ts:.3f} ds={last_sample.ds:.3f}"
					)
					print(
						f"      past_p: ts={past_sample_bi.ts:.3f} ds={past_sample_bi.ds:.3f}"
					)
					print(
						f"      next_p: ts={next_sample_bi_plus_1.ts:.3f} ds={next_sample_bi_plus_1.ds:.3f}"
					)
					print(f"      delta : ts={delta_ts:.3f} ds={delta_ds:.3f}")
					print(
						f"      new_p : ts={new_sample.ts:.3f} ds={new_sample.ds:.3f}"
					)

				elif new_suggestion.stroke_type == "freeform":
					delta_xs = next_sample_bi_plus_1.xs - past_sample_bi.xs
					delta_ys = next_sample_bi_plus_1.ys - past_sample_bi.ys
					delta_zs = next_sample_bi_plus_1.zs - past_sample_bi.zs
					new_xs = last_sample.xs + delta_xs
					new_ys = last_sample.ys + delta_ys
					new_zs = last_sample.zs + delta_zs

					new_sample = Sample(
						position=np.array([0, 0, 0]),
						normal=np.array([0, 1, 0]),
						size=next_sample_bi_plus_1.size,
						pressure=next_sample_bi_plus_1.pressure,
						timestamp=next_sample_bi_plus_1.timestamp,
						curvature=next_sample_bi_plus_1.curvature,
					)
					new_sample.xs = new_xs
					new_sample.ys = new_ys
					new_sample.zs = np.clip(new_zs, 0.0, 1.0)
				else:
					raise ValueError("Unknown stroke type")

				new_position, new_normal = self.parameterizer._params_to_world(
					new_sample, last_stroke, new_suggestion.stroke_type
				)
				new_sample.position = new_position
				new_sample.normal = new_normal
				new_suggestion.add_sample(new_sample)
				num_generated_samples += 1

			if num_generated_samples > 0:
				# self.parameterizer.parameterize_stroke(new_suggestion, camera_lookat)
				suggestions.append(new_suggestion)
				print(
					f"  Generated candidate stroke with {num_generated_samples} samples."
				)
			else:
				print(f"  Failed to generate any valid samples for this candidate.")

		print(f"Generated {len(suggestions)} suggestions")
		for i, s in enumerate(suggestions):
			print(f"Suggestion {i}: {len(s.samples)} samples")

		return suggestions

	def _match_samples(
		self, stroke1: Stroke, stroke2: Stroke
	) -> Optional[Dict[int, int]]:
		"""
		Matches samples between stroke1 and stroke2 using Hungarian algorithm.

		Returns:
			A dictionary mapping index in stroke1 to index in stroke2.
		"""
		if (
			not stroke1.samples
			or not stroke2.samples
			or stroke1.stroke_type != stroke2.stroke_type
		):
			return None

		cost_matrix = np.zeros((len(stroke1.samples), len(stroke2.samples)))
		for i, s1 in enumerate(stroke1.samples):
			for j, s2 in enumerate(stroke2.samples):
				if stroke1.stroke_type == "surface":
					p1 = np.array([s1.ts, s1.ds])
					p2 = np.array([s2.ts, s2.ds])
				elif stroke1.stroke_type == "freeform":
					p1 = np.array([s1.xs, s1.ys, s1.zs])
					p2 = np.array([s2.xs, s2.ys, s2.zs])
				else:
					return None
				cost_matrix[i, j] = np.linalg.norm(p1 - p2) ** 2

		try:
			row_ind, col_ind = linear_sum_assignment(cost_matrix)
			match_map = {r: c for r, c in zip(row_ind, col_ind)}
			return match_map
		except ValueError:
			print("Warning: Sample matching failed (linear_sum_assignment).")
			return None

	def _calculate_energy(
		self,
		candidate_stroke: Stroke,
		current_workflow: Workflow,
		geo_calc: Optional[CachedGeodesicCalculator],
		max_spatial_distance: float,
		max_temporal_difference: float,
	) -> float:
		"""Calculates the energy of a candidate stroke (Equation 11)."""

		if not current_workflow.strokes:
			return 0.0

		min_energy = float("inf")

		for i in range(len(current_workflow.strokes) - 1):
			past_stroke = current_workflow.strokes[i]

			neighborhood_distance_sq = calculate_neighborhood_distance(
				candidate_stroke,
				past_stroke,
				current_workflow,
				geo_calc,
				wp=0.5,
				wa=0.3,
				wt=0.2,
				wn=0.3,
				wc=0.2,
			)
			# print(
			# f"Candidate Stroke Energy - Past Stroke Index: {i}, Neighborhood Distance: {neighborhood_distance}"
			# )

			#  TODO: Θ(bo) constrains... set to zero for now.
			application_constraint = 0.0

			energy = neighborhood_distance_sq + application_constraint

			min_energy = min(min_energy, energy)

		print(f"Calculated Energy for Candidate: {energy}")
		return min_energy

	def synthesize_next_stroke(
		self,
		current_workflow: Workflow,
		mesh_data: MeshData,
		num_iterations: int = 5,
		energy_threshold: float = 100.0,
		neighbor_params: dict = None,
	) -> Optional[Stroke]:
		"""
		Synthesizes the next stroke using initialization and iterative optimization.
		"""
		start_time = time.time()

		self._update_parameterizer_mesh(mesh_data)
		if not self.parameterizer:
			print("Error: Parameterizer not available for synthesis.")
			return None
		try:
			geo_calc = CachedGeodesicCalculator(mesh_data.vertices, mesh_data.faces)
		except Exception as e:
			print(f"Error creating GeodesicCalculator: {e}")
			return None

		if not current_workflow.strokes:
			print("Workflow has no strokes, cannot synthesize.")
			return None

		candidate_strokes = self.initialize_suggestions(current_workflow)
		if not candidate_strokes:
			print("No initial candidates generated.")
			return None

		best_initial_candidate = None
		min_initial_energy = float("inf")

		camera_lookat = get_active_camera_lookat_vector()
		for cand in candidate_strokes:
			self.parameterizer.parameterize_stroke(cand, camera_lookat)

		for candidate in candidate_strokes:
			current_min_dist_sq = float("inf")
			# best_match_stroke_i = None
			for i in range(len(current_workflow.strokes)):
				hist_stroke = current_workflow.strokes[i]
				self.parameterizer.parameterize_stroke(hist_stroke, camera_lookat)

				dist_sq = calculate_neighborhood_distance(
					candidate, hist_stroke, current_workflow, geo_calc, neighbor_params
				)
				if dist_sq < current_min_dist_sq:
					current_min_dist_sq = dist_sq
					# best_match_stroke_i = hist_stroke

			if current_min_dist_sq < min_initial_energy:
				min_initial_energy = current_min_dist_sq
				best_initial_candidate = candidate

		if best_initial_candidate is None:
			print("Could not select an initial candidate.")
			return None

		print(f"Selected initial candidate with energy (dist^2): {min_initial_energy}")

		optimized_stroke = Stroke()
		optimized_stroke.stroke_type = best_initial_candidate.stroke_type
		for sample in best_initial_candidate.samples:
			new_s = Sample(
				sample.position.copy(),
				sample.normal.copy(),
				sample.size,
				sample.pressure,
				sample.timestamp,
				sample.curvature,
			)
			new_s.ts, new_s.ds = sample.ts, sample.ds
			new_s.xs, new_s.ys, new_s.zs = sample.xs, sample.ys, sample.zs
			new_s.camera_lookat = sample.camera_lookat
			optimized_stroke.add_sample(new_s)

		last_stroke = current_workflow.strokes[-1]

		for iteration in range(num_iterations):
			iter_start_time = time.time()
			print(f"\n--- Optimization Iteration {iteration+1}/{num_iterations} ---")

			min_dist_sq = float("inf")
			best_match_stroke_bi = None
			best_match_index_i = -1

			for i in range(len(current_workflow.strokes)):
				hist_stroke = current_workflow.strokes[i]
				dist_sq = calculate_neighborhood_distance(
					optimized_stroke,
					hist_stroke,
					current_workflow,
					geo_calc,
					neighbor_params,
				)
				if dist_sq < min_dist_sq:
					min_dist_sq = dist_sq
					best_match_stroke_bi = hist_stroke
					best_match_index_i = i

			if best_match_stroke_bi is None:
				print("  Warning: Could not find best match stroke 'bi' in iteration.")
				break

			print(
				f"  Found best match 'bi' (index {best_match_index_i}) with dist^2: {min_dist_sq:.4f}"
			)

			if best_match_index_i > 0:
				predecessor_stroke_b_prime_i = current_workflow.strokes[
					best_match_index_i - 1
				]
			else:
				print("  Warning: Best match 'bi' has no predecessor.")
				break

			match_bo_bi = self._match_samples(optimized_stroke, best_match_stroke_bi)
			match_b_prime_b_prime_i = self._match_samples(
				last_stroke, predecessor_stroke_b_prime_i
			)

			if match_bo_bi is None or match_b_prime_b_prime_i is None:
				print("  Warning: Failed to match samples for update.")
				continue

			num_updated = 0
			for idx_s0, s0 in enumerate(optimized_stroke.samples):
				if idx_s0 not in match_bo_bi:
					continue
				idx_si = match_bo_bi[idx_s0]
				si = best_match_stroke_bi.samples[idx_si]

				# TODO: Need inverse map or re-match: last_stroke -> bo
				if idx_s0 >= len(last_stroke.samples):
					continue
				s_prime_0 = last_stroke.samples[idx_s0]

				if idx_s0 not in match_b_prime_b_prime_i:
					continue
				idx_s_prime_i = match_b_prime_b_prime_i[idx_s0]
				s_prime_i = predecessor_stroke_b_prime_i.samples[idx_s_prime_i]

				target_params = {}
				if optimized_stroke.stroke_type == "surface":
					delta_ts = si.ts - s_prime_i.ts
					delta_ds = si.ds - s_prime_i.ds
					target_params["ts"] = s_prime_0.ts + delta_ts
					target_params["ds"] = s_prime_0.ds + delta_ds
				elif optimized_stroke.stroke_type == "freeform":
					delta_xs = si.xs - s_prime_i.xs
					delta_ys = si.ys - s_prime_i.ys
					delta_zs = si.zs - s_prime_i.zs
					target_params["xs"] = s_prime_0.xs + delta_xs
					target_params["ys"] = s_prime_0.ys + delta_ys
					target_params["zs"] = s_prime_0.zs + delta_zs

				if optimized_stroke.stroke_type == "surface":
					s0.ts = target_params["ts"]
					s0.ds = target_params["ds"]
				elif optimized_stroke.stroke_type == "freeform":
					s0.xs = target_params["xs"]
					s0.ys = target_params["ys"]
					s0.zs = target_params["zs"]

				s0.size = si.size
				s0.pressure = si.pressure
				s0.curvature = si.curvature

				num_updated += 1

			print(
				f"  Updated parameters for {num_updated}/{len(optimized_stroke.samples)} samples."
			)

			if num_updated > 0:
				for s0 in optimized_stroke.samples:
					try:
						new_pos, new_norm = self.parameterizer._params_to_world(
							s0, last_stroke, optimized_stroke.stroke_type
						)
						s0.position = new_pos
						s0.normal = new_norm
					except Exception as e:
						print(
							f"  Warning: Inverse parameterization failed during update iter {iteration+1}: {e}"
						)

			print(
				f"  Iteration {iteration+1} took {time.time() - iter_start_time:.2f}s"
			)

		final_energy = (
			calculate_neighborhood_distance(
				optimized_stroke,
				best_match_stroke_bi,
				current_workflow,
				geo_calc,
				neighbor_params,
			)
			if best_match_stroke_bi
			else float("inf")
		)

		print(
			f"\nOptimization finished. Final energy (dist^2): {final_energy:.4f} (Threshold: {energy_threshold})"
		)
		print(f"Total synthesis time: {time.time() - start_time:.2f}s")

		if final_energy > energy_threshold:
			print("Final energy exceeds threshold. No suggestion returned.")
			return None

		if not all(
			hasattr(s, "position") and s.position is not None
			for s in optimized_stroke.samples
		):
			print("Warning: Final optimized stroke has missing positions.")
			for s0 in optimized_stroke.samples:
				if not hasattr(s0, "position") or s0.position is None:
					try:
						new_pos, new_norm = self.parameterizer._params_to_world(
							s0, last_stroke, optimized_stroke.stroke_type
						)
						s0.position = new_pos
						s0.normal = new_norm
					except:
						pass

		return optimized_stroke
