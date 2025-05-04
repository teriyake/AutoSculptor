import time
import copy
import math
import numpy as np  # type: ignore
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from autosculptor.core.data_structures import Stroke, Sample, Workflow
from autosculptor.core.mesh_interface import MeshData, MeshInterface
from autosculptor.analysis.parameterization import StrokeParameterizer
from autosculptor.analysis.similarity import (
	calculate_sample_differential,
	calculate_stroke_sample_match_distance,
	calculate_differential_distance_sq,
	calculate_neighborhood_distance,
	calculate_raw_sample_differential,
)
from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator
from autosculptor.analysis.neighborhood import calculate_stroke_neighborhoods
from autosculptor.maya.utils import get_active_camera_lookat_vector


def rotate_vector(vector: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
	angle = np.linalg.norm(axis_angle)
	if angle < 1e-9:
		return vector
	axis = axis_angle / angle
	cos_angle = np.cos(angle)
	sin_angle = np.sin(angle)
	cross_prod = np.cross(axis, vector)
	dot_prod = np.dot(axis, vector)
	rotated = (
		vector * cos_angle + cross_prod * sin_angle + axis * dot_prod * (1 - cos_angle)
	)

	return rotated


def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
	v0_n = v0 / (np.linalg.norm(v0) + 1e-9)
	v1_n = v1 / (np.linalg.norm(v1) + 1e-9)

	dot = np.dot(v0_n, v1_n)
	dot = np.clip(dot, -1.0, 1.0)

	if abs(dot) > 0.9995:
		result = (1 - t) * v0_n + t * v1_n
		return result / (np.linalg.norm(result) + 1e-9)

	theta_0 = np.arccos(dot)
	theta = theta_0 * t

	sin_theta = np.sin(theta_0)
	if abs(sin_theta) < 1e-9:
		return v0_n

	sin_theta_inv = 1.0 / sin_theta

	w0 = np.sin((1.0 - t) * theta_0) * sin_theta_inv
	w1 = np.sin(t * theta_0) * sin_theta_inv

	result = w0 * v0_n + w1 * v1_n
	return result / (np.linalg.norm(result) + 1e-9)


class StrokeSynthesizer:
	def __init__(self, mesh_data: MeshData):
		self.mesh_data = mesh_data
		self.parameterizer = None

		if mesh_data and hasattr(mesh_data, "vertices"):
			self.parameterizer = StrokeParameterizer(mesh_data)
		else:
			print("Warning: StrokeSynthesizer created without valid mesh data")

	def _calculate_rotation_between_normals(self, normal_source, normal_target):
		"""
		Calculates the rotation needed to align normal_source with normal_target.
		"""
		n_src = normal_source / (np.linalg.norm(normal_source) + 1e-9)
		n_tgt = normal_target / (np.linalg.norm(normal_target) + 1e-9)

		dot_product = np.dot(n_src, n_tgt)

		if np.isclose(dot_product, 1.0):
			return R.identity()
		if np.isclose(dot_product, -1.0):
			axis = np.cross(n_src, np.array([1.0, 0.0, 0.0]))
			if np.linalg.norm(axis) < 1e-6:
				axis = np.cross(n_src, np.array([0.0, 1.0, 0.0]))
			axis /= np.linalg.norm(axis) + 1e-9
			return R.from_rotvec(axis * math.pi)

		axis = np.cross(n_src, n_tgt)
		angle = np.arccos(dot_product)
		axis /= np.linalg.norm(axis) + 1e-9
		return R.from_rotvec(axis * angle)

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

		active_context_indices = current_workflow.get_active_context_indices()
		active_strokes = current_workflow.get_active_context_strokes()
		full_stroke_count = len(current_workflow.strokes)

		if not active_strokes or full_stroke_count < 1:
			print("Initialize Suggestions: Not enough strokes in context or workflow.")
			return suggestions

		last_stroke_b_prime = current_workflow.strokes[-1]
		camera_lookat = get_active_camera_lookat_vector()
		self._ensure_parameterized(last_stroke_b_prime, camera_lookat)
		geo_calc = self.parameterizer.geo_calc

		candidate_indices = []
		for i, past_stroke_bi in enumerate(active_strokes):
			original_index_i = (
				active_context_indices[i] if active_context_indices else i
			)
			next_stroke_index = original_index_i + 1
			if next_stroke_index >= full_stroke_count:
				continue
			next_stroke_bi_plus_1 = current_workflow.strokes[next_stroke_index]

			self._ensure_parameterized(past_stroke_bi, camera_lookat)
			self._ensure_parameterized(next_stroke_bi_plus_1, camera_lookat)

			if last_stroke_b_prime.stroke_type != past_stroke_bi.stroke_type:
				continue
			if past_stroke_bi.stroke_type != next_stroke_bi_plus_1.stroke_type:
				continue
			if not past_stroke_bi.samples or not next_stroke_bi_plus_1.samples:
				continue

			neighbor_dist_sq = calculate_neighborhood_distance(
				bo=last_stroke_b_prime,
				bi=past_stroke_bi,
				workflow=current_workflow,
				geo_calc=geo_calc,
				neighbor_params=None,
				wp=0.1,
				wa=0.8,
				wt=0.5,
				wn=0.8,
				wc=0.2,
			)
			if neighbor_dist_sq != float("inf"):
				next_stroke_index = i + 1
				if next_stroke_index < len(current_workflow.strokes):
					candidate_indices.append(
						{
							"original_index": original_index_i,
							"dist_sq": neighbor_dist_sq,
						}
					)

		candidate_indices.sort(key=lambda x: x["dist_sq"])
		num_candidates_to_use = min(3, len(candidate_indices))
		# print(
		# 	f"Found {len(candidate_indices)} potential matches. Using top {num_candidates_to_use}."
		# )

		for candidate in candidate_indices[:num_candidates_to_use]:
			original_index_i = candidate["original_index"]
			past_stroke_bi = current_workflow.strokes[original_index_i]
			next_stroke_bi_plus_1 = current_workflow.strokes[original_index_i + 1]

			# print(
			# f"\nGenerating candidate from past stroke {original_index_i} (Dist^2: {candidate['dist_sq']:.4f})"
			# )

			if not next_stroke_bi_plus_1.samples:
				print(f"  Skipping: Next stroke {original_index_i+1} has no samples.")
				continue

			match_map_prime_i = self._match_samples(last_stroke_b_prime, past_stroke_bi)
			if match_map_prime_i is None:
				print(
					f"  Skipping: Failed to match samples between last stroke and past stroke {i}."
				)
				continue

			initial_suggestion = Stroke()
			initial_suggestion.stroke_type = last_stroke_b_prime.stroke_type
			initial_suggestion.brush_size = last_stroke_b_prime.brush_size
			initial_suggestion.brush_strength = last_stroke_b_prime.brush_strength
			initial_suggestion.brush_mode = last_stroke_b_prime.brush_mode
			initial_suggestion.brush_falloff = last_stroke_b_prime.brush_falloff

			num_generated_samples = 0
			for idx_prime, idx_i in match_map_prime_i.items():
				s_prime = last_stroke_b_prime.samples[idx_prime]
				s_i = past_stroke_bi.samples[idx_i]

				min_dist = float("inf")
				s_i_plus_1 = None
				for s_next_candidate in next_stroke_bi_plus_1.samples:
					if initial_suggestion.stroke_type == "surface":
						p1 = np.array([s_i.ts, s_i.ds])
						p2 = np.array([s_next_candidate.ts, s_next_candidate.ds])
					else:
						p1 = np.array([s_i.xs, s_i.ys, s_i.zs])
						p2 = np.array(
							[
								s_next_candidate.xs,
								s_next_candidate.ys,
								s_next_candidate.zs,
							]
						)
					dist = np.linalg.norm(p1 - p2)
					if dist < min_dist:
						min_dist = dist
						s_i_plus_1 = s_next_candidate

				if s_i_plus_1 is None:
					print(
						f"  Warning: Could not find corresponding s_{original_index_i+1} for s_{original_index_i} (idx {idx_i}). Skipping sample."
					)
					continue

				delta_position = s_i_plus_1.position - s_i.position
				n_base = s_i.normal / (np.linalg.norm(s_i.normal) + 1e-9)
				n_target = s_i_plus_1.normal / (
					np.linalg.norm(s_i_plus_1.normal) + 1e-9
				)
				axis = np.cross(n_base, n_target)
				axis_norm = np.linalg.norm(axis)
				d_normal_rot = np.zeros(3)
				if axis_norm > 1e-9:
					axis /= axis_norm
					dot_product = np.dot(n_base, n_target)
					angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
					d_normal_rot = axis * angle
				elif np.dot(n_base, n_target) < 0:
					perp_axis = (
						np.array([1.0, 0.0, 0.0])
						if abs(n_base[0]) < 0.9
						else np.array([0.0, 1.0, 0.0])
					)
					axis = np.cross(n_base, perp_axis)
					axis /= np.linalg.norm(axis) + 1e-9
					d_normal_rot = axis * np.pi
				delta_size = s_i_plus_1.size - s_i.size
				delta_pressure = s_i_plus_1.pressure - s_i.pressure
				delta_curvature = s_i_plus_1.curvature - s_i.curvature

				target_position_initial = s_prime.position + delta_position
				target_normal_rotated = rotate_vector(s_prime.normal, d_normal_rot)

				target_size = s_prime.size + delta_size
				target_pressure = np.clip(s_prime.pressure + delta_pressure, 0.0, 1.0)
				target_curvature = s_prime.curvature + delta_curvature
				target_timestamp = s_prime.timestamp

				closest_point_data = MeshInterface.find_closest_point(
					self.parameterizer.mesh_data, target_position_initial
				)

				if closest_point_data and closest_point_data[0] is not None:
					final_position = np.array(closest_point_data[0])
					final_normal = target_normal_rotated
					norm_mag = np.linalg.norm(final_normal)
					if norm_mag > 1e-6:
						final_normal /= norm_mag
					else:
						final_normal = np.array([0.0, 1.0, 0.0])

					new_sample = Sample(
						position=final_position,
						normal=final_normal,
						size=target_size,
						pressure=target_pressure,
						timestamp=target_timestamp,
						curvature=target_curvature,
					)
					new_sample.camera_lookat = (
						s_prime.camera_lookat
						if hasattr(s_prime, "camera_lookat")
						else camera_lookat
					)

					initial_suggestion.add_sample(new_sample)
					num_generated_samples += 1
				else:
					print(
						f"  Warning: Could not project target position for sample init corresponding to s_prime idx {idx_prime}. Skipping."
					)

			if num_generated_samples > 0:
				self._ensure_parameterized(initial_suggestion, camera_lookat)
				suggestions.append(initial_suggestion)
				# print(
				# 	f"  Generated candidate stroke with {num_generated_samples} samples."
				# )
			else:
				print(f"  Failed to generate any valid samples for this candidate.")

		# print(f"Generated {len(suggestions)} suggestions")
		# for i, s in enumerate(suggestions):
		# print(f"Suggestion {i}: {len(s.samples)} samples")

		return suggestions

	def _ensure_parameterized(self, stroke: Stroke, camera_lookat: np.ndarray):
		"""Ensures a stroke has been parameterized."""
		needs_param = True
		if stroke.samples:
			if len(stroke.samples) > 1:
				last_s = stroke.samples[-1]
				if stroke.stroke_type == "surface" and abs(last_s.ts) > 1e-6:
					needs_param = False
				elif stroke.stroke_type == "freeform" and abs(last_s.zs) > 1e-6:
					needs_param = False
			elif len(stroke.samples) == 1:
				needs_param = False

		if needs_param and self.parameterizer:
			# print(f"  Parameterizing stroke {id(stroke)}...")
			try:
				self.parameterizer.parameterize_stroke(stroke, camera_lookat)
			except Exception as e:
				print(f"  Warning: Failed to parameterize stroke {id(stroke)}: {e}")

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

		cam_lookat = get_active_camera_lookat_vector()
		self._ensure_parameterized(stroke1, cam_lookat)
		self._ensure_parameterized(stroke2, cam_lookat)

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
				# TODO: Add attribute differences to cost?
				# cost_matrix[i, j] += 0.1 * (1.0 - np.dot(s1.normal, s2.normal))**2

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

		context_strokes = current_workflow.get_active_context_strokes()
		if not context_strokes:
			print("Warning: No strokes in active context for energy calculation.")
			return float("inf")

		for past_stroke in context_strokes:
			neighborhood_distance_sq = calculate_neighborhood_distance(
				candidate_stroke,
				past_stroke,
				current_workflow,
				geo_calc,
				wp=0.1,
				wa=0.8,
				wt=0.5,
				wn=0.8,
				wc=0.2,
			)
			# print(
			# f"Candidate Stroke Energy - Past Stroke Index: {i}, Neighborhood Distance: {neighborhood_distance}"
			# )

			#  TODO: Θ(bo) constrains... set to zero for now.
			application_constraint = 0.0

			energy = neighborhood_distance_sq + application_constraint

			min_energy = min(min_energy, energy)

		# print(f"Calculated Energy for Candidate: {energy}")
		return min_energy

	def _apply_optimization(
		self, target_stroke: Stroke, source_stroke: Stroke, alpha: float
	) -> Stroke:
		"""Helper to apply one step of parameter interpolation."""
		optimized_stroke = copy.deepcopy(target_stroke)

		match_map = self._match_samples(target_stroke, source_stroke)
		if match_map is None:
			print("Warning: Skipping optimization step due to failed sample matching.")
			return optimized_stroke

		for idx_target, s_target in enumerate(optimized_stroke.samples):
			if idx_target not in match_map:
				continue
			idx_source = match_map[idx_target]
			if idx_source >= len(source_stroke.samples):
				continue
			s_source = source_stroke.samples[idx_source]

			if optimized_stroke.stroke_type == "surface":
				s_target.ts = (1 - alpha) * s_target.ts + alpha * s_source.ts
				s_target.ds = (1 - alpha) * s_target.ds + alpha * s_source.ds
				s_target.ts = np.clip(s_target.ts, 0.0, 1.0)
				s_target.ds = np.clip(s_target.ds, -1.0, 1.0)
			elif optimized_stroke.stroke_type == "freeform":
				s_target.xs = (1 - alpha) * s_target.xs + alpha * s_source.xs
				s_target.ys = (1 - alpha) * s_target.ys + alpha * s_source.ys
				s_target.zs = (1 - alpha) * s_target.zs + alpha * s_source.zs
				s_target.zs = np.clip(s_target.zs, 0.0, 1.0)

			# s_target.size = (1 - alpha) * s_target.size + alpha * s_source.size
			# s_target.pressure = (1 - alpha) * s_target.pressure + alpha * s_source.pressure
			# s_target.curvature = (1 - alpha) * s_target.curvature + alpha * s_source.curvature

		return optimized_stroke

	def _apply_parameter_smoothing(self, stroke: Stroke, iterations: int = 3) -> Stroke:
		"""Applies smoothing to the parameter space of a stroke."""

		if not stroke.samples or len(stroke.samples) < 3:
			print("Skipping smoothing: Not enough samples.")
			return stroke

		if not self.parameterizer:
			return stroke

		print(f"Applying parameter smoothing ({iterations} iterations)...")
		smoothed_stroke = copy.deepcopy(stroke)

		num_samples = len(smoothed_stroke.samples)

		if smoothed_stroke.stroke_type == "surface":
			ts_values = np.array([s.ts for s in smoothed_stroke.samples])
			ds_values = np.array([s.ds for s in smoothed_stroke.samples])

			for _ in range(iterations):
				ts_old = ts_values.copy()
				ds_old = ds_values.copy()
				for i in range(1, num_samples - 1):
					ts_values[i] = (ts_old[i - 1] + ts_old[i] + ts_old[i + 1]) / 3.0
					ds_values[i] = (ds_old[i - 1] + ds_old[i] + ds_old[i + 1]) / 3.0
				ts_values[0], ts_values[-1] = ts_old[0], ts_old[-1]
				ds_values[0], ds_values[-1] = ds_old[0], ds_old[-1]

			for i, sample in enumerate(smoothed_stroke.samples):
				sample.ts = np.clip(ts_values[i], 0.0, 1.0)
				sample.ds = np.clip(ds_values[i], -1.0, 1.0)

		elif smoothed_stroke.stroke_type == "freeform":
			xs_values = np.array([s.xs for s in smoothed_stroke.samples])
			ys_values = np.array([s.ys for s in smoothed_stroke.samples])
			zs_values = np.array([s.zs for s in smoothed_stroke.samples])

			for _ in range(iterations):
				xs_old, ys_old, zs_old = (
					xs_values.copy(),
					ys_values.copy(),
					zs_values.copy(),
				)
				for i in range(1, num_samples - 1):
					xs_values[i] = (xs_old[i - 1] + xs_old[i] + xs_old[i + 1]) / 3.0
					ys_values[i] = (ys_old[i - 1] + ys_old[i] + ys_old[i + 1]) / 3.0
					zs_values[i] = (zs_old[i - 1] + zs_old[i] + zs_old[i + 1]) / 3.0
				xs_values[0], xs_values[-1] = xs_old[0], xs_old[-1]
				ys_values[0], ys_values[-1] = ys_old[0], ys_old[-1]
				zs_values[0], zs_values[-1] = zs_old[0], zs_old[-1]

			for i, sample in enumerate(smoothed_stroke.samples):
				sample.xs = xs_values[i]
				sample.ys = ys_values[i]
				sample.zs = np.clip(zs_values[i], 0.0, 1.0)
		else:
			return stroke

		if not smoothed_stroke.samples:
			return smoothed_stroke

		print("  Updating world coordinates after parameter smoothing...")
		valid_samples = []
		ref_sample = smoothed_stroke.samples[0]

		for i, sample in enumerate(smoothed_stroke.samples):
			try:
				new_pos, new_norm = self.parameterizer._params_to_world(
					sample, stroke, stroke.stroke_type
				)

				if new_pos is not None and not np.any(np.isnan(new_pos)):
					sample.position = new_pos
					if new_norm is not None:
						sample.normal = new_norm
					valid_samples.append(sample)
				else:
					print(
						f"  Warning: _params_to_world returned invalid position for sample {i}. Skipping."
					)
			except Exception as e:
				print(
					f"  Error in _params_to_world during smoothing update for sample {i}: {e}"
				)

		smoothed_stroke.samples = valid_samples
		if not smoothed_stroke.samples:
			print("Warning: Smoothing resulted in an empty stroke after world update.")
			return stroke

		print(
			f"Parameter smoothing complete. Stroke now has {len(smoothed_stroke.samples)} samples."
		)
		return smoothed_stroke

	def synthesize_next_stroke(
		self,
		current_workflow: Workflow,
		mesh_data: MeshData,
		num_iterations: int = 10,
		energy_threshold: float = 50.0,
		neighbor_params: dict = None,
		optimization_alpha=0.2,
		smoothing_iterations: int = 0,
	) -> Optional[Stroke]:
		"""
		Synthesizes the next stroke using initialization and iterative optimization.
		"""
		synthesis_start_time = time.time()
		# print("\n--- Starting Stroke Synthesis ---")

		self._update_parameterizer_mesh(mesh_data)
		if not self.parameterizer:
			print("Error: Parameterizer not available.")
			return None
		geo_calc = self.parameterizer.geo_calc if self.parameterizer else None

		if not current_workflow.strokes or len(current_workflow.strokes) < 2:
			print("Workflow history too short (<2 strokes). Cannot synthesize.")
			return None

		active_context_indices = current_workflow.get_active_context_indices()
		active_strokes = current_workflow.get_active_context_strokes()
		if not active_strokes:
			print("Workflow has no active context strokes. Cannot synthesize.")
			return None

		if len(current_workflow.strokes) == 0:
			print("Error: Workflow somehow became empty.")
			return None

		print(
			f"Synthesizing based on context: {active_context_indices if active_context_indices else 'Full History'}"
		)

		initial_candidates = self.initialize_suggestions(current_workflow)
		if not initial_candidates:
			print("No initial candidates generated.")
			return None

		best_initial_candidate = None
		min_initial_energy = float("inf")
		last_stroke = current_workflow.strokes[-1]
		camera_lookat = get_active_camera_lookat_vector()
		self._ensure_parameterized(last_stroke, camera_lookat)

		# print("Evaluating initial candidates...")
		for candidate in initial_candidates:
			self._ensure_parameterized(candidate, camera_lookat)
			current_min_dist_sq = float("inf")
			for hist_stroke_bi in active_strokes:
				if hist_stroke_bi.stroke_type != candidate.stroke_type:
					continue

				self._ensure_parameterized(hist_stroke_bi, camera_lookat)
				dist_sq = calculate_neighborhood_distance(
					candidate,
					hist_stroke_bi,
					current_workflow,
					geo_calc,
					neighbor_params,
					wp=0.1,
					wa=0.8,
					wt=0.5,
					wn=0.8,
					wc=0.2,
				)
				if dist_sq < current_min_dist_sq:
					current_min_dist_sq = dist_sq

			# print(
			# 	f"  Initial Candidate {initial_candidates.index(candidate)} min energy (dist^2): {current_min_dist_sq:.4f}"
			# )
			if current_min_dist_sq < min_initial_energy:
				min_initial_energy = current_min_dist_sq
				best_initial_candidate = copy.deepcopy(candidate)

		if best_initial_candidate is None:
			print("Could not select initial candidate.")
			return None
		print(
			f"Selected initial candidate with energy (dist^2): {min_initial_energy:.4f}"
		)

		optimized_stroke = best_initial_candidate
		last_iteration_energy = min_initial_energy
		last_stroke_b_prime = current_workflow.strokes[-1]
		for iteration in range(num_iterations):
			iter_start_time = time.time()
			# print(f"\n--- Optimization Iteration {iteration+1}/{num_iterations} ---")
			# print(f"last_iteration_energy: {last_iteration_energy}")

			self._ensure_parameterized(optimized_stroke, camera_lookat)

			min_dist_sq = float("inf")
			best_match_stroke_bi = None
			best_match_original_index_i = -1
			active_context_indices_current = (
				current_workflow.get_active_context_indices()
			)
			active_strokes_current = current_workflow.get_active_context_strokes()
			if not active_strokes_current:
				print(
					"  Warning: Active context became empty during optimization. Stopping."
				)
				break

			for i, hist_stroke_bi in enumerate(active_strokes_current):
				original_index_i = (
					active_context_indices_current[i]
					if active_context_indices_current
					else i
				)
				next_stroke_index = original_index_i + 1
				if next_stroke_index >= len(current_workflow.strokes):
					continue
				hist_stroke_bi_plus_1 = current_workflow.strokes[next_stroke_index]

				if hist_stroke_bi.stroke_type != optimized_stroke.stroke_type:
					continue
				if (
					not hist_stroke_bi_plus_1
					or hist_stroke_bi_plus_1.stroke_type != optimized_stroke.stroke_type
				):
					continue

				self._ensure_parameterized(hist_stroke_bi, camera_lookat)

				if i > 0:
					self._ensure_parameterized(
						current_workflow.strokes[i - 1], camera_lookat
					)
				if i + 1 < len(current_workflow.strokes):
					self._ensure_parameterized(
						current_workflow.strokes[i + 1], camera_lookat
					)

				neighbor_dist_sq = calculate_neighborhood_distance(
					optimized_stroke,
					hist_stroke_bi,
					current_workflow,
					geo_calc,
					neighbor_params,
					wp=1.0,
					wa=0.1,
					wt=0.1,
					wn=0.5,
					wc=0.2,
				)

				if neighbor_dist_sq < min_dist_sq:
					min_dist_sq = neighbor_dist_sq
					best_match_stroke_bi = hist_stroke_bi
					best_match_index_i = original_index_i

			if best_match_stroke_bi is None:
				print("  Warning: No best match 'bi'. Stopping.")
				break
			# if best_match_index_i == 0:
			# print(
			# "  Warning: Best match 'bi' lacks predecessor. Skipping iteration."
			# )
			# continue

			# print(
			# f"  Found best matching historical stroke 'bi' (index {best_match_index_i}) with neighborhood dist^2: {min_dist_sq:.4f}"
			# )

			hist_stroke_bi_plus_1 = current_workflow.strokes[best_match_index_i + 1]
			self._ensure_parameterized(hist_stroke_bi_plus_1, camera_lookat)

			current_optimization_energy = calculate_differential_distance_sq(
				bo=optimized_stroke,
				b_prime=last_stroke_b_prime,
				bi=best_match_stroke_bi,
				bi_plus_1=hist_stroke_bi_plus_1,
				wp=0.1,
				wa=1.5,
				wt=1.0,
				wn=1.0,
				wc=0.8,
			)

			# print(
			# 	f"  Current Optimization Energy (DiffDist^2): {current_optimization_energy:.4f} (Previous: {last_iteration_energy:.4f})"
			# )

			improvement_tolerance = 1e-4
			energy_increase_factor = 1.05

			if iteration > 0:
				if (
					current_optimization_energy
					> last_iteration_energy * energy_increase_factor
				):
					print(
						f"  Warning: Optimization Energy increased significantly from {last_iteration_energy:.4f} to {current_optimization_energy:.4f}. Stopping early."
					)
					break
				if (
					last_iteration_energy - current_optimization_energy
				) < improvement_tolerance:
					print(
						f"  Info: Optimization energy improvement ({last_iteration_energy - current_optimization_energy:.4f}) less than tolerance ({improvement_tolerance}). Stopping early."
					)
					break

			last_iteration_energy = current_optimization_energy

			successor_stroke_b_i_plus_1 = current_workflow.strokes[
				best_match_index_i + 1
			]
			self._ensure_parameterized(successor_stroke_b_i_plus_1, camera_lookat)
			self._ensure_parameterized(best_match_stroke_bi, camera_lookat)

			target_world_updates = {}
			num_targets_calculated = 0

			match_bo_b_prime = self._match_samples(
				optimized_stroke, last_stroke_b_prime
			)
			if match_bo_b_prime is None:
				print(
					"  Warning: Failed matching bo -> b_prime for update step. Skipping iteration."
				)
				continue

			match_b_prime_bi = self._match_samples(
				last_stroke_b_prime, best_match_stroke_bi
			)
			if match_b_prime_bi is None:
				print(
					"  Warning: Failed matching b_prime -> bi for update step. Skipping iteration."
				)
				continue

			match_bo_bi = self._match_samples(optimized_stroke, best_match_stroke_bi)

			if match_bo_bi is None:
				print(
					"  Warning: Failed sample matching bo -> bi. Skipping target calculation."
				)
				continue

			match_i_plus_1_bi = self._match_samples(
				successor_stroke_b_i_plus_1, best_match_stroke_bi
			)
			map_bi_to_bi_plus_1 = {v: k for k, v in match_i_plus_1_bi.items()}

			for idx_bo, s_bo in enumerate(optimized_stroke.samples):
				if idx_bo not in match_bo_b_prime:
					continue

				idx_b_prime = match_bo_b_prime[idx_bo]
				if idx_b_prime >= len(last_stroke_b_prime.samples):
					continue

				s_prime = last_stroke_b_prime.samples[idx_b_prime]

				if idx_b_prime not in match_b_prime_bi:
					print(
						f"  Warning: Cannot find corresponding s_prime for s_i (idx {idx_bi}). Skipping target sample."
					)
					continue

				idx_bi = match_b_prime_bi[idx_b_prime]
				if idx_bi >= len(best_match_stroke_bi.samples):
					continue
				s_i = best_match_stroke_bi.samples[idx_bi]

				if idx_bi not in map_bi_to_bi_plus_1:
					print(
						f"  Cannot find corresponding s_i+1 for s_i (idx {idx_bi}) in update. Skipping target sample."
					)
					continue
				idx_bi_plus_1 = map_bi_to_bi_plus_1[idx_bi]
				if idx_bi_plus_1 >= len(hist_stroke_bi_plus_1.samples):
					continue
				s_i_plus_1 = hist_stroke_bi_plus_1.samples[idx_bi_plus_1]

				try:
					delta_position = s_i_plus_1.position - s_i.position
					n_base = s_i.normal / (np.linalg.norm(s_i.normal) + 1e-9)
					n_target = s_i_plus_1.normal / (
						np.linalg.norm(s_i_plus_1.normal) + 1e-9
					)
					axis = np.cross(n_base, n_target)
					axis_norm = np.linalg.norm(axis)
					d_normal_rot = np.zeros(3)
					if axis_norm > 1e-9:
						axis /= axis_norm
						dot_product = np.dot(n_base, n_target)
						angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
						d_normal_rot = axis * angle
					elif np.dot(n_base, n_target) < 0:
						perp_axis = (
							np.array([1.0, 0.0, 0.0])
							if abs(n_base[0]) < 0.9
							else np.array([0.0, 1.0, 0.0])
						)
						axis = np.cross(n_base, perp_axis)
						axis /= np.linalg.norm(axis) + 1e-9
						d_normal_rot = axis * np.pi
					delta_size = s_i_plus_1.size - s_i.size
					delta_pressure = s_i_plus_1.pressure - s_i.pressure
					delta_curvature = s_i_plus_1.curvature - s_i.curvature
				except Exception as e:
					print(
						f"  Warning: Error calculating historical delta for update: {e}. Skipping sample {idx_bo}"
					)
					continue

				target_position_initial = s_prime.position + delta_position
				target_normal_rotated = rotate_vector(s_prime.normal, d_normal_rot)

				closest_point_data = MeshInterface.find_closest_point(
					self.parameterizer.mesh_data, target_position_initial
				)

				if closest_point_data and closest_point_data[0] is not None:
					target_final_position = np.array(closest_point_data[0])
					target_final_normal = target_normal_rotated
					norm_mag = np.linalg.norm(target_final_normal)
					if norm_mag > 1e-6:
						target_final_normal /= norm_mag
					else:
						target_final_normal = np.array([0.0, 1.0, 0.0])

					target_size = max(0.01, s_prime.size + delta_size)
					target_pressure = np.clip(
						s_prime.pressure + delta_pressure, 0.0, 1.0
					)
					target_curvature = s_prime.curvature + delta_curvature

					target_world_updates[idx_bo] = (
						target_final_position,
						target_final_normal,
						target_size,
						target_pressure,
						target_curvature,
					)
					num_targets_calculated += 1
				else:
					print(
						f"  Could not project target position for update sample {idx_bo}. Skipping."
					)

			if num_targets_calculated == 0:
				print(
					"  Warning: Could not calculate any target parameters based on history match. Skipping update."
				)
				continue
			# print(
			# 	f"  Calculated {num_targets_calculated} target sample parameters for optimization step."
			# )

			param_changed_count = 0
			for idx_bo, s_bo in enumerate(optimized_stroke.samples):
				if idx_bo in target_world_updates:
					(
						target_pos,
						target_norm,
						target_size,
						target_press,
						target_curve,
					) = target_world_updates[idx_bo]

					# s_bo.position = (
					# 1 - optimization_alpha
					# ) * s_bo.position + optimization_alpha * target_pos

					interpolated_pos_linear = (
						1 - optimization_alpha
					) * s_bo.position + optimization_alpha * target_pos

					final_interpolated_pos = interpolated_pos_linear
					final_interpolated_norm = None

					closest_point_data_interp = MeshInterface.find_closest_point(
						self.parameterizer.mesh_data, interpolated_pos_linear
					)
					if (
						closest_point_data_interp
						and closest_point_data_interp[0] is not None
					):
						final_interpolated_pos = np.array(closest_point_data_interp[0])
						projected_normal = np.array(closest_point_data_interp[1])
						norm_mag_proj = np.linalg.norm(projected_normal)
						if norm_mag_proj > 1e-6:
							final_interpolated_norm = projected_normal / norm_mag_proj
						else:
							final_interpolated_norm = None
					else:
						final_interpolated_pos = interpolated_pos_linear

					s_bo.position = final_interpolated_pos

					if final_interpolated_norm is not None:
						s_bo.normal = final_interpolated_norm
					else:
						norm_s_bo = (
							np.linalg.norm(s_bo.normal)
							if s_bo.normal is not None
							else 0
						)
						norm_target = (
							np.linalg.norm(target_norm)
							if target_norm is not None
							else 0
						)
						if norm_s_bo > 1e-6 and norm_target > 1e-6:
							try:
								s_bo.normal = slerp(
									s_bo.normal, target_norm, optimization_alpha
								)
							except Exception as e:
								print(
									f"  Warning: SLERP failed for sample {idx_bo}: {e}. Keeping original normal."
								)
						else:
							pass

					s_bo.size = max(
						0.01,
						(1 - optimization_alpha) * s_bo.size
						+ optimization_alpha * target_size,
					)
					s_bo.pressure = np.clip(
						(1 - optimization_alpha) * s_bo.pressure
						+ optimization_alpha * target_press,
						0.0,
						1.0,
					)
					s_bo.curvature = (
						1 - optimization_alpha
					) * s_bo.curvature + optimization_alpha * target_curve

					param_changed_count += 1
			# print(f"  Parameters interpolated for {param_changed_count} samples.")
			if param_changed_count > 0:
				try:
					# print(
					# 	"  Re-parameterizing optimized stroke after world interpolation..."
					# )
					self._ensure_parameterized(optimized_stroke, camera_lookat)
				except Exception as e:
					print(
						f"  Warning: Failed to re-parameterize after world interpolation: {e}"
					)
			# if param_changed_count == 0: print("  Note: Parameters did not change significantly this iteration.")

			# print(
			# 	f"  Iteration {iteration + 1} took {time.time() - iter_start_time:.3f}s"
			# )

		if smoothing_iterations > 0:
			# print(
			# 	f"\nApplying final parameter smoothing ({smoothing_iterations} iterations)..."
			# )
			optimized_stroke = self._apply_parameter_smoothing(
				optimized_stroke, iterations=smoothing_iterations
			)
		self._ensure_parameterized(optimized_stroke, camera_lookat)

		final_energy = float("inf")
		final_best_match_bi = None
		final_best_match_idx = -1
		min_final_dist_sq = float("inf")
		active_context_indices_final = current_workflow.get_active_context_indices()
		active_strokes_final = current_workflow.get_active_context_strokes()

		if active_strokes_final:
			for i, hist_stroke_bi in enumerate(active_strokes_final):
				if hist_stroke_bi.stroke_type != optimized_stroke.stroke_type:
					continue
				self._ensure_parameterized(hist_stroke_bi, camera_lookat)
				if i > 0:
					self._ensure_parameterized(
						current_workflow.strokes[i - 1], camera_lookat
					)
				if i + 1 < len(current_workflow.strokes):
					self._ensure_parameterized(
						current_workflow.strokes[i + 1], camera_lookat
					)

				dist_sq = calculate_neighborhood_distance(
					optimized_stroke,
					hist_stroke_bi,
					current_workflow,
					geo_calc,
					neighbor_params,
					wp=0.1,
					wa=0.8,
					wt=0.5,
					wn=0.8,
					wc=0.2,
				)
				if dist_sq < min_final_dist_sq:
					min_final_dist_sq = dist_sq
					final_best_match_bi = hist_stroke_bi
					final_best_match_idx = (
						active_context_indices_final[i]
						if active_context_indices_final
						else i
					)

			if final_best_match_bi:
				final_energy = min_final_dist_sq
				# print(f"Final best match is stroke {final_best_match_idx}")
			else:
				print(
					"Warning: Could not find a final best match stroke 'bi'. Using last iteration's energy."
				)
				final_energy = last_iteration_energy
		else:
			print(
				"Warning: Optimization loop did not complete successfully. Using initial energy."
			)
			final_energy = min_initial_energy

		# print(
		# 	f"\nOptimization finished. Final energy (dist^2): {final_energy:.4f} (Threshold: {energy_threshold})"
		# )
		# print(f"Total synthesis time: {time.time() - synthesis_start_time:.2f}s")

		if final_energy > energy_threshold:
			print("Final energy exceeds threshold. No suggestion returned.")
			return None
		if not optimized_stroke.samples:
			print("Final stroke has no samples. No suggestion returned.")
			return None

		valid_samples = []
		needs_final_fixup = False
		for s in optimized_stroke.samples:
			if not (
				hasattr(s, "position")
				and s.position is not None
				and not np.any(np.isnan(s.position))
			):
				needs_final_fixup = True
				break
			valid_samples.append(s)
		if needs_final_fixup:
			print(
				"Warning: Final stroke has invalid positions. Attempting final fixup."
			)
			valid_samples = []
			for so in optimized_stroke.samples:
				try:
					new_pos, new_norm = self.parameterizer._params_to_world(
						so, last_stroke, optimized_stroke.stroke_type
					)
					if new_pos is not None and not np.any(np.isnan(new_pos)):
						so.position = new_pos
						if (
							optimized_stroke.stroke_type == "surface"
							and new_norm is not None
						):
							so.normal = new_norm
						valid_samples.append(so)
					else:
						print(
							"  Skipping sample during final fixup - invalid position generated."
						)
				except Exception as e:
					print(f"  Skipping sample during final fixup - error: {e}")
		optimized_stroke.samples = valid_samples
		if not optimized_stroke.samples:
			print(
				"Final stroke has no valid samples after fixup. No suggestion returned."
			)
			return None

		return optimized_stroke

	def clone_workflow(
		self,
		source_strokes: List[Stroke],
		source_anchor_pos: np.ndarray,
		source_anchor_normal: np.ndarray,
		target_anchor_pos: np.ndarray,
		target_anchor_normal: np.ndarray,
		target_mesh_data: MeshData,
		scale_factor: float = 1.0,
	) -> Optional[Workflow]:
		"""
		Clones a list of source strokes to a target location and orientation.

		Args:
			source_strokes: List of Stroke objects to clone.
			source_anchor_pos: 3D position of the source anchor.
			source_anchor_normal: Normal vector at the source anchor.
			target_anchor_pos: 3D position of the target anchor.
			target_anchor_normal: Normal vector at the target anchor.
			target_mesh_data: The MeshData representing the current state of the
							  mesh where the clone will be placed.
			scale_factor: Optional scaling factor for stroke size/attributes.

		Returns:
			A new Workflow object containing the cloned strokes, or None if cloning fails.
		"""
		print("\n--- Starting Workflow Cloning ---")
		if not source_strokes:
			print("Clone Error: No source strokes provided.")
			return None
		if not self.parameterizer:
			print("Clone Error: StrokeParameterizer not initialized.")
			return None
		if not target_mesh_data or not hasattr(target_mesh_data, "vertices"):
			print("Clone Error: Invalid target mesh data provided.")
			return None

		self._update_parameterizer_mesh(target_mesh_data)
		if not self.parameterizer:
			print("Clone Error: Parameterizer failed to initialize/update.")
			return None

		translation = target_anchor_pos - source_anchor_pos
		try:
			rotation = self._calculate_rotation_between_normals(
				source_anchor_normal, target_anchor_normal
			)
			rotation_matrix = rotation.as_matrix()
			print(f"  Translation: {translation}")
			print(f"  Rotation Axis/Angle: {rotation.as_rotvec()}")
		except Exception as e:
			print(f"Clone Error: Failed to calculate rotation: {e}")
			return None

		cloned_workflow = Workflow()
		last_cloned_sample: Optional[Sample] = None
		last_source_sample: Optional[Sample] = None

		for stroke_idx, source_stroke in enumerate(source_strokes):
			if not source_stroke.samples:
				print(f"  Skipping empty source stroke {stroke_idx}.")
				continue

			print(f"\n  Cloning stroke {stroke_idx} ({source_stroke.stroke_type})...")
			cloned_stroke = Stroke()
			cloned_stroke.stroke_type = source_stroke.stroke_type
			cloned_stroke.brush_size = (source_stroke.brush_size or 1.0) * scale_factor
			cloned_stroke.brush_strength = source_stroke.brush_strength
			cloned_stroke.brush_mode = source_stroke.brush_mode
			cloned_stroke.brush_falloff = source_stroke.brush_falloff

			dummy_cam_lookat = np.array([0.0, 0.0, 1.0])
			self._ensure_parameterized(source_stroke, dummy_cam_lookat)

			for sample_idx, source_sample in enumerate(source_stroke.samples):
				cloned_sample = Sample(
					position=np.zeros(3),
					normal=np.zeros(3),
					size=source_sample.size * scale_factor,
					pressure=source_sample.pressure,
					timestamp=source_sample.timestamp,
					curvature=source_sample.curvature,
				)
				if (
					hasattr(source_sample, "camera_lookat")
					and source_sample.camera_lookat is not None
				):
					cloned_sample.camera_lookat = rotation.apply(
						source_sample.camera_lookat
					)
				else:
					cloned_sample.camera_lookat = dummy_cam_lookat

				if sample_idx == 0 and stroke_idx == 0:
					cloned_sample.position = target_anchor_pos
					cloned_sample.normal = target_anchor_normal
					print(
						f"    Sample 0 (First): Placed at target anchor {cloned_sample.position}"
					)

				elif sample_idx == 0 and stroke_idx > 0:
					if last_source_sample is None:
						print(
							"    Error: Cannot determine start of stroke > 0 without previous source sample."
						)
						continue

					delta_pos_world = (
						source_sample.position - last_source_sample.position
					)
					try:
						delta_rot = self._calculate_rotation_between_normals(
							last_source_sample.normal, source_sample.normal
						)
					except Exception as e:
						print(
							f"    Warning: Could not calculate delta rotation for stroke start: {e}. Using identity."
						)
						delta_rot = R.identity()

					if last_cloned_sample is None:
						print(
							"    Error: Cannot determine start of stroke > 0 without previous cloned sample."
						)
						continue

					transformed_delta_pos = rotation.apply(delta_pos_world)
					initial_cloned_pos = (
						last_cloned_sample.position + transformed_delta_pos
					)

					initial_cloned_normal = delta_rot.apply(last_cloned_sample.normal)

					closest_data = MeshInterface.find_closest_point(
						target_mesh_data, initial_cloned_pos
					)
					if closest_data and closest_data[0] is not None:
						cloned_sample.position = np.array(closest_data[0])
						if cloned_stroke.stroke_type == "surface":
							cloned_sample.normal = np.array(closest_data[1])
						else:
							norm_mag = np.linalg.norm(initial_cloned_normal)
							cloned_sample.normal = (
								initial_cloned_normal / (norm_mag + 1e-9)
								if norm_mag > 1e-9
								else np.array([0, 1, 0])
							)

						print(
							f"    Sample 0 (Stroke {stroke_idx}): Placed relative to previous stroke end -> {cloned_sample.position}"
						)
					else:
						print(
							f"    Warning: Failed to snap start of stroke {stroke_idx} to mesh. Using calculated position."
						)
						cloned_sample.position = initial_cloned_pos
						norm_mag = np.linalg.norm(initial_cloned_normal)
						cloned_sample.normal = (
							initial_cloned_normal / (norm_mag + 1e-9)
							if norm_mag > 1e-9
							else np.array([0, 1, 0])
						)
				else:
					if last_source_sample is None or last_cloned_sample is None:
						print(
							f"    Error: Missing previous sample context for sample {sample_idx}. Skipping."
						)
						continue

					if cloned_stroke.stroke_type == "surface":
						try:
							diff_dict = calculate_raw_sample_differential(
								source_sample, last_source_sample, "surface"
							)
							d_ts = diff_dict["d_ts"]
							d_ds = diff_dict["d_ds"]
							d_normal_rot_vec = diff_dict["d_normal_rot"]
						except Exception as e:
							print(
								f"    Warning: Failed to calculate raw differential for sample {sample_idx}: {e}. Using world delta."
							)
							delta_pos_world = (
								source_sample.position - last_source_sample.position
							)
							try:
								delta_rot = self._calculate_rotation_between_normals(
									last_source_sample.normal, source_sample.normal
								)
							except:
								delta_rot = R.identity()

							transformed_delta_pos = rotation.apply(delta_pos_world)
							initial_cloned_pos = (
								last_cloned_sample.position + transformed_delta_pos
							)
							initial_cloned_normal = delta_rot.apply(
								last_cloned_sample.normal
							)

							closest_data = MeshInterface.find_closest_point(
								target_mesh_data, initial_cloned_pos
							)
							if closest_data and closest_data[0] is not None:
								cloned_sample.position = np.array(closest_data[0])
								cloned_sample.normal = np.array(closest_data[1])
							else:
								print(
									f"    Warning: Fallback world delta failed to snap sample {sample_idx}. Skipping."
								)
								continue
							print(
								f"    Sample {sample_idx}: Calculated via world delta fallback -> {cloned_sample.position}"
							)

						else:
							target_ts = last_cloned_sample.ts + d_ts
							target_ds = last_cloned_sample.ds + d_ds

							predicted_normal = rotate_vector(
								last_cloned_sample.normal, d_normal_rot_vec
							)
							norm_mag = np.linalg.norm(predicted_normal)
							predicted_normal = (
								predicted_normal / (norm_mag + 1e-9)
								if norm_mag > 1e-9
								else last_cloned_sample.normal
							)

							local_tangent = np.cross(
								np.array([0, 0, 1]), last_cloned_sample.normal
							)
							if np.linalg.norm(local_tangent) < 1e-6:
								local_tangent = np.array([1, 0, 0])
							local_tangent /= np.linalg.norm(local_tangent)
							local_bitangent = np.cross(
								last_cloned_sample.normal, local_tangent
							)

							delta_pos_world = (
								source_sample.position - last_source_sample.position
							)
							projected_delta_tangent = (
								np.dot(delta_pos_world, local_tangent) * local_tangent
							)
							projected_delta_bitangent = (
								np.dot(delta_pos_world, local_bitangent)
								* local_bitangent
							)
							local_plane_delta = (
								projected_delta_tangent + projected_delta_bitangent
							)

							transformed_local_delta = rotation.apply(local_plane_delta)

							initial_cloned_pos = (
								last_cloned_sample.position + transformed_local_delta
							)

							closest_data = MeshInterface.find_closest_point(
								target_mesh_data, initial_cloned_pos
							)
							if closest_data and closest_data[0] is not None:
								cloned_sample.position = np.array(closest_data[0])
								cloned_sample.normal = np.array(closest_data[1])
								print(
									f"    Sample {sample_idx}: Calculated via local frame delta -> {cloned_sample.position}"
								)
							else:
								print(
									f"    Warning: Failed to snap sample {sample_idx} using local frame delta. Skipping."
								)
								continue

					elif cloned_stroke.stroke_type == "freeform":
						delta_pos_world = (
							source_sample.position - last_source_sample.position
						)
						try:
							delta_rot = self._calculate_rotation_between_normals(
								last_source_sample.normal, source_sample.normal
							)
						except Exception as e:
							print(
								f"    Warning: Could not calculate delta rotation for freeform sample {sample_idx}: {e}. Using identity."
							)
							delta_rot = R.identity()

						transformed_delta_pos = rotation.apply(delta_pos_world)
						cloned_sample.position = (
							last_cloned_sample.position + transformed_delta_pos
						)

						cloned_sample.normal = delta_rot.apply(
							last_cloned_sample.normal
						)
						norm_mag = np.linalg.norm(cloned_sample.normal)
						cloned_sample.normal = (
							cloned_sample.normal / (norm_mag + 1e-9)
							if norm_mag > 1e-9
							else np.array([0, 1, 0])
						)
						print(
							f"    Sample {sample_idx}: Calculated via freeform world delta -> {cloned_sample.position}"
						)

				cloned_stroke.add_sample(cloned_sample)
				last_cloned_sample = cloned_sample
				last_source_sample = source_sample

			if cloned_stroke.samples:
				try:
					final_cam_lookat = (
						cloned_stroke.samples[-1].camera_lookat
						if hasattr(cloned_stroke.samples[-1], "camera_lookat")
						else dummy_cam_lookat
					)
					self._ensure_parameterized(cloned_stroke, final_cam_lookat)
					cloned_workflow.add_stroke(cloned_stroke)
					print(
						f"  Finished cloning stroke {stroke_idx} with {len(cloned_stroke.samples)} samples."
					)
				except Exception as e:
					print(
						f"  Warning: Failed to re-parameterize cloned stroke {stroke_idx}: {e}"
					)
					cloned_workflow.add_stroke(cloned_stroke)
			else:
				print(f"  Warning: Cloned stroke {stroke_idx} ended up empty.")

		print(
			f"--- Workflow Cloning Finished: {len(cloned_workflow.strokes)} strokes cloned ---"
		)
		return cloned_workflow
