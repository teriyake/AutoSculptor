"""
Core data structures
"""

from typing import Optional, List
import numpy as np  # type: ignore

# from autosculptor.core.brush import BrushMode


class Sample:
	"""
	Represents a single sample in a brush stroke.
	Based on the 'Sample' representation from Section 4.1 of the paper.
	"""

	def __init__(
		self, position, normal=None, size=1.0, pressure=1.0, timestamp=0, curvature=0.0
	):
		"""
		Initialize a new Sample.

		Args:
			position (np.ndarray): 3D position of the sample
			normal (np.ndarray, optional): Surface normal at the sample position
			size (float, optional): Size of the brush at this sample
			pressure (float, optional): Pressure applied at this sample
			timestamp (float, optional): Time when this sample was created
		"""
		self.position = np.array(position, dtype=np.float32)
		self.normal = (
			np.array(normal, dtype=np.float32)
			if normal is not None
			else np.array([0, 1, 0], dtype=np.float32)
		)
		self.size = float(size)
		self.pressure = float(pressure)
		self.timestamp = float(timestamp)
		self.curvature = float(curvature)

		self.ts = 0.0
		self.ds = 0.0
		self.xs = 0.0
		self.ys = 0.0
		self.zs = 0.0
		self.camera_lookat = np.array([0, 0, 1], dtype=np.float32)

	def __repr__(self):
		return (
			f"Sample(pos={self.position}, size={self.size}, pressure={self.pressure})"
		)


class Stroke:
	"""
	Represents a sequence of samples forming a brush stroke.
	"""

	def __init__(self):
		"""Initialize an empty brush stroke."""
		self.samples = []
		self.stroke_type = None
		self.start_time = None
		self.end_time = None

		# Brush properties
		self.brush_size: Optional[float] = None
		self.brush_strength: Optional[float] = None
		self.brush_mode: Optional[str] = None
		self.brush_falloff: Optional[str] = None

	def add_sample(self, sample):
		"""
		Add a sample to the stroke.

		Args:
			sample (Sample): The sample to add
		"""
		if self.start_time is None:
			self.start_time = sample.timestamp

		self.samples.append(sample)
		self.end_time = sample.timestamp

	def get_samples(self):
		"""
		Get all samples in the stroke.

		Returns:
			list: List of Sample objects
		"""
		return self.samples

	def __len__(self):
		return len(self.samples)

	def __repr__(self):
		return f"Stroke(type={self.stroke_type}, samples={len(self.samples)})"


class Workflow:
	"""
	Represents a sequence of strokes forming a sculpting workflow.
	"""

	def __init__(self):
		"""Initialize an empty workflow."""
		self.strokes = []
		self.region = None
		self.active_context_indices: Optional[List[int]] = None

	def add_stroke(self, stroke):
		"""
		Add a stroke to the workflow.

		Args:
			stroke (Stroke): The stroke to add
		"""
		self.strokes.append(stroke)

	def get_strokes(self):
		"""
		Get all strokes in the workflow.

		Returns:
			list: List of Stroke objects
		"""
		return self.strokes

	def set_active_context(self, indices: List[int]):
		"""Sets the active context using a list of valid stroke indices."""
		valid_indices = [i for i in indices if 0 <= i < len(self.strokes)]
		if valid_indices:
			self.active_context_indices = sorted(list(set(valid_indices)))
			print(
				f"Workflow: Active context set to indices: {self.active_context_indices}"
			)
		else:
			print(
				"Workflow: Attempted to set empty or invalid context. Clearing context."
			)
			self.clear_active_context()

	def clear_active_context(self):
		"""Clears the active context, making the full history active."""
		self.active_context_indices = None
		print("Workflow: Active context cleared.")

	def get_active_context_strokes(self) -> List[Stroke]:
		"""Returns the list of strokes in the active context, or all strokes if no context is set."""
		if self.active_context_indices is None:
			return self.strokes
		else:
			valid_context_strokes = []
			valid_indices = []
			for index in self.active_context_indices:
				if 0 <= index < len(self.strokes):
					valid_context_strokes.append(self.strokes[index])
					valid_indices.append(index)
			if len(valid_indices) != len(self.active_context_indices):
				print("Workflow: Context indices updated due to stroke removal.")
				self.active_context_indices = valid_indices if valid_indices else None

			return (
				valid_context_strokes if self.active_context_indices else self.strokes
			)

	def get_active_context_indices(self) -> Optional[List[int]]:
		"""Returns the list of indices in the active context, or None."""
		if self.active_context_indices is not None:
			valid_indices = [
				i for i in self.active_context_indices if 0 <= i < len(self.strokes)
			]
			if len(valid_indices) != len(self.active_context_indices):
				print("Workflow: Context indices updated due to stroke removal.")
				self.active_context_indices = valid_indices if valid_indices else None
		return self.active_context_indices

	def __len__(self):
		return len(self.strokes)

	def __repr__(self):
		context_info = (
			f" (Context: {self.active_context_indices})"
			if self.active_context_indices
			else " (Context: Full)"
		)
		return f"Workflow(strokes={len(self.strokes)}{context_info})"
