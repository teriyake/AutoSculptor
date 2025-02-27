"""
Core data structures
"""

import numpy as np


class Sample:
    """
    Represents a single sample in a brush stroke.
    Based on the 'Sample' representation from Section 4.1 of the paper.
    """

    def __init__(self, position, normal=None, size=1.0, pressure=1.0, timestamp=0):
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

    def __len__(self):
        return len(self.strokes)

    def __repr__(self):
        return f"Workflow(strokes={len(self.strokes)})"
