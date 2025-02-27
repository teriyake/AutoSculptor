"""
Base Brush class
"""

from abc import ABC, abstractmethod
from enum import Enum
from .data_structures import Sample, Stroke


class BrushMode(Enum):
    """Enumeration for brush operation modes."""

    ADD = 1
    SUBTRACT = 2
    SMOOTH = 3


class Brush(ABC):
    """Abstract base class for all brushes."""

    def __init__(self, size=1.0, strength=1.0, mode=BrushMode.ADD, falloff="smooth"):
        """
        Initialize a brush.

        Args:
            size (float): Size/radius of the brush
            strength (float): Intensity of the brush effect
            mode (BrushMode): Operation mode (ADD, SUBTRACT, SMOOTH)
            falloff (str): Falloff type ('smooth', 'linear', 'constant')
        """
        self.size = size
        self.strength = strength
        self.mode = mode
        self.falloff = falloff
        self.current_stroke = None

    def begin_stroke(self):
        """
        Begin a new stroke and return it.

        Returns:
            Stroke: The newly created stroke
        """
        self.current_stroke = Stroke()
        return self.current_stroke

    def end_stroke(self):
        """
        End the current stroke and return it.

        Returns:
            Stroke: The completed stroke
        """
        stroke = self.current_stroke
        self.current_stroke = None
        return stroke

    def calculate_falloff(self, distance):
        """
        Calculate the falloff value based on the distance from brush center.

        Args:
            distance (float): Distance from the brush center

        Returns:
            float: Falloff value between 0 and 1
        """
        normalized_dist = min(distance / self.size, 1.0)

        if self.falloff == "constant":
            return 1.0 if normalized_dist < 1.0 else 0.0
        elif self.falloff == "linear":
            return max(0.0, 1.0 - normalized_dist)
        else:  # "smooth"
            return max(0.0, 1.0 - (normalized_dist * normalized_dist))

    def get_displacement_vector(self, normal):
        """
        Get the displacement vector based on the brush mode.

        Args:
            normal (np.ndarray): Surface normal at the point

        Returns:
            np.ndarray: Vector to displace the vertex
        """
        if self.mode == BrushMode.ADD:
            return normal
        elif self.mode == BrushMode.SUBTRACT:
            return -normal
        else:
            return None

    @abstractmethod
    def add_sample(self, position, normal=None, pressure=1.0, timestamp=0):
        """
        Add a sample to the current stroke.

        Args:
            position: 3D position of the sample
            normal: Surface normal at the position
            pressure: Pressure applied at this sample
            timestamp: Time when this sample was created

        Returns:
            Sample: The created sample
        """
        pass

    @abstractmethod
    def apply_to_mesh(self, mesh, sample):
        """
        Apply the brush effect to a mesh at a given sample.

        Args:
            mesh: The mesh to modify
            sample: The sample to apply

        Returns:
            bool: True if the mesh was modified
        """
        pass
