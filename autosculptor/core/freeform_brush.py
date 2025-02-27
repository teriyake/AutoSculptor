"""
Freeform brush
"""

import numpy as np
from .brush import Brush, BrushMode
from .data_structures import Sample


class FreeformBrush(Brush):
    """
    Freeform brush that operates in 3D space.
    This brush allows sculpting in areas away from the mesh surface.
    """

    def __init__(self, size=1.0, strength=1.0, mode=BrushMode.ADD, falloff="smooth"):
        """
        Initialize a new FreeformBrush.

        Args:
            size (float): Size of the brush
            strength (float): Strength of the brush effect
            mode (BrushMode): Operation mode (ADD, SUBTRACT, SMOOTH)
            falloff (str): Falloff type ('smooth', 'linear', 'constant')
        """
        super().__init__(size, strength, mode, falloff)

    def add_sample(self, position, normal=None, pressure=1.0, timestamp=0):
        """
        Add a sample to the current stroke.

        Args:
            position (tuple or list): 3D position of the sample
            normal (tuple or list, optional): Direction of the brush (can be derived from camera)
            pressure (float, optional): Pressure applied at this sample
            timestamp (float, optional): Time when this sample was created

        Returns:
            Sample: The created and added sample
        """
        if self.current_stroke is None:
            self.begin_stroke()

        sample = Sample(position, normal, self.size, pressure, timestamp)
        self.current_stroke.add_sample(sample)
        self.current_stroke.stroke_type = "freeform"

        return sample

    def apply_to_mesh(self, mesh, sample):
        """
        Apply the freeform brush effect to a mesh at a given sample.

        Args:
            mesh: The mesh to modify (MeshData or SimpleMesh)
            sample (Sample): The sample to apply

        Returns:
            bool: True if the mesh was modified
        """
        # For freeform brush, we affect vertices within a sphere
        try:
            # Check if we're using Maya's mesh interface
            if hasattr(mesh, "maya_mesh_fn"):
                # Maya implementation
                print(
                    f"Applying freeform brush at {sample.position} with strength {self.strength}"
                )
                # This would be implemented using Maya's API to modify vertices
                return True

            else:
                # SimpleMesh implementation for testing
                vertices_in_radius = mesh.find_vertices_in_radius(
                    sample.position, sample.size
                )

                if not vertices_in_radius:
                    return False

                if self.mode == BrushMode.SMOOTH:
                    # For smooth mode
                    return mesh.smooth_vertices(
                        [idx for idx, _ in vertices_in_radius],
                        strength=self.strength * sample.pressure,
                    )
                else:
                    # For add/subtract mode
                    indices = []
                    displacements = []

                    # Use sample normal if provided, otherwise use direction from vertex to sample
                    for idx, dist in vertices_in_radius:
                        vertex_pos = mesh.vertices[idx]

                        # Calculate direction
                        if (
                            sample.normal is not None
                            and np.linalg.norm(sample.normal) > 0
                        ):
                            direction = np.array(sample.normal)
                            direction = direction / np.linalg.norm(direction)
                        else:
                            # Default: direction away from sample position (like inflate)
                            direction = vertex_pos - np.array(sample.position)
                            norm = np.linalg.norm(direction)
                            if norm > 0.0001:
                                direction = direction / norm
                            else:
                                # Skip vertices that are too close to the center
                                continue

                        # Apply direction based on the mode (add/subtract)
                        if self.mode == BrushMode.SUBTRACT:
                            direction = -direction

                        # Calculate falloff and displacement
                        falloff = self.calculate_falloff(dist)
                        displacement = (
                            direction * falloff * self.strength * sample.pressure
                        )

                        indices.append(idx)
                        displacements.append(displacement)

                    return mesh.displace_vertices(indices, displacements)

        except Exception as e:
            print(f"Error applying freeform brush: {str(e)}")
            return False
