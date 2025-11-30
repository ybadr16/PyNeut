# src/mesh.py
import numpy as np

class MeshTally:
    def __init__(self, x_bounds, y_bounds, z_bounds, dims):
        """
        Initialize a 3D Regular Mesh for tallying dose/flux.

        Args:
            x_bounds: (min, max) tuple for X axis.
            y_bounds: (min, max) tuple for Y axis.
            z_bounds: (min, max) tuple for Z axis.
            dims: (nx, ny, nz) tuple of integers (number of voxels).
        """
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.z_min, self.z_max = z_bounds
        self.nx, self.ny, self.nz = dims

        # Calculate voxel size
        self.dx = (self.x_max - self.x_min) / self.nx
        self.dy = (self.y_max - self.y_min) / self.ny
        self.dz = (self.z_max - self.z_min) / self.nz

        # The 3D data array (Zero dose initially)
        # We use (z, y, x) ordering for easier standard Python looping,
        # but we handle the export order carefully.
        self.data = np.zeros((self.nx, self.ny, self.nz))

    def score(self, x, y, z, weight):
        """
        Add a weight contribution to the voxel containing (x, y, z).
        """
        # Calculate integer indices (0-based)
        i = int((x - self.x_min) / self.dx)
        j = int((y - self.y_min) / self.dy)
        k = int((z - self.z_min) / self.dz)

        # Check if point is inside the mesh bounds
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            self.data[i, j, k] += weight

    def write_vtk(self, filename):
        """
        Export the mesh data to a legacy VTK file (Structured Points).
        This file can be opened in ParaView.
        """
        print(f"Exporting Mesh Tally to {filename}...")

        with open(filename, 'w') as f:
            # VTK Header
            f.write("# vtk DataFile Version 2.0\n")
            f.write("Monte Carlo Mesh Tally - Energy Deposition\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")

            # Dimensions are points (N+1)
            f.write(f"DIMENSIONS {self.nx+1} {self.ny+1} {self.nz+1}\n")
            f.write(f"ORIGIN {self.x_min} {self.y_min} {self.z_min}\n")
            f.write(f"SPACING {self.dx} {self.dy} {self.dz}\n")

            # Cell Data (N voxels)
            f.write(f"CELL_DATA {self.nx * self.ny * self.nz}\n")
            f.write("SCALARS energy_deposition float 1\n")
            f.write("LOOKUP_TABLE default\n")

            # Write data flattened
            # VTK expects X index to vary fastest, then Y, then Z.
            # Our array is [x, y, z].
            # numpy.flatten(order='F') ensures the first index (x) varies fastest.
            flat_data = self.data.flatten(order='F')

            for val in flat_data:
                f.write(f"{val:.4e}\n")

        print("Export complete.")
