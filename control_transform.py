import h5py
import numpy as np

with h5py.File("Control.mat", "r") as f:
    print("Available variables:")
    for key in f.keys():
        print(f" - {key}")

    # Load the matrix only once
    raw_matrix = f['max_power_matrix'][()]  # shape: (n_wavelengths, n_angles)
    angle_values = f['angle_values'][()].squeeze()
    wavelengths = f['wavelengths'][()].squeeze()

# Transpose the matrix to (angle, wavelength) if needed
# Check current shape:
print("Original shape:", raw_matrix.shape)

# If shape is (n_lambda, n_angle), you want to transpose
if raw_matrix.shape[0] == len(wavelengths):
    matrix = raw_matrix.T
else:
    matrix = raw_matrix  # already correct

# Now shape is (n_angles, n_wavelengths)
print("Fixed shape:", matrix.shape)

# Example access
print("First angle, all wavelengths:", matrix[0, :])
print("First angle's first wavelength power:", matrix[0, 0])

print(matrix.shape)
print(max(matrix[0]))
