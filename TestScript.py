import h5py

Data_Matrix = []

with h5py.File("Control.mat", 'r') as f:
        # Optional: Print all available variable names
        print("Available variables:")
        for key in f.keys():
            print(f" - {key}")

        # Replace these with your actual variable names
        Data_Matrix.append(f['max_power_matrix'][()])

#print(Data_Matrix)
print(len(Data_Matrix[0]))
print(len(Data_Matrix[0][0]))
print(len(Data_Matrix[0][0][0]))

# array should be [angles, wavelengths, power value]
