import h5py
import matplotlib.pyplot as plt
import numpy as np


paths = ["Control.mat", "MgF2.mat", "SiO2.mat", "ZrO2.mat"]

Materials = ["Control", "MgF2", "SiO2", "ZrO2"]
Colors = ["black","blue","orange","green"]

global Angle_Values
Data_Matrix = []
Thickness_Values = []
Wavelengths = []

print(paths[3])
for i in [1,2,3]:
    print('hallo')
    with h5py.File(paths[i], 'r') as f:
        # Optional: Print all available variable names
        print("Available variables:")
        for key in f.keys():
            print(f" - {key}")

        # Replace these with your actual variable names
        Angle_Values = f['angle_values'][()]
        if (i != 0):
            Data_Matrix.append(f['max_power_matrix'][()])
        else:
            Data_Matrix.append([f['max_power_matrix'][()] for _ in range(33)])
        Thickness_Values = f['thickness_values'][()]
        Wavelengths = f['wavelengths'][()]

print("okeh")
print(len(Data_Matrix))
print(len(Data_Matrix[0]))
print(len(Data_Matrix[0][0]))
print(len(Data_Matrix[0][0][0]))

Thickness_Values = Thickness_Values[0]
Wavelengths = Wavelengths[0]
Angle_Values = Angle_Values[0]
print("Thickness values")
print(type(Thickness_Values))
print(Thickness_Values)
print("Wavelength values")
print(Wavelengths)
print("Angle values")
print(Angle_Values)


Data_Matrix =  [
        [
            [
                [
                    Data_Matrix[a][b][c][d]
                    for b in range(len(Data_Matrix[0]))
                ]
                for c in range(len(Data_Matrix[0][0]))
            ]
            for d in range(len(Data_Matrix[0][0][0]))
        ]
        for a in range(len(Data_Matrix))
    ]

#Data_Matrix[0] = [Data_Matrix[0] for _ in range(33)]

print("hi")
print(len(Data_Matrix))
print(len(Data_Matrix[0]))
print(len(Data_Matrix[0][0]))
print(len(Data_Matrix[0][0][0]))
print("vs")
print(len(Data_Matrix))
print(len(Data_Matrix[1]))
print(len(Data_Matrix[1][0]))
print(len(Data_Matrix[1][0][0]))



# ------------------- Thickness vs. Power -------------------------
def MaxPowerPerThickness(Material):
    list = []
    temp = []
    for i in Data_Matrix[Material]:
        list.append(max(i[0]))
    return list

print(len(MaxPowerPerThickness(1)))

for i in range(0,3):
    plt.plot(Thickness_Values, MaxPowerPerThickness(i), label=(Materials[i] + ': Max Power'), color=Colors[i], linestyle='-')
plt.savefig("Thickness vs. Power.png")

# ------------------- Angle & Thickness vs. Power -------------------------
def MaxPowerPerAngleAndThickness(Material, Angle, Thickness):
    return max(Data_Matrix[Material][Thickness][Angle])

List = []
for i in range(len(Thickness_Values)):
    for j in range(len(Angle_Values)):
        List.append([Thickness_Values[i],Angle_Values[j],MaxPowerPerAngleAndThickness(2,j,i)])

print("hi" + str(MaxPowerPerAngleAndThickness(0,0,0)))
data = np.array(List)

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]


plt.tricontourf(x,y,z, levels=100, cmap='viridis') # 'viridis' colormap, 'lower' origin for y-axis
plt.colorbar(label='Intensity') # Add a colorbar to show the intensity scale
plt.title('2D Color Intensity Plot')
plt.xlabel('Thickness')
plt.ylabel('Angle')
plt.show()
plt.savefig("Angle & Thickness vs. Power Heatmap.png")

