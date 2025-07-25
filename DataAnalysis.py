import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
from control_transform import matrix

paths = ["Control.mat", "MgF2.mat", "SiO2.mat", "ZrO2.mat"]

Materials = ["Control", "MgF2", "SiO2", "ZrO2"]
Colors = ["black","blue","orange","green"]

global Angle_Values
Data_Matrix = []
Thickness_Values = []
Wavelengths = []

##print(paths[3])
for i in [1,2,3]:
    #print('hallo')
    with h5py.File(paths[i], 'r') as f:
        # Optional: ##print all available variable names
        ##print("Available variables:")
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

##print("okeh")
##print(len(Data_Matrix))
##print(len(Data_Matrix[0]))
##print(len(Data_Matrix[0][0]))
##print(len(Data_Matrix[0][0][0]))

Thickness_Values = Thickness_Values[0]
Wavelengths = Wavelengths[0]
Angle_Values = Angle_Values[0]
##print("Thickness values")
##print(type(Thickness_Values))
##print(Thickness_Values)
##print("Wavelength values")
##print(Wavelengths)
##print("Angle values")
##print(Angle_Values)


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

##print("hi")
##print(len(Data_Matrix))
##print(len(Data_Matrix[0]))
##print(len(Data_Matrix[0][0]))
##print(len(Data_Matrix[0][0][0]))
##print("vs")
##print(len(Data_Matrix))
##print(len(Data_Matrix[1]))
##print(len(Data_Matrix[1][0]))
##print(len(Data_Matrix[1][0][0]))
##print("vs")
##print(len(Data_Matrix))
##print(len(Data_Matrix[2]))
##print(len(Data_Matrix[2][0]))
##print(len(Data_Matrix[2][0][0]))



# ------------------- Thickness vs. Power -------------------------
def MaxPowerPerThickness(Material):
    list = []
    temp = []
    for i in Data_Matrix[Material]:
        list.append(max(i[0]))
    return list

##print(len(MaxPowerPerThickness(1)))

control_power_points = [0.00039033198061802157] * 33

for i in range(0,3):
    plt.plot(Thickness_Values, MaxPowerPerThickness(i), label=(Materials[i+1]), color=Colors[i+1], linestyle='-')

plt.plot(Thickness_Values, control_power_points, label="Control", color="black", linestyle="--")
plt.plot(80e-9, 0.00059203, 'ro')
plt.plot(150e-9, 0.000526959, 'ro')
plt.plot(150e-9, 0.0005133519, 'ro')
plt.xlabel('Thickness (m)')
plt.ylabel('Power')
plt.legend()
plt.tight_layout()
plt.title('Thickness vs. Power (All 4)')
plt.savefig("Thickness vs. Power (All 4).png")

# ------------------- Angle & Thickness vs. Power -------------------------
def MaxPowerPerAngleAndThickness(Material, Angle, Thickness):
    ###print(max(Data_Matrix[Material][Thickness][Angle]))
    return max(Data_Matrix[Material][Thickness][Angle])

List = []
for i in range(len(Thickness_Values)):
    for j in range(len(Angle_Values)):
        List.append([Thickness_Values[i],Angle_Values[j],MaxPowerPerAngleAndThickness(2,j,i)])

##print("hi" + str(MaxPowerPerAngleAndThickness(0,0,0)))




data = np.array(List)  # Replace with your actual data variable
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# --- Step 2: Create 2D grid ---
x_unique = np.unique(x)
y_unique = np.unique(y)

X, Y = np.meshgrid(x_unique, y_unique)
Z = np.full_like(X, np.nan, dtype=np.float64)

for xi, yi, zi in zip(x, y, z):
    x_idx = np.where(x_unique == xi)[0][0]
    y_idx = np.where(y_unique == yi)[0][0]
    Z[y_idx, x_idx] = zi  # row = y index, col = x index

# --- Step 3: Interpolate to higher resolution grid ---
interp = RectBivariateSpline(y_unique, x_unique, Z)

# Create finer grid
x_fine = np.linspace(x_unique.min(), x_unique.max(), 600)
y_fine = np.linspace(y_unique.min(), y_unique.max(), 600)
Z_fine = interp(y_fine, x_fine).T

# --- Step 4: Plot the smooth heatmap ---
plt.figure(figsize=(8, 6))
plt.imshow(Z_fine, extent=(x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()),
           origin='lower', aspect='auto', cmap='viridis')
#plt.scatter(x, y, c=z, cmap='viridis', s=40, edgecolors='white', linewidth=0.8, zorder=4)
plt.colorbar(label='Intensity')
plt.xlabel('Thickness (m)')
plt.ylabel('Incident Angle (°)')
plt.title('ZrO2: Thickness, Incidence Angle vs. Max Power')
plt.tight_layout()
plt.show()
plt.savefig("ZrO2: Thickness, Incidence Angle vs. Max Power.png")


#---------------- Thickness and Angle vs Spread --------------------
def spectral_spread_fixed_center(wavelengths, intensities, center=905e-9, window=3e-9):
    # Ensure input arrays are NumPy arrays
    wavelengths = np.asarray(wavelengths)
    intensities = np.asarray(intensities)

    # Select range around center
    mask = (wavelengths >= center - window) & (wavelengths <= center + window)
    wl = wavelengths[mask]
    inten = intensities[mask]
    
    if len(wl) == 0:
        raise ValueError("No wavelengths within specified window.")

    # Normalize intensities
    p = inten / np.sum(inten)

    # Weighted spread around center wavelength
    spread = np.sqrt(np.sum(p * (wl - center)**2))
    return spread

def SpectralSpreadAtThicknessAndAngle(Material, Angle, Thickness):
    return max(Data_Matrix[Material][Thickness][Angle])

print(Wavelengths)
List = []
for i in range(len(Thickness_Values)):
    for j in range(len(Angle_Values)):
        List.append([Thickness_Values[i],Angle_Values[j],spectral_spread_fixed_center(Wavelengths,Data_Matrix[0][i][j])])




data = np.array(List)  # Replace with your actual data variable
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# --- Step 2: Create 2D grid ---
x_unique = np.unique(x)
y_unique = np.unique(y)

X, Y = np.meshgrid(x_unique, y_unique)
Z = np.full_like(X, np.nan, dtype=np.float64)

for xi, yi, zi in zip(x, y, z):
    x_idx = np.where(x_unique == xi)[0][0]
    y_idx = np.where(y_unique == yi)[0][0]
    Z[y_idx, x_idx] = zi  # row = y index, col = x index

# --- Step 3: Interpolate to higher resolution grid ---
interp = RectBivariateSpline(y_unique, x_unique, Z)

# Create finer grid
x_fine = np.linspace(x_unique.min(), x_unique.max(), 600)
y_fine = np.linspace(y_unique.min(), y_unique.max(), 600)
Z_fine = interp(y_fine, x_fine)

# --- Step 4: Plot the smooth heatmap ---
plt.figure(figsize=(8, 6))
plt.imshow(Z_fine, extent=(x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()),
           origin='lower', aspect='auto', cmap='viridis')
#plt.scatter(x, y, c=z, cmap='viridis', s=40, edgecolors='white', linewidth=0.8, zorder=4)
plt.colorbar(label='Spread')
plt.xlabel('Thickness (m)')
plt.ylabel('Incident Angle (°)')
plt.title('MgF2: Thickness, Incidence Angle vs. Spectral Spread')
plt.tight_layout()
plt.show()
plt.savefig("MgF2: Thickness, Incidence Angle vs. Spectral Spread.png")


#---------------- Thicknesses vs Power and Spectral Spread -------------------------------

fig, ax1 = plt.subplots(figsize=(8, 4))

for i in range(0,3):
    ax1.plot(Thickness_Values, MaxPowerPerThickness(i), label=(Materials[i+1] + " Power"), color=Colors[i], linestyle='-')



ax1.set_xlabel('Thickness (m)')
ax1.set_ylabel('Power')

ax2 = ax1.twinx()
ax2.set_ylabel("Spectral Spread")

print("HOG RIDERRR: " + str(Data_Matrix[0][0][0]))
for i in range(0,3):
    List = []
    for x in range(0, 33):
        List.append(spectral_spread_fixed_center(Wavelengths, Data_Matrix[i][x][0]))
    ax2.plot(Thickness_Values, List, label=(Materials[i+1] + " Spread"), color=Colors[i], linestyle=':')

fig.legend()

plt.title('Thickness vs. Power, Spectral Spread (All Materials)')
plt.savefig("Thickness vs. Power, Spectral Spread (All Materials).png")


# ------------------------------- Optimal Thicknesses -------------------------------
def FindBestThickness(Material):
    Best = 0
    Power = 0
    Index = 0
    a = 0
    for i in Data_Matrix[Material]:
        if max(i[0]) > Best:
            Best = max(i[0]) # HERERERHEREHHERHEHERHER
            Index = a
        a+=1
    List = []
    List.append(Index)
    List.append(Best)
    return List
OptimalThicknesses = [["MgF2", Thickness_Values[FindBestThickness(0)[0]] * 1000000000, FindBestThickness(0)[1] * 100000],
                      ["SiO2", Thickness_Values[FindBestThickness(1)[0]] * 1000000000, FindBestThickness(1)[1] * 100000],
                      ["ZrO2", Thickness_Values[FindBestThickness(2)[0]] * 1000000000, FindBestThickness(2)[1] * 100000]
                      ]

columns = ['Material', 'Thickness (nm)', 'Power at Normal Angle']

fig, ax = plt.subplots(figsize=(6, 2)) 
table = ax.table(cellText=OptimalThicknesses, colLabels=columns, loc='center', cellLoc="left", colLoc="left",bbox=[0, 0, 1, 0.9])
ax.axis('off')  
plt.title("Optimal Thickness Table",)
plt.show()
plt.tight_layout()
plt.savefig("Optimal Thickness Table (All Materials)")
# Add table

#---------------- Optimal Thickness: Angle & Wavelength vs Spread --------------------
# def spectral_spread_fixed_center(wavelengths, intensities, center=905e-9, window=3e-9):
#     # Ensure input arrays are NumPy arrays
#     wavelengths = np.asarray(wavelengths)
#     intensities = np.asarray(intensities)

#     # Select range around center
#     mask = (wavelengths >= center - window) & (wavelengths <= center + window)
#     wl = wavelengths[mask]
#     inten = intensities[mask]
    
#     if len(wl) == 0:
#         raise ValueError("No wavelengths within specified window.")

#     # Normalize intensities
#     p = inten / np.sum(inten)

#     # Weighted spread around center wavelength
#     spread = np.sqrt(np.sum(p * (wl - center)**2))
#     return spread

def SpectralSpreadAtThicknessAndAngle(Material, Angle, Thickness):
    return max(Data_Matrix[Material][Thickness][Angle])

print("DEEDY BLAD", Thickness_Values[20])
List = []
for i in range(len(Angle_Values)):
    for j in range(len(Wavelengths)):
        List.append([Angle_Values[i],Wavelengths[j],Data_Matrix[2][20][i][j]])




data = np.array(List)  # Replace with your actual data variable
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# --- Step 2: Create 2D grid ---
x_unique = np.unique(x)
y_unique = np.unique(y)

X, Y = np.meshgrid(x_unique, y_unique)
Z = np.full_like(X, np.nan, dtype=np.float64)

for xi, yi, zi in zip(x, y, z):
    x_idx = np.where(x_unique == xi)[0][0]
    y_idx = np.where(y_unique == yi)[0][0]
    Z[y_idx, x_idx] = zi  # row = y index, col = x index

# --- Step 3: Interpolate to higher resolution grid ---
interp = RectBivariateSpline(y_unique, x_unique, Z)

# Create finer grid
x_fine = np.linspace(x_unique.min(), x_unique.max(), 600)
y_fine = np.linspace(y_unique.min(), y_unique.max(), 600)
Z_fine = interp(y_fine, x_fine)

# --- Step 4: Plot the smooth heatmap ---
plt.figure(figsize=(8, 6))
plt.imshow(Z_fine, extent=(x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()),
           origin='lower', aspect='auto', cmap='viridis')
#plt.scatter(x, y, c=z, cmap='viridis', s=40, edgecolors='white', linewidth=0.8, zorder=4)
plt.colorbar(label='Spread')
plt.xlabel('Incident Angle (°)')
plt.ylabel('Wavelength (nm)')
plt.title('ZrO2: Incident Angle, Wavelengh vs. Power')
plt.tight_layout()
plt.show()
plt.savefig("ZrO2: Incident Angle, Wavelengh vs. Power.png")

# -------------------------- Blue/Red shift -------------------------------


print("SNEAKY GOLEM: " + str(Wavelengths))
print("shibai sekyaa: " + str(np.array(Data_Matrix[0][20][0])))
print("skibidi: " + str(Wavelengths.shape))

print("x shape:", np.array(Wavelengths).shape)
print("y shape:", np.array(Data_Matrix[0][20][0]).shape)

plt.clf()
plt.plot(Wavelengths, (np.array(Data_Matrix[0][20][0]) - np.array(Data_Matrix[0][20][0]).min()) / (np.array(Data_Matrix[0][20][0]).max() - np.array(Data_Matrix[0][20][0]).min()), color="blue", label="MgF2", linestyle="-")
plt.plot(Wavelengths, (np.array(Data_Matrix[1][20][0]) - np.array(Data_Matrix[1][20][0]).min()) / (np.array(Data_Matrix[1][20][0]).max() - np.array(Data_Matrix[1][20][0]).min()), color="orange", label="SiO2", linestyle="-")
plt.plot(Wavelengths, (np.array(Data_Matrix[2][20][0]) - np.array(Data_Matrix[2][20][0]).min()) / (np.array(Data_Matrix[2][20][0]).max() - np.array(Data_Matrix[2][20][0]).min()), color="green", label="ZrO2", linestyle="-")
plt.plot(Wavelengths, (matrix[0] - matrix[0]) / (matrix[0].max() - matrix[0].min()), color="black", label="Control", linestyle=":")
plt.title("Intensity of Wavelengths")
plt.legend()
plt.xlabel("Wavelengths (m)")
plt.ylabel("Power")
plt.savefig("Intensity of Wavelengths (All Materials vs. Control).png")