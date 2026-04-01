import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def find_roi_index(v_min, v_max, arr):
    """
    MATLAB:
      [c_min, i_min] = min((double(arr)-double(v_min)).^2);
      [c_max, i_max] = min((double(arr)-double(v_max)).^2);
      new_arr = arr(i_min:i_max);

    Python notes:
    - indices are 0-based
    - MATLAB slicing i_min:i_max is inclusive; Python is exclusive on the end
    """
    arr = np.asarray(arr, dtype=np.float64)
    i_min = int(np.argmin((arr - float(v_min)) ** 2))
    i_max = int(np.argmin((arr - float(v_max)) ** 2))

    if i_min > i_max:
        i_min, i_max = i_max, i_min

    new_arr = arr[i_min : i_max + 1]  # inclusive end
    return i_min, i_max, new_arr


# --- Load .mat ---
mat_path = "/home/student/Documents/datasets/k-radar/1/radar_zyx_cube/cube_00417.mat"
arr_zyx_struct = loadmat(mat_path)

# In MATLAB you had: arr_zyx_struct.arr_zyx
# In scipy, it's typically a dict key:
arr_zyx = arr_zyx_struct["arr_zyx"]  # shape: (len_z, len_y, len_x)

len_z, len_y, len_x = arr_zyx.shape

cnt_minus_1 = 0
cnt_minus = 0

# --- Axis bins ---
x_min = 0.0
x_per_bin = 0.4
x_max = 100.0 - x_per_bin

y_min = -80.0
y_per_bin = 0.4
y_max = 80.0 - y_per_bin

z_min = -30.0
z_per_bin = 0.4
z_max = 30.0 - z_per_bin

# MATLAB colon operator includes end if it lands exactly; mimic with a small epsilon
eps = 1e-9
arr_x = np.arange(x_min, x_max + eps, x_per_bin)
arr_y = np.arange(y_min, y_max + eps, y_per_bin)
arr_z = np.arange(z_min, z_max + eps, z_per_bin)

# --- Version 1: literal translation (nested loops) ---
new_arr_xy = np.zeros((len_y, len_x), dtype=np.float64)

for i_y in range(len_y):
    for i_x in range(len_x):
        temp_arr_z = arr_zyx[:, i_y, i_x]  # already 1D in NumPy

        pw_sum = 0.0
        cnt_none_minus_1 = 0

        for i_z in range(len_z):
            v = temp_arr_z[i_z]

            if v < 0:
                cnt_minus += 1
                continue
            if v == -1:
                cnt_minus_1 += 1
                continue

            pw_sum += v
            cnt_none_minus_1 += 1

        # avoid divide-by-zero if all entries were skipped
        new_arr_xy[i_y, i_x] = pw_sum / cnt_none_minus_1 if cnt_none_minus_1 > 0 else np.nan

# --- Plot (like surf(...); view(2)) ---
plt.figure(figsize=(8, 12))
X, Y = np.meshgrid(arr_x, arr_y)

Z_db = 10.0 * np.log10(new_arr_xy)
#Z_db = np.nan_to_num(Z_db, nan=0.0)
import matplotlib as mpl
cmap = mpl.cm.viridis.copy()
cmap.set_bad(color="magenta")

print(f"Z_db.shape {Z_db.shape} {Z_db[200]} max/min {np.max(Z_db)}/{np.min(Z_db)}")
#plt.pcolormesh(X, Y, Z_db, shading="auto", cmap="jet")
#plt.pcolormesh(Z_db, shading="auto", cmap="jet")
plt.imshow(Z_db, aspect="equal", vmin=100, vmax=140, cmap="jet")
#plt.colorbar()
#plt.colorbar(label="Power (dB)")
#plt.xlim(0, 100)
#plt.ylim(-80, 80)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.colorbar(label="10*log10(power)")
plt.title("Full Range (top-down)")
plt.show()

print("cnt_minus:", cnt_minus)
print("cnt_minus_1:", cnt_minus_1)

