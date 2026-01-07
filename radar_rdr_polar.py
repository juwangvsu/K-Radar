# read radar rdr_polar_3d npy_file, save range-angle bev 
# npy shape (2,256,107,37) channel 1 is pw. ch 2 is dopper
# pw is mean over doppler channel 1:63. pw is already normalized

# use_log True plot seems better
# radar_zyx_cube/cube_00621.mat  raw pw measurement, unlogged, -1 for out of fov, otherwise range from single digit to 1e12, dynamic range about 50db
# radar_tesseract/tesseract_00621.mat raw pw measurement, unloged
# radar_bev_image/radar_bev_100_00621.png

# processed:
# RadarTensor/rdr_polar_3d/polar3d_00621.npy stock max 74.06, min 0.00033, most likely normalized but not logged since the min value is not negative. if logged, will probably see some negative db value. 
# RadarTensor/rdr_polar_3d/new_all/1/polar3d_00621.npy local gen,  

import argparse
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def compute_bev(arr_rae: np.ndarray) -> np.ndarray:
    """
    Compute BEV (Bird's Eye View) image from arr_rae (r,a, e).

      - Take mean
    """
    r, a, e = arr_rae.shape

    # Mark invalid values (negative or -1) as NaN so they are ignored in mean
    arr = arr_rae.astype(np.float32)
    arr[arr < 0] = np.nan

    # Mean over elevation axis ignoring NaNs
    bev = np.nanmean(arr, axis=2)  # shape: (Y, X)
    print(f"bev.shape {bev.shape}")

    # Replace NaN (where all Z were invalid) with 0
    bev = np.nan_to_num(bev, nan=0.0)

    return bev


def save_bev_image(bev: np.ndarray, out_path: str, use_log: bool = True):
    """
    Save BEV as image.
    MATLAB used 10*log10(new_arr_xy) when plotting.
    """
    img = bev.copy()
    print(f"img.shape {img.shape} {use_log}")

    if use_log:
        # Avoid log(0)
        img = 10.0 * np.log10(np.maximum(img, 1e-12))
    print(f"img max/min {np.max(img)} / {np.min(img)}")

    #plt.figure(figsize=(10, 6))
    #plt.figure()
    plt.imshow(img, origin="lower", aspect="equal")
    plt.colorbar(label="Power (dB)" if use_log else "Mean Power")
    plt.title("BEV Image (Mean along Z-axis)")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=50)
    plt.close()

def do_dir(dirname):
    import os
    data = {}
    for fname in os.listdir(dirname):
        if fname.endswith(".npy"):
            path = os.path.join(dirname, fname)
            key = os.path.splitext(fname)[0]  # filename without .npy
            data = np.load(path)

            print(f"max/min {np.max(data)} / {np.min(data)}")

            arr_rae = data[0]
            if arr_rae.ndim != 3:
                raise ValueError(f"must be 3D. Got shape: {arr_rae.shape}")
            bev = compute_bev(arr_rae)

            base, _ = os.path.splitext(path)
            out_path = base.split('/')[-1] + "_bev.png"
            save_bev_image(bev, out_path, use_log=False)


def main():
    parser = argparse.ArgumentParser(description="Generate BEV image from MATLAB arr_drae tensor.")
    parser.add_argument("--polar_file", default="/home/student/Documents/datasets/k-radar/RadarTensor/rdr_polar_3d/1/polar3d_00319.npy", help="Path to .mat file containing variable arrDREA")
    parser.add_argument("--out", default=None, help="Output PNG path (default: same name + _bev.png)")
    parser.add_argument("--use_log", action="store_true", help="Do not apply 10*log10 scaling")
    parser.add_argument("--show", action="store_true", help="Display image interactively")
    args = parser.parse_args()

    polar_file = args.polar_file
    if os.path.isdir(polar_file):
        do_dir(polar_file)
        return

    data = np.load(polar_file)

    arr_rae = data[0]
    if arr_rae.ndim != 3:
        raise ValueError(f"must be 3D. Got shape: {arr_rae.shape}")
    bev = compute_bev(arr_rae)

    if args.out is None:
        base, _ = os.path.splitext(polar_file)
        out_path = base.split('/')[-1] + "_bev.png"
    else:
        out_path = args.out

    save_bev_image(bev, out_path, use_log=args.use_log)

    print(f"Saved BEV image to: {out_path}, arr_rae.shape {arr_rae.shape} arr_drea {arr_rae[0:10,0,0]}")

    if args.show:
        # Re-display image
        img = 10.0 * np.log10(np.maximum(bev, 1e-12)) if not args.no_log else bev
        plt.figure(figsize=(10, 6))
        plt.imshow(img, origin="lower", aspect="auto")
        plt.colorbar()
        plt.title("BEV Image (Mean along Z-axis)")
        plt.show()


if __name__ == "__main__":
    main()

