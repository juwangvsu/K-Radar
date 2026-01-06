# read radar rdr_polar_3d npy_file, save range-angle bev 

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
    print(f"img.shape {img.shape}")

    if use_log:
        # Avoid log(0)
        img = 10.0 * np.log10(np.maximum(img, 1e-12))

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
    parser.add_argument("--no-log", action="store_true", help="Do not apply 10*log10 scaling")
    parser.add_argument("--show", action="store_true", help="Display image interactively")
    args = parser.parse_args()

    polar_file = args.polar_file
    if os.path.isdir(polar_file):
        do_dir(polar_file)

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

    save_bev_image(bev, out_path, use_log=False)

    print(f"Saved BEV image to: {out_path}, arr_rae.shape {arr_rae.shape} arr_drea {arr_rae[0:10,0]}")

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

