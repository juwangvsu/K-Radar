# read zyx_cub /.mat, save as png. the zyx don't have doppler channel
# zyx shape (150, 400, 250)
# zyx max: 5e15, min: -1 for out of fov
# https://github.com/juwangvsu/K-Radar/blob/main/tools/mfiles/gen_3_get_zyx_cube.m
# mean dopller, bilinear interp

import argparse
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def compute_bev(arr_zyx: np.ndarray) -> np.ndarray:
    """
    Compute BEV (Bird's Eye View) image from arr_zyx (Z, Y, X).

    MATLAB intent:
      - Iterate over y,x
      - Sum valid Z values
      - Ignore values == -1 or < 0
      - Take mean
    """
    # arr_zyx shape: (Z, Y, X)
    z, y, x = arr_zyx.shape
    print(f"arr_zyx.shape {arr_zyx.shape}")

    # Mark invalid values (negative or -1) as NaN so they are ignored in mean
    arr = arr_zyx.astype(np.float32)
    arr[arr < 0] = np.nan

    # Mean over Z axis ignoring NaNs
    bev = np.nanmean(arr, axis=0)  # shape: (Y, X)

    # Replace NaN (where all Z were invalid) with 0
    bev = np.nan_to_num(bev, nan=0.0)

    bev= bev/1e11

    return bev


def save_bev_image(bev: np.ndarray, out_path: str, use_log: bool = True):
    """
    Save BEV as image.
    MATLAB used 10*log10(new_arr_xy) when plotting.
    """
    img = bev.copy()
    print(f"img.shape {img.shape}, use_log {use_log}")

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
    plt.savefig(out_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate BEV image from MATLAB arr_zyx cube.")
    parser.add_argument("--mat_file", default="/home/student/Documents/datasets/k-radar/radar_zyx_cube/cube_00012.mat", help="Path to .mat file containing variable arr_zyx")
    parser.add_argument("--out", default=None, help="Output PNG path (default: same name + _bev.png)")
    parser.add_argument("--no-log", action="store_true", help="Do not apply 10*log10 scaling")
    parser.add_argument("--show", action="store_true", help="Display image interactively")
    args = parser.parse_args()

    mat_file = args.mat_file
    data = scipy.io.loadmat(mat_file)

    if "arr_zyx" not in data:
        raise KeyError(f"MAT file does not contain 'arr_zyx'. Keys found: {list(data.keys())}")

    arr_zyx = data["arr_zyx"]
    if arr_zyx.ndim != 3:
        raise ValueError(f"arr_zyx must be 3D. Got shape: {arr_zyx.shape}")

    # arr_zyx.shape (150, 400, 250)
    
    bev = compute_bev(arr_zyx)

    if args.out is None:
        base, _ = os.path.splitext(mat_file)
        base = base.split('/')[-1]
        out_path = base + "_bev.png"
    else:
        out_path = args.out

    save_bev_image(bev, out_path, use_log=True)

    print(f"Saved BEV image to: {out_path}, arr_zyx.shape {arr_zyx.shape} max arr_zyx {np.max(arr_zyx, axis=1)}")

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

