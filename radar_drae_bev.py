# read radar drae 4d tensor matlab mat file, 
# save range-angle bev, only using doppler ch 0, then mean over elevation
# save zyx cube, channel mean over dopplar 1:63,
# resultant zyx is pw measurement, unlogged.

import argparse
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def compute_zyx(arr_drea: np.ndarray) -> np.ndarray:
    arr = arr_rea.astype(np.float32)


    arr[arr < 0] = np.nan


def compute_bev(arr_rea: np.ndarray) -> np.ndarray:
    """
    Compute BEV (Bird's Eye View) image from arr_rea (r, e, a).

    MATLAB intent:
      - Iterate over y,x
      - Sum valid Z values
      - Ignore values == -1 or < 0
      - Take mean
    """
    # arr_zyx shape: (Z, Y, X)
    r, e, a = arr_rea.shape

    # Mark invalid values (negative or -1) as NaN so they are ignored in mean
    arr = arr_rea.astype(np.float32)
    arr[arr < 0] = np.nan

    # Mean over elevation axis ignoring NaNs
    bev = np.nanmean(arr, axis=1)  # shape: (Y, X)
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


def main():
    parser = argparse.ArgumentParser(description="Generate BEV image from MATLAB arr_drae tensor.")
    parser.add_argument("--mat_file", default="/home/student/Documents/datasets/k-radar/1/radar_tesseract/tesseract_00417.mat", help="Path to .mat file containing variable arrDREA")
    parser.add_argument("--out", default=None, help="Output PNG path (default: same name + _bev.png)")
    parser.add_argument("--no-log", action="store_true", help="Do not apply 10*log10 scaling")
    parser.add_argument("--show", action="store_true", help="Display image interactively")
    args = parser.parse_args()

    mat_file = args.mat_file
    data = scipy.io.loadmat(mat_file)
    print(f"data.keys {data.keys()}")

    if "arrDREA" not in data:
        raise KeyError(f"MAT file does not contain 'arr_zyx'. Keys found: {list(data.keys())}")

    arr_drea = data["arrDREA"]
    if arr_drea.ndim != 4:
        raise ValueError(f"arr_zyx must be 3D. Got shape: {arr_zyx.shape}")

    zyx = compute_zyx(arr_drea)
    # arr_zyx.shape (150, 400, 250)
    
    bev = compute_bev(arr_drea[0])

    if args.out is None:
        base, _ = os.path.splitext(mat_file)
        out_path = base.split('/')[-1] + "_bev.png"
    else:
        out_path = args.out

    save_bev_image(bev, out_path, use_log=(not args.no_log))

    print(f"Saved BEV image to: {out_path}, arr_drea.shape {arr_drea.shape} arr_drea[0] {arr_drea[0:10,200,80:180]}")

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

