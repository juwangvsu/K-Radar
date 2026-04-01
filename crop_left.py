'''
look through a folder you specify

for every *.png

crop the left half (width // 2)

save it with the same filename into a sibling/output folder named cam-front-left (created if missing)
'''
#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image


def crop_left_half(in_path: Path, out_path: Path) -> None:
    with Image.open(in_path) as im:
        w, h = im.size
        box = (0, 0, w // 2, h)  # left half
        cropped = im.crop(box)

        # Preserve format and avoid surprises with alpha palettes, etc.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out_path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop the left half of all .png images in a folder and save to cam-front-left/"
    )
    parser.add_argument(
        "root_folder",
        type=Path,
        help="Path to the folder containing .png images",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders (output keeps relative paths under cam-front-left/)",
    )

    args = parser.parse_args()
    root_dir: Path = args.root_folder

    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {in_dir}")

    out_dir = root_dir / "cam-front-left"
    in_dir = root_dir / "cam-front"

    pattern = "**/*.png" if args.recursive else "*.png"
    png_files = sorted(in_dir.glob(pattern))

    # If recursive, avoid processing outputs we created in a prior run.
    png_files = [p for p in png_files if out_dir not in p.parents]

    if not png_files:
        print(f"No .png files found in: {in_dir}")
        return 0

    processed = 0
    for src in png_files:
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        try:
            crop_left_half(src, dst)
            processed += 1
        except Exception as e:
            print(f"[WARN] Failed: {src} ({e})")

    print(f"Processed {processed}/{len(png_files)} images.")
    print(f"Output folder: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

