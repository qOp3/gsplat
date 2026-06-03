"""Convert COLMAP text model (cameras/images/points3D .txt) to binary (.bin).

Usage:
    python examples/colmap_txt_to_bin.py --sparse_dir data/<scene>/sparse/0
    python examples/colmap_txt_to_bin.py --data_dir  data/<scene>          # auto-finds sparse/0

The binary format is required because the installed pycolmap uses Python-2-style
map() calls in its text parser, which silently breaks under Python 3 (np.array
wraps the iterator as a 0-d object instead of converting it to a numeric array).
"""

import argparse
import os
import struct

CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}

INVALID_POINT3D = (1 << 64) - 1  # uint64 sentinel for "no 3D point"


def _data_lines(path):
    with open(path) as f:
        return [l.rstrip("\n") for l in f if l.strip() and not l.strip().startswith("#")]


def convert_cameras(sparse_dir):
    txt = os.path.join(sparse_dir, "cameras.txt")
    bin_ = os.path.join(sparse_dir, "cameras.bin")
    if not os.path.exists(txt):
        raise FileNotFoundError(txt)

    cameras = {}
    for line in _data_lines(txt):
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        w, h = int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        cameras[cam_id] = (model, w, h, params)

    with open(bin_, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, (model, w, h, params) in cameras.items():
            f.write(struct.pack("<IiQQ", cam_id, CAMERA_MODEL_IDS[model], w, h))
            f.write(struct.pack(f"<{len(params)}d", *params))

    print(f"cameras : {len(cameras)} -> {bin_}")
    return cameras


def convert_images(sparse_dir):
    txt = os.path.join(sparse_dir, "images.txt")
    bin_ = os.path.join(sparse_dir, "images.bin")
    if not os.path.exists(txt):
        raise FileNotFoundError(txt)

    lines = _data_lines(txt)
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected even number of data lines in images.txt, got {len(lines)}")

    images = []
    for i in range(0, len(lines), 2):
        h = lines[i].split()
        pts_raw = lines[i + 1].split() if lines[i + 1].strip() else []
        if len(pts_raw) % 3 != 0:
            raise ValueError(f"Image {h[9]}: points2D count not divisible by 3")
        pts = [
            (
                float(pts_raw[j]),
                float(pts_raw[j + 1]),
                INVALID_POINT3D if int(pts_raw[j + 2]) == -1 else int(pts_raw[j + 2]),
            )
            for j in range(0, len(pts_raw), 3)
        ]
        images.append(
            (int(h[0]), float(h[1]), float(h[2]), float(h[3]), float(h[4]),
             float(h[5]), float(h[6]), float(h[7]), int(h[8]), h[9], pts)
        )

    with open(bin_, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for (img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts) in images:
            f.write(struct.pack("<I4d3dI", img_id, qw, qx, qy, qz, tx, ty, tz, cam_id))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", len(pts)))
            for (x, y, p3d_id) in pts:
                f.write(struct.pack("<ddQ", x, y, p3d_id))

    print(f"images  : {len(images)} -> {bin_}")
    return images


def convert_points3D(sparse_dir):
    txt = os.path.join(sparse_dir, "points3D.txt")
    bin_ = os.path.join(sparse_dir, "points3D.bin")
    if not os.path.exists(txt):
        raise FileNotFoundError(txt)

    pts3d = []
    for line in _data_lines(txt):
        p = line.split()
        track = [(int(p[j]), int(p[j + 1])) for j in range(8, len(p), 2)]
        pts3d.append(
            (int(p[0]), float(p[1]), float(p[2]), float(p[3]),
             int(p[4]), int(p[5]), int(p[6]), float(p[7]), track)
        )

    with open(bin_, "wb") as f:
        f.write(struct.pack("<Q", len(pts3d)))
        for (pid, x, y, z, r, g, b, err, track) in pts3d:
            f.write(struct.pack("<Q3d3Bd", pid, x, y, z, r, g, b, err))
            f.write(struct.pack("<Q", len(track)))
            for (iid, pidx) in track:
                f.write(struct.pack("<II", iid, pidx))

    print(f"points3D: {len(pts3d)} -> {bin_}")
    return pts3d


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sparse_dir", help="Path to sparse/0 directory containing .txt files")
    group.add_argument("--data_dir", help="Dataset root; script auto-locates sparse/0 or sparse/")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify with pycolmap after writing (default: on)")
    args = parser.parse_args()

    if args.data_dir:
        for candidate in ["sparse/0", "sparse"]:
            d = os.path.join(args.data_dir, candidate)
            if os.path.isdir(d):
                sparse_dir = d
                break
        else:
            raise FileNotFoundError(f"No sparse/0 or sparse/ directory under {args.data_dir}")
    else:
        sparse_dir = args.sparse_dir

    print(f"Converting: {sparse_dir}")
    convert_cameras(sparse_dir)
    convert_images(sparse_dir)
    convert_points3D(sparse_dir)

    if args.verify:
        try:
            from pycolmap import SceneManager
            m = SceneManager(sparse_dir)
            m.load_cameras()
            m.load_images()
            m.load_points3D()
            print(f"Verified : {len(m.cameras)} cams, {len(m.images)} imgs, {m.points3D.shape[0]} pts")
        except ImportError:
            print("pycolmap not available; skipping verification")


if __name__ == "__main__":
    main()
