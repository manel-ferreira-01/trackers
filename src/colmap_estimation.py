"""
Pairwise relative pose estimation using pycolmap (5-point / essential matrix).

Integrates with the existing pipeline:
    obs_mat      [2F, N]  tracked keypoints (u,v interleaved per frame)
    Ks           [F,3,3]  per-frame intrinsics
"""

import torch
import numpy as np
import pycolmap
from pathlib import Path
import tempfile
from typing import Literal


def _get_rigid3d(obj):
    """Return a Rigid3d from either a property or a callable (API changed across versions)."""
    return obj() if callable(obj) else obj


def _make_camera(K: np.ndarray, width: int = 1, height: int = 1) -> pycolmap.Camera:
    """Build a pycolmap PINHOLE Camera from a 3x3 K matrix."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return pycolmap.Camera(
        model="PINHOLE",
        width=width,
        height=height,
        params=[fx, fy, cx, cy],
    )


def _tensor_to_jpeg(tensor: torch.Tensor, path: Path) -> None:
    """
    Write a CHW or HWC float[0,1] / uint8[0,255] tensor as a JPEG.
    Accepts: [C,H,W], [H,W,C], [H,W].  Requires Pillow.
    """
    from PIL import Image

    t = tensor.detach().cpu()
    if t.dtype != torch.uint8:
        t = (t.clamp(0.0, 1.0) * 255).to(torch.uint8)
    if t.ndim == 3 and t.shape[0] in (1, 3):   # CHW → HWC
        t = t.permute(1, 2, 0)
    if t.ndim == 3 and t.shape[2] == 1:         # [H,W,1] → [H,W]
        t = t.squeeze(2)
    Image.fromarray(t.numpy()).save(str(path), format="JPEG", quality=95)


# ── Pairwise pose (unchanged) ─────────────────────────────────────────────────

def estimate_pairwise_pose_colmap(
    obs_mat: torch.Tensor,
    Ks: torch.Tensor,
    frame_i: int,
    frame_j: int,
    ransac_max_error: float = 4.0,
) -> dict:
    """
    Estimate relative pose between two frames using pycolmap's essential matrix
    estimator (LORANSAC + 5-point algorithm).

    Args:
        obs_mat:          [2F, N] tracked keypoints (u, v interleaved per frame)
        Ks:               [F, 3, 3] or [1, 3, 3] camera intrinsics
        frame_i:          index of the first frame  (will be treated as reference)
        frame_j:          index of the second frame
        ransac_max_error: RANSAC inlier threshold in pixels

    Returns:
        dict with keys: R, t, num_inliers, inlier_ratio, tvg
            R  - [3,3] float64 tensor  (cam_j rotation  relative to cam_i)
            t  - [3]   float64 tensor  (cam_j translation relative to cam_i, unit-norm)
    """
    pts_i = obs_mat[frame_i * 2: frame_i * 2 + 2, :].T.cpu().numpy().astype(np.float64)
    pts_j = obs_mat[frame_j * 2: frame_j * 2 + 2, :].T.cpu().numpy().astype(np.float64)

    N = pts_i.shape[0]
    matches = np.stack([np.arange(N), np.arange(N)], axis=1).astype(np.uint32)

    K_i = Ks[min(frame_i, Ks.shape[0] - 1)].cpu().numpy().astype(np.float64)
    K_j = Ks[min(frame_j, Ks.shape[0] - 1)].cpu().numpy().astype(np.float64)

    cam_i = _make_camera(K_i)
    cam_j = _make_camera(K_j)

    options = pycolmap.TwoViewGeometryOptions(
        compute_relative_pose=True,
        ransac=pycolmap.RANSACOptions(max_error=ransac_max_error),
    )

    tvg = pycolmap.estimate_calibrated_two_view_geometry(
        cam_i, pts_i,
        cam_j, pts_j,
        matches=matches,
        options=options,
    )

    num_inliers = len(tvg.inlier_matches)
    inlier_ratio = num_inliers / N if N > 0 else 0.0

    pose = _get_rigid3d(tvg.cam2_from_cam1)
    R = torch.tensor(pose.rotation.matrix(), dtype=torch.float64)
    t = torch.tensor(pose.translation, dtype=torch.float64)

    return {
        "R": R,
        "t": t,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "tvg": tvg,
    }


# ── Full COLMAP SIFT pipeline ─────────────────────────────────────────────────

def run_colmap_sift_pipeline(
    frames: torch.Tensor,
    Ks: torch.Tensor | None,
    frame_indices: list[int] | None = None,
    *,
    matching_strategy: Literal["exhaustive", "sequential"] = "exhaustive",
    sequential_overlap: int = 10,
    sift_num_features: int = 8192,
    ransac_max_error: float = 4.0,
    camera_mode: Literal["single", "per_frame"] = "single",
    refine_focal_length: bool = False,
    refine_principal_point: bool = False,
) -> dict:
    """
    Full COLMAP pipeline from raw image tensors to registered camera poses,
    using COLMAP's own SIFT for feature extraction and matching.

    Tested against pycolmap 4.0.4.

    Args:
        frames:               [F, C, H, W] or [F, H, W, C], float[0,1] or uint8[0,255].
        Ks:                   [F,3,3] or [1,3,3] calibrated intrinsics, or None to let
                              COLMAP estimate (uses SIMPLE_RADIAL model).
        frame_indices:        Integer labels used for image filenames. Defaults to 0…F-1.
        matching_strategy:    "exhaustive"  – all pairs; fine for ≤ ~100 frames.
                              "sequential"  – sliding window; use for video.
        sequential_overlap:   Window half-size for sequential matching.
        sift_num_features:    Max SIFT keypoints per image.
        ransac_max_error:     Pixel threshold for geometric verification.
        camera_mode:          "single"    – one shared camera (typical for monocular video).
                              "per_frame" – one camera per image.
        refine_focal_length:  Let BA adjust focal length (ignored when Ks is None).
        refine_principal_point: Let BA adjust principal point.

    Returns:
        dict:
            cam_lists        – list of F [3,4] float64 tensors  (NaN = not registered)
                               Each is [R | t] s.t.  x_cam = R @ X_world + t
            num_registered   – number of frames successfully registered
            reconstruction   – pycolmap.Reconstruction  (use with reconstruction_reprojection_error)
    """
    F = frames.shape[0]
    if frame_indices is None:
        frame_indices = list(range(F))
    assert len(frame_indices) == F, "len(frame_indices) must equal frames.shape[0]"

    # Resolve H, W regardless of CHW vs HWC layout
    if frames.ndim != 4:
        raise ValueError(f"frames must be 4-D [F,C,H,W] or [F,H,W,C], got {tuple(frames.shape)}")
    if frames.shape[1] in (1, 3):       # CHW
        img_h, img_w = int(frames.shape[2]), int(frames.shape[3])
    else:                                # HWC
        img_h, img_w = int(frames.shape[1]), int(frames.shape[2])

    colmap_camera_mode = (
        pycolmap.CameraMode.SINGLE if camera_mode == "single"
        else pycolmap.CameraMode.PER_IMAGE
    )

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir    = Path(_tmpdir)
        image_dir = tmpdir / "images"
        image_dir.mkdir()
        db_path   = tmpdir / "colmap.db"
        sparse_dir = tmpdir / "sparse"
        sparse_dir.mkdir()

        # ── 1. Write images to disk ────────────────────────────────────────────
        names = []
        for i, fi in enumerate(frame_indices):
            name = f"frame_{fi:04d}.jpg"
            names.append(name)
            _tensor_to_jpeg(frames[i], image_dir / name)

        # ── 2. Build ImageReaderOptions ────────────────────────────────────────
        reader_options = pycolmap.ImageReaderOptions()
        if Ks is not None:
            K0 = Ks[0].cpu().numpy().astype(np.float64)
            reader_options.camera_model  = "PINHOLE"
            reader_options.camera_params = f"{K0[0,0]},{K0[1,1]},{K0[0,2]},{K0[1,2]}"
        else:
            reader_options.camera_model  = "SIMPLE_RADIAL"
            reader_options.camera_params = ""

        # ── 3. Feature extraction ──────────────────────────────────────────────
        #   FeatureExtractionOptions wraps SiftExtractionOptions at .sift
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.sift.max_num_features = sift_num_features
        extraction_options.use_gpu = torch.cuda.is_available()

        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(image_dir),
            camera_mode=colmap_camera_mode,
            reader_options=reader_options,
            extraction_options=extraction_options,
        )

        # Patch per-frame Ks into DB after extraction (per_frame mode only)
        if Ks is not None and camera_mode == "per_frame":
            _overwrite_camera_params_db(db_path, names, Ks, frame_indices, img_w, img_h)

        # ── 4. Feature matching ────────────────────────────────────────────────
        #   FeatureMatchingOptions wraps SiftMatchingOptions at .sift
        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.use_gpu = torch.cuda.is_available()

        verification_options = pycolmap.TwoViewGeometryOptions()
        verification_options.ransac = pycolmap.RANSACOptions(max_error=ransac_max_error)
        verification_options.compute_relative_pose = True

        if matching_strategy == "exhaustive":
            pycolmap.match_exhaustive(
                database_path=str(db_path),
                matching_options=matching_options,
                verification_options=verification_options,
            )

        elif matching_strategy == "sequential":
            pairing_options = pycolmap.SequentialPairingOptions()
            pairing_options.overlap = sequential_overlap
            pycolmap.match_sequential(
                database_path=str(db_path),
                matching_options=matching_options,
                pairing_options=pairing_options,
                verification_options=verification_options,
            )

        else:
            raise ValueError(f"Unknown matching_strategy: {matching_strategy!r}")

        # ── 5. Incremental mapping ─────────────────────────────────────────────
        #   IncrementalPipelineOptions is the top-level options object in 4.0.4;
        #   it contains a .mapper (IncrementalMapperOptions) and .ba fields.
        pipeline_options = pycolmap.IncrementalPipelineOptions()

        # Intrinsic refinement control lives on the mapper sub-object
        mapper_opts = pipeline_options.mapper
        mapper_opts.abs_pose_refine_focal_length = refine_focal_length
        mapper_opts.abs_pose_refine_extra_params  = refine_principal_point

        maps = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(image_dir),
            output_path=str(sparse_dir),
            options=pipeline_options,
        )

        if not maps:
            return {
                "cam_lists":      [torch.full((3, 4), float("nan"), dtype=torch.float64)] * F,
                "num_registered": 0,
                "reconstruction": None,
            }

        # Largest connected component
        rec = maps[max(maps, key=lambda k: maps[k].num_reg_images())]

        # ── 6. Extract [R|t] pose tensors ──────────────────────────────────────
        name_to_pose: dict[str, torch.Tensor] = {}
        for img in rec.images.values():
            pose = _get_rigid3d(img.cam_from_world)
            R = torch.tensor(pose.rotation.matrix(), dtype=torch.float64)
            t = torch.tensor(pose.translation,       dtype=torch.float64)
            name_to_pose[img.name] = torch.cat([R, t.unsqueeze(1)], dim=1)

        cam_lists = [
            name_to_pose.get(n, torch.full((3, 4), float("nan"), dtype=torch.float64))
            for n in names
        ]

        return {
            "cam_lists":      cam_lists,
            "num_registered": rec.num_reg_images(),
            "reconstruction": rec,
        }
        # TemporaryDirectory cleaned up here — images and DB are deleted


# ── Injected-track multiview (kept for compatibility) ─────────────────────────

def _write_dummy_jpeg(path: Path) -> None:
    """Write the smallest valid 1×1 JPEG so COLMAP can open the file."""
    _JPEG_1x1 = bytes([
        0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,0x01,0x00,0x00,0x01,
        0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,
        0x07,0x07,0x07,0x09,0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
        0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,0x24,0x2E,0x27,0x20,
        0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,
        0x39,0x3D,0x38,0x32,0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
        0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,0x01,0x05,0x01,0x01,
        0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,
        0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
        0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,0x01,0x02,0x03,0x00,
        0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,
        0x81,0x91,0xA1,0x08,0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,
        0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,0x29,0x2A,0x34,0x35,
        0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,
        0x56,0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,
        0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x92,0x93,0x94,
        0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,
        0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,
        0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,0xE3,0xE4,0xE5,0xE6,
        0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA,0xFF,0xDA,
        0x00,0x08,0x01,0x01,0x00,0x00,0x3F,0x00,0xFB,0xD7,0xFF,0xD9,
    ])
    path.write_bytes(_JPEG_1x1)


def estimate_multiview_pose_colmap_incremental(
    obs_mat: torch.Tensor,
    Ks: torch.Tensor,
    frame_indices: list[int],
    ransac_max_error: float = 2.0,
) -> dict:
    """
    Run pycolmap incremental SfM on N frames using pre-computed tracks from obs_mat.
    Uses injected keypoints/matches rather than COLMAP's own SIFT.

    Args:
        obs_mat:       [2F, N] tracked keypoints (u, v interleaved per frame)
        Ks:            [F,3,3] or [1,3,3] camera intrinsics
        frame_indices: which global frame indices the rows of obs_mat refer to
        ransac_max_error: RANSAC inlier threshold in pixels
    """
    import sqlite3
    from itertools import combinations

    F = len(frame_indices)
    N = obs_mat.shape[1]

    _MAX_IMAGE_ID = 2147483647
    def _pair_id(a, b):
        lo, hi = min(a, b), max(a, b)
        return _MAX_IMAGE_ID * lo + hi

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        db_path = tmpdir / "colmap.db"

        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE cameras (
                camera_id          INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                model              INTEGER NOT NULL,
                width              INTEGER NOT NULL,
                height             INTEGER NOT NULL,
                params             BLOB,
                prior_focal_length INTEGER NOT NULL
            );
            CREATE TABLE images (
                image_id  INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name      TEXT    NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL,
                prior_qw  REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
                prior_tx  REAL, prior_ty REAL, prior_tz REAL
            );
            CREATE TABLE keypoints (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows     INTEGER NOT NULL,
                cols     INTEGER NOT NULL,
                data     BLOB
            );
            CREATE TABLE descriptors (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows     INTEGER NOT NULL,
                cols     INTEGER NOT NULL,
                data     BLOB
            );
            CREATE TABLE matches (
                pair_id INTEGER PRIMARY KEY NOT NULL,
                rows    INTEGER NOT NULL,
                cols    INTEGER NOT NULL,
                data    BLOB
            );
            CREATE TABLE two_view_geometries (
                pair_id INTEGER PRIMARY KEY NOT NULL,
                rows    INTEGER NOT NULL,
                cols    INTEGER NOT NULL,
                data    BLOB,
                config  INTEGER NOT NULL,
                F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB
            );
        """)

        camera_ids = []
        for i in range(F):
            K = Ks[min(frame_indices[i], Ks.shape[0] - 1)].cpu().numpy().astype(np.float64)
            params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
            cur = conn.execute(
                "INSERT INTO cameras(model, width, height, params, prior_focal_length) "
                "VALUES (?, ?, ?, ?, ?)",
                (1, int(K[0,2] * 2), int(K[1,2] * 2), params.tobytes(), 0),
            )
            camera_ids.append(cur.lastrowid)

        image_ids = []
        for i in range(F):
            name = f"frame_{frame_indices[i]:04d}.jpg"
            cur = conn.execute(
                "INSERT INTO images(name, camera_id) VALUES (?, ?)",
                (name, camera_ids[i]),
            )
            image_ids.append(cur.lastrowid)
            _write_dummy_jpeg(tmpdir / name)

        identity_matches = np.stack([np.arange(N), np.arange(N)], axis=1).astype(np.uint32)

        for i in range(F):
            # Fix: use frame_indices[i] not i when indexing obs_mat
            kpts = obs_mat[frame_indices[i]*2: frame_indices[i]*2+2, :].T \
                          .cpu().numpy().astype(np.float32)
            conn.execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (image_ids[i], N, 2, kpts.tobytes()),
            )
            conn.execute(
                "INSERT INTO descriptors(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (image_ids[i], 0, 128, b""),
            )

        pairs_lines = []
        for i, j in combinations(range(F), 2):
            conn.execute(
                "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (_pair_id(image_ids[i], image_ids[j]), N, 2, identity_matches.tobytes()),
            )
            pairs_lines.append(
                f"frame_{frame_indices[i]:04d}.jpg frame_{frame_indices[j]:04d}.jpg"
            )

        conn.commit()
        conn.close()

        pairs_path = tmpdir / "pairs.txt"
        pairs_path.write_text("\n".join(pairs_lines))

        pycolmap.verify_matches(
            str(db_path),
            str(pairs_path),
            options=pycolmap.TwoViewGeometryOptions(
                compute_relative_pose=True,
                ransac=pycolmap.RANSACOptions(max_error=ransac_max_error),
            ),
        )

        output_path = tmpdir / "sparse"
        output_path.mkdir()

        maps = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(tmpdir),
            output_path=str(output_path),
        )

        if not maps:
            return {"cam_lists": [torch.full((3, 4), float("nan"), dtype=torch.float64)] * F}

        rec = maps[max(maps, key=lambda k: maps[k].num_reg_images())]

        name_to_pose = {}
        for img in rec.images.values():
            pose = _get_rigid3d(img.cam_from_world)
            R = torch.tensor(pose.rotation.matrix(), dtype=torch.float64)
            t = torch.tensor(pose.translation, dtype=torch.float64)
            name_to_pose[img.name] = torch.cat([R, t.unsqueeze(1)], dim=1)

        cam_lists = []
        for i in range(F):
            name = f"frame_{frame_indices[i]:04d}.jpg"
            cam_lists.append(name_to_pose.get(
                name, torch.full((3, 4), float("nan"), dtype=torch.float64)
            ))

        return {
            "cam_lists":      cam_lists,
            "num_registered": rec.num_reg_images(),
            "reconstruction": rec,
        }


# ── Shared helpers ────────────────────────────────────────────────────────────

def _overwrite_camera_params_db(
    db_path: Path,
    names: list[str],
    Ks: torch.Tensor,
    frame_indices: list[int],
    img_w: int,
    img_h: int,
) -> None:
    """
    After extract_features, patch the DB cameras with the supplied per-frame Ks.
    Only needed when camera_mode='per_frame' and Ks is not None.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    name_to_fi = {n: fi for n, fi in zip(names, frame_indices)}

    rows = conn.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    for _image_id, name, camera_id in rows:
        fi = name_to_fi.get(name)
        if fi is None:
            continue
        K = Ks[min(fi, Ks.shape[0] - 1)].cpu().numpy().astype(np.float64)
        params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
        conn.execute(
            "UPDATE cameras SET model=1, width=?, height=?, params=? WHERE camera_id=?",
            (img_w, img_h, params.tobytes(), camera_id),
        )

    conn.commit()
    conn.close()


def reconstruction_reprojection_error(rec) -> dict:
    """
    Compute reprojection error from a pycolmap Reconstruction object.

    Returns dict with:
        mean_error   - mean reprojection error over all observations (pixels)
        median_error - median reprojection error (pixels)
        errors       - np.ndarray of all per-observation errors
        per_image    - dict {image_name: mean_error} for each registered image
    """
    errors = []
    per_image = {}

    for img in rec.images.values():
        cam = rec.cameras[img.camera_id]
        pose = _get_rigid3d(img.cam_from_world)
        R = torch.tensor(pose.rotation.matrix(), dtype=torch.float64)
        t = torch.tensor(pose.translation, dtype=torch.float64)

        p = cam.params  # PINHOLE: [fx, fy, cx, cy]
        K = torch.tensor([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=torch.float64)

        img_errors = []
        for p2d in img.points2D:
            if p2d.point3D_id not in rec.points3D:
                continue
            X   = torch.tensor(rec.points3D[p2d.point3D_id].xyz, dtype=torch.float64)
            obs = torch.tensor(p2d.xy, dtype=torch.float64)
            X_cam  = R @ X + t
            x_proj = (K @ X_cam)[:2] / (K @ X_cam)[2]
            img_errors.append((x_proj - obs).norm().item())

        per_image[img.name] = float(np.mean(img_errors)) if img_errors else float("nan")
        errors.extend(img_errors)

    arr = np.array(errors) if errors else np.array([float("nan")])
    return {
        "mean_error":   float(np.mean(arr)),
        "median_error": float(np.median(arr)),
        "errors":       arr,
        "per_image":    per_image,
    }