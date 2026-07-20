#!/usr/bin/env python3
"""Benchmark the incremental_global_mapper drip-feed against one-shot GLOMAP.

Given a fully populated COLMAP database (features + verified matches for all
images, e.g. a session database from the spacemapper server), this script:

  1. Builds an initial reconstruction from the first --start_images images
     with the one-shot `global_mapper`.
  2. Drip-feeds the remaining images one at a time through
     `incremental_global_mapper`, mimicking the production upload loop: for
     each step a subset database containing only the images "uploaded" so
     far is created, `view_graph_calibrator` runs on it, and the previous
     output is used as the prior reconstruction. Wall time per add is
     recorded. Run once per requested variant (e.g. full solve vs windowed).
  3. Runs the one-shot `global_mapper` on the complete database as the
     reference.
  4. Reports, per variant: ATE RMSE vs the one-shot reference (after Sim3
     alignment, for comparison only), RPE (relative pose error over
     consecutive image pairs), 3D point count, mean reprojection error, and
     per-add wall time, as a markdown table.

Requires numpy. No pycolmap dependency: sparse models are read with the
standalone binary readers below.

Example:
  python3 incremental_benchmark.py \
      --database_path session/database.db \
      --output_dir /tmp/bench \
      --colmap_exe build/src/colmap/exe/colmap \
      --variants full,window8 \
      --start_images 5
"""

import argparse
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# COLMAP pair-id encoding (colmap/util/types.h).
MAX_NUM_IMAGES = 2147483647


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    return pair_id // MAX_NUM_IMAGES, pair_id % MAX_NUM_IMAGES


# ── Binary sparse-model readers (images.bin / points3D.bin) ────────────────


def _read(fid, fmt):
    return struct.unpack(fmt, fid.read(struct.calcsize(fmt)))


def read_images_bin(path: Path) -> dict[int, dict]:
    """Returns {image_id: {"qvec", "tvec", "name"}} (cam_from_world)."""
    images = {}
    with open(path, "rb") as fid:
        (num_reg,) = _read(fid, "<Q")
        for _ in range(num_reg):
            image_id, qw, qx, qy, qz, tx, ty, tz, _cam = _read(
                fid, "<IdddddddI")
            name = b""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_points2D,) = _read(fid, "<Q")
            fid.read(24 * num_points2D)  # x, y (double) + point3D_id (uint64)
            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "name": name.decode("utf-8"),
            }
    return images


def read_points3D_bin(path: Path) -> tuple[int, float]:
    """Returns (num_points, mean_reprojection_error)."""
    errors = []
    with open(path, "rb") as fid:
        (num_points,) = _read(fid, "<Q")
        for _ in range(num_points):
            _pid, _x, _y, _z, _r, _g, _b, error = _read(fid, "<QdddBBBd")
            errors.append(error)
            (track_len,) = _read(fid, "<Q")
            fid.read(8 * track_len)
    return num_points, (float(np.mean(errors)) if errors else float("nan"))


def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def camera_center(image) -> np.ndarray:
    return -qvec_to_rotmat(image["qvec"]).T @ image["tvec"]


# ── Metrics ────────────────────────────────────────────────────────────────


def umeyama_sim3(src: np.ndarray, tgt: np.ndarray):
    """Least-squares Sim3 (s, R, t) with tgt ≈ s * R @ src + t."""
    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt
    cov = tgt_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_src = (src_c**2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_tgt - s * R @ mu_src
    return s, R, t


def ate_rmse(eval_images: dict, ref_images: dict) -> float:
    """ATE RMSE of camera centres after Sim3 alignment (comparison only)."""
    common = sorted(set(eval_images) & set(ref_images))
    if len(common) < 3:
        return float("nan")
    src = np.array([camera_center(eval_images[i]) for i in common])
    tgt = np.array([camera_center(ref_images[i]) for i in common])
    s, R, t = umeyama_sim3(src, tgt)
    aligned = (s * (R @ src.T)).T + t
    return float(np.sqrt(((aligned - tgt) ** 2).sum(axis=1).mean()))


def rpe_stats(eval_images: dict, ref_images: dict) -> tuple[float, float]:
    """RPE over consecutive common images: (mean rot err deg, mean rel
    translation direction err deg)."""
    common = sorted(set(eval_images) & set(ref_images))
    rot_errs, dir_errs = [], []
    for a, b in zip(common[:-1], common[1:]):
        def rel(images):
            Ra, ta = qvec_to_rotmat(images[a]["qvec"]), images[a]["tvec"]
            Rb, tb = qvec_to_rotmat(images[b]["qvec"]), images[b]["tvec"]
            R_rel = Rb @ Ra.T
            t_rel = tb - R_rel @ ta
            return R_rel, t_rel

        R1, t1 = rel(eval_images)
        R2, t2 = rel(ref_images)
        dR = R1 @ R2.T
        angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        rot_errs.append(angle)
        n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
        if n1 > 1e-9 and n2 > 1e-9:
            cosang = np.clip(np.dot(t1 / n1, t2 / n2), -1, 1)
            dir_errs.append(np.degrees(np.arccos(cosang)))
    return (
        float(np.mean(rot_errs)) if rot_errs else float("nan"),
        float(np.mean(dir_errs)) if dir_errs else float("nan"),
    )


# ── Database subsetting ────────────────────────────────────────────────────


def list_images_by_id(database_path: Path) -> list[tuple[int, str]]:
    db = sqlite3.connect(database_path)
    try:
        return db.execute(
            "SELECT image_id, name FROM images ORDER BY image_id").fetchall()
    finally:
        db.close()


def _remove_db_files(db_path: Path) -> None:
    """Removes a database and any journal sidecars a previous writer left
    behind — a fresh copy next to a stale -wal/-shm file reads as corrupt."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()


def make_subset_db(full_db: Path, subset_db: Path, keep_ids: set[int]) -> None:
    """Copies the database and deletes all rows referencing images outside
    keep_ids, mimicking a production database that has only seen the first
    k uploads."""
    _remove_db_files(subset_db)
    shutil.copyfile(full_db, subset_db)
    db = sqlite3.connect(subset_db)
    try:
        placeholders = ",".join(str(i) for i in keep_ids)
        for table, col in [
            ("images", "image_id"),
            ("keypoints", "image_id"),
            ("descriptors", "image_id"),
        ]:
            try:
                db.execute(
                    f"DELETE FROM {table} WHERE {col} NOT IN ({placeholders})")
            except sqlite3.OperationalError:
                pass  # table absent in some schema versions
        for table in ("matches", "two_view_geometries"):
            pair_ids = [r[0] for r in db.execute(f"SELECT pair_id FROM {table}")]
            to_delete = [
                (p,)
                for p in pair_ids
                if not set(pair_id_to_image_ids(p)) <= keep_ids
            ]
            db.executemany(f"DELETE FROM {table} WHERE pair_id = ?", to_delete)
        try:
            db.execute(
                "DELETE FROM pose_priors WHERE corr_data_id NOT IN "
                f"({placeholders})"
            )
        except sqlite3.OperationalError:
            pass
        db.commit()
    finally:
        db.close()


# ── Pipeline runners ───────────────────────────────────────────────────────


def run(cmd: list[str], log_path: Path) -> float:
    t0 = time.perf_counter()
    with open(log_path, "ab") as log:
        log.write(("\n$ " + " ".join(map(str, cmd)) + "\n").encode())
        subprocess.run(list(map(str, cmd)), check=True, stdout=log, stderr=log)
    return time.perf_counter() - t0


def find_model_dir(output_path: Path) -> Path:
    if (output_path / "0" / "cameras.bin").exists():
        return output_path / "0"
    if (output_path / "cameras.bin").exists():
        return output_path
    raise FileNotFoundError(f"no cameras.bin under {output_path}")


def run_one_shot(colmap, db, image_path, out_dir, log) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = run(
        [colmap, "global_mapper", "--database_path", db,
         "--image_path", image_path, "--output_path", out_dir],
        log,
    )
    return dt


def drip_feed(colmap, full_db, image_path, work_dir, image_ids, start_images,
              extra_args, log) -> tuple[Path, list[float]]:
    """Returns (final model dir, per-add wall times)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    subset_db = work_dir / "database.db"
    prior_dir = work_dir / "prior"

    # Initial reconstruction from the first start_images images.
    keep = set(image_ids[:start_images])
    make_subset_db(full_db, subset_db, keep)
    init_out = work_dir / "init"
    init_out.mkdir(exist_ok=True)
    run([colmap, "global_mapper", "--database_path", subset_db,
         "--image_path", image_path, "--output_path", init_out], log)
    if prior_dir.exists():
        shutil.rmtree(prior_dir)
    shutil.copytree(find_model_dir(init_out), prior_dir)

    times = []
    for i in range(start_images, len(image_ids)):
        keep.add(image_ids[i])
        make_subset_db(full_db, subset_db, keep)
        run([colmap, "view_graph_calibrator", "--database_path", subset_db],
            log)
        add_out = work_dir / "add"
        if add_out.exists():
            shutil.rmtree(add_out)
        add_out.mkdir()
        dt = run(
            [colmap, "incremental_global_mapper",
             "--database_path", subset_db,
             "--image_path", image_path,
             "--prior_reconstruction_path", prior_dir,
             "--output_path", add_out] + extra_args,
            log,
        )
        times.append(dt)
        model = find_model_dir(add_out)
        # Promote, preserving anchors.txt like the production server does
        # (it moves the whole model directory).
        shutil.rmtree(prior_dir)
        shutil.move(str(model), str(prior_dir))
        print(f"  [{i + 1}/{len(image_ids)}] add: {dt:.2f}s", flush=True)
    return prior_dir, times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--database_path", required=True, type=Path)
    ap.add_argument("--image_path", type=Path, default=Path("."),
                    help="Image dir (only used for point colors; optional)")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--colmap_exe", type=Path, default=Path("colmap"))
    ap.add_argument("--start_images", type=int, default=5)
    ap.add_argument("--max_images", type=int, default=0,
                    help="Cap the number of images (0 = all)")
    ap.add_argument(
        "--variants", default="full,window8",
        help="Comma list: 'full' (window 0) and/or 'windowN[.M]' for "
             "optimize_window_size N with full_solve_interval M")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / "benchmark.log"

    image_rows = list_images_by_id(args.database_path)
    image_ids = [r[0] for r in image_rows]
    if args.max_images:
        image_ids = image_ids[: args.max_images]
    n = len(image_ids)
    print(f"{n} images in database; drip-feeding from {args.start_images}")

    # Reference: one-shot global mapper over everything.
    ref_dir = args.output_dir / "one_shot"
    print("Running one-shot global_mapper reference ...")
    ref_db = args.output_dir / "ref_database.db"
    make_subset_db(args.database_path, ref_db, set(image_ids))
    run([args.colmap_exe, "view_graph_calibrator", "--database_path", ref_db],
        log)
    one_shot_time = run_one_shot(
        args.colmap_exe, ref_db, args.image_path, ref_dir, log)
    ref_model = find_model_dir(ref_dir)
    ref_images = read_images_bin(ref_model / "images.bin")
    ref_points, ref_reproj = read_points3D_bin(ref_model / "points3D.bin")

    rows = []
    rows.append({
        "variant": "one-shot global_mapper",
        "reg": len(ref_images),
        "points": ref_points,
        "reproj": ref_reproj,
        "ate": 0.0,
        "rpe_rot": 0.0,
        "rpe_dir": 0.0,
        "mean_add": float("nan"),
        "p95_add": float("nan"),
        "total": one_shot_time,
    })

    for variant in args.variants.split(","):
        variant = variant.strip()
        extra = []
        if variant == "full":
            pass
        elif variant.startswith("window"):
            spec = variant[len("window"):]
            win, _, interval = spec.partition(".")
            extra += ["--optimize_window_size", win]
            if interval:
                extra += ["--full_solve_interval", interval]
        else:
            print(f"unknown variant '{variant}'", file=sys.stderr)
            return 1

        print(f"Drip-feed variant '{variant}' ...")
        work = args.output_dir / f"drip_{variant}"
        model_dir, times = drip_feed(
            args.colmap_exe, args.database_path, args.image_path, work,
            image_ids, args.start_images, extra, log)
        images = read_images_bin(model_dir / "images.bin")
        points, reproj = read_points3D_bin(model_dir / "points3D.bin")
        rpe_rot, rpe_dir = rpe_stats(images, ref_images)
        rows.append({
            "variant": f"drip-feed {variant}",
            "reg": len(images),
            "points": points,
            "reproj": reproj,
            "ate": ate_rmse(images, ref_images),
            "rpe_rot": rpe_rot,
            "rpe_dir": rpe_dir,
            "mean_add": float(np.mean(times)) if times else float("nan"),
            "p95_add": float(np.percentile(times, 95)) if times else float("nan"),
            "total": float(np.sum(times)) if times else float("nan"),
        })

    # Markdown table.
    md = [
        f"# incremental_global_mapper benchmark — {n} images\n",
        "| variant | reg images | 3D points | mean reproj px | ATE rmse | "
        "RPE rot deg | RPE dir deg | mean s/add | p95 s/add | total s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            "| {variant} | {reg} | {points} | {reproj:.3f} | {ate:.4f} | "
            "{rpe_rot:.3f} | {rpe_dir:.3f} | {mean_add:.2f} | {p95_add:.2f} "
            "| {total:.1f} |".format(**r))
    table = "\n".join(md) + "\n"
    print("\n" + table)
    (args.output_dir / "results.md").write_text(table)
    print(f"Written to {args.output_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
