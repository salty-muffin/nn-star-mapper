"""Bounded global search for the best pose inside artist-set ranges.

``project.py`` is for *hand-picking* a pose and seeing the fit; this is for
*searching*. You constrain each of the six pose axes to a range -- "tilt-az may
swing between -40 and 40 degrees, distance stays pinned at 12" -- and the solver
finds the pose inside that box that lands the most neurons on stars. It is the
artist-controlled middle ground between fully hand-picked and a blind automatic
sweep: tighten the box for control, widen it to let the computer surprise you.

Why a *global* optimiser
------------------------
The score is non-convex and piecewise-flat: the Hungarian assignment makes it
jump in steps as neurons cross gates, so a local method (Nelder-Mead) just rolls
into whatever optimum sits next to its start. We instead run
``differential_evolution`` -- a derivative-free global search that samples the
whole box and respects per-axis bounds natively -- then a short Nelder-Mead
polish for the last sub-pixel. An axis whose range is a single value is *pinned*
and dropped from the search vector entirely.

Flux weighting
--------------
Optionally the objective prefers *bright* stars, which read better on screen.
Each star gets a weight ``1 + alpha * w`` where ``w`` is its normalised flux
(rank- or log-scaled, since raw flux has a brutal dynamic range), folded into
both the assignment and the maximised objective via ``score_pose``. ``alpha = 0``
recovers the pure geometric fit; raise it to bias toward brighter stars.

Two ways to set the box
-----------------------
* From scratch -- each axis flag is an absolute ``lo:hi`` range (or a bare number
  to pin that axis):

      uv run python optimize.py --tilt-az -40:40 --tilt-el -40:40 --distance 12

* Around a base pose (``--pose-in``) -- the loaded pose is the centre of the box
  and each bare-number flag is the *max deviation* (±) from it, so you can polish
  a hand-picked pose without letting it wander: ``--tilt-az 10`` searches
  ``centre ± 10``, ``0`` pins the axis, and ``lo:hi`` still forces an absolute
  range for that one axis:

      uv run python optimize.py --pose-in data/pose_02.json \\
          --tilt-az 10 --tilt-el 10 --roll 15 --pos-x 150 --pos-y 150 --distance 0 \\
          --flux-alpha 1.0 --overlay data/opt.png --pose-out data/pose.json
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

from project import (
    Pose,
    Scene,
    _load_plate,
    load_scene,
    project,
    render_overlay,
    score_pose,
)

# Pose axis order used everywhere here and in Pose.as_vector / from_vector.
AXES = ("tilt_az", "tilt_el", "roll", "pos_x", "pos_y", "distance")

# Default ± deviation per axis when a base pose is loaded but the flag is omitted.
DEFAULT_DEV = {
    "tilt_az": 15.0,
    "tilt_el": 15.0,
    "roll": 20.0,
    "pos_x": 200.0,
    "pos_y": 200.0,
    "distance": 3.0,
}


# --------------------------------------------------------------------------- #
# Range parsing + flux weighting
# --------------------------------------------------------------------------- #
class Range(click.ParamType):
    """An axis spec: ``lo:hi`` (absolute interval) or a bare number.

    A bare number is returned as a ``float`` and interpreted later by context: a
    ± deviation when a base pose is loaded, otherwise a pinned value. An interval
    is returned as a ``(lo, hi)`` tuple and is always absolute.
    """

    name = "range"

    def convert(self, value, param, ctx) -> float | tuple[float, float]:
        if isinstance(value, (float, tuple)):  # default already resolved
            return value
        try:
            if ":" in str(value):
                lo, hi = (float(x) for x in str(value).split(":", 1))
                return (lo, hi) if lo <= hi else (hi, lo)
            return float(value)
        except ValueError:
            self.fail(f"{value!r} is not 'lo:hi' or a number", param, ctx)


def resolve_bounds(
    name: str,
    spec: float | tuple[float, float] | None,
    center: float | None,
    default_abs: tuple[float, float],
) -> tuple[float, float]:
    """Turn an axis flag into concrete ``(lo, hi)`` search bounds.

    ``lo:hi`` is always absolute. A bare number is a ± deviation around
    ``center`` when a base pose is loaded (``0`` pins the axis), or a pinned value
    when there is no center. An omitted flag (``spec is None``) falls back to the
    default deviation around the center, or to ``default_abs`` with no center.
    """
    if isinstance(spec, tuple):
        return spec
    if center is not None:
        dev = DEFAULT_DEV[name] if spec is None else abs(spec)
        return (center - dev, center + dev)
    if spec is None:
        return default_abs
    return (spec, spec)


def flux_weights(flux: np.ndarray, mode: str, alpha: float) -> np.ndarray | None:
    """Per-star multiplier ``1 + alpha * w`` with ``w`` a normalised brightness.

    ``rank`` maps stars to evenly-spaced weights by brightness order (robust to
    the few-very-bright-stars dynamic range); ``log`` normalises log-flux to
    [0, 1]; ``none`` (or ``alpha == 0``) disables weighting entirely.
    """
    if alpha == 0.0 or mode == "none" or len(flux) == 0:
        return None
    if mode == "rank":
        w = np.argsort(np.argsort(flux)).astype(float) / max(len(flux) - 1, 1)
    elif mode == "log":
        lf = np.log(np.maximum(flux, np.finfo(float).tiny))
        span = lf.max() - lf.min()
        w = (lf - lf.min()) / span if span > 0 else np.zeros_like(lf)
    else:  # pragma: no cover - click choices guard this
        raise click.BadParameter(f"unknown flux norm {mode!r}")
    return 1.0 + alpha * w


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def search_pose(
    scene: Scene,
    bounds: list[tuple[float, float]],
    star_tree: cKDTree,
    sigma: float,
    eps: float,
    gate: float,
    flux_w: np.ndarray | None,
    maxiter: int,
    popsize: int,
    seed: int | None,
) -> Pose:
    """Maximise the (flux-weighted) objective over the box; pinned axes are fixed.

    Axes whose range has zero width are held constant and removed from the search
    so ``differential_evolution`` only explores the genuinely free dimensions.
    A Nelder-Mead polish then tightens the global result.
    """
    base = np.array([lo for lo, _ in bounds])  # pinned axes keep this value
    free = [i for i, (lo, hi) in enumerate(bounds) if hi > lo]

    def to_full(x_free: np.ndarray) -> np.ndarray:
        full = base.copy()
        full[free] = x_free
        return full

    def neg_objective(x_free: np.ndarray) -> float:
        pose = Pose.from_vector(to_full(np.atleast_1d(x_free)))
        pts = project(scene.neurons, pose, scene.focal_px)
        return -score_pose(pts, scene.stars, star_tree, sigma, eps, gate, flux_w).objective  # fmt: skip

    if not free:  # every axis pinned -- nothing to search
        return Pose.from_vector(base)

    free_bounds = [bounds[i] for i in free]
    result = differential_evolution(
        neg_objective,
        bounds=free_bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=False,  # our objective is non-smooth; gradient polish is useless
        tol=1e-4,
        init="sobol",
    )
    # Derivative-free local polish on the free axes from the global best.
    polished = minimize(
        neg_objective,
        result.x,
        method="Nelder-Mead",
        options={"xatol": 1e-2, "fatol": 1e-4, "maxiter": 2000},
    )
    best = polished.x if polished.fun <= result.fun else result.x
    return Pose.from_vector(to_full(np.atleast_1d(best)))


def projected_extent(scene: Scene, pose: Pose) -> tuple[float, float]:
    """Bounding-box (width, height) in px of the projected lattice.

    Reported so a flux-hungry objective can't quietly collapse the lattice onto a
    tight cluster of bright stars without it showing up here.
    """
    pts = project(scene.neurons, pose, scene.focal_px)
    span = pts.max(axis=0) - pts.min(axis=0)
    return float(span[0]), float(span[1])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
RANGE = Range()


@click.command()
@click.option("--network", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("data/network.json"), show_default=True, help="Neuron geometry JSON from netgen.py.")  # fmt: skip
@click.option("--stars", "stars_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("data/stars.json"), show_default=True, help="Detected-star JSON from detect_stars.py.")  # fmt: skip
@click.option("--image", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Plate image for the overlay backdrop (default: path in stars JSON).")  # fmt: skip
@click.option("--top-n", type=int, default=None, help="Use only the brightest N stars as targets (default: all).")  # fmt: skip
@click.option("--focal-px", type=float, default=None, help="Override focal length in pixels (default: value from stars JSON).")  # fmt: skip
@click.option("--pose-in", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Base pose JSON; axis flags become ± deviations around it.")  # fmt: skip
@click.option("--tilt-az", type=RANGE, default=None, help="Azimuth tilt: ±deviation from --pose-in, or 'lo:hi' absolute degrees.")  # fmt: skip
@click.option("--tilt-el", type=RANGE, default=None, help="Elevation tilt: ±deviation from --pose-in, or 'lo:hi' absolute degrees.")  # fmt: skip
@click.option("--roll", type=RANGE, default=None, help="Roll: ±deviation from --pose-in, or 'lo:hi' absolute degrees.")  # fmt: skip
@click.option("--pos-x", type=RANGE, default=None, help="Image x: ±deviation from --pose-in, or 'lo:hi' absolute px.")  # fmt: skip
@click.option("--pos-y", type=RANGE, default=None, help="Image y: ±deviation from --pose-in, or 'lo:hi' absolute px.")  # fmt: skip
@click.option("--distance", type=RANGE, default=None, help="Distance: ±deviation from --pose-in, or 'lo:hi' absolute.")  # fmt: skip
@click.option("--flux-alpha", type=float, default=0.0, show_default=True, help="Brightness bias: 0 = pure geometry, higher favours brighter stars.")  # fmt: skip
@click.option("--flux-norm", type=click.Choice(["rank", "log", "none"]), default="rank", show_default=True, help="How star flux is normalised before weighting.")  # fmt: skip
@click.option("--sigma", type=float, default=8.0, show_default=True, help="Soft-match falloff in px; several near-misses beat one exact hit.")  # fmt: skip
@click.option("--eps", type=float, default=6.0, show_default=True, help="Hard inlier tolerance in px (for the reported count).")  # fmt: skip
@click.option("--gate", type=float, default=None, help="Max neuron-star pairing distance in px (default: 4*sigma).")  # fmt: skip
@click.option("--maxiter", type=int, default=80, show_default=True, help="differential_evolution generations.")  # fmt: skip
@click.option("--popsize", type=int, default=15, show_default=True, help="differential_evolution population multiplier.")  # fmt: skip
@click.option("--seed", type=int, default=None, help="Random seed for a reproducible search.")  # fmt: skip
@click.option("--overlay", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write a static overlay image of the best projection over the plate.")  # fmt: skip
@click.option("--pose-out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write the best pose + score to JSON for downstream steps.")  # fmt: skip
@click.option("--display-scale", type=float, default=0.5, show_default=True, help="Downsample factor for the overlay backdrop only.")  # fmt: skip
def main(
    network: Path,
    stars_path: Path,
    image: Path | None,
    top_n: int | None,
    focal_px: float | None,
    pose_in: Path | None,
    tilt_az: float | tuple[float, float] | None,
    tilt_el: float | tuple[float, float] | None,
    roll: float | tuple[float, float] | None,
    pos_x: float | tuple[float, float] | None,
    pos_y: float | tuple[float, float] | None,
    distance: float | tuple[float, float] | None,
    flux_alpha: float,
    flux_norm: str,
    sigma: float,
    eps: float,
    gate: float | None,
    maxiter: int,
    popsize: int,
    seed: int | None,
    overlay: Path | None,
    pose_out: Path | None,
    display_scale: float,
) -> None:
    """Search for the best pose inside the given per-axis ranges."""
    scene = load_scene(network, stars_path, top_n, focal_px)
    if gate is None:
        gate = 4.0 * sigma

    base = Pose(**json.loads(pose_in.read_text())["pose"]) if pose_in else None
    centers = {n: (getattr(base, n) if base else None) for n in AXES}
    default_abs = {
        "tilt_az": (-45.0, 45.0),
        "tilt_el": (-45.0, 45.0),
        "roll": (-180.0, 180.0),
        "pos_x": (0.0, float(scene.width)),
        "pos_y": (0.0, float(scene.height)),
        "distance": (8.0, 30.0),
    }
    specs = dict(zip(AXES, (tilt_az, tilt_el, roll, pos_x, pos_y, distance)))
    bounds = [resolve_bounds(n, specs[n], centers[n], default_abs[n]) for n in AXES]

    star_tree = cKDTree(scene.stars)
    flux_w = flux_weights(scene.star_flux, flux_norm, flux_alpha)

    free = [
        f"{n} [{lo:g},{hi:g}]" for n, (lo, hi) in zip(AXES, bounds) if hi > lo
    ]
    around = f" around {pose_in}" if base else ""
    click.echo(
        f"searching {len(free)} axes{around}: {', '.join(free) or 'none'}\n"
        f"  flux bias alpha={flux_alpha:g} ({flux_norm})"
    )
    pose = search_pose(scene, bounds, star_tree, sigma, eps, gate, flux_w, maxiter, popsize, seed)  # fmt: skip

    pts = project(scene.neurons, pose, scene.focal_px)
    score = score_pose(pts, scene.stars, star_tree, sigma, eps, gate, flux_w)
    ext_w, ext_h = projected_extent(scene, pose)
    click.echo(
        f"best: soft={score.soft:.2f}  objective={score.objective:.2f}  "
        f"inliers(<{eps:g}px)={score.inliers}  matched={len(score.matches)}\n"
        f"  projected extent: {ext_w:.0f} x {ext_h:.0f} px\n"
        f"  pose: tilt_az={pose.tilt_az:.3f} tilt_el={pose.tilt_el:.3f} "
        f"roll={pose.roll:.3f} pos_x={pose.pos_x:.1f} pos_y={pose.pos_y:.1f} "
        f"distance={pose.distance:.3f}"
    )

    if overlay is not None:
        plate = _load_plate(scene, image, stars_path)
        overlay.parent.mkdir(parents=True, exist_ok=True)
        render_overlay(scene, pose, score, sigma, eps, plate, display_scale, overlay)
        click.echo(f"Wrote {overlay}")

    if pose_out is not None:
        pose_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pose": asdict(pose),
            "focal_px": scene.focal_px,
            "image_size": [scene.width, scene.height],
            "search": {
                "pose_in": str(pose_in) if pose_in else None,
                "bounds": {AXES[i]: list(b) for i, b in enumerate(bounds)},
                "flux_alpha": flux_alpha,
                "flux_norm": flux_norm,
                "seed": seed,
            },
            "score": {
                "soft": score.soft,
                "objective": score.objective,
                "inliers": score.inliers,
                "eps": eps,
                "sigma": sigma,
                "matched": len(score.matches),
            },
            "matches": [[int(n), int(s)] for n, s in score.matches],
        }
        pose_out.write_text(json.dumps(payload, indent=2))
        click.echo(f"Wrote {pose_out}")


if __name__ == "__main__":
    main()
