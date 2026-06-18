"""Elastic snap: nudge neurons onto stars after the rigid fit, organically.

This is step 5 of the plan. A perfectly rigid lattice will almost never land
every neuron on a star, even at a well-chosen pose -- ``project.py`` typically
gets a handful of exact inliers and many near-misses. Here we take a solved pose
(from ``project.py --pose-out``) and *deform* the lattice so the matched neurons
sit exactly on their stars, spreading the deformation smoothly across each layer
so the result still reads as the same network -- just gently warped, which looks
organic rather than broken.

Why this is well posed
----------------------
Blender renders the final shot from the 3D geometry at the solved pose, so to
put a neuron on a star we must move it in *3D* such that it *reprojects* onto the
star pixel. The only neuron motion that changes its screen position without
changing its depth is a shift in the camera's frontoparallel plane (perpendicular
to the view ray); a move along the ray just rescales and is invisible. So we:

1. Project the rigid lattice at the pose and assign neurons to stars (Hungarian,
   within a snap gate -- more generous than the rigid inlier tolerance, to catch
   stragglers).
2. Solve a smooth per-neuron **pixel-shift field** ``s`` that takes each matched
   neuron's projection onto its star, regularised so neighbours in the same layer
   move together (a graph-Laplacian / elastic-sheet smoothness term). Unmatched
   neurons carry no data term; they just follow their neighbours.
3. Convert each pixel shift to a frontoparallel 3D displacement at that neuron's
   own depth and write the deformed lattice back in lattice coordinates.

Because step 3 is exact (a pixel shift ``s_i`` moves the projection by exactly
``s_i``), the smoothness/data trade-off in step 2 is the *only* thing that pulls
matched neurons off their stars -- raise ``--smooth`` for a stiffer, more
faithful lattice with looser hits; lower it for tighter hits with more warp.

    uv run python snap.py --pose-in data/pose_02.json --overlay data/snap.png \
        --output data/network_snapped.json
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from project import (
    Pose,
    Scene,
    _rotation,
    load_scene,
    project,
    render_overlay,
    score_pose,
)


# --------------------------------------------------------------------------- #
# Smoothness graph
# --------------------------------------------------------------------------- #
def grid_neighbor_edges(network_path: Path) -> list[tuple[int, int]]:
    """4-connected neighbour pairs *within* each layer, by (layer, row, col).

    The dense inter-layer edges that draw the visible fan are too stiff a
    smoothness prior -- coupling every node of one layer to every node of the
    next would drag the whole network rigidly. Instead each layer is treated as
    its own elastic sheet: a neuron is tied only to its up/down/left/right grid
    neighbours, so layers can warp independently and locally.
    """
    net = json.loads(network_path.read_text())
    by_addr: dict[tuple[int, int, int], int] = {
        (n["layer"], n["row"], n["col"]): n["id"] for n in net["neurons"]
    }
    edges: list[tuple[int, int]] = []
    for (layer, row, col), i in by_addr.items():
        for dr, dc in ((1, 0), (0, 1)):  # only +row / +col to avoid duplicates
            j = by_addr.get((layer, row + dr, col + dc))
            if j is not None:
                edges.append((i, j))
    return edges


def laplacian(n: int, edges: list[tuple[int, int]]) -> sp.csr_matrix:
    """Graph Laplacian L (= degree - adjacency) for ``n`` nodes over ``edges``."""
    if not edges:
        return sp.csr_matrix((n, n))
    e = np.array(edges)
    i, j = e[:, 0], e[:, 1]
    data = np.concatenate([np.ones(len(e)), np.ones(len(e)), -np.ones(len(e)), -np.ones(len(e))])  # fmt: skip
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([i, j, j, i])
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


# --------------------------------------------------------------------------- #
# The snap
# --------------------------------------------------------------------------- #
def snap_neurons(
    scene: Scene,
    pose: Pose,
    matches: np.ndarray,
    smooth_edges: list[tuple[int, int]],
    smooth: float,
    anchor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (deformed neurons (N,3), per-neuron pixel shift applied (N,2)).

    Solves, per image axis independently,

        min_s  sum_matched (s_i - target_i)^2
               + smooth * sum_edges (s_i - s_j)^2
               + anchor * sum_i s_i^2

    where ``target_i`` is the pixel offset from neuron i's rigid projection to
    its assigned star. The anchor term keeps the system positive-definite even
    for a layer that drew no matches (it then simply stays put) and lightly
    damps runaway shifts. The solved pixel shifts are converted to frontoparallel
    3D moves at each neuron's depth and folded back into lattice coordinates.
    """
    n = len(scene.neurons)
    r = _rotation(pose.tilt_az, pose.tilt_el, pose.roll)
    t = np.array([0.0, 0.0, pose.distance])
    cam = scene.neurons @ r.T + t
    proj = project(scene.neurons, pose, scene.focal_px)

    # Per-neuron data term: matched neurons are pulled toward their star.
    w = np.zeros(n)
    target = np.zeros((n, 2))
    for ni, si in matches:
        w[ni] = 1.0
        target[ni] = scene.stars[si] - proj[ni]

    a = sp.diags(w + anchor) + smooth * laplacian(n, smooth_edges)
    rhs = (w[:, None] * target)  # data term only contributes where w>0
    shift = spsolve(a.tocsc(), rhs)
    shift = np.atleast_2d(shift)
    if shift.shape[0] != n:  # spsolve squeezes a single column
        shift = shift.T

    # A pixel shift s moves the projection by exactly s; realise it as a
    # frontoparallel camera-space move at the neuron's own depth, then map the
    # camera points back into lattice coordinates (R is orthonormal).
    cam_shift = np.zeros_like(cam)
    cam_shift[:, 0] = shift[:, 0] * cam[:, 2] / scene.focal_px
    cam_shift[:, 1] = shift[:, 1] * cam[:, 2] / scene.focal_px
    new_cam = cam + cam_shift
    new_neurons = (new_cam - t) @ r
    return new_neurons, shift


def write_snapped(
    network_path: Path,
    out_path: Path,
    new_neurons: np.ndarray,
    pose: Pose,
    smooth: float,
) -> None:
    """Write the deformed lattice in the same schema netgen.py emits."""
    net = json.loads(network_path.read_text())
    for neuron, (x, y, z) in zip(net["neurons"], new_neurons):
        neuron["x"], neuron["y"], neuron["z"] = float(x), float(y), float(z)
    net["snap"] = {
        "source_network": str(network_path),
        "pose": {
            k: getattr(pose, k)
            for k in ("tilt_az", "tilt_el", "roll", "pos_x", "pos_y", "distance")
        },
        "smooth": smooth,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(net, indent=2))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--network", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("data/network.json"), show_default=True, help="Rigid neuron geometry JSON from netgen.py.")  # fmt: skip
@click.option("--stars", "stars_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("data/stars.json"), show_default=True, help="Detected-star JSON from detect_stars.py.")  # fmt: skip
@click.option("--pose-in", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Solved pose JSON from project.py --pose-out.")  # fmt: skip
@click.option("--image", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Plate image for the overlay backdrop (default: path in stars JSON).")  # fmt: skip
@click.option("--top-n", type=int, default=None, help="Use only the brightest N stars as targets (default: all).")  # fmt: skip
@click.option("--focal-px", type=float, default=None, help="Override focal length in pixels (default: value from stars JSON).")  # fmt: skip
@click.option("--snap-gate", type=float, default=40.0, show_default=True, help="Max neuron-star pairing distance in px; stragglers within this get snapped.")  # fmt: skip
@click.option("--smooth", type=float, default=0.05, show_default=True, help="Elastic stiffness; higher = stiffer/more coherent warp but looser hits.")  # fmt: skip
@click.option("--anchor", type=float, default=1e-3, show_default=True, help="Damping that keeps unmatched layers in place and the solve stable.")  # fmt: skip
@click.option("--sigma", type=float, default=8.0, show_default=True, help="Soft-score falloff in px for the before/after report.")  # fmt: skip
@click.option("--eps", type=float, default=6.0, show_default=True, help="Hard inlier tolerance in px for the before/after report.")  # fmt: skip
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=Path("data/network_snapped.json"), show_default=True, help="Where to write the deformed lattice.")  # fmt: skip
@click.option("--overlay", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write an overlay image of the snapped projection over the plate.")  # fmt: skip
@click.option("--display-scale", type=float, default=0.5, show_default=True, help="Downsample factor for the overlay backdrop only.")  # fmt: skip
def main(
    network: Path,
    stars_path: Path,
    pose_in: Path,
    image: Path | None,
    top_n: int | None,
    focal_px: float | None,
    snap_gate: float,
    smooth: float,
    anchor: float,
    sigma: float,
    eps: float,
    output: Path,
    overlay: Path | None,
    display_scale: float,
) -> None:
    """Deform the lattice so matched neurons land on their stars, organically."""
    scene = load_scene(network, stars_path, top_n, focal_px)
    pose = Pose(**json.loads(pose_in.read_text())["pose"])
    star_tree = cKDTree(scene.stars)

    # Re-derive the assignment at the (generous) snap gate so stragglers count.
    rigid_pts = project(scene.neurons, pose, scene.focal_px)
    before = score_pose(rigid_pts, scene.stars, star_tree, sigma, eps, snap_gate)
    if not len(before.matches):
        raise click.ClickException(
            f"No neuron-star pairs within {snap_gate:g}px; nothing to snap."
        )

    smooth_edges = grid_neighbor_edges(network)
    new_neurons, shift = snap_neurons(
        scene, pose, before.matches, smooth_edges, smooth, anchor
    )

    snapped_scene = Scene(
        neurons=new_neurons,
        edges=scene.edges,
        stars=scene.stars,
        star_flux=scene.star_flux,
        width=scene.width,
        height=scene.height,
        focal_px=scene.focal_px,
    )
    snapped_pts = project(new_neurons, pose, scene.focal_px)
    after = score_pose(snapped_pts, scene.stars, star_tree, sigma, eps, snap_gate)

    # Residual on the originally-matched neurons after deformation.
    resid = np.hypot(*(snapped_pts[before.matches[:, 0]] - scene.stars[before.matches[:, 1]]).T)  # fmt: skip
    moved = np.hypot(*(new_neurons - scene.neurons).T[:2])
    click.echo(
        f"{len(scene.neurons)} neurons, {len(before.matches)} snapped "
        f"(gate {snap_gate:g}px)\n"
        f"  inliers(<{eps:g}px): {before.inliers} -> {after.inliers}    "
        f"soft: {before.soft:.2f} -> {after.soft:.2f}\n"
        f"  matched residual px: mean {resid.mean():.2f}  max {resid.max():.2f}\n"
        f"  3D displacement: mean {moved.mean():.3f}  max {moved.max():.3f} "
        f"(lattice units)"
    )

    write_snapped(network, output, new_neurons, pose, smooth)
    click.echo(f"Wrote {output}")

    if overlay is not None:
        from project import _load_plate

        plate = _load_plate(snapped_scene, image, stars_path)
        render_overlay(
            snapped_scene, pose, after, sigma, eps, plate, display_scale, overlay
        )
        click.echo(f"Wrote {overlay}")


if __name__ == "__main__":
    main()
