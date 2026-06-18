"""Project the 3D neuron lattice onto the plate and overlay it on the stars.

This is the hand-picked-orientation prototype from the project plan: it loads the
shared neuron geometry (``netgen.py`` output) and the detected stars
(``detect_stars.py`` output), projects the rigid lattice through a *known*
pinhole camera at a hand-set pose, and draws the result over the plate so you
can *see* the alignment problem before any automated search.

Camera model
------------
The lens is known (``f`` in pixels comes from the stars JSON), so we use a full
pinhole projection -- not weak perspective. Intrinsics are fixed: focal length
``f`` and principal point at the image centre. The only free parameters are the
*pose* of the lattice in front of the camera:

* ``tilt-az`` / ``tilt-el`` -- the two out-of-plane tilt angles (which oblique
  direction the camera views the lattice from). These dominate the projected
  *shape*.
* ``roll`` -- in-plane rotation.
* ``pos-x`` / ``pos-y`` -- where the lattice centre lands in the image (pixels).
* ``distance`` -- depth of the lattice centre; the single on-screen-scale knob
  (``f`` is fixed, so distance alone sets how big the lattice reads). The
  lattice's depth/aspect is baked into the geometry by ``netgen.py`` and is not
  a knob here.

Scoring
-------
For a given pose we assign each neuron to at most one star (Hungarian, so a
bright star cannot soak up many neurons) and score the fit with a *soft,
saturating* reward ``exp(-(d/sigma)**2)`` summed over matched pairs. Because the
reward saturates near 1, several near-misses beat a single exact hit -- which is
the look we want (edges reading as lines between stars). A hard inlier count
(neurons within ``eps`` px of their star) is reported alongside for legibility.

    uv run python project.py --interactive
    uv run python project.py --tilt-el 35 --roll 12 --distance 18 \
        --pos-x 3024 --pos-y 1701 --pose-out data/pose.json
    uv run python project.py --optimize --overlay data/overlay.png
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    """Everything the projector needs, pulled from the two JSON files."""

    neurons: np.ndarray  # (N, 3) float, lattice-local coordinates (x, y, z)
    edges: np.ndarray  # (E, 2) int, neuron-id pairs
    stars: np.ndarray  # (M, 2) float, star pixel coordinates (x, y)
    star_flux: np.ndarray  # (M,) float, relative brightness
    width: int
    height: int
    focal_px: float


def load_scene(
    network_path: Path,
    stars_path: Path,
    top_n: int | None,
    focal_override: float | None,
) -> Scene:
    """Load and align the network geometry and detected stars.

    ``top_n`` keeps only the brightest stars as candidate targets (faster, and
    neurons land on stars that actually read on screen). ``focal_override`` wins
    over the value recorded in the stars JSON, for absorbing calibration slop.
    """
    net = json.loads(network_path.read_text())
    neurons = np.array(
        [[n["x"], n["y"], n["z"]] for n in net["neurons"]], dtype=np.float64
    )
    edges = np.array(net["edges"], dtype=np.int64)

    det = json.loads(stars_path.read_text())
    stars = np.array([[s["x"], s["y"]] for s in det["stars"]], dtype=np.float64)
    flux = np.array([s["flux"] for s in det["stars"]], dtype=np.float64)
    # detect_stars already writes brightest-first, but don't rely on it.
    order = np.argsort(flux)[::-1]
    stars, flux = stars[order], flux[order]
    if top_n is not None:
        stars, flux = stars[:top_n], flux[:top_n]

    focal = focal_override if focal_override is not None else det.get("focal_length_px")
    if not focal:
        raise click.ClickException(
            "No focal length: pass --focal-px or run detect_stars with --focal-length."
        )

    return Scene(
        neurons=neurons,
        edges=edges,
        stars=stars,
        star_flux=flux,
        width=int(det["width"]),
        height=int(det["height"]),
        focal_px=float(focal),
    )


# --------------------------------------------------------------------------- #
# Pose + projection
# --------------------------------------------------------------------------- #
@dataclass
class Pose:
    """Hand-set lattice pose. Angles in degrees, positions/distance in pixels.

    ``pos_x``/``pos_y`` default to the image centre when built via
    :meth:`centred`. ``distance`` is in the same units as the lattice
    coordinates' implied depth; with ``f`` fixed it is the on-screen-scale knob.
    """

    tilt_az: float = 0.0
    tilt_el: float = 0.0
    roll: float = 0.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    distance: float = 12.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.tilt_az,
                self.tilt_el,
                self.roll,
                self.pos_x,
                self.pos_y,
                self.distance,
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> Pose:
        return cls(*[float(x) for x in v])


def _rotation(tilt_az: float, tilt_el: float, roll: float) -> np.ndarray:
    """Build a rotation matrix from azimuth, elevation and roll (degrees).

    The out-of-plane tilt is one rotation about an axis lying *in* the image
    plane -- ``tilt_el`` about the horizontal image axis X, ``tilt_az`` about the
    vertical image axis Y -- via the exponential map (Rodrigues). Doing the two
    tilts as a single axis-angle rotation rather than chained Euler angles avoids
    gimbal lock: rotation about the horizontal axis stays available at any
    azimuth (chained Euler collapses ``el`` into ``roll`` at ``az = 90``).
    ``roll`` then spins the result in the image plane about the view axis Z.
    """
    el, az, ro = np.radians([tilt_el, tilt_az, roll])
    w = np.array([el, az, 0.0])  # tilt rotation vector: X=elevation, Y=azimuth
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        r_tilt = np.eye(3)
    else:
        k = w / theta
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        r_tilt = np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)
    cz, sz = np.cos(ro), np.sin(ro)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ r_tilt


def format_flags(pose: Pose) -> str:
    """Render a pose as the exact CLI flag string to paste back into a run."""
    return (
        f"--tilt-az {pose.tilt_az:g} --tilt-el {pose.tilt_el:g} --roll {pose.roll:g} "
        f"--pos-x {pose.pos_x:.1f} --pos-y {pose.pos_y:.1f} --distance {pose.distance:g}"
    )


def copy_to_clipboard(text: str) -> str | None:
    """Copy ``text`` to the system clipboard; return the tool used, or None.

    Tries the common Linux clipboard CLIs (Wayland then X11). The caller also
    prints the string, so a missing tool degrades gracefully to copy-by-hand.
    """
    import shutil
    import subprocess

    candidates = [
        ("wl-copy", ["wl-copy"]),
        ("xclip", ["xclip", "-selection", "clipboard"]),
        ("xsel", ["xsel", "--clipboard", "--input"]),
    ]
    for tool, args in candidates:
        if shutil.which(tool):
            try:
                subprocess.run(
                    args,
                    input=text.encode(),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return tool
            except Exception:  # noqa: BLE001 - any failure falls through to next tool
                continue
    return None


def project(neurons: np.ndarray, pose: Pose, focal_px: float) -> np.ndarray:
    """Project lattice-local neurons to image pixels via a fixed pinhole.

    The lattice is rotated, then its centre placed at depth ``distance`` along
    the camera's view axis (+Z, away from the camera). Pixels use the image
    convention with y increasing downwards and the principal point at
    (``pos_x``, ``pos_y``) -- matching detect_stars' ``origin="upper"`` plate.
    """
    r = _rotation(pose.tilt_az, pose.tilt_el, pose.roll)
    cam = neurons @ r.T
    cam[:, 2] += pose.distance
    z = np.maximum(cam[:, 2], 1e-6)  # guard against the lattice crossing the lens
    u = pose.pos_x + focal_px * cam[:, 0] / z
    v = pose.pos_y + focal_px * cam[:, 1] / z
    return np.column_stack([u, v])


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class Score:
    soft: float  # sum of exp(-(d/sigma)^2) over matched pairs
    inliers: int  # matched pairs within eps px
    matches: np.ndarray  # (K, 2) int, (neuron_idx, star_idx)
    distances: np.ndarray  # (K,) px, matched-pair distances


def score_pose(
    points: np.ndarray,
    stars: np.ndarray,
    star_tree: cKDTree,
    sigma: float,
    eps: float,
    gate: float,
) -> Score:
    """Soft-score a projected template against the stars with unique matches.

    Each neuron is a candidate for the few stars within ``gate`` px (a kd-tree
    query keeps this cheap with thousands of stars); the best one-to-one
    assignment is found with the Hungarian algorithm maximizing the saturating
    reward ``exp(-(d/sigma)^2)``. Neurons with no star inside ``gate`` simply go
    unmatched -- not every neuron must find a star.
    """
    # Candidate (neuron, star) pairs: only stars within the gate matter, since
    # the reward is ~0 beyond a few sigma anyway.
    neigh = star_tree.query_ball_point(points, r=gate)
    rows, cols, rewards = [], [], []
    for ni, stars_near in enumerate(neigh):
        for si in stars_near:
            d = float(np.hypot(*(points[ni] - stars[si])))
            rows.append(ni)
            cols.append(si)
            rewards.append(np.exp(-((d / sigma) ** 2)))
    if not rows:
        return Score(0.0, 0, np.empty((0, 2), int), np.empty(0))

    rows = np.array(rows)
    cols = np.array(cols)
    rewards = np.array(rewards)

    # Hungarian over a compact dense reward matrix on just the involved rows/cols.
    uniq_n, n_idx = np.unique(rows, return_inverse=True)
    uniq_s, s_idx = np.unique(cols, return_inverse=True)
    reward_mat = np.zeros((len(uniq_n), len(uniq_s)))
    reward_mat[n_idx, s_idx] = rewards
    r_sel, c_sel = linear_sum_assignment(reward_mat, maximize=True)

    matches, dists, soft = [], [], 0.0
    for r_i, c_i in zip(r_sel, c_sel):
        if reward_mat[r_i, c_i] <= 0:
            continue  # assignment padding, not a real candidate pair
        ni, si = int(uniq_n[r_i]), int(uniq_s[c_i])
        d = float(np.hypot(*(points[ni] - stars[si])))
        matches.append((ni, si))
        dists.append(d)
        soft += float(np.exp(-((d / sigma) ** 2)))

    dists = np.array(dists)
    inliers = int(np.count_nonzero(dists <= eps)) if len(dists) else 0
    return Score(soft, inliers, np.array(matches, dtype=int).reshape(-1, 2), dists)


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def optimize_pose(
    scene: Scene,
    start: Pose,
    star_tree: cKDTree,
    sigma: float,
    eps: float,
    gate: float,
) -> Pose:
    """Locally maximise the soft score from a hand-picked starting pose.

    Hand-pick the orientation, then let Nelder-Mead snap the placement/scale to
    land more neurons. Derivative-free because the Hungarian assignment makes the
    score piecewise-flat. Angles, pixel offsets and distance live on very
    different scales, so the simplex is seeded with per-parameter steps.
    """

    def neg_score(v: np.ndarray) -> float:
        pose = Pose.from_vector(v)
        pts = project(scene.neurons, pose, scene.focal_px)
        return -score_pose(pts, scene.stars, star_tree, sigma, eps, gate).soft

    x0 = start.as_vector()
    steps = np.array([5.0, 5.0, 5.0, 50.0, 50.0, max(1.0, 0.1 * start.distance)])
    simplex = np.vstack([x0, x0 + np.diag(steps)])
    res = minimize(
        neg_score,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": simplex,
            "xatol": 1e-2,
            "fatol": 1e-4,
            "maxiter": 4000,
        },
    )
    return Pose.from_vector(res.x)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _load_plate(scene: Scene, image_path: Path | None, stars_path: Path):
    """Best-effort load of the plate image for the backdrop; None if missing."""
    if image_path is None:
        recorded = json.loads(stars_path.read_text()).get("image")
        image_path = Path(recorded) if recorded else None
    if image_path is None or not image_path.exists():
        return None
    import imageio.v3 as iio

    return np.asarray(iio.imread(image_path))


def render_overlay(
    scene: Scene,
    pose: Pose,
    score: Score,
    sigma: float,
    eps: float,
    plate,
    display_scale: float,
    out_path: Path,
) -> None:
    """Draw the projected lattice + matches over the plate (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    pts = project(scene.neurons, pose, scene.focal_px)

    dpi = 100
    fig = plt.figure(figsize=(scene.width / dpi, scene.height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, scene.width)
    ax.set_ylim(scene.height, 0)  # y-down to match image coordinates
    ax.set_axis_off()

    if plate is not None:
        from astropy.visualization import ZScaleInterval

        disp = plate
        if disp.ndim == 3:
            disp = disp[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        if display_scale != 1.0:
            step = max(1, int(round(1 / display_scale)))
            disp = disp[::step, ::step]
        vmin, vmax = ZScaleInterval().get_limits(disp)
        ax.imshow(
            disp,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=(0, scene.width, scene.height, 0),
            zorder=0,
        )
    else:
        ax.scatter(scene.stars[:, 0], scene.stars[:, 1], s=6, c="0.5", zorder=0)

    # Edges as a faint fan so the dense lattice stays readable.
    segs = pts[scene.edges]
    ax.add_collection(
        LineCollection(segs, colors="tab:cyan", linewidths=0.5, alpha=0.5, zorder=1)
    )

    matched = set(int(n) for n in score.matches[:, 0]) if len(score.matches) else set()
    is_matched = np.array([i in matched for i in range(len(pts))])
    ax.scatter(
        pts[~is_matched, 0],
        pts[~is_matched, 1],
        s=14,
        facecolors="none",
        edgecolors="tab:orange",
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(pts[is_matched, 0], pts[is_matched, 1], s=14, c="lime", zorder=3)

    # A line from each matched neuron to its star, green when inside eps.
    if len(score.matches):
        link = np.stack([pts[score.matches[:, 0]], scene.stars[score.matches[:, 1]]], 1)
        colors = ["lime" if d <= eps else "yellow" for d in score.distances]
        ax.add_collection(LineCollection(link, colors=colors, linewidths=0.6, zorder=2))

    ax.text(
        12,
        28,
        f"soft={score.soft:.1f}  inliers<{eps:g}px={score.inliers}/{len(pts)}"
        f"  sigma={sigma:g}",
        color="white",
        fontsize=12,
        family="monospace",
        bbox=dict(facecolor="black", alpha=0.6, pad=4),
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run_interactive(
    scene: Scene, pose: Pose, star_tree, sigma: float, eps: float, gate: float, plate
) -> Pose:
    """Open a live slider view to drag the pose over the plate.

    Returns the final pose so it can still be written out on close. Falls back
    with a clear message if no interactive matplotlib backend is available.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.widgets import Button, Slider

    if plt.get_backend().lower() == "agg":
        raise click.ClickException(
            "No interactive matplotlib backend; drop --interactive and use CLI flags."
        )

    state = {"pose": pose}
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.40)
    ax.set_xlim(0, scene.width)
    ax.set_ylim(scene.height, 0)
    ax.set_aspect("equal")

    if plate is not None:
        from astropy.visualization import ZScaleInterval

        disp = plate
        if disp.ndim == 3:
            disp = disp[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        vmin, vmax = ZScaleInterval().get_limits(disp)
        ax.imshow(
            disp,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=(0, scene.width, scene.height, 0),
        )
    else:
        ax.scatter(scene.stars[:, 0], scene.stars[:, 1], s=6, c="0.5")

    edge_coll = LineCollection([], colors="tab:cyan", linewidths=0.4, alpha=0.3)
    ax.add_collection(edge_coll)
    (node_plot,) = ax.plot([], [], "o", mfc="none", mec="tab:orange", ms=4)
    (hit_plot,) = ax.plot([], [], "o", color="lime", ms=4)
    title = ax.set_title("")

    specs = [
        ("tilt_az", -180, 180),
        ("tilt_el", -180, 180),
        ("roll", -180, 180),
        ("pos_x", 0, scene.width),
        ("pos_y", 0, scene.height),
        ("distance", 1, max(60, pose.distance * 3)),
    ]
    sliders = {}
    for i, (name, lo, hi) in enumerate(specs):
        sax = plt.axes((0.12, 0.02 + i * 0.045, 0.78, 0.03))
        sliders[name] = Slider(sax, name, lo, hi, valinit=getattr(pose, name))

    def redraw(_=None) -> None:
        p = Pose(**{n: sliders[n].val for n in sliders})
        state["pose"] = p
        pts = project(scene.neurons, p, scene.focal_px)
        sc = score_pose(pts, scene.stars, star_tree, sigma, eps, gate)
        edge_coll.set_segments(pts[scene.edges])
        matched = set(int(n) for n in sc.matches[:, 0]) if len(sc.matches) else set()
        m = np.array([i in matched for i in range(len(pts))])
        node_plot.set_data(pts[~m, 0], pts[~m, 1])
        hit_plot.set_data(pts[m, 0], pts[m, 1])
        title.set_text(f"soft={sc.soft:.1f}  inliers<{eps:g}px={sc.inliers}/{len(pts)}")
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(redraw)

    # Copy the current pose as a ready-to-paste CLI flag string.
    copy_ax = plt.axes((0.12, 0.31, 0.30, 0.045))
    copy_btn = Button(copy_ax, "copy flags to clipboard")

    def on_copy(_event) -> None:
        flags = format_flags(state["pose"])
        tool = copy_to_clipboard(flags)
        note = f"copied via {tool}" if tool else "no clipboard tool found; copy below"
        print(f"{flags}\n  [{note}]")
        copy_btn.label.set_text("copied!" if tool else "copy failed - see terminal")
        fig.canvas.draw_idle()

    copy_btn.on_clicked(on_copy)

    redraw()
    plt.show()
    return state["pose"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--network", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("data/network.json"), show_default=True, help="Neuron geometry JSON from netgen.py.")  # fmt: skip
@click.option("--stars", "stars_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("data/stars.json"), show_default=True, help="Detected-star JSON from detect_stars.py.")  # fmt: skip
@click.option("--image", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Plate image for the backdrop (default: path recorded in the stars JSON).")  # fmt: skip
@click.option("--top-n", type=int, default=None, help="Use only the brightest N stars as targets (default: all).")  # fmt: skip
@click.option("--focal-px", type=float, default=None, help="Override focal length in pixels (default: value from stars JSON).")  # fmt: skip
@click.option("--tilt-az", type=float, default=0.0, show_default=True, help="Out-of-plane azimuth tilt (degrees).")  # fmt: skip
@click.option("--tilt-el", type=float, default=0.0, show_default=True, help="Out-of-plane elevation tilt (degrees).")  # fmt: skip
@click.option("--roll", type=float, default=0.0, show_default=True, help="In-plane rotation (degrees).")  # fmt: skip
@click.option("--pos-x", type=float, default=None, help="Image x of the lattice centre, px (default: image centre).")  # fmt: skip
@click.option("--pos-y", type=float, default=None, help="Image y of the lattice centre, px (default: image centre).")  # fmt: skip
@click.option("--distance", type=float, default=12.0, show_default=True, help="Depth of the lattice centre; the on-screen-scale knob.")  # fmt: skip
@click.option("--sigma", type=float, default=8.0, show_default=True, help="Soft-match falloff in px; several near-misses beat one exact hit.")  # fmt: skip
@click.option("--eps", type=float, default=6.0, show_default=True, help="Hard inlier tolerance in px (for the reported count).")  # fmt: skip
@click.option("--gate", type=float, default=None, help="Max neuron-star pairing distance in px (default: 4*sigma).")  # fmt: skip
@click.option("--pose-in", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Seed pose from a previously written pose JSON.")  # fmt: skip
@click.option("--optimize", is_flag=True, help="Locally maximise the soft score from the (hand-set) start pose.")  # fmt: skip
@click.option("--interactive", is_flag=True, help="Open a live slider window to drag the pose.")  # fmt: skip
@click.option("--overlay", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write a static overlay image of the projection over the plate.")  # fmt: skip
@click.option("--pose-out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write the final pose + score to JSON for downstream steps.")  # fmt: skip
@click.option("--display-scale", type=float, default=0.5, show_default=True, help="Downsample factor for the overlay backdrop only (projection stays full-res).")  # fmt: skip
def main(
    network: Path,
    stars_path: Path,
    image: Path | None,
    top_n: int | None,
    focal_px: float | None,
    tilt_az: float,
    tilt_el: float,
    roll: float,
    pos_x: float | None,
    pos_y: float | None,
    distance: float,
    sigma: float,
    eps: float,
    gate: float | None,
    pose_in: Path | None,
    optimize: bool,
    interactive: bool,
    overlay: Path | None,
    pose_out: Path | None,
    display_scale: float,
) -> None:
    """Project the neuron lattice onto the plate and overlay it on the stars."""
    scene = load_scene(network, stars_path, top_n, focal_px)
    if gate is None:
        gate = 4.0 * sigma

    # Build the starting pose: pose-in seeds it; an explicitly passed flag
    # overrides that field; otherwise fall back to the flag default (and the
    # image centre as the default landing point).
    ctx = click.get_current_context()

    def given(name: str) -> bool:
        return ctx.get_parameter_source(name).name == "COMMANDLINE"

    if pose_in is not None:
        pose = Pose(**json.loads(pose_in.read_text())["pose"])
    else:
        pose = Pose()
    for name, value in [
        ("tilt_az", tilt_az),
        ("tilt_el", tilt_el),
        ("roll", roll),
        ("pos_x", pos_x),
        ("pos_y", pos_y),
        ("distance", distance),
    ]:
        if pose_in is None or given(name):
            setattr(pose, name, value)
    if pose.pos_x is None:
        pose.pos_x = scene.width / 2.0
    if pose.pos_y is None:
        pose.pos_y = scene.height / 2.0

    star_tree = cKDTree(scene.stars)

    if interactive:
        plate = _load_plate(scene, image, stars_path)
        pose = run_interactive(scene, pose, star_tree, sigma, eps, gate, plate)

    if optimize:
        before = score_pose(project(scene.neurons, pose, scene.focal_px), scene.stars, star_tree, sigma, eps, gate)  # fmt: skip
        pose = optimize_pose(scene, pose, star_tree, sigma, eps, gate)
        after = score_pose(project(scene.neurons, pose, scene.focal_px), scene.stars, star_tree, sigma, eps, gate)  # fmt: skip
        click.echo(
            f"optimise: soft {before.soft:.2f} -> {after.soft:.2f}, "
            f"inliers {before.inliers} -> {after.inliers}"
        )

    pts = project(scene.neurons, pose, scene.focal_px)
    score = score_pose(pts, scene.stars, star_tree, sigma, eps, gate)
    click.echo(
        f"{len(scene.neurons)} neurons, {len(scene.stars)} stars | "
        f"soft={score.soft:.2f}  inliers(<{eps:g}px)={score.inliers}  "
        f"matched={len(score.matches)}"
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
            "score": {
                "soft": score.soft,
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
