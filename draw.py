"""Draw the network's edges as a clean plate-sized image, ready to composite.

Where ``project.py``/``snap.py`` render diagnostic overlays (stars, match links,
score text) on top of the plate, this is the *finished-look* renderer: it draws
nothing but the network's connections, projected through a solved pose, onto a
flat background -- at the original plate's exact pixel dimensions. Both the edge
colour and the background colour are full RGBA, so you can render glowing lines
on a transparent canvas and do the fade as a straight composite downstream.

The projection is identical to ``project.py`` (same pinhole, same pose). Point
``--network`` at the snapped lattice (``snap.py`` output) and every connection
draws as a line between the real stars its neurons now sit on -- the
constellation, with no diagnostic clutter. The snapped file already records the
pose it was snapped through, so ``--pose-in`` is optional; pass it only to draw a
different lattice (e.g. the rigid ``netgen.py`` one) through a separate pose
JSON. Edges use matplotlib's antialiased ``LineCollection`` to match the previews.

    uv run python draw.py --network data/network_snapped.json \
        --output data/drawing.png \
        --edge-color "0.6,0.85,1.0,0.9" --bg-color "0,0,0,0" --line-width 2
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from project import Pose, load_scene, project


# --------------------------------------------------------------------------- #
# Pose loading
# --------------------------------------------------------------------------- #
def load_pose(data: dict) -> Pose | None:
    """Pull a Pose from a JSON payload, tolerating either layout.

    ``project.py --pose-out`` writes the pose at top level (``{"pose": {...}}``),
    while ``snap.py`` records the pose it snapped through under its provenance
    block (``{"snap": {"pose": {...}}}``). Accept both so the snapped network
    file can double as the pose source. Returns None if neither is present.
    """
    if isinstance(data.get("pose"), dict):
        return Pose(**data["pose"])
    snap = data.get("snap")
    if isinstance(snap, dict) and isinstance(snap.get("pose"), dict):
        return Pose(**snap["pose"])
    return None


# --------------------------------------------------------------------------- #
# Colour parsing
# --------------------------------------------------------------------------- #
class Color(click.ParamType):
    """An RGBA colour: ``r,g,b[,a]`` (0-1 or 0-255), a hex string, or a name.

    Comma form is the most explicit -- e.g. ``0.6,0.85,1,0.9`` or
    ``153,217,255,230``; if any component exceeds 1 the whole tuple is taken as
    0-255 and rescaled. Anything else (``#rrggbbaa``, ``#rgb``, ``cyan``) is
    passed to matplotlib's colour parser. The result is always an ``(r,g,b,a)``
    tuple in 0-1.
    """

    name = "color"

    def convert(self, value, param, ctx) -> tuple[float, float, float, float]:
        if isinstance(value, tuple):
            return value
        s = str(value).strip()
        if "," in s:
            try:
                parts = [float(p) for p in s.split(",")]
            except ValueError:
                self.fail(f"{value!r}: non-numeric component in comma colour", param, ctx)  # fmt: skip
            if len(parts) not in (3, 4):
                self.fail(f"{value!r}: expected 3 or 4 comma-separated numbers", param, ctx)  # fmt: skip
            if max(parts) > 1.0:  # treat as 0-255
                parts = [p / 255.0 for p in parts]
            if len(parts) == 3:
                parts.append(1.0)
            return (parts[0], parts[1], parts[2], parts[3])
        from matplotlib.colors import to_rgba

        try:
            return tuple(to_rgba(s))
        except ValueError:
            self.fail(f"{value!r} is not a recognised colour", param, ctx)


class Resolution(click.ParamType):
    """A target output size: ``WIDTH`` or ``WIDTHxHEIGHT`` (e.g. ``1920x1080``).

    Returns ``(width, height|None)``. With only a width, the height is left to
    the caller to derive from the plate's aspect ratio, so the drawing keeps its
    proportions; with both, the canvas extent is taken verbatim while the content
    is still scaled uniformly (by the width ratio), never stretched.
    """

    name = "resolution"

    def convert(self, value, param, ctx) -> tuple[int, int | None]:
        if isinstance(value, tuple):
            return value
        s = str(value).strip().lower().replace(" ", "")
        parts = s.split("x") if "x" in s else [s]
        try:
            dims = [int(p) for p in parts]
        except ValueError:
            self.fail(f"{value!r}: expected WIDTH or WIDTHxHEIGHT integers", param, ctx)  # fmt: skip
        if any(d <= 0 for d in dims):
            self.fail(f"{value!r}: dimensions must be positive", param, ctx)
        if len(dims) == 1:
            return (dims[0], None)
        if len(dims) == 2:
            return (dims[0], dims[1])
        self.fail(f"{value!r}: expected WIDTH or WIDTHxHEIGHT", param, ctx)


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def render_edges(
    scene,
    pose: Pose,
    edge_color: tuple[float, float, float, float],
    bg_color: tuple[float, float, float, float],
    line_width: float,
    out_path: Path,
    out_width: int,
    out_height: int,
    dpi: int = 100,
) -> None:
    """Draw the projected edges onto a flat RGBA background at ``out_width`` x ``out_height``.

    The pose is solved against the full plate, so projected points are in plate
    pixels; they are scaled uniformly by ``out_width / scene.width`` so the
    drawing keeps the same position and size *proportionally* at any output
    resolution (never stretched -- the same factor applies to both axes).

    ``line_width`` is in *output pixels* (converted to points internally so the
    stroke is the requested width regardless of ``dpi``). The background is laid
    down as an explicit full-canvas rectangle so its alpha is honoured: the
    figure itself is saved transparent, and anything the rectangle doesn't cover
    (and any edge alpha) carries straight through to the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    scale = out_width / scene.width
    pts = project(scene.neurons, pose, scene.focal_px) * scale

    fig = plt.figure(figsize=(out_width / dpi, out_height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, out_width)
    ax.set_ylim(out_height, 0)  # y-down to match image coordinates
    ax.set_axis_off()

    # Background as a drawn rectangle so its RGBA (incl. alpha) is preserved;
    # the figure/axes patches are left transparent via savefig(transparent=True).
    ax.add_patch(
        Rectangle((0, 0), out_width, out_height, facecolor=bg_color, edgecolor="none", zorder=0)  # fmt: skip
    )

    lw_points = line_width * 72.0 / dpi  # pixels -> points at this dpi
    segs = pts[scene.edges]
    ax.add_collection(
        LineCollection(segs, colors=[edge_color], linewidths=lw_points, zorder=1)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--network", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("data/network.json"), show_default=True, help="Neuron geometry JSON; point at snap.py output for the finished on-stars look.")  # fmt: skip
@click.option("--stars", "stars_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("data/stars.json"), show_default=True, help="Detected-star JSON; supplies focal length and plate dimensions.")  # fmt: skip
@click.option("--pose-in", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Pose JSON to project through (default: read the pose from the snapped --network file).")  # fmt: skip
@click.option("--focal-px", type=float, default=None, help="Override focal length in pixels (default: value from stars JSON).")  # fmt: skip
@click.option("--edge-color", type=Color(), default="0.6,0.85,1.0,0.9", show_default=True, help="Edge RGBA: r,g,b[,a] (0-1 or 0-255), #rrggbbaa, or a colour name.")  # fmt: skip
@click.option("--bg-color", type=Color(), default="0,0,0,0", show_default=True, help="Background RGBA (default: fully transparent).")  # fmt: skip
@click.option("--line-width", type=float, default=2.0, show_default=True, help="Edge stroke width in output pixels.")  # fmt: skip
@click.option("--resolution", type=Resolution(), default=None, help="Output size WIDTH or WIDTHxHEIGHT (default: plate size); content scales proportionally.")  # fmt: skip
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=Path("data/drawing.png"), show_default=True, help="Where to write the rendered PNG.")  # fmt: skip
@click.option("--dpi", type=int, default=100, show_default=True, help="Render dpi; output dimensions are held exact regardless.")  # fmt: skip
def main(
    network: Path,
    stars_path: Path,
    pose_in: Path,
    focal_px: float | None,
    edge_color: tuple[float, float, float, float],
    bg_color: tuple[float, float, float, float],
    line_width: float,
    resolution: tuple[int, int | None] | None,
    output: Path,
    dpi: int,
) -> None:
    """Draw the network's connections onto a plate-sized RGBA canvas."""
    scene = load_scene(network, stars_path, None, focal_px)

    # The pose comes from --pose-in if given, else from the snap block written
    # into the network file by snap.py -- so a snapped lattice is self-contained.
    source = pose_in if pose_in is not None else network
    pose = load_pose(json.loads(source.read_text()))
    if pose is None:
        raise click.ClickException(
            f"No pose found in {source}. Pass --pose-in pointing at a "
            "project.py --pose-out JSON, or a snapped --network from snap.py."
        )

    # Default to the plate size; a width-only --resolution preserves aspect.
    if resolution is None:
        out_width, out_height = scene.width, scene.height
    else:
        out_width, out_height = resolution
        if out_height is None:
            out_height = round(scene.height * out_width / scene.width)

    render_edges(
        scene, pose, edge_color, bg_color, line_width, output, out_width, out_height, dpi
    )
    click.echo(f"{len(scene.edges)} edges over {out_width}x{out_height} -> {output}")


if __name__ == "__main__":
    main()
