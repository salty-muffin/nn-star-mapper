"""Draw the network's edges as a clean plate-sized image, ready to composite.

Where ``project.py``/``snap.py`` render diagnostic overlays (stars, match links,
score text) on top of the plate, this is the *finished-look* renderer: it draws
nothing but the network's connections, projected through a solved pose, onto a
flat background -- at the original plate's exact pixel dimensions. Both the edge
colour and the background colour are full RGBA, so you can render glowing lines
on a transparent canvas and do the fade as a straight composite downstream.

The projection is identical to ``project.py`` (same pinhole, same pose), so point
the ``--network`` at the snapped lattice (``snap.py`` output) and ``--pose-in`` at
the pose it was snapped through, and every connection draws as a line between the
real stars its neurons now sit on -- the constellation, with no diagnostic
clutter. The edges are drawn with matplotlib's antialiased ``LineCollection`` so
they match the look of the previews.

    uv run python draw.py --network data/network_snapped.json \
        --pose-in data/pose_02.json --output data/drawing.png \
        --edge-color "0.6,0.85,1.0,0.9" --bg-color "0,0,0,0" --line-width 2
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from project import Pose, load_scene, project


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
    dpi: int = 100,
) -> None:
    """Draw the projected edges onto a flat RGBA background at plate resolution.

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

    pts = project(scene.neurons, pose, scene.focal_px)

    fig = plt.figure(figsize=(scene.width / dpi, scene.height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, scene.width)
    ax.set_ylim(scene.height, 0)  # y-down to match image coordinates
    ax.set_axis_off()

    # Background as a drawn rectangle so its RGBA (incl. alpha) is preserved;
    # the figure/axes patches are left transparent via savefig(transparent=True).
    ax.add_patch(
        Rectangle((0, 0), scene.width, scene.height, facecolor=bg_color, edgecolor="none", zorder=0)  # fmt: skip
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
@click.option("--pose-in", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Pose JSON to project through (project.py --pose-out).")  # fmt: skip
@click.option("--focal-px", type=float, default=None, help="Override focal length in pixels (default: value from stars JSON).")  # fmt: skip
@click.option("--edge-color", type=Color(), default="0.6,0.85,1.0,0.9", show_default=True, help="Edge RGBA: r,g,b[,a] (0-1 or 0-255), #rrggbbaa, or a colour name.")  # fmt: skip
@click.option("--bg-color", type=Color(), default="0,0,0,0", show_default=True, help="Background RGBA (default: fully transparent).")  # fmt: skip
@click.option("--line-width", type=float, default=2.0, show_default=True, help="Edge stroke width in output pixels.")  # fmt: skip
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=Path("data/drawing.png"), show_default=True, help="Where to write the rendered PNG.")  # fmt: skip
@click.option("--dpi", type=int, default=100, show_default=True, help="Render dpi; plate dimensions are held exact regardless.")  # fmt: skip
def main(
    network: Path,
    stars_path: Path,
    pose_in: Path,
    focal_px: float | None,
    edge_color: tuple[float, float, float, float],
    bg_color: tuple[float, float, float, float],
    line_width: float,
    output: Path,
    dpi: int,
) -> None:
    """Draw the network's connections onto a plate-sized RGBA canvas."""
    scene = load_scene(network, stars_path, None, focal_px)
    pose = Pose(**json.loads(pose_in.read_text())["pose"])

    render_edges(scene, pose, edge_color, bg_color, line_width, output, dpi)
    click.echo(
        f"{len(scene.edges)} edges over {scene.width}x{scene.height} -> {output}"
    )


if __name__ == "__main__":
    main()
