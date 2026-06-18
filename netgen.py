"""Generate the 3D neuron lattice + edge list for the network.

This is the SHARED source of truth: the *same* geometry must feed both the
solver (which projects it onto detected stars) and the Blender build (which
renders it). To stay importable everywhere -- including Blender's bundled
interpreter -- the core generator is pure Python with no third-party deps. The
JSON it writes is the interchange format; the solver wraps the neuron list in a
numpy array, Blender reads it to place objects.

A network is a stack of layers along the depth (Z) axis. Each layer is a grid
of neurons in the X-Y plane. Neurons in adjacent layers are densely connected
(every node to every node), which reads as the classic feedforward fan of
lines. The whole lattice is centered on the origin.

Layer spec: a comma list of grid shapes, e.g. ``8x8,6x6,6x6,4x4,2x5``. A bare
integer ``N`` means an N-tall single column (``Nx1``).

    uv run python netgen.py
    uv run python netgen.py --layers 8x8,6x6,4x4 --preview data/network_preview.png
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class Neuron:
    """One node, with both its global id and its (layer, row, col) address."""

    id: int
    layer: int
    row: int
    col: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class LayerInfo:
    """Where a layer's neurons live in the flat neuron list."""

    index: int
    rows: int
    cols: int
    start: int  # first global neuron id in this layer
    count: int


@dataclass(frozen=True)
class Network:
    neurons: list[Neuron]
    layers: list[LayerInfo]
    edges: list[tuple[int, int]]  # (id, id) pairs, densely between adjacent layers


def parse_layers(text: str) -> list[tuple[int, int]]:
    """Parse ``"8x8,6x6,5"`` into ``[(8, 8), (6, 6), (5, 1)]``.

    Each token is ``ROWSxCOLS``; a bare integer ``N`` is shorthand for ``Nx1``.
    """
    shapes: list[tuple[int, int]] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            if "x" in token.lower():
                r_str, c_str = token.lower().split("x")
                rows, cols = int(r_str), int(c_str)
            else:
                rows, cols = int(token), 1
        except ValueError as exc:
            raise click.BadParameter(
                f"Bad layer token {token!r}; use ROWSxCOLS (e.g. 8x8) or N."
            ) from exc
        if rows < 1 or cols < 1:
            raise click.BadParameter(f"Layer {token!r} must have positive dimensions.")
        shapes.append((rows, cols))
    if not shapes:
        raise click.BadParameter("No layers parsed.")
    return shapes


def generate(
    shapes: list[tuple[int, int]],
    layer_spacing: float = 1.0,
    neuron_spacing: float = 1.0,
) -> Network:
    """Build the centered 3D lattice and dense inter-layer edges.

    Layers are stacked along Z at ``layer_spacing`` apart; within a layer the
    grid is spaced ``neuron_spacing`` apart in X (cols) and Y (rows). The whole
    network is centered on the origin.
    """
    n_layers = len(shapes)
    z0 = (n_layers - 1) / 2.0

    neurons: list[Neuron] = []
    layers: list[LayerInfo] = []
    for li, (rows, cols) in enumerate(shapes):
        start = len(neurons)
        z = (li - z0) * layer_spacing
        cx = (cols - 1) / 2.0
        cy = (rows - 1) / 2.0
        for r in range(rows):
            for c in range(cols):
                neurons.append(
                    Neuron(
                        id=len(neurons),
                        layer=li,
                        row=r,
                        col=c,
                        x=(c - cx) * neuron_spacing,
                        y=(r - cy) * neuron_spacing,
                        z=z,
                    )
                )
        layers.append(LayerInfo(li, rows, cols, start, len(neurons) - start))

    edges: list[tuple[int, int]] = []
    for a, b in zip(layers, layers[1:]):
        for i in range(a.start, a.start + a.count):
            for j in range(b.start, b.start + b.count):
                edges.append((i, j))

    return Network(neurons=neurons, layers=layers, edges=edges)


def to_dict(net: Network) -> dict:
    return {
        "layers": [asdict(l) for l in net.layers],
        "neurons": [asdict(n) for n in net.neurons],
        "edges": [list(e) for e in net.edges],
    }


def save_preview(net: Network, path: Path) -> None:
    """Write a 3D preview of neurons + edges (needs matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise click.ClickException("Preview needs matplotlib.") from exc

    # Plot axes are (depth, x, y) = (z, x, y); use this same ordering for both
    # the nodes and the edge segments so lines actually connect the dots.
    def plot_xyz(n: Neuron) -> tuple[float, float, float]:
        return (n.z, n.x, n.y)

    px = [p[0] for p in map(plot_xyz, net.neurons)]
    py = [p[1] for p in map(plot_xyz, net.neurons)]
    pz = [p[2] for p in map(plot_xyz, net.neurons)]

    segments = [
        (plot_xyz(net.neurons[i]), plot_xyz(net.neurons[j])) for i, j in net.edges
    ]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Line3DCollection(segments, colors="tab:blue", linewidths=0.3, alpha=0.3)
    )
    ax.scatter(px, py, pz, c="white", edgecolors="black", s=20, depthshade=False)
    ax.set_xlabel("z (depth)")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    # Equal scale on every axis: box aspect proportional to the data extents,
    # so one unit is the same visual length on all three axes.
    spans = [max(a) - min(a) or 1.0 for a in (px, py, pz)]
    ax.set_box_aspect(spans)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option("--layers", default="8x8,6x6,6x6,4x4,2x5", show_default=True, help="Comma list of layer grids, e.g. 8x8,6x6,4x4 (bare N means Nx1).")  # fmt: skip
@click.option("--layer-spacing", type=float, default=1.0, show_default=True, help="Distance between adjacent layers along the depth axis.")  # fmt: skip
@click.option("--neuron-spacing", type=float, default=1.0, show_default=True, help="In-plane spacing between neurons within a layer.")  # fmt: skip
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=Path("data/network.json"), show_default=True, help="Where to write the network geometry JSON.")  # fmt: skip
@click.option("--preview", type=click.Path(dir_okay=False, path_type=Path),  default=None, help="Optional path for a 3D preview image.")  # fmt: skip
def main(
    layers: str,
    layer_spacing: float,
    neuron_spacing: float,
    output: Path,
    preview: Path | None,
) -> None:
    """Generate the neuron lattice + edges and write to JSON."""
    shapes = parse_layers(layers)
    net = generate(shapes, layer_spacing=layer_spacing, neuron_spacing=neuron_spacing)
    click.echo(
        f"{len(shapes)} layers, {len(net.neurons)} neurons, {len(net.edges)} edges."
    )

    payload = {
        "spec": {
            "layers": layers,
            "layer_spacing": layer_spacing,
            "neuron_spacing": neuron_spacing,
        },
        **to_dict(net),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    click.echo(f"Wrote {output}")

    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        save_preview(net, preview)
        click.echo(f"Wrote {preview}")


if __name__ == "__main__":
    main()
