"""Detect stars in a locked-off night-sky plate.

Runs photutils' DAOStarFinder over a single image and writes subpixel star
positions plus a relative brightness (``flux``) to JSON. No astrometry / WCS is
involved -- we only need pixel coordinates of the dots so the solver can land
neurons on them.

``flux`` here is DAOStarFinder's background-subtracted, kernel-normalized summed
intensity: a *relative, linear* brightness (bigger = brighter). It is not a
calibrated magnitude. We use it only to rank stars and optionally keep the
brightest N.

Focal length does not affect detection. It is accepted purely so the solver
downstream can read ``f`` (in pixels) from one place; it is echoed into the
output metadata and never used here.

    uv run python detect_stars.py data/plate.tif
    uv run python detect_stars.py data/plate.tif --top-n 200 --preview data/stars_preview.png
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import imageio.v3 as iio
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

# Rec. 709 luminance weights (perceptual grayscale from RGB).
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def load_luminance(path: Path) -> np.ndarray:
    """Read an image as a 2D float32 luminance array.

    Handles grayscale, RGB and RGBA inputs of any bit depth. An alpha channel,
    if present, is dropped.
    """
    img = np.asarray(iio.imread(path)).astype(np.float32)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] >= 3:
            return img[..., :3] @ _LUMA
        return img[..., 0]
    raise click.ClickException(f"Unsupported image shape {img.shape} from {path}")


def detect(
    image: np.ndarray,
    fwhm: float,
    threshold_sigma: float,
) -> np.ndarray:
    """Return an (N, 3) array of (x, y, flux), brightest first.

    Background level/spread are estimated with sigma-clipped statistics and the
    detection threshold is set ``threshold_sigma`` standard deviations above the
    background.
    """
    _, median, std = sigma_clipped_stats(image, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = finder(image - median)
    if sources is None:
        return np.empty((0, 3), dtype=np.float64)

    stars = np.column_stack(
        [sources["x_centroid"], sources["y_centroid"], sources["flux"]]
    )
    # Brightest first so --top-n is a simple slice.
    return stars[np.argsort(stars[:, 2])[::-1]]


def save_preview(image: np.ndarray, stars: np.ndarray, path: Path) -> None:
    """Write an annotated preview marking each detected star (needs matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise click.ClickException("Preview needs matplotlib.") from exc

    vmin, vmax = ZScaleInterval().get_limits(image)
    dpi = 100
    h, w = image.shape
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    ax.scatter(
        stars[:, 0],
        stars[:, 1],
        s=80,
        facecolors="none",
        edgecolors="red",
        linewidths=0.8,
    )
    ax.set_axis_off()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))  # fmt: skip
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=Path("data/stars.json"), show_default=True, help="Where to write the detected-star JSON.")  # fmt: skip
@click.option("--fwhm", type=float, default=3.0, show_default=True, help="Expected star FWHM in pixels (DAOStarFinder kernel size).")  # fmt: skip
@click.option("--threshold-sigma", type=float, default=5.0, show_default=True, help="Detection threshold, in std-devs above the background.")  # fmt: skip
@click.option("--top-n", type=int, default=None, help="Keep only the brightest N stars (default: keep all).")  # fmt: skip
@click.option("--focal-length", type=float, default=None, help="Lens focal length in mm. Metadata only -- not used for detection.")  # fmt: skip
@click.option("--sensor-width", type=float, default=36.0, show_default=True, help="Sensor width in mm, used with --focal-length to record f in pixels.")  # fmt: skip
@click.option("--preview", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional path for an annotated preview image.")  # fmt: skip
def main(
    image_path: Path,
    output: Path,
    fwhm: float,
    threshold_sigma: float,
    top_n: int | None,
    focal_length: float | None,
    sensor_width: float,
    preview: Path | None,
) -> None:
    """Detect stars in IMAGE_PATH and write (x, y, flux) to JSON."""
    image = load_luminance(image_path)
    height, width = image.shape

    stars = detect(image, fwhm=fwhm, threshold_sigma=threshold_sigma)
    total = len(stars)
    if top_n is not None:
        stars = stars[:top_n]
    click.echo(f"Detected {total} stars; keeping {len(stars)}.")

    focal_length_px = None
    if focal_length is not None:
        focal_length_px = focal_length / sensor_width * width

    result = {
        "image": str(image_path),
        "width": width,
        "height": height,
        "params": {
            "fwhm": fwhm,
            "threshold_sigma": threshold_sigma,
            "top_n": top_n,
        },
        "focal_length_mm": focal_length,
        "sensor_width_mm": sensor_width if focal_length is not None else None,
        "focal_length_px": focal_length_px,
        "stars": [
            {"x": float(x), "y": float(y), "flux": float(f)} for x, y, f in stars
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    click.echo(f"Wrote {output}")

    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        save_preview(image, stars, preview)
        click.echo(f"Wrote {preview}")


if __name__ == "__main__":
    main()
