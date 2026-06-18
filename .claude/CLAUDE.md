# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

For a film shot, a 3D rendering of a small neural network blends into / fades out over a
static starry night sky. As the network fades away, (at least some of) its neurons should
sit _exactly_ on real stars in the sky, so that the network's edges read as lines drawn
**between actual stars** — like a constellation made of the network's connections.

## Engine & tooling

This project uses Python and Blender. Prefer Python for solver work and Blender Python (`bpy`) for render/export work. `uv` is used for the environment.

## Guidelines

- Treat this repository as a planning-first workspace for a Python solver plus Blender-based rendering pipeline.
- Start from [README.md](README.md) and [plan.md](plan.md); they are the source of truth for scope, algorithms, and open questions.
- Keep changes minimal and aligned with the plan; do not invent project structure, dependencies, or runtime assumptions that are not in the docs.
- Preserve `nn_starfield/netgen.py` as the canonical geometry source once code exists; do not duplicate neuron layout logic across modules.
- When implementing solver logic, respect the plan’s constraints: unique star assignments, score by inliers within tolerance, and avoid collapse-to-one-star behavior.
- If camera, sensor, plate, or focal-length metadata is missing, stop and surface that gap before hard-coding intrinsics.
- If you add setup or workflow details, link to existing docs instead of restating them here.
- Keep this file concise; update it only when a repo convention would otherwise be easy to miss.
