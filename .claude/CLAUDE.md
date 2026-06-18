# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

For a film shot, a 3D rendering of a small neural network blends into / fades out over a
static starry night sky. As the network fades away, (at least some of) its neurons should
sit _exactly_ on real stars, so the network's edges read as lines drawn **between actual
stars** — a constellation made of the network's connections.

## Engine & tooling

Python for solver work; Blender Python (`bpy`, Blender's bundled interpreter) for
render/export. `uv` manages the environment.

```sh
uv sync                      # create/refresh the venv from pyproject + uv.lock
uv add numpy scipy           # add a runtime dependency (writes pyproject + lock)
uv run python -m nn_starfield.solve   # run a module once code exists
uv run pytest                # run tests (once a suite exists)
uv run pytest path/to/test.py::test_name   # run a single test
```
