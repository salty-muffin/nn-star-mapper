# Neural Network → Starfield Alignment — Project Plan

> Pickup note for Claude Code: this is a spec carried over from a planning conversation.
> **Language: Python.** The alignment/solver tooling is Python; the final geometry is
> generated and rendered in Blender (also driven by a Python script). Read the whole
> plan before writing code — the "Pitfalls" and "Open questions" sections change the design.

---

## The idea

For a film shot, a 3D rendering of a small neural network blends into / fades out over a
static starry night sky. As the network fades away, (at least some of) its neurons should
sit *exactly* on real stars in the sky, so that the network's edges read as lines drawn
**between actual stars** — like a constellation made of the network's connections.

- **Network:** a simple MNIST-style feedforward net, visualized in 3D. ~3 hidden layers,
  max layer size 8×8 nodes. Neurons as points/spheres, edges as lines between layers.
- **Sky:** a 50mm view of the night sky, locked-off camera, **stars do not move**
  (no sidereal drift in the shot — treat it as a single static plate).
- **Effect:** network is rendered as an overlay that fades; on fade-out the lattice has
  been positioned/oriented so neurons coincide with star positions.

**Core technical problem:** find the camera **pose (rotation + translation)** and
**focal length** that project the rigid 3D neuron set onto real 2D star positions in the
frame — *without knowing in advance which neuron maps to which star* (unknown
correspondences).

---

## Inspirations / prior art (the named problem)

This is **pose estimation with unknown correspondences** — don't reinvent it.

- **Blind PnP / SoftPOSIT** (David, DeMenthon, et al.) — simultaneously solve pose and
  correspondence between a known 3D point set and a 2D image point set. This is exactly
  our case.
- **astrometry.net plate-solving** — matches star fields by hashing 4-star asterisms
  ("quads") into similarity-invariant features. We borrow the *matching trick*, not the
  full WCS/astrometric solve — we only need pixel positions of dots, not real RA/Dec.
- **Geometric hashing** — general technique behind the quad approach; robust 2D
  point-pattern matching under similarity transforms.
- **photutils (`DAOStarFinder`)** — astronomy-grade star centroid + flux detection;
  gives clean subpixel star positions from the plate.

---

## Why this is tractable (DOF analysis)

The full problem looks like "unknown camera + unknown correspondences," but it collapses:

1. **Focal length is essentially known.** 50mm on a known sensor → fixed FOV → fixed `f`
   in pixels at the video resolution. Keep it *slightly* free only at the final refine
   step to absorb calibration slop.
2. **Most pose DOF fall out of a 2D fit.** The projected *shape* of the lattice is governed
   almost entirely by the **out-of-plane tilt** (2 angles = direction the camera views the
   lattice from). In-plane rotation, image translation, and overall scale (≈ distance,
   since `f` is known) are just a **2D similarity transform** fitted *after* projecting.
3. **So the brute-force search is only ~2 DOF** (the tilt), plus optionally 1 for
   perspective strength if we want true perspective over weak-perspective. Everything else
   is solved analytically inside the loop.

Inner loop sketch:
```
for tilt in sample_orientations():          # 2 DOF sweep
    P = project(neurons, tilt, f)           # 2D template
    sim, inliers = match_2d(P, stars, eps)  # RANSAC pairs / quad-hash + Hungarian
    keep best by len(inliers)
refine (R, t, f) on best correspondences    # LM / CMA-ES, true perspective
```

---

## Approach / pipeline

### 1. Star detection (Python)
- Input: one representative frame (the locked plate).
- `photutils.DAOStarFinder` → subpixel centroids + flux. (OpenCV blob detection is a
  fallback; DAOStarFinder is cleaner.)
- Output: array of `(x, y, brightness)`. No WCS needed.
- Optionally keep only the **brightest N stars** as candidate targets — faster search and
  neurons land on stars that actually read on screen.

### 2. Neuron model (Python)
- Generate 3D neuron coordinates from the network spec (layers along one axis, grid per
  layer). Also generate the edge list (which neuron pairs connect).
- **Single source of truth:** the *same* generator must feed both the solver and the
  Blender build, so geometry never diverges. Factor it into a shared module.

### 3. Tilt sweep + 2D matcher (the solver)
- For each sampled orientation: project neurons → 2D template.
- Match template against detected stars to recover the best in-plane **similarity**
  (translation, rotation, scale) and count **inliers** (neurons within ε px of a *unique*
  star).
- Matcher options:
  - **Pair-RANSAC:** any template pair ↔ star pair fixes a similarity (4 DOF). Simple.
  - **Quad / triangle geometric hashing** (astrometry.net style): similarity-invariant
    hashes of 4-point asterisms; faster + more robust with many stars.
- **Hungarian assignment** for uniqueness so one bright star can't soak up many neurons.

### 4. Refine
- Take best hypothesis; refine continuous `(R, t, f)` on the established correspondences.
- Loss: Chamfer / matched-pair distance. **Levenberg–Marquardt** on correspondences, or
  **CMA-ES** if staying derivative-free. Recovers true perspective, tightens the fit.

### 5. Elastic snap (probably necessary — see Pitfalls)
- A perfectly rigid 8×8-style lattice will almost certainly **not** hit all stars on a
  random sky. After maximizing rigid inliers, **snap stragglers**: nudge each unmatched
  neuron to its nearest star within a tolerance, with light regularization so the lattice
  doesn't distort grotesquely. Small deviations read as organic; edges still clearly draw
  lines between stars — which is the whole effect. ("At least some of them" — lean into
  this.)

### 6. Blender export & render (Python in Blender)
- Set camera focal length / sensor to solved intrinsics at the video's resolution.
- Place the (possibly snapped) network at the solved pose.
- Render RGBA over black; do the fade as a composite in Resolve once projection matches.

---

## Pitfalls to design against
The naive objective "minimize sum of distances to nearest star" **cheats**. Guard with:
- **(a)** Unique assignments (Hungarian) — no many-neurons-to-one-star.
- **(b)** Maximize **inlier count within ε**, not raw distance.
- **(c)** Fix or lower-bound the projected extent so the solver can't collapse the lattice
  onto a dense cluster.
- **(d)** Optionally restrict targets to the brightest N stars.

---

## Two workflows — decide which (or do both)
- **Auto:** full tilt-sweep lets the computer find the orientation that genuinely maximizes
  real star matches. Most "honest" matches, less art control.
- **Hand-picked:** choose the oblique orientation you find most beautiful, fix it, and only
  solve 2D placement + elastic snap. Fast, total control over the look.
Recommended: prototype hand-picked first (fast feedback on the *look*), then add the
auto sweep.

---

## Suggested repo structure
```
nn_starfield/
  netgen.py        # SHARED neuron coords + edges from spec (truth source)
  detect_stars.py  # DAOStarFinder → (x,y,flux)
  project.py       # camera/projection math, weak-persp + true persp
  match.py         # tilt sweep, pair-RANSAC / quad-hash, Hungarian, scoring
  refine.py        # LM / CMA-ES polish on correspondences
  snap.py          # elastic snap of unmatched neurons + regularization
  solve.py         # orchestrates: detect → sweep → refine → snap → dump pose
  blender_build.py # runs inside Blender: consumes netgen + solved pose, renders
  data/            # plate frame(s), detected stars cache, solved pose json
```
Suggested libs: `numpy`, `scipy` (optimize, spatial.cKDTree, optimize.linear_sum_assignment),
`photutils` + `astropy`, optionally `opencv-python`, `cma` (if using CMA-ES). Blender uses
its bundled Python (`bpy`).

---

## Open questions to resolve at pickup
1. **Neuron count & topology.** Exact layer sizes (input/hidden/output) and total node
   count? This decides whether quad-hashing is worth it over plain pair-RANSAC. (Spec so
   far: ~3 hidden layers, max 8×8 = 64 at the widest point.)
2. **Which neurons must match?** All of them, only hidden layers, only "front face,"
   or just a chosen subset? Drives how aggressive the snap is.
3. **Edges vs. nodes priority.** Is the constellation read carried by *neurons-on-stars*
   or by *edges-between-stars*? If edges, we might match a subset and let lines do the work.
4. **Plate provenance.** Real sky photo, a stock/star-catalog render, or invented stars?
   If real, confirm camera/sensor so `f` in pixels is exact. If invented, we can *design*
   a sky the lattice fits perfectly and skip most of the solver.
5. **Static confirmed?** Assuming a single locked plate (no sidereal motion). If the stars
   *do* drift across the shot, the pose is time-varying and the plan grows a tracking step.
6. **Weak-persp vs. true persp.** Is the lattice shallow enough (small depth vs. distance)
   that weak-perspective is visually fine, or do we need full perspective in the refine?
7. **Look target.** Spheres or flat points for neurons? Glowing edges? This is mostly a
   Blender/Resolve concern but affects what "on a star" means visually (centroid alignment
   tolerance ε).
8. **Tolerance ε.** How many pixels counts as "on the star"? Tied to how big neurons render
   and how forgiving the snap should be.

---

## First concrete step at pickup
Start with `netgen.py` + `detect_stars.py` + a hand-picked-orientation `project.py`
prototype that overlays the projected lattice on the plate so you can *see* the alignment
problem before automating the search. Decide Q1 and Q4 first — they unblock everything else.
