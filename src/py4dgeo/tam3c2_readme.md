# Time-Adaptive M3C2 (TAM3C2)

A time-adaptive extension of the M3C2 (Multiscale Model-to-Model Cloud Comparison) algorithm for sparse, multi-temporal LiDAR / point cloud time series.

This document explains the **mathematical formulation**, and **implementation details** of `py4dgeo/tam3c2.py`, line by line where relevant. It is written so that someone can understand what the algorithm does and *why each piece is there*.

---

## 1. Why time-adaptive?

Standard M3C2 (Lague et al., 2013) computes the signed distance between two epochs of a point cloud along a locally estimated surface normal:

1. For each *corepoint* `c`, estimate a normal `n` from a **spherical** neighborhood of radius `r_normal` in the reference cloud. （PCA)
2. Project all reference and target points lying inside a cylinder (radius `cyl_radius`, length `2·max_distance`, axis `n`) onto `n`.
3. Distance = `mean(target_projection) − mean(reference_projection)`.
4. Level of Detection (LoD₉₅) is derived from the spread of the two projections and a registration error.

This works well when **each epoch contains enough points** in every neighborhood. It breaks down when:

- The sensor is sparse (e.g., a single ULS pass per day, partial coverage).
- The point cloud has occlusions or shadows that vary between epochs.
- The surface is small relative to the scan-line spacing.

In those cases, an isolated epoch's neighborhood may contain only a handful of points — too few to robustly estimate either a normal or a stable mean projection. The variance of the M3C2 mean blows up and the resulting time series becomes noisy or full of NaNs.

**Key idea of TAM3C2.** When the time axis is densely sampled (e.g., 365 daily epochs), neighboring epochs in time can be *borrowed* to supplement a sparse single-epoch neighborhood. We aggregate points from `t ± Δt` until we have enough points (`required_points`), capped at a maximum time window. This happens in two distinct phases:

- **Normal Estimation (once per reference):** Normals are estimated from a time-aggregated **spherical** neighborhood centered on the reference time. This is a one-time, multi-scale "contest" to find the most stable normal per corepoint.
- **Distance Calculation (for each target):** M3C2 means / LoD are computed from a time-aggregated **cylindrical** neighborhood, whose axis is the stable normal found in the previous step.

The trade-off is **temporal smoothing** vs. **spatial completeness**. TAM3C2 controls this trade-off with two ideas:

- **Per-target windowing:** The maximum half-window scales with the time gap between reference and target, `max_window = |Δt_target| · max_window_ratio`. Targets close to the reference borrow little; targets far away can borrow more. Both sides use the *same* per-target cap.
- **Multi-scale selection:** `normal_radii` and `max_window_ratio` may each be a list. The combination that yields the **most planar** aggregated reference neighborhood is chosen *per corepoint*, once.

---

## 2. Where TAM3C2 plugs into py4dgeo

```
SpatiotemporalAnalysis
   .reference_epoch = ref
   .corepoints      = corepoints
   .m3c2            = TAM3C2(...)        # <-- M3C2LikeAlgorithm subclass
   .add_epochs(*others)                  # calls m3c2.calculate_distances(ref, t_k) for each k
```

TAM3C2 inherits from `py4dgeo.m3c2.M3C2LikeAlgorithm`, which means it conforms to the same interface as the built-in M3C2:

- `calculate_distances(epoch1, epoch2)` returns `(distances, uncertainties)` for the configured corepoints.
- The `uncertainties` array is a *structured ndarray* with fields `lodetection`, `spread1`, `num_samples1`, `spread2`, `num_samples2` — same schema py4dgeo's downstream tools (smoothing, 4D-OBC region growing) expect.

Each call to `add_epochs(target_k)` triggers exactly one call to `TAM3C2.calculate_distances(ref, target_k)`. The result becomes column *k* of `analysis.distances` of shape `(n_corepoints, n_targets)`.

---

## 3. Top-level data flow

```
calculate_distances(ref, target):
    # STEP 1: One-time multi-scale contest to find the best normal for each corepoint.
    # This runs only the first time `calculate_distances` is called for a given reference.
    1. _ensure_ref_scale_cache(ref)
         # For each corepoint, test every combination of the provided scales.
         For every corepoint, for every (normal_radius, window_ratio) combo:
           # Aggregate points in a SPHERE, also time-adaptive, centered on t_ref.
           # This uses a large time window to give all scales a fair chance.
           aggregate a SPHERE on the REF side using a full-range window;
           compute planarity from PCA on that sphere.
         # The combo that yields the most planar surface wins.
         Keep the combo that maximizes planarity per corepoint.
         # Store the winning normal and scale choice for later use.
         Store normals[i] and opt_scale_idx[i].

    # STEP 2: For the given target, calculate distances using the pre-selected best scales.
    # This runs for every target epoch.
    2. time_gap = |t_target - t_ref|
       For each corepoint i:
         # Retrieve the "winner" scale combination for this corepoint from the cache.
         (sr, wr)    = scale_combinations[opt_scale_idx[i]]
         max_window = time_gap * wr

         # Aggregate points in a CYLINDER, using the cached best normal as the axis.
         # This is done independently for the reference and target time centers.
         aggregate a CYLINDER on REF side  (around t_ref, axis = normals[i])
         aggregate a CYLINDER on TARGET side (around t_target, axis = normals[i])

         # If enough points were found, compute the final M3C2 distance.
         If both sides have >= required_points:
             compute weights w_ref, w_tgt (NONE / LINEAR / GAUSSIAN)
             dist, uncertainty = _m3c2_and_lod(...)
             distances[i] = dist
             uncertainties[i] = uncertainty
         Else:
             leave NaN

    # STEP 3 & 4: Store diagnostics and return results.
    3. Append per-target diagnostics arrays to self._diag.
    4. Return distances, uncertainties.
```

---

## 4. Component details

### 4.1 Spherical aggregation — `_aggregate_sphere(...)`

This function is used for the **normal estimation step**, which is performed as part of the multi-scale selection process (see Section 4.4).

**Why a sphere?** To robustly estimate a surface normal with PCA, we need an *isotropic* neighborhood—one that extends equally in all directions from the corepoint. This ensures that the underlying surface patch is sampled without directional bias, allowing PCA to accurately find the plane of best fit. A sphere is the ideal shape for this.

Algorithm:

1. Sort all epoch indices by `|epoch_time − center_time|` → `order`. This is the "greedy nearest-first in time" rule.
2. Walk `order`:
   - Skip `exclude_idx` (typically the index of the reference epoch itself, so it is not double-counted in subsequent steps — though actually here it is included; `exclude_idx` is `None` in current scale-selection use).
   - If `|Δt| > max_window`, break (sorted, so all remaining are also out).
   - Run `cKDTree.query_ball_point(cp, radius)` against that epoch's cloud, then prune by exact squared distance.
   - Append the points and book-keeping arrays (Δt, source epoch index).
   - Count this epoch as `n_before` or `n_after` depending on the sign of Δt.
   - Stop as soon as the cumulative point count `total >= required_points`.

Returns the stacked arrays plus four diagnostics (`n_before`, `n_after`, `w_used` = the largest `|Δt|` actually used).

**Why "greedy nearest-first"?** It minimizes the average temporal distance of the aggregated points: we always prefer points closer in time to the center. This makes the temporal weighting (Section 4.5) more meaningful, as points from distant epochs are only included when closer ones are insufficient.

### 4.2 Cylindrical aggregation — `_aggregate_cylinder(...)`

This function is used for the **actual M3C2 distance computation**, gathering points from both the reference and target time periods.

**Why a cylinder?** Once a stable normal vector is determined, the goal of M3C2 is to measure change *along that specific direction*. A cylinder is the perfect *anisotropic* shape for this task. It includes all points within a given radius `cyl_radius` of the normal axis, effectively creating a "core sample" of the point cloud that is relevant to the measurement direction, while ignoring points that are far away perpendicular to the normal.

Same outer loop as the sphere version, but the per-epoch geometry test is the M3C2 cylinder:

- Pre-fetch candidates with a `cKDTree` ball query of radius `√(cyl_radius² + max_distance²)` (bounding sphere of the cylinder).
- For each candidate `p`, compute
  - along-axis projection `along = (p − cp) · n`,
  - perpendicular distance squared `perp² = ||p − cp||² − along²`.
- Keep `p` iff `perp² ≤ cyl_radius²` and `|along| ≤ max_distance`.

Cylinder orientation reuses the reference normal selected in step 4.4, so reference and target are projected onto the **same axis** — this is essential for M3C2 to compare like-with-like.

### 4.3 PCA normal & planarity — `_planarity_and_normal(pts)`

Standard plane-fit via covariance eigen-decomposition:

- Center points: `X = pts − mean(pts)`.
- Covariance: `C = (XᵀX) / N`.
- Eigenvalues sorted ascending: `λ₀ ≤ λ₁ ≤ λ₂`.
- **Normal** = eigenvector of `λ₀` (smallest), flipped to point along `orientation_vector` (default `[0,0,1]`).
- **Planarity** (Demantké et al. 2011 definition):

  $$
  \mathrm{planarity} = \frac{\lambda_1 - \lambda_0}{\lambda_2}
  $$

  Higher = the point set is closer to a flat patch. Used as the per-corepoint scale selection score in 4.4.

Edge case: if `len(pts) < 3`, return `(-inf, orientation_vector)` — that scale will not be picked unless every scale fails.

### 4.4 Scale selection — `_ensure_ref_scale_cache(ref_epoch)`

This is the heart of the multi-scale approach. It runs **only once** per reference epoch and acts as a "contest" to find the most suitable spatial scale (`normal_radius`) and temporal scale (`max_window_ratio`) for *each corepoint individually*. The results are cached for all subsequent distance calculations involving that reference epoch.

It builds:
- `_ref_normals`: shape `(n_cp, 3)`, the chosen normal per corepoint.
- `_opt_scale_idx`: shape `(n_cp,)`, the index into `_scale_combinations` identifying the winning combo.
- `_scale_combinations`: list of `(spatial_radius, window_ratio)` pairs — the full Cartesian product of `normal_radii × max_window_ratio`.

**Contest Logic:** For each corepoint `i` and each scale combination `(sr, wr)`:
1. `pts = _aggregate_sphere(corepoints[i], t_ref, ref_idx, sr, wr·time_range)`
2. `planarity, normal = _planarity_and_normal(pts)`
3. The `planarity` score is recorded. After all combos are tested, the one with the highest score wins.
4. The `normal` and scale index `(sr, wr)` from the winning combo are stored in the cache for corepoint `i`.

**Rationale for `time_range`:** For the scale selection contest only, `max_window` is set from the **total time range** of the entire dataset (`epochs_timeseries`), not the per-target gap. This gives every scale combination access to the maximum possible temporal context during the planarity competition. A combination with a large `window_ratio` is not artificially handicapped simply because the first target epoch happens to be close to the reference. It ensures a fair and consistent competition to find the truly best intrinsic scale for the surface at that corepointnge)`
2. `planarity, normal = _planarity_and_normal(pts)`
3. If `planarity > best[i]`: update `best[i]`, `normals[i]`, `opt_idx[i]`.

Corepoints that never collect ≥3 points (so `best` stays `-inf`) fall back to the `orientation_vector` and the default scale index 0; their distances will likely be NaN in practice.

**What if only one scale is provided?** If you provide a single `normal_radii` and a single `max_window_ratio`, the `_scale_combinations` list will contain only one element. The multi-scale "contest" becomes a "no-contest" election: this single combination is chosen for all corepoints. The algorithm gracefully handles this, effectively behaving as a single-scale method.

### 4.5 Weighting — `_compute_weights(dt_array, pts, cp, window_size, spatial_r)`

Three options selected by `Weighting`:

- **`NONE`** — returns `None`; downstream code treats this as unweighted means and variances.

- **`LINEAR`** — triangular kernels in both space and time:

  $$
  w_s = \max\!\bigl(0,\, 1 - \tfrac{\lVert p-c \rVert}{r_{\text{spatial}}}\bigr), \qquad
  w_t = \max\!\bigl(0,\, 1 - \tfrac{|\Delta t|}{W}\bigr), \qquad
  w = w_s \cdot w_t.
  $$

- **`GAUSSIAN`** — isotropic Gaussian in **rescaled** normalized coordinates $(u_s, u_t/r_{st})$ with $u_s = \lVert p-c\rVert/r_{\text{spatial}}$ and $u_t = |\Delta t|/W$:

  $$
  w_s = \exp\!\left(-\,\frac{1}{2\sigma^2}\,u_s^{\,2}\right), \qquad
  w_t = \exp\!\left(-\,\frac{1}{2\sigma^2}\Bigl(\tfrac{u_t}{r_{st}}\Bigr)^{2}\right), \qquad
  w = w_s \cdot w_t.
  $$

  Here `σ = sigma_ratio` (kernel width in units of the normalized radius/window), and `r_st = space_time_ratio` is the **spacetime anisotropy ratio**:
  * `r_st = 1` — symmetric trust in space and time.
  * `r_st > 1` — temporal kernel decays *slower* than the spatial one → more temporal aggregation → lower LoD, higher risk of mixing real change into the spread.
  * `r_st < 1` — temporal kernel decays *faster* → closer to single-epoch M3C2.

  See §7.1 for an automated, data-driven way to pick `r_st`.

If the weights collapse to all-zero (e.g., numerical edge case), they are reset to all-ones to avoid division by zero downstream.

### 4.6 Weighted distance and LoD — `_m3c2_and_lod(cp, normal, ref_pts, ref_w, tgt_pts, tgt_w)`

First, project both sides onto the normal:

$$
p_r = (\text{ref\_pts} - c)\cdot n, \qquad p_t = (\text{tgt\_pts} - c)\cdot n
$$

For the **unweighted** case (`ref_w is None`):

$$
m_r = \overline{p_r},\quad m_t = \overline{p_t},\quad
s_r = \sigma(p_r),\quad s_t = \sigma(p_t),\quad
n_r = |p_r|,\quad n_t = |p_t|.
$$

For the **weighted** case:

$$
m_r = \frac{\sum w_r\,p_r}{\sum w_r}, \qquad
s_r = \sqrt{\frac{\sum w_r (p_r - m_r)^2}{\sum w_r}}.
$$

Effective sample size (Kish):

$$
n_r = N_{\text{eff}}^{(r)} = \frac{\bigl(\sum w_r\bigr)^2}{\sum w_r^{\,2}}.
$$

Same expressions for the target side with `w_t, p_t`. Effective sample size collapses to the raw count when all weights are equal, and shrinks when a few points dominate.

The **signed M3C2 distance** and the **Level of Detection at 95%**:

$$
\boxed{\;\mathrm{dist} = m_t - m_r\;}
$$

$$
\boxed{\;\mathrm{LoD}_{95} = 1.96 \cdot \sqrt{\frac{s_r^{2}}{n_r} + \frac{s_t^{2}}{n_t}} + e_{\text{reg}}\;}
$$

where `e_reg = registration_error` is added as a constant offset (independent of point count, modeling rigid co-registration uncertainty).

Output `uncertainty` is a one-row structured record with fields `(lodetection, spread1, num_samples1, spread2, num_samples2)` so that py4dgeo's downstream code (which expects the same fields as classic M3C2) works without modification.

### 4.7 Main loop — `calculate_distances(epoch1, epoch2)`

1. Ensures the scale cache is built (only does work the first time).
2. Computes `time_gap = |t_target − t_ref|`. If this is zero (degenerate), falls back to the full time range so the per-target window does not collapse to zero.
3. Pre-allocates output arrays full of NaN.
4. Per corepoint:
   - Look up the chosen `(spatial_radius, window_ratio)` and form `max_window = time_gap · window_ratio`.
   - Aggregate ref-cylinder and tgt-cylinder *with the SAME `max_window`*. This is Design B: both sides get the same per-target cap, so neither side "spreads more" in time than the other for a given target.
   - Validate: both sides must have at least `required_points` aggregated points.
   - If valid: compute weights, run `_m3c2_and_lod`, store.
   - Else: leave NaN (downstream smoothing / OBC code is NaN-aware).
5. Append diagnostics: `window_used_*`, `n_before_*`, `n_after_*`, `n_points_*` arrays, plus the target timestamp.
6. (Optional) if `keep_neighborhoods=True`, store the full aggregated point sets and weights — useful for plotting / debugging but memory-heavy on large grids.

The result `(distances, uncertainties)` is returned to `SpatiotemporalAnalysis`, which assigns it to one column of its persistent storage.

### 4.8 Diagnostics

`diagnostics()` returns a dict where every per-target field is a `(n_cp, n_targets)` matrix obtained via `np.column_stack(list_of_columns)`. Fields:

| key                  | meaning                                                          |
| -------------------- | ---------------------------------------------------------------- |
| `target_timestamps`  | python list of `datetime`, length `n_targets`                    |
| `window_used_ref`    | largest `|Δt|` actually used on the ref side                     |
| `window_used_tgt`    | same, target side                                                |
| `n_before_ref/tgt`   | number of distinct **epochs** contributing from before `t_*`     |
| `n_after_ref/tgt`    | number of distinct **epochs** contributing from after `t_*`      |
| `n_points_ref/tgt`   | total aggregated **point** count per side                        |
| `scale_idx`          | `(n_cp,)` — chosen scale-combination index per corepoint         |

`save_diagnostics(path)` serializes everything (timestamps as ISO strings) to a single `.npz`.

---

## 5. Notation table

| Symbol            | Code variable                            | Meaning                                                 |
| ----------------- | ---------------------------------------- | ------------------------------------------------------- |
| `c`               | `cp`                                     | Corepoint position (R³)                                 |
| `n`               | `normal`, `self._ref_normals[i]`         | Local surface normal                                    |
| `t_ref, t_tgt`    | `ref_time, tgt_time`                     | Unix-time of reference / target epoch                   |
| `Δt`              | `dt`, `dt_array`                         | `epoch_time − center_time`                              |
| `W`               | `max_window`                             | Per-target half-window for aggregation, seconds         |
| `r_spatial`       | `spatial_r` / `sr`                       | Spherical normal radius for current corepoint           |
| `wr`              | `window_ratio` / `wr`                    | Selected per-corepoint window ratio                     |
| `N_eff`           | `n_r`, `n_t` (weighted branch)           | Kish effective sample size                              |
| `s_r, s_t`        | `s_r`, `s_t`                             | Weighted (or unweighted) std of along-normal projection |
| `m_r, m_t`        | `m_r`, `m_t`                             | Weighted (or unweighted) mean of along-normal projection|
| `e_reg`           | `self.registration_error`                | Constant co-registration uncertainty added to LoD       |
| `required_points` | `self.required_points`                   | Minimum points required per side                        |
| planarity         | `pl` in `_planarity_and_normal`          | `(λ₁ − λ₀)/λ₂`                                          |

---

## 6. Helper functions (outside the class)

### `read_epochs_from_folder(folder, suffixes=(".las",".laz",".xyz"))`

Reads every file in `folder` whose extension matches `suffixes` and whose filename contains a timestamp `YYMMDD` or `YYMMDD_HHMMSS` (regex `_TS_RE`). Year is interpreted as `2000 + YY`. Returns a list of `py4dgeo.Epoch` sorted by `.timestamp`. Files without a parseable timestamp are skipped with an info log.

### `extract_reference_and_others(epochs, reference_timestamp)`

Splits a sorted epoch list into the single reference epoch (matched by `==` on `timestamp`) and the rest. Raises `ValueError` if no match.

### `sample_corepoints(epoch, method, voxel_size=…, n_samples=…, seed=0)`

Two modes:

- `"voxel"`: hash points into voxels of side `voxel_size`, return the **centroid** of each non-empty voxel. The hash uses three large primes XORed together — purely a fast group-by primitive, not a cryptographic hash.
- `"random"`: uniformly choose `n_samples` points without replacement.

Returns a `(N, 3)` ndarray suitable for `SpatiotemporalAnalysis.corepoints`.

---

## 7. Worked example (matches the notebook in `jupyter/test_S1_uls_tam3c2.ipynb`)

```python
from datetime import datetime
import py4dgeo
from py4dgeo import (
    TAM3C2, Weighting,
    read_epochs_from_folder, extract_reference_and_others, sample_corepoints,
)
from py4dgeo.segmentation import temporal_averaging

# 1. Load 365 daily epochs
epochs = read_epochs_from_folder(r"C:\...\S1_uls_downsampled")

# 2. Pick reference and (optionally) crop to a sub-range
ref_t = datetime(2020, 1, 5)
sorted_eps = sorted(epochs, key=lambda e: e.timestamp)
ref_idx = next(i for i, e in enumerate(sorted_eps) if e.timestamp == ref_t)
epochs = sorted_eps[:ref_idx + 1 + 120]                    # ref + 120 days after
ref, others = extract_reference_and_others(epochs, ref_t)

# 3. Corepoints (voxel-centered) on the reference cloud
corepoints = sample_corepoints(ref, method="voxel", voxel_size=1.5)

# 4. Build TAM3C2 — multi-scale on BOTH normal radius and window ratio
tam = TAM3C2(
    epochs_timeseries=epochs,
    max_window_ratio=[0.2, 0.3],
    normal_radii=[0.5, 1.0, 1.5],
    required_points=10,
    weighting=Weighting.GAUSSIAN,
    sigma_ratio=1.0,
    space_time_ratio=1.0,             # see §7.1 for selection
    keep_neighborhoods=False,
    corepoints=corepoints,
    cyl_radius=1.0,
    max_distance=5.0,
    registration_error=0.02,
)

# 5. Plug into SpatiotemporalAnalysis and compute
analysis = py4dgeo.SpatiotemporalAnalysis("S1_uls_tam3c2.zip", force=True)
analysis.reference_epoch = ref
analysis.corepoints      = corepoints
analysis.m3c2            = tam
analysis.add_epochs(*others)            # one call per target -> calculate_distances

# 6. Temporal smoothing + (downstream) 4D-OBC segmentation
analysis.smoothed_distances = temporal_averaging(
    analysis.distances, smoothing_window=3
)

# 7. Inspect aggregation behavior
diag = tam.diagnostics()
print(diag["n_points_ref"].shape)       # (n_corepoints, n_targets)
tam.save_diagnostics("S1_uls_tam3c2_diag.npz")
```

After step 5, `analysis.distances` has shape `(n_corepoints, n_targets)` and `analysis.uncertainties` has the matching structured-dtype array. From there, any standard py4dgeo workflow (4D-OBC region growing, plotting per-corepoint time series, etc.) applies unchanged.

### 7.1 Choosing `space_time_ratio` — `sweep_space_time_ratio` and `estimate_space_time_ratio`

`space_time_ratio` is the most impactful TAM3C2 parameter. Increasing it widens the *effective* temporal aggregation window and lowers LoD; pushing it too far lets real surface evolution leak into the M3C2 within-neighbourhood spread that the LoD formula uses. The module ships a data-driven selector that walks this trade-off using only quantities M3C2 already computes (no ground truth required).

**Why we look at the spread on stable corepoints.** The within-neighbourhood spread that M3C2 reports (`uncertainties.spread1` / `spread2`) has two contributions: the genuine geometric roughness of the surface inside the neighbourhood — which is exactly what the M3C2 LoD formula is designed to absorb — and any actual surface motion that the temporal aggregation pulls into the same neighbourhood. On a corepoint that does not change over the whole time series, the second contribution should stay close to zero, *unless* the temporal weighting is generous enough to drag in microscopic drift. Watching how the mean spread on stable corepoints evolves as `space_time_ratio` grows therefore isolates this contamination, without needing any ground-truth change information.

**Selection rule.** Call *stable-spread* the mean of `spread1` and `spread2` taken over the set of stable corepoints (those whose maximum absolute distance across all targets stays below `stable_threshold`) and over all evaluated targets. Compute it for every candidate ratio, take the smallest candidate ratio as the **baseline** (it does the least temporal smoothing, so its stable-spread approximates the pure geometric-noise floor), and pick the **largest** candidate ratio whose stable-spread is at most `(1 + tolerance)` times that baseline (default `tolerance = 0.1`, i.e. at most 10% above the baseline). If no candidate qualifies, the candidate with the smallest stable-spread is returned and `constraint_satisfied = False` is flagged.

**`sweep_space_time_ratio(ratios, *, epochs_timeseries, reference_epoch, target_epochs, corepoints, tam_kwargs=None)`** — low-level driver. For every ratio in `ratios` it builds a fresh `TAM3C2(space_time_ratio=r, **tam_kwargs)` and calls `calculate_distances` on every target, returning a list of dicts with the full `(n_cp, n_targets)` matrices of distances, `spread1` / `spread2`, `lod95`, `num_samples1` / `num_samples2`.

**`estimate_space_time_ratio(*, epochs_timeseries, reference_epoch, target_epochs, corepoints, tam_kwargs=None, candidate_ratios=(0.5, 1.0, 2.0, 4.0), tolerance=0.1, stable_threshold=0.05)`** — runs the sweep, derives the stable mask from the smallest-ratio run, computes the stable-spread for every candidate, applies the selection rule above, and returns

| key | meaning |
|---|---|
| `best_ratio` | the chosen `space_time_ratio` |
| `report` | list[dict] with `space_time_ratio`, `rho` (= stable-spread divided by the baseline stable-spread), `mean_spread`, `mean_lod95`, `mean_num_samples` |
| `sweep_results` | raw output of `sweep_space_time_ratio` (for plotting per-corepoint maps) |
| `stable_mask` | bool array of corepoints used in the baseline |
| `baseline_ratio` | the smallest candidate ratio (used as the stable-spread reference) |
| `baseline_mean_spread` | stable-spread at `baseline_ratio` |
| `constraint_satisfied` | `True` iff at least one candidate stayed within `1 + tolerance` of the baseline |
| `tolerance`, `stable_threshold` | echoed inputs |

Minimal usage:

```python
from py4dgeo import estimate_space_time_ratio

tam_kwargs = dict(
    max_window_ratio=max_window_ratio,
    normal_radii=normal_radii,
    required_points=required_points,
    weighting=Weighting.GAUSSIAN,
    sigma_ratio=sigma_ratio,
    cyl_radius=cyl_radius,
    max_distance=max_distance,
    registration_error=registration_error,
)

result = estimate_space_time_ratio(
    epochs_timeseries=epochs,
    reference_epoch=reference_epoch,
    target_epochs=other_epochs[:30],   # subset for speed
    corepoints=corepoints,
    tam_kwargs=tam_kwargs,
    candidate_ratios=(0.5, 1.0, 2.0, 4.0),
    tolerance=0.1,
    stable_threshold=0.05,             # = obc_height_threshold in the main pipeline
)

print(result["best_ratio"], result["constraint_satisfied"])
for r in result["report"]:
    print(f"  r_st={r['space_time_ratio']:.2f}  rho={r['rho']:.3f}  "
          f"mean LoD95={r['mean_lod95']:.4f}  N_eff={r['mean_num_samples']:.1f}")
```

A worked example with diagnostic plots (relative stable-spread, mean LoD$_{95}$, mean $N_{\text{eff}}$ as functions of `space_time_ratio`) lives in `jupyter/space_time_ratio_analysis.ipynb`.

> **Tip — pick targets with non-trivial time gaps.** `max_window = |t_target − t_ref| · max_window_ratio`. Targets very close to the reference (gap ≲ one inter-epoch step) yield empty cylinders and all-NaN spreads, which `estimate_space_time_ratio` then has to ignore via `nanmean`. Prefer evenly spaced targets across the full time range, e.g. `np.linspace(0, len(others)-1, 30, dtype=int)`.

---

## 8. Why the design choices are the way they are

- **Aggregation order = nearest in time first.** Maximizes information content per point added; also makes the temporal Gaussian weight meaningful (later additions are exponentially down-weighted by `w_t`).

- **Independent ref/target aggregation with the same `max_window` (Design B).** Symmetric in time. An alternative ("Design A") would aggregate both sides under a single combined window centered at `(t_ref + t_tgt)/2`, but that mixes ref and tgt time roles and complicates the LoD bookkeeping. Independent aggregation keeps the M3C2 statistic well-defined on each side.

- **`max_window = time_gap · max_window_ratio`.** Naturally couples temporal smoothing to the question being asked: short-time changes (small `time_gap`) get little temporal smoothing; long-time comparisons (large `time_gap`) can afford to borrow more. With a list of `max_window_ratio` values, each corepoint picks the ratio that maximizes reference planarity — i.e., the smallest window that still gives a stable plane fit.

- **Scale selection done once on the reference.** Avoids per-target instability (different targets might otherwise pick different normals for the same corepoint, which would break the interpretability of the resulting time series). Done on a planarity score because that is what M3C2's normal estimate cares about.

- **Effective sample size (Kish) in LoD.** Using `N` instead of `N_eff` would under-estimate uncertainty when a few points dominate the weights (typical with Gaussian weighting around a corepoint that sits near the cylinder edge). `N_eff` keeps LoD honest in that case.

- **Structured uncertainty dtype.** Identical schema to classic M3C2's output, so py4dgeo's smoothing and segmentation (`RegionGrowingAlgorithm`, etc.) consume TAM3C2 output without changes.

---

## 9. Limitations / known constraints

- All epochs in `epochs_timeseries` must be loaded into memory and have `Epoch.timestamp` set. KDTrees are built lazily on first use but never released — large datasets can be memory-heavy.
- Scale selection is **per corepoint** but **constant in time**. A surface that changes character over the analysis window (e.g., dune that becomes vegetated) will still use the reference-time normal. This is by design — to keep the time series interpretable — but users should be aware.
- `max_window_ratio ∈ (0, 1]`. A value of `1.0` allows the half-window to reach across the *entire* `|t_target − t_ref|` gap; values > 1 are rejected at init for safety.
- For `t_target == t_ref` (e.g., a sanity check call), `time_gap` falls back to the full time range so aggregation does not collapse to zero — but this case has no physical meaning and is mainly defensive.

---

## 10. File map

```
src/py4dgeo/tam3c2.py
├── class Weighting(Enum)                            # NONE / LINEAR / GAUSSIAN
├── class TAM3C2(M3C2LikeAlgorithm)
│   ├── __init__                                     # config + diag containers
│   ├── _build_index                                 # KDTree per epoch + epoch_times
│   ├── _find_epoch_index
│   ├── _aggregate_sphere                            # 4.1
│   ├── _aggregate_cylinder                          # 4.2
│   ├── _planarity_and_normal                        # 4.3
│   ├── _ensure_ref_scale_cache                      # 4.4
│   ├── directions                                   # returns cached normals
│   ├── _compute_weights                             # 4.5
│   ├── _m3c2_and_lod                                # 4.6
│   ├── calculate_distances                          # 4.7 — main entry from py4dgeo
│   ├── diagnostics                                  # 4.8
│   └── save_diagnostics
├── class LossFunctionCalculator                     # 7.1 — LoD / error loss
├── sweep_alpha                                      # 7.1 — driver
├── _parse_timestamp_from_filename                   # YYMMDD[_HHMMSS]
├── read_epochs_from_folder
├── extract_reference_and_others
└── sample_corepoints                                # 'voxel' or 'random'
```

---

## 11. References

- Lague, D., Brodu, N., & Leroux, J. (2013). *Accurate 3D comparison of complex topography with terrestrial laser scanner: Application to the Rangitikei canyon (N-Z).* ISPRS Journal of Photogrammetry and Remote Sensing, 82, 10-26.
- Demantké, J., Mallet, C., David, N., & Vallet, B. (2011). *Dimensionality-based scale selection in 3D LiDAR point clouds.* ISPRS Workshop Laser Scanning 2011.
- Kish, L. (1965). *Survey Sampling.* John Wiley & Sons. (Effective sample size for weighted estimators.)
- py4dgeo: https://github.com/3dgeo-heidelberg/py4dgeo
