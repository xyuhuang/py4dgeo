"""Time-Adaptive M3C2 (TAM3C2).

Time-adaptive extension of M3C2: where a single epoch is too sparse to
estimate a normal or a stable mean, neighborhoods are aggregated across
nearby epochs in time.

- Inherits ``M3C2LikeAlgorithm``, so it plugs into
  ``py4dgeo.SpatiotemporalAnalysis.add_epochs``.
- Each ``calculate_distances(ref, target)`` aggregates a neighborhood around
  ``t_ref`` and one around ``t_target`` independently (greedy nearest-first,
  until ``required_points`` is reached or ``max_window = |t_target - t_ref| *
  max_window_ratio`` is exhausted), then computes the M3C2 distance and LoD95.
- Weighting is NONE / LINEAR / GAUSSIAN; weighted LoD uses
  ``N_eff = (sum w)^2 / sum(w^2)``. Points of the epoch being measured
  (``dt == 0``) bypass both kernels and enter at full weight, so only borrowed
  points are ever down-weighted.
- ``normal_radii`` and ``max_window_ratio`` may be lists; the best
  (radius, ratio) pair is selected per corepoint on the reference epoch by
  planarity of an aggregated spherical neighborhood.
- Per-call diagnostics are available via ``diagnostics()`` /
  ``save_diagnostics()``.

Related modules: :mod:`py4dgeo.tam3c2_ici` (data-adaptive borrowing depth),
:mod:`py4dgeo.tabaseline` (the aggregation baseline and the fixed-window
control), :mod:`py4dgeo.tam3c2_utils` (IO, corepoint sampling, normals),
:mod:`py4dgeo.tam3c2_lod` (variance-component LoD),
:mod:`py4dgeo.tam3c2_stratio` (choosing ``space_time_ratio``).

Typical workflow::

    import py4dgeo
    from py4dgeo import (TAM3C2, Weighting, read_epochs_from_folder,
                         extract_reference_and_others, sample_corepoints)

    epochs = read_epochs_from_folder(folder)
    ref, others = extract_reference_and_others(epochs, reference_timestamp)
    corepoints = sample_corepoints(ref, method="voxel", voxel_size=1.5)

    tam = TAM3C2(
        epochs_timeseries=epochs, max_window_ratio=0.2, normal_radii=[0.5, 1.0],
        required_points=10, weighting=Weighting.GAUSSIAN, corepoints=corepoints,
        cyl_radius=1.0, max_distance=5.0, registration_error=0.02,
    )

    analysis = py4dgeo.SpatiotemporalAnalysis("out.zip", force=True)
    analysis.reference_epoch = ref
    analysis.corepoints = corepoints
    analysis.m3c2 = tam
    analysis.add_epochs(*others)
"""

import logging
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from py4dgeo.epoch import Epoch
from py4dgeo.m3c2 import M3C2LikeAlgorithm
from py4dgeo.util import Py4DGeoError

# Re-exported so that `from py4dgeo.tam3c2 import ...` keeps working for the
# helpers that now live in tam3c2_utils.
from py4dgeo.tam3c2_utils import (  # noqa: F401
    _parse_timestamp_from_filename,
    clean_corepoint_normals,
    extract_reference_and_others,
    read_epochs_from_folder,
    sample_corepoints,
)

logger = logging.getLogger("py4dgeo")

#: Structured dtype of the per-corepoint uncertainty every TAM3C2-family method returns.
UNCERTAINTY_DTYPE = [
    ("lodetection", "<f8"),
    ("spread1", "<f8"),
    ("num_samples1", "<f8"),
    ("spread2", "<f8"),
    ("num_samples2", "<f8"),
]

#: Per-target, per-corepoint diagnostic arrays, in the order of ``diagnostics()``.
_PER_TARGET_DIAG = (
    "window_used_ref",
    "window_used_tgt",
    "n_before_ref",
    "n_after_ref",
    "n_before_tgt",
    "n_after_tgt",
    "n_points_ref",
    "n_points_tgt",
    "scale_idx_per_target",
)


class Weighting(Enum):
    """Weighting scheme applied."""

    NONE = "none"
    LINEAR = "linear"
    GAUSSIAN = "gaussian"


class _Neighborhood:
    """Points gathered epoch by epoch around one center time.

    ``result()`` gives the 6-tuple every ``_aggregate_*`` method returns:
    ``(pts, dt, eidx, n_before, n_after, w_used)`` - points, their
    ``epoch_time - center_time``, their source epoch index, the number of
    contributing epochs before / after the center time, and the largest
    ``|dt|`` used. An empty neighborhood gives ``(None, None, None, 0, 0, 0.0)``.
    """

    def __init__(self):
        self.pts, self.dt, self.eidx = [], [], []
        self.n_before = self.n_after = 0
        self.w_used = 0.0
        self.total = 0

    def add(self, pts, dt, ei):
        self.pts.append(pts)
        self.dt.append(np.full(len(pts), dt))
        self.eidx.append(np.full(len(pts), ei, dtype=np.int32))
        if dt < 0:
            self.n_before += 1
        elif dt > 0:
            self.n_after += 1
        self.w_used = max(self.w_used, abs(dt))
        self.total += len(pts)

    def result(self):
        if self.total == 0:
            return None, None, None, 0, 0, 0.0
        return (
            np.vstack(self.pts),
            np.concatenate(self.dt),
            np.concatenate(self.eidx),
            self.n_before,
            self.n_after,
            self.w_used,
        )


class TAM3C2(M3C2LikeAlgorithm):
    """Time-Adaptive M3C2.

    Parameters
    ----------
    epochs_timeseries : list[py4dgeo.Epoch]
        Pool of epochs used for time-adaptive aggregation. Must include the
        reference and all targets that will later be passed via
        ``calculate_distances``. Each epoch must have ``timestamp`` set.
    max_window_ratio : float or list of float
        Fraction of ``|t_target - t_ref|`` used as the maximum half-window for
        time-adaptive aggregation. May be a list for multi-scale selection.
    normal_radii : float or list of float
        Spherical radius (or radii) used for PCA-based normal estimation on
        the aggregated reference neighborhood.
    required_points : int
        Target point count used to stop temporal aggregation on each side
        (ref / target). If the maximum time window is reached first, a distance
        can still be calculated as long as at least two points are available
        on each side. The same count is the floor for the normal: where the
        scale-selection sphere never reaches it, the normal is NaN and the core
        point reports no distance.
    weighting : Weighting
        ``Weighting.NONE`` / ``LINEAR`` / ``GAUSSIAN``.
    sigma_ratio : float
        Gaussian kernel width (in units of the spatial radius / time window).
    space_time_ratio : float
        Unitless **spacetime anisotropy ratio** in normalized coordinates.
        The temporal coordinate is rescaled to ``u_t / space_time_ratio``
        before the isotropic kernel is applied. Larger values stretch the
        effective temporal kernel (larger contribution from temporally distant
        points, but no change to which epochs are selected). This may lower LoD
        while increasing the risk of temporal blurring. Default ``1.0``
        treats normalized space and time symmetrically.
        Recommended workflow: pick a value with
        :func:`py4dgeo.estimate_space_time_ratio`.
    orientation_vector : array-like of shape (3,)
        Reference up direction used to orient normals.
    include_center_epoch : bool
        If True, include the epoch centered on the aggregation time in the
        spherical and cylindrical neighborhoods. If False, skip that epoch and
        aggregate only from neighboring epochs. Default True.
    spatial_weighting : bool
        If True, apply the spatial part of the weighting kernel using the
        radial distance perpendicular to the M3C2 normal. If False, apply only
        the temporal part of the weighting kernel. Points of the center epoch
        itself are exempt from both kernels: they always enter at full weight
        1, so borrowed points can only ever be down-weighted relative to them.
        Default True.
    temporal_weighting : bool
        If True, apply the temporal part of the weighting kernel using the
        time offset from the reference or target evaluation epoch. If False,
        temporal weights are fixed to one while spatial weights, when enabled,
        are retained. Default True. Set ``spatial_weighting=True`` and
        ``temporal_weighting=False`` for a spatial-only ablation.
    keep_neighborhoods : bool
        If True, store per-(corepoint, target) aggregated points/weights in
        ``self._neighborhoods`` (memory-heavy; for debugging/visualization and
        for :mod:`py4dgeo.tam3c2_lod`).
    **kwargs
        Forwarded to ``M3C2LikeAlgorithm`` (``corepoints``, ``cyl_radius``,
        ``max_distance``, ``registration_error``, ...).
    """

    def __init__(
        self,
        *,
        epochs_timeseries: List[Epoch],
        max_window_ratio,
        normal_radii,
        required_points: int = 10,
        weighting: Weighting = Weighting.GAUSSIAN,
        sigma_ratio: float = 1.0,
        space_time_ratio: float = 1.0,
        orientation_vector=(0.0, 0.0, 1.0),
        include_center_epoch=True,
        spatial_weighting: bool = True,
        temporal_weighting: bool = True,
        keep_neighborhoods: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not epochs_timeseries:
            raise Py4DGeoError("TAM3C2: epochs_timeseries must be non-empty.")
        if any(getattr(e, "timestamp", None) is None for e in epochs_timeseries):
            raise Py4DGeoError(
                "TAM3C2: every epoch in epochs_timeseries must have a timestamp."
            )
        if sigma_ratio <= 0:
            raise Py4DGeoError(f"TAM3C2: sigma_ratio must be > 0, got {sigma_ratio}.")
        if space_time_ratio <= 0:
            raise Py4DGeoError(
                f"TAM3C2: space_time_ratio must be > 0, got {space_time_ratio}."
            )

        self.epochs_timeseries = sorted(epochs_timeseries, key=lambda e: e.timestamp)
        self.max_window_ratio = max_window_ratio
        self.normal_radii = normal_radii
        self.required_points = int(required_points)
        self.weighting = Weighting(weighting)
        self.sigma_ratio = float(sigma_ratio)
        self.space_time_ratio = float(space_time_ratio)
        self.orientation_vector = np.array(orientation_vector, dtype=float).reshape(3)
        self.include_center_epoch = bool(include_center_epoch)
        self.spatial_weighting = bool(spatial_weighting)
        self.temporal_weighting = bool(temporal_weighting)
        self.keep_neighborhoods = bool(keep_neighborhoods)

        # Index cache
        self._kdtrees: Optional[List[cKDTree]] = None
        self._epoch_times: Optional[np.ndarray] = None

        # Window the next scale selection should use, set by calculate_distances to the
        # reference/target gap. None -> fall back to the full series length.
        self._scale_time_gap: Optional[float] = None

        # Per-corepoint scale-selection cache, keyed by (reference epoch, gap)
        self._ref_normals: Optional[np.ndarray] = None  # (n_cp, 3)
        self._opt_scale_idx: Optional[np.ndarray] = None  # (n_cp,)
        self._scale_combinations: Optional[List[Tuple[float, float]]] = None
        self._ref_cache_key = None

        # Diagnostics. Every _PER_TARGET_DIAG entry is a list with one (n_cp,)
        # array per target; "scale_idx" is the selection of the latest target.
        self._diag = {"target_timestamps": [], "scale_idx": None}
        self._diag.update({k: [] for k in _PER_TARGET_DIAG})
        # Optional store of full aggregated neighborhoods, one list per target
        self._neighborhoods: Optional[list] = [] if keep_neighborhoods else None

    @property
    def name(self):
        return "TAM3C2"

    @staticmethod
    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [float(v) for v in x]
        return [float(x)]

    def _build_index(self, trees=True):
        if self._epoch_times is None:
            self._epoch_times = np.array(
                [e.timestamp.timestamp() for e in self.epochs_timeseries]
            )
        if trees and self._kdtrees is None:
            self._kdtrees = [cKDTree(e.cloud) for e in self.epochs_timeseries]

    def _find_epoch_index(self, epoch: Epoch) -> Optional[int]:
        for i, e in enumerate(self.epochs_timeseries):
            if e is epoch or e.timestamp == epoch.timestamp:
                return i
        return None

    def _time_gap(self, ref_time, tgt_time):
        """``|t_target - t_ref|``; the full series length when the two coincide."""
        gap = abs(tgt_time - ref_time)
        if gap == 0.0:
            gap = float(self._epoch_times.max() - self._epoch_times.min()) or 1.0
        return gap

    def _check_window_ratios(self):
        ratios = self._as_list(self.max_window_ratio)
        for wr in ratios:
            if not 0.0 < wr <= 1.0:
                raise Py4DGeoError(f"max_window_ratio must lie in (0, 1], got {wr}")
        return ratios

    def _nearest_epoch_order(self, center_time, exclude_idx, max_window):
        """Yield epochs from nearest to farthest in time.

        On an exact ``|dt|`` tie the side taken is the OPPOSITE of the one
        taken last, so ties alternate rather than always favouring one
        direction; the very first tie (no previous side) goes to *before*. The
        center epoch is yielded first when present and not excluded, and the
        walk may continue on one side alone at a time-series boundary.

        :class:`~py4dgeo.tabaseline.TAbaselineOriginal` has its own greedy
        walk that breaks ties towards *after* instead. That is deliberate: it
        reproduces the aggregation code the published evaluation ran, so the
        two rules are kept separate rather than merged.
        """
        times = self._epoch_times
        n = len(times)
        before = int(np.searchsorted(times, center_time, side="left")) - 1
        after = before + 1

        while after < n and times[after] == center_time:
            if after != exclude_idx:
                yield after
            after += 1

        last_side = None
        while True:
            if before == exclude_idx:
                before -= 1
            if after == exclude_idx:
                after += 1
            if before < 0 and after >= n:
                return

            before_dt = abs(times[before] - center_time) if before >= 0 else np.inf
            after_dt = abs(times[after] - center_time) if after < n else np.inf
            if min(before_dt, after_dt) > max_window:
                return

            if np.isclose(before_dt, after_dt, rtol=0.0, atol=1e-9):
                take_before = last_side != "before"
            else:
                take_before = before_dt < after_dt
            # A tie can straddle the window edge: take whichever side is inside.
            if take_before and before_dt > max_window:
                take_before = False
            elif not take_before and after_dt > max_window:
                take_before = True

            if take_before:
                yield before
                before -= 1
                last_side = "before"
            else:
                yield after
                after += 1
                last_side = "after"

    def _aggregate_greedy(self, order, center_time, select):
        """Add ``select(ei)`` epoch by epoch in ``order`` until ``required_points``."""
        nb = _Neighborhood()
        for ei in order:
            pts = select(ei)
            if pts is None:
                continue
            nb.add(pts, self._epoch_times[ei] - center_time, ei)
            if nb.total >= self.required_points:
                break
        return nb.result()

    def _sphere_points_for_epoch(self, cp, ei, radius):
        """Points of epoch ``ei`` within ``radius`` of ``cp``, or None."""
        idxs = self._kdtrees[ei].query_ball_point(cp, radius)
        if not idxs:
            return None
        pts = self.epochs_timeseries[ei].cloud[idxs]
        pts = pts[np.sum((pts - cp) ** 2, axis=1) <= radius * radius]
        return pts if len(pts) else None

    def _aggregate_sphere(self, cp, center_time, exclude_idx, radius, max_window):
        """Greedy nearest-first temporal aggregation of a spherical neighborhood.

        Returns ``(pts, dt, eidx, n_before, n_after, w_used)``, see
        :class:`_Neighborhood`.
        """
        return self._aggregate_greedy(
            self._nearest_epoch_order(center_time, exclude_idx, max_window),
            center_time,
            lambda ei: self._sphere_points_for_epoch(cp, ei, radius),
        )

    def _cylinder_points_for_epoch(
        self, cp, normal, ei, cyl_radius, max_distance, bounding_r=None
    ):
        """Points of epoch ``ei`` inside the cylinder around ``cp`` along ``normal``.

        Cylinder geometry follows py4dgeo M3C2 convention: a point ``p`` is
        inside iff the along-normal projection ``(p-c).n`` has absolute value
        ``<= max_distance`` and the perpendicular distance is ``<= cyl_radius``.

        Returns ``(pts, along)`` - the selected points and their along-normal
        projections - or ``(None, None)`` when this epoch contributes nothing.
        """
        if not np.isfinite(normal).all():
            # The scale selection found no supported normal here, so the cylinder
            # has no axis: the epoch contributes nothing and the distance stays NaN.
            # Handled here rather than by skipping the corepoint, because
            # TAM3C2ICI counts on exactly two aggregation calls per corepoint.
            return None, None

        if bounding_r is None:
            bounding_r = np.sqrt(cyl_radius * cyl_radius + max_distance * max_distance)

        idxs = self._kdtrees[ei].query_ball_point(cp, bounding_r)
        if not idxs:
            return None, None
        pts = self.epochs_timeseries[ei].cloud[idxs]
        v = pts - cp
        along = v @ normal
        perp_sq = np.einsum("ij,ij->i", v, v) - along * along
        mask = (perp_sq <= cyl_radius * cyl_radius) & (np.abs(along) <= max_distance)
        if not mask.any():
            return None, None
        return pts[mask], along[mask]

    def _aggregate_cylinder(
        self, cp, normal, center_time, exclude_idx, cyl_radius, max_distance, max_window
    ):
        """Greedy nearest-first temporal aggregation of a cylinder along ``normal``.

        Aggregation stops as soon as ``required_points`` is reached. See
        :class:`~py4dgeo.tam3c2_ici.TAM3C2ICI` for the data-adaptive
        alternative that keeps borrowing past that floor until the epochs
        stop agreeing.
        """
        bounding_r = np.sqrt(cyl_radius * cyl_radius + max_distance * max_distance)
        return self._aggregate_greedy(
            self._nearest_epoch_order(center_time, exclude_idx, max_window),
            center_time,
            lambda ei: self._cylinder_points_for_epoch(
                cp, normal, ei, cyl_radius, max_distance, bounding_r
            )[0],
        )

    def _planarity_and_normal(self, pts):
        if pts is None or len(pts) < 3:
            return -np.inf, self.orientation_vector.copy()
        centered = pts - pts.mean(axis=0)
        cov = (centered.T @ centered) / len(centered)
        evals, evecs = np.linalg.eigh(cov)  # ascending
        l0, l1, l2 = evals
        planarity = float((l1 - l0) / l2) if l2 > 1e-12 else 0.0
        normal = evecs[:, 0]
        if normal @ self.orientation_vector < 0:
            normal = -normal
        return planarity, normal

    def _ensure_ref_scale_cache(self, ref_epoch: Epoch):
        """Per-corepoint multi-scale selection on the reference epoch.

        ``max_window`` is the same one the cylinder uses, ``|t_target - t_ref| *
        max_window_ratio``, taken from ``self._scale_time_gap``: borrowing for the normal
        should reach as far as the interval being measured, not as far as the whole series
        happens to run. The normal therefore depends on the target, and the cache key
        includes the gap. Callers that invoke this directly without setting
        ``_scale_time_gap`` fall back to the full series length.
        """
        gap = self._scale_time_gap
        cache_key = (id(ref_epoch), None if gap is None else round(float(gap), 6))
        if self._ref_normals is not None and self._ref_cache_key == cache_key:
            return

        self._build_index()
        if self.corepoints is None:
            raise Py4DGeoError("TAM3C2: corepoints must be set before calculation.")

        n_cp = self.corepoints.shape[0]
        ref_idx = self._find_epoch_index(ref_epoch)
        scale_exclude_idx = None if self.include_center_epoch else ref_idx
        ref_time = ref_epoch.timestamp.timestamp()

        normal_radii = self._as_list(self.normal_radii)
        if any(sr <= 0 for sr in normal_radii):
            raise Py4DGeoError(f"normal_radii must be positive, got {normal_radii}")
        combos = [(sr, wr) for sr in normal_radii for wr in self._check_window_ratios()]
        self._scale_combinations = combos

        # Same basis as the cylinder: the reference/target gap.
        time_range = float(gap) if gap else float(
            self._epoch_times.max() - self._epoch_times.min()
        )
        if time_range <= 0:
            time_range = 1.0

        normals = np.zeros((n_cp, 3))
        opt_idx = np.zeros(n_cp, dtype=np.int32)
        best_pl = np.full(n_cp, -np.inf)

        logger.info(
            "TAM3C2: scale selection on reference (%d combos, %d corepoints)",
            len(combos),
            n_cp,
        )

        # A plane has three parameters, so three points fit it exactly and leave no
        # residual to judge the fit by. Requiring ``required_points`` keeps the fit
        # over-determined - the same argument that sets the floor for the distance.
        min_pts = max(3, self.required_points)

        for k, (sr, wr) in enumerate(combos):
            mw = time_range * wr
            for i in range(n_cp):
                pts = self._aggregate_sphere(
                    self.corepoints[i], ref_time, scale_exclude_idx, sr, mw
                )[0]
                if pts is None or len(pts) < min_pts:
                    continue
                pl, nrm = self._planarity_and_normal(pts)
                if pl > best_pl[i]:
                    best_pl[i] = pl
                    normals[i] = nrm
                    opt_idx[i] = k

        # Corepoints that never collected enough points: no normal, hence no distance.
        no_data = ~np.isfinite(best_pl)
        if np.any(no_data):
            logger.warning(
                "TAM3C2: %d/%d corepoints never reached %d points for the normal "
                "(no distance will be reported there).",
                int(no_data.sum()),
                n_cp,
                min_pts,
            )
            normals[no_data] = np.nan

        self._ref_normals = normals
        self._opt_scale_idx = opt_idx
        self._ref_cache_key = cache_key

    def directions(self):
        if self._ref_normals is None:
            raise Py4DGeoError(
                f"{self.name}.directions() called before any distance computation."
            )
        return self._ref_normals

    def _compute_weights(self, dt_array, pts, cp, normal, window_size, spatial_r):
        """Compute per-point spatial/temporal weights, normalized to sum one.

        ``window_size`` is the temporal kernel's reference scale. Callers must
        pass the ACTUAL max |dt| reached by this aggregation (``w_used`` from
        ``_aggregate_cylinder``), not the theoretical ``max_window`` cap - the
        cap is frequently never reached (required_points stops the search
        early), so using it here would make the kernel far wider than the real
        spread of borrowed points, flattening the temporal weights to ~1.
        """
        if self.weighting == Weighting.NONE:
            return None
        offsets = pts - cp
        along = offsets @ normal
        radial_sq = np.einsum("ij,ij->i", offsets, offsets) - along * along
        d_spatial = np.sqrt(np.maximum(radial_sq, 0.0))
        d_time = np.abs(dt_array)

        if self.weighting == Weighting.LINEAR:
            def kernel(u):
                return np.clip(1.0 - u, 0.0, 1.0)
        else:  # GAUSSIAN
            sigma = self.sigma_ratio

            def kernel(u):
                return np.exp(-(u ** 2) / (2.0 * sigma * sigma))

        # Spacetime anisotropy: an isotropic kernel on (u_s, u_t / space_time_ratio)
        # is the same as stretching the temporal reach by space_time_ratio.
        win_eff = window_size * self.space_time_ratio

        w = np.ones(len(pts))
        if self.spatial_weighting:
            w = w * kernel(d_spatial / spatial_r)
        if self.temporal_weighting and win_eff > 0:
            w = w * kernel(d_time / win_eff)

        # dt == 0 points are the epoch itself, not borrowed evidence: no kernel
        # at all, weight 1, so borrowed points can only be down-weighted.
        w = np.where(d_time == 0.0, 1.0, w)

        # Each side sums to one independently (ref and target are separate calls).
        return self._normalize_weights(w)

    @staticmethod
    def _normalize_weights(w):
        """Return finite non-negative weights whose sum is exactly one."""
        w = np.asarray(w, dtype=float)
        weight_sum = float(np.sum(w))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:  # degenerate
            w = np.ones_like(w, dtype=float)
            weight_sum = float(len(w))
        return w / weight_sum

    @staticmethod
    def _side_stats(p, w):
        """Mean, spread and (effective) sample count of one side's projections."""
        if w is None:
            return p.mean(), p.std(ddof=1), float(len(p))
        sw = w.sum()
        m = float(np.sum(p * w) / sw)
        s = float(np.sqrt(np.sum(w * (p - m) ** 2) / sw))
        sw2 = float(np.sum(w * w))
        n = (sw * sw) / sw2 if sw2 > 0 else float(len(p))
        # The mean was estimated from these same points, so one degree of freedom is
        # spent and the spread above is biased low. The unweighted branch gets this from
        # np.std(ddof=1) and lib/distances.cpp from dividing by rows() - 1; with weights
        # the count to use is the effective one, which for equal weights reduces to
        # exactly N / (N - 1). Only the spread is corrected - n keeps dividing the
        # variance, which is what makes a down-weighted point count for less.
        if n > 1.0:
            s *= float(np.sqrt(n / (n - 1.0)))
        return m, s, n

    def _m3c2_and_lod(self, cp, normal, ref_pts, ref_w, tgt_pts, tgt_w):
        m_r, s_r, n_r = self._side_stats((ref_pts - cp) @ normal, ref_w)
        m_t, s_t, n_t = self._side_stats((tgt_pts - cp) @ normal, tgt_w)
        n_r = max(n_r, 1.0)
        n_t = max(n_t, 1.0)

        # LoD95 = 1.96 * (SE + e_reg), registration error INSIDE the factor, as in
        # Lague et al. (2013) and py4dgeo's C++ kernel (lib/distances.cpp).
        lod95 = 1.96 * (
            float(np.sqrt(s_r * s_r / n_r + s_t * s_t / n_t))
            + float(self.registration_error)
        )
        uncertainty = np.array(
            [(lod95, s_r, n_r, s_t, n_t)], dtype=UNCERTAINTY_DTYPE
        )[0]
        return float(m_t - m_r), uncertainty

    def _record_diagnostics(self, target_timestamp, per_cp, scale_idx):
        """Append one target's per-corepoint diagnostics."""
        self._diag["target_timestamps"].append(target_timestamp)
        for key, values in per_cp.items():
            self._diag[key].append(values)
        self._diag["scale_idx"] = np.asarray(scale_idx, dtype=np.int32).copy()
        self._diag["scale_idx_per_target"].append(self._diag["scale_idx"])

    @staticmethod
    def _empty_diag_arrays(n_cp):
        """Zeroed per-corepoint diagnostic arrays for one target."""
        return {
            k: np.zeros(n_cp) if k.startswith("window") else np.zeros(n_cp, np.int32)
            for k in _PER_TARGET_DIAG
            if k != "scale_idx_per_target"
        }

    @staticmethod
    def _store_pack_diag(diag, i, ref_pack, tgt_pack):
        """Write corepoint ``i``'s aggregation summary into the diagnostic arrays."""
        for side, pack in (("ref", ref_pack), ("tgt", tgt_pack)):
            pts, _, _, n_before, n_after, w_used = pack
            diag[f"window_used_{side}"][i] = w_used
            diag[f"n_before_{side}"][i] = n_before
            diag[f"n_after_{side}"][i] = n_after
            diag[f"n_points_{side}"][i] = 0 if pts is None else len(pts)

    def calculate_distances(self, epoch1, epoch2, searchtree=None):
        """Time-adaptive M3C2 distance between ``epoch1`` (ref) and ``epoch2`` (tgt).

        Called once per target by ``SpatiotemporalAnalysis.add_epochs``.
        """
        if self.cyl_radius is None:
            raise Py4DGeoError(f"{self.name} requires cyl_radius (float).")
        if self.corepoints is None:
            raise Py4DGeoError(f"{self.name} requires corepoints to be set.")

        self._build_index()
        n_cp = self.corepoints.shape[0]
        ref_exclude_idx = None if self.include_center_epoch else self._find_epoch_index(epoch1)
        tgt_exclude_idx = None if self.include_center_epoch else self._find_epoch_index(epoch2)
        ref_time = epoch1.timestamp.timestamp()
        tgt_time = epoch2.timestamp.timestamp()

        # Per-target max_window is based on the time gap between ref and target.
        # Normals use the same window, so the gap has to be known before scale selection.
        time_gap = self._time_gap(ref_time, tgt_time)
        self._scale_time_gap = time_gap
        self._ensure_ref_scale_cache(epoch1)

        distances = np.full(n_cp, np.nan)
        uncertainties = np.full(n_cp, np.nan, dtype=UNCERTAINTY_DTYPE)
        diag = self._empty_diag_arrays(n_cp)
        nbhd_record = [] if self.keep_neighborhoods else None

        for i in range(n_cp):
            cp = self.corepoints[i]
            normal = self._ref_normals[i]
            sr, wr = self._scale_combinations[int(self._opt_scale_idx[i])]
            max_window = time_gap * wr

            ref_pack = self._aggregate_cylinder(
                cp, normal, ref_time, ref_exclude_idx,
                self.cyl_radius, self.max_distance, max_window,
            )
            tgt_pack = self._aggregate_cylinder(
                cp, normal, tgt_time, tgt_exclude_idx,
                self.cyl_radius, self.max_distance, max_window,
            )
            self._store_pack_diag(diag, i, ref_pack, tgt_pack)
            ref_pts, ref_dt, ref_eidx, _, _, w_r = ref_pack
            tgt_pts, tgt_dt, tgt_eidx, _, _, w_t = tgt_pack

            if ref_pts is None or tgt_pts is None or len(ref_pts) < 2 or len(tgt_pts) < 2:
                if nbhd_record is not None:
                    nbhd_record.append(None)
                continue

            # Bandwidth = the ACTUAL max |dt| reached (w_r / w_t), not the
            # max_window cap - see _compute_weights.
            ref_w = self._compute_weights(ref_dt, ref_pts, cp, normal, w_r, self.cyl_radius)
            tgt_w = self._compute_weights(tgt_dt, tgt_pts, cp, normal, w_t, self.cyl_radius)
            distances[i], uncertainties[i] = self._m3c2_and_lod(
                cp, normal, ref_pts, ref_w, tgt_pts, tgt_w
            )

            if nbhd_record is not None:
                nbhd_record.append(
                    {
                        "ref_pts": ref_pts,
                        "ref_w": ref_w,
                        "ref_dt": ref_dt,
                        "ref_eidx": ref_eidx,
                        "tgt_pts": tgt_pts,
                        "tgt_w": tgt_w,
                        "tgt_dt": tgt_dt,
                        "tgt_eidx": tgt_eidx,
                        "spatial_r": sr,
                        "window_ratio": wr,
                        "max_window": max_window,
                    }
                )

        self._record_diagnostics(epoch2.timestamp, diag, self._opt_scale_idx)
        if nbhd_record is not None:
            self._neighborhoods.append(nbhd_record)

        return distances, uncertainties

    def diagnostics(self):
        """Return diagnostics.

        Per-target entries are ``(n_cp, n_targets)`` matrices (``None`` before
        the first target); ``scale_idx`` is the ``(n_cp,)`` scale selection of
        the latest target - the selection depends on the reference/target gap,
        so ``scale_idx_per_target`` keeps all of them.
        """
        out = {
            "target_timestamps": list(self._diag["target_timestamps"]),
            "scale_idx": (
                None if self._diag["scale_idx"] is None else self._diag["scale_idx"].copy()
            ),
        }
        for key, vals in self._diag.items():
            if key not in out:  # subclasses add their own per-target lists
                out[key] = np.column_stack(vals) if vals else None
        return out

    def save_diagnostics(self, path):
        """Save diagnostic matrices to an ``.npz`` file."""
        diag = self.diagnostics()
        arrays = {k: v for k, v in diag.items() if isinstance(v, np.ndarray)}
        if diag["target_timestamps"]:
            arrays["target_timestamps"] = np.array(
                [t.isoformat() for t in diag["target_timestamps"]]
            )
        np.savez(path, **arrays)


__all__ = ["Weighting", "TAM3C2", "UNCERTAINTY_DTYPE"]
