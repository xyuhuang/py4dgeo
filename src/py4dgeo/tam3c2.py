"""
Time-Adaptive M3C2 (TAM3C2)
============================

Time-adaptive extension of M3C2: when the point density in a single epoch is
too low to robustly estimate normals or compute M3C2 distances, neighborhoods
are aggregated across nearby epochs in time.

Design summary
--------------
- Inherits ``py4dgeo.M3C2LikeAlgorithm`` so it plugs directly into
  ``py4dgeo.SpatiotemporalAnalysis.add_epochs``.
- For each call ``calculate_distances(ref, target)``:
    * Aggregate a neighborhood around each corepoint at ``t_ref`` (greedy
      nearest-first across epochs, until ``required_points`` reached or
      ``max_window`` exceeded).
    * Aggregate the same way around ``t_target``.
    * Each side is aggregated **independently** (Design B): ``max_window`` is
      ``|t_target - t_ref| * max_window_ratio``, applied to both sides as the
      per-target cap.
    * Compute M3C2 distance + LoD95 (supports NONE / LINEAR / GAUSSIAN
      weighting; weighted LoD uses N_eff = (sum w)^2 / sum(w^2)).
- Multi-scale support: ``normal_radii`` and ``max_window_ratio`` may be lists.
  Optimal (normal_radius, window_ratio) per corepoint is selected once on
  the reference epoch using planarity of an aggregated spherical neighborhood.
- Per-call diagnostics (window used, n_before/n_after epochs, point counts)
  are accumulated and available via ``diagnostics()`` / ``save_diagnostics()``.

Typical workflow
----------------

    from py4dgeo.tam3c2 import (
        TAM3C2, Weighting,
        read_epochs_from_folder, extract_reference_and_others, sample_corepoints,
    )
    import py4dgeo

    epochs = read_epochs_from_folder(folder)
    ref, others = extract_reference_and_others(epochs, reference_timestamp)
    corepoints = sample_corepoints(ref, method="voxel", voxel_size=1.5)

    tam = TAM3C2(
        epochs_timeseries=epochs,
        max_window_ratio=0.2,        # or [0.1, 0.2, 0.3] for multi-scale
        normal_radii=[0.5, 1.0, 1.5],
        required_points=10,
        weighting=Weighting.GAUSSIAN,
        sigma_ratio=1.0,
        space_time_ratio=1.0,        # spacetime anisotropy ratio (unitless)
        corepoints=corepoints,
        cyl_radius=1.0,
        max_distance=5.0,
        registration_error=0.02,
    )

    analysis = py4dgeo.SpatiotemporalAnalysis("out.zip", force=True)
    analysis.reference_epoch = ref
    analysis.corepoints = corepoints
    analysis.m3c2 = tam
    analysis.add_epochs(*others)
"""

import logging
import os
import re
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

import py4dgeo
from py4dgeo.epoch import Epoch
from py4dgeo.m3c2 import M3C2LikeAlgorithm
from py4dgeo.util import Py4DGeoError

logger = logging.getLogger("py4dgeo")

class Weighting(Enum):
    """Weighting scheme applied."""

    NONE = "none"
    LINEAR = "linear"
    GAUSSIAN = "gaussian"

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
        Minimum aggregated point count per side (ref / target) for the
        distance to be considered valid. Aggregation stops as soon as this is
        reached.
    weighting : Weighting
        ``Weighting.NONE`` / ``LINEAR`` / ``GAUSSIAN``.
    sigma_ratio : float
        Gaussian kernel width (in units of the spatial radius / time window).
    space_time_ratio : float
        Unitless **spacetime anisotropy ratio** in normalized coordinates.
        The temporal coordinate is rescaled to ``u_t / space_time_ratio``
        before the isotropic Gaussian is applied. Larger values stretch the
        effective temporal window (more temporal aggregation, lower LoD but
        higher risk of mixing real change into the spread). Default ``1.0``
        treats normalized space and time symmetrically.
        Recommended workflow: pick a value with
        :func:`estimate_space_time_ratio`.
    orientation_vector : array-like of shape (3,)
        Reference up direction used to orient normals.
    include_center_epoch : bool
        If True, include the epoch centered on the aggregation time in the
        spherical and cylindrical neighborhoods. If False, skip that epoch and
        aggregate only from neighboring epochs. Default True.
    spatial_weighting : bool
        If True, apply the spatial part of the weighting kernel using the
        radial distance perpendicular to the M3C2 normal. If False, apply only
        the temporal part of the weighting kernel. Center-epoch points keep
        unit weight in both cases. Default True.
    keep_neighborhoods : bool
        If True, store per-(corepoint, target) aggregated points/weights in
        ``self._neighborhoods`` (memory-heavy; for debugging/visualization).
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
        orientation_vector=np.array([0.0, 0.0, 1.0]),
        include_center_epoch=True,
        spatial_weighting: bool = True,
        keep_neighborhoods: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not epochs_timeseries:
            raise Py4DGeoError("TAM3C2: epochs_timeseries must be non-empty.")
        for e in epochs_timeseries:
            if getattr(e, "timestamp", None) is None:
                raise Py4DGeoError(
                    "TAM3C2: every epoch in epochs_timeseries must have a timestamp."
                )

        self.epochs_timeseries = sorted(epochs_timeseries, key=lambda e: e.timestamp)
        self.max_window_ratio = max_window_ratio
        self.normal_radii = normal_radii
        self.required_points = int(required_points)
        self.weighting = (
            weighting if isinstance(weighting, Weighting) else Weighting(weighting)
        )
        self.sigma_ratio = float(sigma_ratio)
        if space_time_ratio <= 0:
            raise Py4DGeoError(
                f"TAM3C2: space_time_ratio must be > 0, got {space_time_ratio}."
            )
        self.space_time_ratio = float(space_time_ratio)
        self.orientation_vector = np.asarray(orientation_vector, dtype=float).reshape(3)
        self.include_center_epoch = bool(include_center_epoch)
        self.spatial_weighting = bool(spatial_weighting)
        self.keep_neighborhoods = bool(keep_neighborhoods)

        # Index cache
        self._kdtrees: Optional[List[cKDTree]] = None
        self._epoch_times: Optional[np.ndarray] = None

        # Per-corepoint scale-selection cache (built once per reference)
        self._ref_normals: Optional[np.ndarray] = None  # (n_cp, 3)
        self._opt_scale_idx: Optional[np.ndarray] = None  # (n_cp,)
        self._scale_combinations: Optional[List[Tuple[float, float]]] = None
        self._ref_cache_key: Optional[int] = None

        # Per-target diagnostics. Each entry is a 1D array of length n_cp.
        self._diag = {
            "target_timestamps": [],
            "window_used_ref": [],
            "window_used_tgt": [],
            "n_before_ref": [],
            "n_after_ref": [],
            "n_before_tgt": [],
            "n_after_tgt": [],
            "n_points_ref": [],
            "n_points_tgt": [],
            "scale_idx": [],  # (n_cp,) chosen scale index per cp (same for every target)
        }
        # Optional store of full aggregated neighborhoods
        self._neighborhoods: Optional[list] = [] if keep_neighborhoods else None

    @property
    def name(self):
        return "TAM3C2"

    @staticmethod
    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [float(v) for v in x]
        return [float(x)]

    def _build_index(self):
        if self._kdtrees is None:
            self._kdtrees = [cKDTree(e.cloud) for e in self.epochs_timeseries]
            self._epoch_times = np.array(
                [e.timestamp.timestamp() for e in self.epochs_timeseries]
            )

    def _find_epoch_index(self, epoch: Epoch) -> Optional[int]:
        for i, e in enumerate(self.epochs_timeseries):
            if e is epoch or e.timestamp == epoch.timestamp:
                return i
        return None

    def _aggregate_sphere(self, cp, center_time, exclude_idx, radius, max_window):
        """Greedy nearest-first temporal aggregation of a spherical neighborhood.

        Returns
        -------
        pts : (N,3) ndarray or None
        dt_arr : (N,) ndarray of (epoch_time - center_time) per point, or None
        eidx : (N,) ndarray of source epoch indices, or None
        n_before, n_after : int
            Number of *epochs* contributing from before/after the center time.
        w_used : float
            Largest |Î”t| (seconds) actually used.
        """
        times = self._epoch_times
        order = np.argsort(np.abs(times - center_time))

        chunks_pts, chunks_dt, chunks_eidx = [], [], []
        n_before = n_after = 0
        w_used = 0.0
        total = 0

        for ei in order:
            if exclude_idx is not None and ei == exclude_idx:
                continue
            dt = times[ei] - center_time
            adt = abs(dt)
            if adt > max_window:
                break

            idxs = self._kdtrees[ei].query_ball_point(cp, radius)
            if not idxs:
                continue
            pts = self.epochs_timeseries[ei].cloud[idxs]
            d2 = np.sum((pts - cp) ** 2, axis=1)
            pts = pts[d2 <= radius * radius]
            if len(pts) == 0:
                continue

            chunks_pts.append(pts)
            chunks_dt.append(np.full(len(pts), dt))
            chunks_eidx.append(np.full(len(pts), ei, dtype=np.int32))
            if dt < 0:
                n_before += 1
            else:
                n_after += 1
            if adt > w_used:
                w_used = adt
            total += len(pts)
            if total >= self.required_points:
                break

        if total == 0:
            return None, None, None, 0, 0, 0.0
        return (
            np.vstack(chunks_pts),
            np.concatenate(chunks_dt),
            np.concatenate(chunks_eidx),
            n_before,
            n_after,
            w_used,
        )

    def _aggregate_cylinder(
        self, cp, normal, center_time, exclude_idx, cyl_radius, max_distance, max_window
    ):
        """Greedy nearest-first temporal aggregation of a cylinder along ``normal``.

        Cylinder geometry follows py4dgeo M3C2 convention: a point ``p`` is
        inside iff the along-normal projection ``(p-c).n`` has absolute value
        ``<= max_distance`` and the perpendicular distance is ``<= cyl_radius``.
        """
        times = self._epoch_times
        order = np.argsort(np.abs(times - center_time))
        bounding_r = np.sqrt(cyl_radius * cyl_radius + max_distance * max_distance)

        chunks_pts, chunks_dt, chunks_eidx = [], [], []
        n_before = n_after = 0
        w_used = 0.0
        total = 0

        for ei in order:
            if exclude_idx is not None and ei == exclude_idx:
                continue
            dt = times[ei] - center_time
            adt = abs(dt)
            if adt > max_window:
                break

            idxs = self._kdtrees[ei].query_ball_point(cp, bounding_r)
            if not idxs:
                continue
            pts = self.epochs_timeseries[ei].cloud[idxs]
            v = pts - cp
            along = v @ normal
            perp_sq = np.einsum("ij,ij->i", v, v) - along * along
            mask = (perp_sq <= cyl_radius * cyl_radius) & (
                np.abs(along) <= max_distance
            )
            pts = pts[mask]
            if len(pts) == 0:
                continue

            chunks_pts.append(pts)
            chunks_dt.append(np.full(len(pts), dt))
            chunks_eidx.append(np.full(len(pts), ei, dtype=np.int32))
            if dt < 0:
                n_before += 1
            else:
                n_after += 1
            if adt > w_used:
                w_used = adt
            total += len(pts)
            if total >= self.required_points:
                break

        if total == 0:
            return None, None, None, 0, 0, 0.0
        return (
            np.vstack(chunks_pts),
            np.concatenate(chunks_dt),
            np.concatenate(chunks_eidx),
            n_before,
            n_after,
            w_used,
        )


    def _planarity_and_normal(self, pts):
        if pts is None or len(pts) < 3:
            return -np.inf, self.orientation_vector.copy()
        centered = pts - pts.mean(axis=0)
        cov = (centered.T @ centered) / len(centered)
        evals, evecs = np.linalg.eigh(cov)  # ascending
        l0, l1, l2 = evals[0], evals[1], evals[2]
        planarity = float((l1 - l0) / l2) if l2 > 1e-12 else 0.0
        normal = evecs[:, 0]
        if normal @ self.orientation_vector < 0:
            normal = -normal
        return planarity, normal


    def _ensure_ref_scale_cache(self, ref_epoch: Epoch):
        """Per-corepoint multi-scale selection on the reference epoch.

        For scale selection, ``max_window`` is computed from the total time
        range of ``epochs_timeseries`` (not per-target), so that all scales
        have access to the full pool.
        """
        cache_key = id(ref_epoch)
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
        window_ratios = self._as_list(self.max_window_ratio)
        for sr in normal_radii:
            if sr <= 0:
                raise Py4DGeoError(f"normal_radii must be positive, got {sr}")
        for wr in window_ratios:
            if not 0.0 < wr <= 1.0:
                raise Py4DGeoError(
                    f"max_window_ratio must lie in (0, 1], got {wr}"
                )

        combos = [(sr, wr) for sr in normal_radii for wr in window_ratios]
        self._scale_combinations = combos

        # Use full time range as the basis for scale selection's max_window.
        time_range = float(self._epoch_times.max() - self._epoch_times.min())
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

        for k, (sr, wr) in enumerate(combos):
            mw = time_range * wr
            for i in range(n_cp):
                pts, _, _, _, _, _ = self._aggregate_sphere(
                    self.corepoints[i], ref_time, scale_exclude_idx, sr, mw
                )
                if pts is None or len(pts) < 3:
                    continue
                pl, nrm = self._planarity_and_normal(pts)
                if pl > best_pl[i]:
                    best_pl[i] = pl
                    normals[i] = nrm
                    opt_idx[i] = k

        # Fallback for corepoints that never collected enough points
        no_data = ~np.isfinite(best_pl)
        if np.any(no_data):
            logger.warning(
                "TAM3C2: %d/%d corepoints had no scale-selection neighborhood (using orientation vector).",
                int(no_data.sum()),
                n_cp,
            )
            normals[no_data] = self.orientation_vector

        self._ref_normals = normals
        self._opt_scale_idx = opt_idx
        self._ref_cache_key = cache_key

        # Record chosen scale index once (same for every target).
        if not self._diag["scale_idx"]:
            self._diag["scale_idx"] = opt_idx.copy()

    def directions(self):
        if self._ref_normals is None:
            raise Py4DGeoError(
                "TAM3C2.directions() called before any distance computation."
            )
        return self._ref_normals


    def _compute_weights(
        self, dt_array, pts, cp, normal,
        window_size, spatial_r,
    ):
        if self.weighting == Weighting.NONE:
            return None
        offsets = pts - cp
        along = offsets @ normal
        radial_sq = np.einsum("ij,ij->i", offsets, offsets) - along * along
        d_spatial = np.sqrt(np.maximum(radial_sq, 0.0))
        d_time = np.abs(dt_array)

        # Effective temporal window after spacetime-anisotropy rescaling:
        # in normalized (u_s, u_t) coordinates we apply an isotropic kernel
        # to (u_s, u_t / space_time_ratio).  Equivalently, stretch the time
        # window by space_time_ratio:  w_t_eff = window_size * r_st.
        r_st = self.space_time_ratio
        win_eff = window_size * r_st

        if self.weighting == Weighting.LINEAR:
            if self.spatial_weighting:
                ws = np.clip(1.0 - d_spatial / spatial_r, 0.0, 1.0)
            else:
                ws = np.ones_like(d_spatial)
            if win_eff > 0:
                wt = np.clip(1.0 - d_time / win_eff, 0.0, 1.0)
            else:
                wt = np.ones_like(d_time)
            w = ws * wt
        else:  # GAUSSIAN
            sigma = self.sigma_ratio if self.sigma_ratio > 0 else 1.0
            if self.spatial_weighting:
                ws = np.exp(-((d_spatial / spatial_r) ** 2) / (2.0 * sigma * sigma))
            else:
                ws = np.ones_like(d_spatial)
            if win_eff > 0:
                wt = np.exp(-((d_time / win_eff) ** 2) / (2.0 * sigma * sigma))
            else:
                wt = np.ones_like(d_time)
            w = ws * wt

        if not np.any(w > 0):  # degenerate
            w = np.ones_like(w)
        return w

    def _m3c2_and_lod(self, cp, normal, ref_pts, ref_w, tgt_pts, tgt_w):
        pr = (ref_pts - cp) @ normal
        pt = (tgt_pts - cp) @ normal

        if ref_w is None:
            m_r = pr.mean()
            m_t = pt.mean()
            s_r = pr.std(ddof=1)
            s_t = pt.std(ddof=1)
            n_r = float(len(pr))
            n_t = float(len(pt))
        else:
            sw_r = ref_w.sum()
            sw_t = tgt_w.sum()
            m_r = float(np.sum(pr * ref_w) / sw_r)
            m_t = float(np.sum(pt * tgt_w) / sw_t)
            s_r = float(np.sqrt(np.sum(ref_w * (pr - m_r) ** 2) / sw_r))
            s_t = float(np.sqrt(np.sum(tgt_w * (pt - m_t) ** 2) / sw_t))
            sw2_r = float(np.sum(ref_w * ref_w))
            sw2_t = float(np.sum(tgt_w * tgt_w))
            n_r = (sw_r * sw_r) / sw2_r if sw2_r > 0 else float(len(pr))
            n_t = (sw_t * sw_t) / sw2_t if sw2_t > 0 else float(len(pt))

        n_r = max(n_r, 1.0)
        n_t = max(n_t, 1.0)
        dist = float(m_t - m_r)
        lod95 = 1.96 * float(np.sqrt(s_r * s_r / n_r + s_t * s_t / n_t)) + float(
            self.registration_error
        )

        # Create a structured array for uncertainty
        uncertainty = np.array(
            [(lod95, s_r, n_r, s_t, n_t)],
            dtype=[
                ("lodetection", "<f8"),
                ("spread1", "<f8"),
                ("num_samples1", "<i8"),
                ("spread2", "<f8"),
                ("num_samples2", "<i8"),
            ],
        )[0]

        return dist, uncertainty


    def calculate_distances(self, epoch1, epoch2, searchtree=None):
        """Time-adaptive M3C2 distance between ``epoch1`` (ref) and ``epoch2`` (tgt).

        Called once per target by ``SpatiotemporalAnalysis.add_epochs``.
        """
        if self.cyl_radius is None:
            raise Py4DGeoError("TAM3C2 requires cyl_radius (float).")
        if self.corepoints is None:
            raise Py4DGeoError("TAM3C2 requires corepoints to be set.")

        self._ensure_ref_scale_cache(epoch1) # scale selection on reference epoch (sphere aggregation)
        n_cp = self.corepoints.shape[0]

        ref_idx = self._find_epoch_index(epoch1)
        tgt_idx = self._find_epoch_index(epoch2)
        ref_exclude_idx = None if self.include_center_epoch else ref_idx
        tgt_exclude_idx = None if self.include_center_epoch else tgt_idx
        ref_time = epoch1.timestamp.timestamp()
        tgt_time = epoch2.timestamp.timestamp()

        # Per-target max_window is based on the time gap between ref and target.
        time_gap = abs(tgt_time - ref_time)
        if time_gap == 0.0:
            # Degenerate: target == ref. Fall back to full time range.
            time_gap = float(self._epoch_times.max() - self._epoch_times.min())
            if time_gap == 0.0:
                time_gap = 1.0

        distances = np.full(n_cp, np.nan)
        uncertainties = np.full(
            n_cp,
            np.nan,
            dtype=[
                ("lodetection", "<f8"),
                ("spread1", "<f8"),
                ("num_samples1", "<f8"),
                ("spread2", "<f8"),
                ("num_samples2", "<f8"),
            ],
        )

        wur = np.zeros(n_cp)
        wut = np.zeros(n_cp)
        nbr = np.zeros(n_cp, dtype=np.int32)
        nar = np.zeros(n_cp, dtype=np.int32)
        nbt = np.zeros(n_cp, dtype=np.int32)
        nat = np.zeros(n_cp, dtype=np.int32)
        npr = np.zeros(n_cp, dtype=np.int32)
        npt = np.zeros(n_cp, dtype=np.int32)

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
            ref_pts, ref_dt, ref_eidx, b_r, a_r, w_r = ref_pack
            tgt_pts, tgt_dt, tgt_eidx, b_t, a_t, w_t = tgt_pack

            wur[i] = w_r
            wut[i] = w_t
            nbr[i] = b_r
            nar[i] = a_r
            nbt[i] = b_t
            nat[i] = a_t
            npr[i] = 0 if ref_pts is None else len(ref_pts)
            npt[i] = 0 if tgt_pts is None else len(tgt_pts)

            valid = (
                ref_pts is not None
                and tgt_pts is not None
                and len(ref_pts) >= 2
                and len(tgt_pts) >= 2
            )

            if valid:
                ref_w = self._compute_weights(
                    ref_dt, ref_pts, cp, normal, 
                    max_window, self.cyl_radius,
                )
                tgt_w = self._compute_weights(
                    tgt_dt, tgt_pts, cp, normal, 
                    max_window, self.cyl_radius,
                )
                d, u = self._m3c2_and_lod(cp, normal, ref_pts, ref_w, tgt_pts, tgt_w)
                distances[i] = d
                uncertainties[i] = u

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
            elif nbhd_record is not None:
                nbhd_record.append(None)

        self._diag["target_timestamps"].append(epoch2.timestamp)
        self._diag["window_used_ref"].append(wur)
        self._diag["window_used_tgt"].append(wut)
        self._diag["n_before_ref"].append(nbr)
        self._diag["n_after_ref"].append(nar)
        self._diag["n_before_tgt"].append(nbt)
        self._diag["n_after_tgt"].append(nat)
        self._diag["n_points_ref"].append(npr)
        self._diag["n_points_tgt"].append(npt)
        if nbhd_record is not None:
            self._neighborhoods.append(nbhd_record)

        return distances, uncertainties


    def diagnostics(self):
        """Return per-target diagnostics as ``(n_cp, n_targets)`` matrices."""
        out = {}
        for key, vals in self._diag.items():
            if key == "target_timestamps":
                out[key] = list(vals)
            elif key == "scale_idx":
                out[key] = (
                    np.asarray(vals).copy() if isinstance(vals, np.ndarray) else None
                )
            elif vals:
                out[key] = np.column_stack(vals)
            else:
                out[key] = None
        return out

    def save_diagnostics(self, path):
        """Save diagnostic matrices to an ``.npz`` file."""
        diag = self.diagnostics()
        arrays = {
            k: v
            for k, v in diag.items()
            if isinstance(v, np.ndarray)
        }
        ts = diag.get("target_timestamps")
        if ts:
            arrays["target_timestamps"] = np.array(
                [t.isoformat() for t in ts]
            )
        np.savez(path, **arrays)



# Match YYMMDD or YYMMDD_HHMMSS in a filename
_TS_RE = re.compile(r"(?P<date>\d{6})(?:[_-](?P<time>\d{6}))?")


def _parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Parse a YYMMDD or YYMMDD_HHMMSS timestamp from a filename."""
    base = os.path.basename(filename)
    m = _TS_RE.search(base)
    if not m:
        return None
    d = m.group("date")
    t = m.group("time")
    try:
        year = 2000 + int(d[0:2])
        month = int(d[2:4])
        day = int(d[4:6])
        if t:
            return datetime(year, month, day, int(t[0:2]), int(t[2:4]), int(t[4:6]))
        return datetime(year, month, day)
    except ValueError:
        return None


def read_epochs_from_folder(
    folder: str,
    suffixes: Tuple[str, ...] = (".las", ".laz", ".xyz"),
    xyz_kwargs: Optional[dict] = None,
    las_kwargs: Optional[dict] = None,
) -> List[Epoch]:
    """Read every point cloud in ``folder`` whose filename carries a timestamp.

    File extensions ``.las``/``.laz`` are read via ``py4dgeo.read_from_las``,
    ``.xyz`` via ``py4dgeo.read_from_xyz``. The parsed timestamp is set on
    each Epoch. Epochs are returned sorted by timestamp.
    """
    xyz_kwargs = xyz_kwargs or {}
    las_kwargs = las_kwargs or {}

    files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(suffixes)
    )
    epochs: List[Epoch] = []
    for fn in files:
        ts = _parse_timestamp_from_filename(fn)
        if ts is None:
            logger.info("Skipping (no timestamp in filename): %s", fn)
            continue
        path = os.path.join(folder, fn)
        try:
            if fn.lower().endswith((".las", ".laz")):
                epoch = py4dgeo.read_from_las(path, **las_kwargs)
            else:
                epoch = py4dgeo.read_from_xyz(path, **xyz_kwargs)
        except Exception as ex:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", fn, ex)
            continue
        epoch.timestamp = ts
        epochs.append(epoch)

    epochs.sort(key=lambda e: e.timestamp)
    return epochs


def extract_reference_and_others(
    epochs: List[Epoch], reference_timestamp: datetime
) -> Tuple[Epoch, List[Epoch]]:
    """Split an epoch list into ``(reference_epoch, other_epochs_sorted)``."""
    sorted_eps = sorted(epochs, key=lambda e: e.timestamp)
    ref = next(
        (e for e in sorted_eps if e.timestamp == reference_timestamp), None
    )
    if ref is None:
        raise ValueError(
            f"Reference epoch with timestamp {reference_timestamp} not found."
        )
    others = [e for e in sorted_eps if e is not ref]
    return ref, others


def sample_corepoints(
    epoch: Epoch,
    method: str = "voxel",
    voxel_size: float = 1.0,
    n_samples: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample corepoints from an Epoch.

    method
    ------
    ``"voxel"``
        Voxel down-sampling. Returns the centroid of points in each non-empty
        voxel of size ``voxel_size``.
    ``"random"``
        Random selection of ``n_samples`` points.
    """
    cloud = np.asarray(epoch.cloud, dtype=float)

    if method == "voxel":
        coords = np.floor(cloud / float(voxel_size)).astype(np.int64)
        keys = (
            (coords[:, 0].astype(np.int64) * np.int64(73856093))
            ^ (coords[:, 1].astype(np.int64) * np.int64(19349663))
            ^ (coords[:, 2].astype(np.int64) * np.int64(83492791))
        )
        order = np.argsort(keys, kind="stable")
        keys_sorted = keys[order]
        cloud_sorted = cloud[order]
        split = np.where(np.diff(keys_sorted) != 0)[0] + 1
        groups = np.split(cloud_sorted, split)
        return np.stack([g.mean(axis=0) for g in groups])

    if method == "random":
        rng = np.random.default_rng(seed)
        n = n_samples if n_samples is not None else len(cloud)
        n = min(n, len(cloud))
        idx = rng.choice(len(cloud), size=n, replace=False)
        return cloud[idx]

    raise ValueError(f"Unknown corepoint sampling method: {method!r}")


def sweep_space_time_ratio(
    ratios,
    *,
    epochs_timeseries: List[Epoch],
    reference_epoch: Epoch,
    target_epochs: List[Epoch],
    corepoints: np.ndarray,
    tam_kwargs: Optional[dict] = None,
) -> List[dict]:
    """Run TAM3C2 once per candidate ``space_time_ratio`` value.

    For each ratio a fresh :class:`TAM3C2` is built (every other parameter
    coming from ``tam_kwargs``) and :meth:`TAM3C2.calculate_distances` is
    called for every target in ``target_epochs``.  The per-corepoint /
    per-target distance, spread, LoD and effective-sample-count matrices are
    collected so that downstream code (e.g.
    :func:`estimate_space_time_ratio`) can derive selection criteria.

    Parameters
    ----------
    ratios : iterable of float
        Candidate ``space_time_ratio`` values to evaluate.
    epochs_timeseries : list[Epoch]
        Pool of epochs given to TAM3C2 for time-adaptive aggregation.
    reference_epoch : Epoch
    target_epochs : list[Epoch]
    corepoints : np.ndarray
    tam_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`TAM3C2` (everything
        except ``space_time_ratio``, ``epochs_timeseries`` and
        ``corepoints``).  Must include the M3C2 geometry parameters
        (``cyl_radius``, ``max_distance``, ``registration_error``, ...).

    Returns
    -------
    list[dict]
        One dict per ratio with keys:
        ``space_time_ratio``, ``distances``, ``spread1``, ``spread2``,
        ``lod95``, ``num_samples1``, ``num_samples2`` (all
        ``(n_corepoints, n_targets)`` arrays).
    """
    kwargs = dict(tam_kwargs or {})
    for k in ("space_time_ratio", "epochs_timeseries", "corepoints"):
        kwargs.pop(k, None)

    n_cp = len(corepoints)
    n_tgt = len(target_epochs)
    results: List[dict] = []
    for r in ratios:
        tam = TAM3C2(
            epochs_timeseries=epochs_timeseries,
            space_time_ratio=float(r),
            corepoints=corepoints,
            **kwargs,
        )
        D = np.full((n_cp, n_tgt), np.nan)
        S1 = np.full((n_cp, n_tgt), np.nan)
        S2 = np.full((n_cp, n_tgt), np.nan)
        L = np.full((n_cp, n_tgt), np.nan)
        N1 = np.full((n_cp, n_tgt), np.nan)
        N2 = np.full((n_cp, n_tgt), np.nan)
        for k, tgt in enumerate(target_epochs):
            d, u = tam.calculate_distances(reference_epoch, tgt)
            D[:, k] = d
            S1[:, k] = u["spread1"]
            S2[:, k] = u["spread2"]
            L[:, k] = u["lodetection"]
            N1[:, k] = u["num_samples1"]
            N2[:, k] = u["num_samples2"]
        results.append(
            {
                "space_time_ratio": float(r),
                "distances": D,
                "spread1": S1,
                "spread2": S2,
                "lod95": L,
                "num_samples1": N1,
                "num_samples2": N2,
            }
        )
    return results


def estimate_space_time_ratio(
    *,
    epochs_timeseries: List[Epoch],
    reference_epoch: Epoch,
    target_epochs: List[Epoch],
    corepoints: np.ndarray,
    tam_kwargs: Optional[dict] = None,
    candidate_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    tolerance: float = 0.1,
    stable_threshold: float = 0.05,
) -> dict:
    r"""Select ``space_time_ratio`` from a **spread-inflation** criterion.

    Rationale
    ---------
    The per-side along-normal spreads ``spread1`` / ``spread2`` returned by
    M3C2 mix two contributions on a stable corepoint:

    .. math::

        \mathrm{Var}(p) \;=\;
        \sigma_{\mathrm{geom}}^{2}
        \;+\; \mathrm{Var}_{e}\bigl(\bar{p}_{e}\bigr)

    that is, intra-epoch geometric roughness plus inter-epoch drift
    contamination introduced when we average across time.  Increasing
    ``space_time_ratio`` (= more temporal aggregation) lowers LoD but lets
    the second term leak in.  We therefore pick the **largest** ratio whose
    relative spread inflation stays under ``tolerance``.

    Algorithm
    ---------
    1. Run :func:`sweep_space_time_ratio` over ``candidate_ratios``.
    2. Use the **smallest** ratio in the sweep as the geometric-noise
       baseline.  Corepoints with ``max_t |d(cp, t)| < stable_threshold``
       at that baseline are considered *stable*.
    3. For every ratio compute the spread-inflation ratio

       .. math::

           \rho(r_{st}) \;=\;
           \frac{\langle s(r_{st}) \rangle_\text{stable}}
                {\langle s(r_{st}^{\min}) \rangle_\text{stable}},
           \quad
           s = \tfrac{1}{2}(s_\text{ref} + s_\text{tgt}).

    4. Pick the **largest** ratio with :math:`\rho(r_{st}) \le 1 +`
       ``tolerance``.  If no ratio qualifies the one with the smallest
       :math:`\rho` is returned (and ``constraint_satisfied`` is ``False``).

    Parameters
    ----------
    candidate_ratios : tuple of float
        Default ``(0.5, 1.0, 2.0, 4.0)``.
    tolerance : float
        Maximum allowed relative spread inflation.  Default ``0.1`` = 10 %.
    stable_threshold : float
        Maximum |distance| (in metres) for a corepoint to be classed as
        stable when building the baseline.  Default ``0.05``.

    Returns
    -------
    dict with keys
        ``best_ratio``           : float, the selected ``space_time_ratio``
        ``report``               : list[dict], per-ratio ``rho``, mean LoD,
                                   mean effective sample count, mean spread
        ``sweep_results``        : list[dict], raw output of
                                   :func:`sweep_space_time_ratio`
        ``stable_mask``          : np.ndarray[bool], the stable corepoints
        ``baseline_ratio``       : float, the ratio used as baseline
        ``baseline_mean_spread`` : float
        ``constraint_satisfied`` : bool, whether at least one candidate met
                                   the tolerance
    """
    if not 0.0 < tolerance:
        raise ValueError(f"tolerance must be > 0, got {tolerance!r}")
    if not 0.0 < stable_threshold:
        raise ValueError(
            f"stable_threshold must be > 0, got {stable_threshold!r}"
        )

    candidates = tuple(sorted(float(r) for r in candidate_ratios))
    if not candidates:
        raise ValueError("candidate_ratios must contain at least one value.")

    sweep = sweep_space_time_ratio(
        candidates,
        epochs_timeseries=epochs_timeseries,
        reference_epoch=reference_epoch,
        target_epochs=target_epochs,
        corepoints=corepoints,
        tam_kwargs=tam_kwargs,
    )
    # Already in increasing ratio order because we sorted candidates.

    base = sweep[0]
    with np.errstate(invalid="ignore"):
        max_abs = np.nanmax(np.abs(base["distances"]), axis=1)
    stable = np.isfinite(max_abs) & (max_abs < stable_threshold)
    if not stable.any():
        raise Py4DGeoError(
            "estimate_space_time_ratio: no stable corepoints found "
            f"(max |d| < {stable_threshold} m).  Increase stable_threshold "
            "or check the dataset."
        )

    def _mean_spread(entry):
        s1 = entry["spread1"][stable]
        s2 = entry["spread2"][stable]
        return float(0.5 * (np.nanmean(s1) + np.nanmean(s2)))

    baseline_spread = _mean_spread(base)
    if baseline_spread <= 0.0 or not np.isfinite(baseline_spread):
        raise Py4DGeoError(
            "estimate_space_time_ratio: baseline mean spread is "
            f"{baseline_spread!r}; cannot compute inflation ratio."
        )

    report: List[dict] = []
    for entry in sweep:
        s_mean = _mean_spread(entry)
        rho = s_mean / baseline_spread
        report.append(
            {
                "space_time_ratio": entry["space_time_ratio"],
                "rho": float(rho),
                "mean_spread": s_mean,
                "mean_lod95": float(np.nanmean(entry["lod95"])),
                "mean_num_samples": float(
                    0.5
                    * (
                        np.nanmean(entry["num_samples1"])
                        + np.nanmean(entry["num_samples2"])
                    )
                ),
            }
        )

    threshold = 1.0 + float(tolerance)
    eligible = [r for r in report if r["rho"] <= threshold]
    if eligible:
        best = max(eligible, key=lambda r: r["space_time_ratio"])
        satisfied = True
    else:
        best = min(report, key=lambda r: r["rho"])
        satisfied = False

    return {
        "best_ratio": best["space_time_ratio"],
        "report": report,
        "sweep_results": sweep,
        "stable_mask": stable,
        "baseline_ratio": float(base["space_time_ratio"]),
        "baseline_mean_spread": baseline_spread,
        "constraint_satisfied": satisfied,
        "tolerance": float(tolerance),
        "stable_threshold": float(stable_threshold),
    }



__all__ = [
    "Weighting",
    "TAM3C2",
    "sweep_space_time_ratio",
    "estimate_space_time_ratio",
    "read_epochs_from_folder",
    "extract_reference_and_others",
    "sample_corepoints",
]

