"""Time-aggregation baselines for TAM3C2 comparisons.

``TAbaselineOriginal`` aggregates a vertical cylinder (XY radius plus Z limit)
one epoch at a time, stacks the per-corepoint neighborhoods into one cloud per
side, and runs standard M3C2 on those two clouds. It reproduces the aggregation
code used in the published TAM3C2 evaluation - greedy on ``|dt|``, duplicates
kept, normals left to py4dgeo. ``FixedWindowAggregation`` is the non-adaptive
control it was compared against.

Both report ``uncertainty['lodetection']`` as-is: they select their final
neighborhood inside the C++ kernel on a stacked cloud, which the
variance-component machinery in :mod:`py4dgeo.tam3c2_lod` cannot verify it has
reproduced, so no correction is applied to them.
"""

import logging

import numpy as np

import py4dgeo
from py4dgeo.epoch import Epoch
from py4dgeo.tam3c2 import TAM3C2, UNCERTAINTY_DTYPE
from py4dgeo.util import Py4DGeoError

logger = logging.getLogger("py4dgeo")


class _StackedM3C2(TAM3C2):
    """Shared plumbing of the baselines that run plain M3C2 on stacked clouds."""

    def _run_stacked_m3c2(self, cloud_ref, cloud_tgt, epoch1, epoch2):
        """Standard unweighted M3C2 between two stacked clouds.

        py4dgeo estimates the normals on the stacked reference cloud itself,
        without the NaN hygiene of :func:`clean_corepoint_normals` - that is what
        the published evaluation did, and the distances depend on it.
        Returns ``(distances, uncertainties)`` and stores the normals.
        """
        agg_ref, agg_tgt = Epoch(cloud_ref), Epoch(cloud_tgt)
        agg_ref.timestamp = epoch1.timestamp
        agg_tgt.timestamp = epoch2.timestamp
        m3c2 = py4dgeo.M3C2(
            epochs=(agg_ref, agg_tgt),
            corepoints=self.corepoints,
            normal_radii=self._as_list(self.normal_radii),
            cyl_radius=self.cyl_radius,
            max_distance=self.max_distance,
            registration_error=self.registration_error,
            orientation_vector=self.orientation_vector,
        )
        distances, raw = m3c2.run()
        # Copy into the common float dtype so N fields never get truncated downstream.
        uncertainties = np.full(self.corepoints.shape[0], np.nan, dtype=UNCERTAINTY_DTYPE)
        for field in uncertainties.dtype.names:
            uncertainties[field] = raw[field]
        self._set_normals(m3c2.directions(), epoch1)
        return distances, uncertainties

    def _set_normals(self, normals, epoch1):
        self._ref_normals = np.asarray(normals, dtype=float)
        self._ref_cache_key = id(epoch1)


class TAbaselineOriginal(_StackedM3C2):
    """Unweighted time-aggregation baseline, one epoch per aggregation step.

    A separate method from both unweighted TAM3C2 (nearest-first adaptive
    aggregation without weights) and weighted TAM3C2. For every corepoint and
    for the reference and target sides independently:

    1. Select points inside a cylinder perpendicular to the horizontal plane:
       the XY distance to the corepoint must not exceed ``cyl_radius`` and
       the absolute Z difference must not exceed ``max_distance``.
    2. Include the center epoch when requested, then add epochs ONE at a time,
       greedily by ``|dt|`` with ties going to the *after* side.
       ``required_points`` is tested after every single epoch, so aggregation
       stops as soon as it is met; otherwise it continues until the maximum
       temporal window is exhausted.
    3. Keep the two sides separate: reference points are merged only with the
       temporal neighborhoods selected around the reference time, target
       points analogously. Within each side, the selected neighborhoods of all
       corepoints are stacked; overlapping neighborhoods are deliberately
       retained to match the original TA-baseline workflow.
    4. Run standard M3C2 between the aggregated reference and target clouds to
       estimate a normal and distance at every corepoint. ``required_points``
       is only a stopping condition for temporal aggregation; it is not an
       additional output-validity condition.

    ``max_window_ratio`` may be a list; the largest value is used.
    """

    @property
    def name(self):
        return "TAbaselineOriginal"

    def _epoch_order_ties_after(self, center_time, exclude_idx, max_window):
        """Center epoch, then greedy on ``|dt|``; exact ties go to the *after* side.

        Reproduces the two-pointer loop of the published evaluation. Not routed
        through :meth:`TAM3C2._nearest_epoch_order`, which breaks ties the
        other way.
        """
        times = self._epoch_times
        n = len(times)

        center_matches = np.flatnonzero(times == center_time)
        center_idx = int(center_matches[0]) if len(center_matches) else None
        if center_idx is not None and center_idx != exclude_idx:
            yield center_idx

        before = int(np.searchsorted(times, center_time, side="left")) - 1
        after = before + 1
        if after == center_idx:
            after += 1

        while True:
            if before == exclude_idx:
                before -= 1
            if after == exclude_idx:
                after += 1
            cost_after = abs(times[after] - center_time) if after < n else np.inf
            cost_before = abs(times[before] - center_time) if before >= 0 else np.inf
            if cost_after > max_window:
                cost_after = np.inf
            if cost_before > max_window:
                cost_before = np.inf
            if not (np.isfinite(cost_after) or np.isfinite(cost_before)):
                return
            if cost_after <= cost_before:
                yield after
                after += 1
            else:
                yield before
                before -= 1

    def _vertical_cylinder_points(self, cp, ei, bounding_r):
        """Points of epoch ``ei`` in the XY-radius / Z-limit cylinder, or None."""
        idxs = self._kdtrees[ei].query_ball_point(cp, bounding_r)
        if not idxs:
            return None
        pts = self.epochs_timeseries[ei].cloud[idxs]
        offsets = pts - cp
        horizontal_sq = np.einsum("ij,ij->i", offsets[:, :2], offsets[:, :2])
        mask = (horizontal_sq <= self.cyl_radius * self.cyl_radius) & (
            np.abs(offsets[:, 2]) <= self.max_distance
        )
        return pts[mask] if mask.any() else None

    def _aggregate_vertical_cylinder(self, cp, center_time, exclude_idx, max_window):
        """Returns ``(pts, dt, epoch_idx, n_before, n_after, w_used)``."""
        bounding_r = np.sqrt(self.cyl_radius**2 + self.max_distance**2)
        return self._aggregate_greedy(
            self._epoch_order_ties_after(center_time, exclude_idx, max_window),
            center_time,
            lambda ei: self._vertical_cylinder_points(cp, ei, bounding_r),
        )

    def calculate_distances(self, epoch1, epoch2, searchtree=None):
        """Aggregate vertical-cylinder neighborhoods per corepoint, then run M3C2.

        Adaptive neighborhoods are first constructed independently for every
        corepoint and side, all non-empty neighborhoods are then stacked into
        one aggregated reference cloud and one aggregated target cloud, and a
        standard unweighted M3C2 run computes normals, distances and LoD95 for
        all corepoints. Points shared by overlapping corepoint neighborhoods
        therefore occur repeatedly in the aggregated clouds.
        """
        if self.corepoints is None:
            raise Py4DGeoError("TAbaselineOriginal requires corepoints to be set.")
        if self.cyl_radius is None or float(self.cyl_radius) <= 0.0:
            raise Py4DGeoError("TAbaselineOriginal requires cyl_radius > 0.")
        if self.max_distance is None or float(self.max_distance) <= 0.0:
            raise Py4DGeoError("TAbaselineOriginal requires max_distance > 0.")

        self._build_index()
        window_ratio = max(self._check_window_ratios())
        n_cp = self.corepoints.shape[0]
        self._scale_combinations = [(self.cyl_radius, window_ratio)]
        self._opt_scale_idx = np.zeros(n_cp, dtype=np.int32)

        ref_exclude_idx = None if self.include_center_epoch else self._find_epoch_index(epoch1)
        tgt_exclude_idx = None if self.include_center_epoch else self._find_epoch_index(epoch2)
        ref_time = epoch1.timestamp.timestamp()
        tgt_time = epoch2.timestamp.timestamp()
        max_window = self._time_gap(ref_time, tgt_time) * window_ratio

        diag = self._empty_diag_arrays(n_cp)
        nbhd_record = [] if self.keep_neighborhoods else None
        ref_neighborhoods, tgt_neighborhoods = [], []

        for i in range(n_cp):
            cp = self.corepoints[i]
            ref_pack = self._aggregate_vertical_cylinder(cp, ref_time, ref_exclude_idx, max_window)
            tgt_pack = self._aggregate_vertical_cylinder(cp, tgt_time, tgt_exclude_idx, max_window)
            self._store_pack_diag(diag, i, ref_pack, tgt_pack)
            ref_pts, ref_dt, ref_eidx = ref_pack[:3]
            tgt_pts, tgt_dt, tgt_eidx = tgt_pack[:3]
            if ref_pts is not None:
                ref_neighborhoods.append(ref_pts)
            if tgt_pts is not None:
                tgt_neighborhoods.append(tgt_pts)

            if nbhd_record is not None:
                nbhd_record.append(
                    {
                        "ref_pts": ref_pts,
                        "ref_w": None,
                        "ref_dt": ref_dt,
                        "ref_eidx": ref_eidx,
                        "tgt_pts": tgt_pts,
                        "tgt_w": None,
                        "tgt_dt": tgt_dt,
                        "tgt_eidx": tgt_eidx,
                        "cyl_radius": self.cyl_radius,
                        "max_distance": self.max_distance,
                        "window_ratio": window_ratio,
                        "max_window": max_window,
                    }
                )

        if not ref_neighborhoods or not tgt_neighborhoods:
            distances = np.full(n_cp, np.nan)
            uncertainties = np.full(n_cp, np.nan, dtype=UNCERTAINTY_DTYPE)
            self._set_normals(np.tile(self.orientation_vector, (n_cp, 1)), epoch1)
        else:
            # Overlapping corepoint neighborhoods are deliberately kept as
            # separate copies, matching the published aggregation workflow.
            distances, uncertainties = self._run_stacked_m3c2(
                np.vstack(ref_neighborhoods), np.vstack(tgt_neighborhoods), epoch1, epoch2
            )

        self._record_diagnostics(epoch2.timestamp, diag, self._opt_scale_idx)
        if nbhd_record is not None:
            self._neighborhoods.append(nbhd_record)
        return distances, uncertainties


class FixedWindowAggregation(_StackedM3C2):
    """Non-adaptive control: stack ``window_size`` whole epochs, then run M3C2.

    The reference and target epochs are each replaced by the concatenation of
    the ``k`` epochs before them, themselves, and the ``k`` epochs after them
    (``window_size = 2k + 1``), taken by **position in the time series**, not
    by a time window - so the aggregated span varies with how the series is
    sampled. No spatial selection happens at aggregation time; the whole clouds
    go in, and the standard M3C2 cylinder does all the selecting afterwards.

    This is the fixed-window baseline the adaptive methods were compared
    against: it applies the same temporal smoothing everywhere, including where
    the data were already dense enough not to need it.

    ``required_points``, ``max_window_ratio`` and the weighting parameters are
    inherited but unused. Normals come from py4dgeo on the aggregated reference
    cloud, without NaN cleaning - same as :class:`TAbaselineOriginal`.
    """

    def __init__(self, *, window_size: int = 9, **kwargs):
        kwargs.setdefault("max_window_ratio", 1.0)
        kwargs.setdefault("normal_radii", 0.5)
        super().__init__(**kwargs)
        window_size = int(window_size)
        if window_size < 1 or window_size % 2 == 0:
            raise Py4DGeoError(
                f"FixedWindowAggregation: window_size must be a positive odd "
                f"integer, got {window_size}."
            )
        self.window_size = window_size

    @property
    def name(self):
        return "FixedWindowAggregation"

    def _stack_window(self, center_time):
        """Stack the ``window_size`` epochs centred (by position) on ``center_time``."""
        times = self._epoch_times
        matches = np.flatnonzero(times == center_time)
        if not len(matches):
            raise Py4DGeoError(
                "FixedWindowAggregation: the evaluated epoch must be part of "
                "epochs_timeseries (the window is taken by position)."
            )
        center_idx = int(matches[0])
        side = (self.window_size - 1) // 2
        positions = range(max(0, center_idx - side), min(len(times), center_idx + side + 1))
        clouds = [np.asarray(self.epochs_timeseries[i].cloud) for i in positions]
        # dt per point, so the epochs can be told apart afterwards
        dts = np.concatenate(
            [np.full(len(c), times[i] - center_time) for i, c in zip(positions, clouds)]
        )
        n_before = sum(1 for i in positions if i < center_idx)
        n_after = sum(1 for i in positions if i > center_idx)
        return np.vstack(clouds), dts, n_before, n_after, float(np.abs(dts).max())

    def calculate_distances(self, epoch1, epoch2, searchtree=None):
        if self.corepoints is None:
            raise Py4DGeoError("FixedWindowAggregation requires corepoints.")
        if self.cyl_radius is None:
            raise Py4DGeoError("FixedWindowAggregation requires cyl_radius.")

        self._build_index(trees=False)  # whole epochs are stacked, no spatial queries
        n_cp = self.corepoints.shape[0]
        cloud_ref, dt_ref, b_r, a_r, w_r = self._stack_window(epoch1.timestamp.timestamp())
        cloud_tgt, dt_tgt, b_t, a_t, w_t = self._stack_window(epoch2.timestamp.timestamp())
        if self.keep_neighborhoods:
            self._agg_clouds = {"ref": (cloud_ref, dt_ref), "tgt": (cloud_tgt, dt_tgt)}

        distances, uncertainties = self._run_stacked_m3c2(cloud_ref, cloud_tgt, epoch1, epoch2)

        ones = np.ones(n_cp)
        diag = {
            "window_used_ref": ones * w_r,
            "window_used_tgt": ones * w_t,
            "n_before_ref": (ones * b_r).astype(np.int32),
            "n_after_ref": (ones * a_r).astype(np.int32),
            "n_before_tgt": (ones * b_t).astype(np.int32),
            "n_after_tgt": (ones * a_t).astype(np.int32),
            "n_points_ref": uncertainties["num_samples1"],
            "n_points_tgt": uncertainties["num_samples2"],
        }
        self._record_diagnostics(epoch2.timestamp, diag, np.zeros(n_cp, dtype=np.int32))
        return distances, uncertainties


__all__ = ["TAbaselineOriginal", "FixedWindowAggregation"]
