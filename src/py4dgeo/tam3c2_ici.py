"""TAM3C2 with a data-adaptive borrowing depth (ICI stopping rule).

Split out of :mod:`py4dgeo.tam3c2` so that the published method and the
post-publication variants can be maintained separately. See
:class:`TAM3C2ICI` for the rule itself.
"""

import logging

import numpy as np

from py4dgeo.tam3c2 import TAM3C2, _Neighborhood
from py4dgeo.util import Py4DGeoError

logger = logging.getLogger("py4dgeo")


class TAM3C2ICI(TAM3C2):
    """TAM3C2 with a data-adaptive borrowing depth instead of a temporal kernel.

    Motivation
    ----------
    The temporal kernel of :class:`TAM3C2` needs to know how much the surface
    actually moved between the centre time and a candidate epoch. That quantity
    is not recoverable where aggregation is needed most: at a sparse corepoint
    every epoch holds 0-2 points, so per-epoch means, per-epoch spreads and
    local rates of change are all unmeasurable, and the centre epoch may not
    even exist to anchor against. Sizing the kernel on the reach that was
    actually used instead makes the outermost epoch's weight the constant
    ``exp(-1/(2*sigma_ratio**2))`` regardless of how far away it really is - at
    a borrowing depth near two epochs the kernel collapses to a single hard
    shrinkage factor towards plain M3C2.

    What *is* measurable is whether the running estimate stays put as epochs are
    added: the accumulated sample grows even when no single epoch is usable on
    its own. This class therefore drops the temporal kernel and spends the
    freed degree of freedom on **how deep to borrow**.

    Rule
    ----
    Epochs enter in the nearest-first order of
    :meth:`TAM3C2._nearest_epoch_order`. After each accepted epoch ``k`` the
    accumulated along-normal projections give

    ``D_k = [ mean_k - kappa * SE_k , mean_k + kappa * SE_k ]``

    with ``SE_k = std_k / sqrt(N_k)`` the standard error of the mean. The
    running intersection of all ``D_j``, ``j <= k``, is maintained; a candidate
    epoch whose interval empties that intersection is rolled back and
    aggregation stops. This is Lepski's intersection-of-confidence-intervals
    rule (Goldenshluger-Lepski; Katkovnik's ICI-LPA) applied to borrowing depth
    rather than to a smoothing bandwidth.

    ``kappa`` is dimensionless. No epoch timestamp enters the test - ``dt``
    only ever orders the candidates - so second, hourly, monthly and yearly
    series behave identically, and irregular sampling needs no special case.

    Behaviour by change regime
    --------------------------
    - Drift below the noise floor: the mean does not move enough to empty the
      intersection, so borrowing runs to ``max_window``. This is the right
      answer, not a failure to detect - such borrowing buys variance at
      negligible bias.
    - Steady motion with near-symmetric borrowing: the ``-1``/``+1`` pairing of
      the nearest-first order cancels first-order drift inside the mean, the
      test sees nothing, and borrowing continues. Asymmetry (series boundary,
      irregular sampling) breaks that cancellation, the mean does move, and the
      rule brakes - which is exactly the case a symmetric-pairing baseline
      cannot detect.
    - Fast, one-sided change: detected within an epoch or two, so depth stays at
      the ``required_points`` floor. Borrowing further would average two
      genuinely different surfaces.

    Coverage policy
    ---------------
    ``required_points`` becomes a floor rather than a target: below it epochs
    are accepted unconditionally (a conflict only raises the ``ici_early`` flag)
    so completeness is never traded away, and above it the data decide. This
    also means depth routinely exceeds ``required_points``, unlike in the
    parent class.

    Intended with ``weighting=Weighting.NONE``: every accepted point carries
    weight one, so the Kish effective sample size equals ``N`` and the LoD95
    contraction earned by borrowing deeper is real rather than an artefact of
    reweighting.

    Parameters
    ----------
    ici_kappa : float
        Half-width of the confidence intervals in standard errors. Larger is
        more permissive (deeper borrowing). Default 2.0.
    """

    #: Accumulated points needed before the interval test may fire. Below this
    #: the half-width is dominated by the error in its own sigma estimate
    #: (relative standard error ``1/sqrt(2*(n-1))``) and the rule stops on noise.
    MIN_POINTS_FOR_TEST = 8

    def __init__(self, *, ici_kappa: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        if ici_kappa <= 0:
            raise Py4DGeoError(
                f"TAM3C2ICI: ici_kappa must be > 0, got {ici_kappa}."
            )
        self.ici_kappa = float(ici_kappa)
        # Filled by _aggregate_cylinder, drained by calculate_distances. Two
        # entries per corepoint (reference side first, then target side).
        self._ici_calls = []
        for key in ("ici_stopped_ref", "ici_stopped_tgt", "ici_early_ref", "ici_early_tgt"):
            self._diag[key] = []

    @property
    def name(self):
        return "TAM3C2ICI"

    def _aggregate_cylinder(
        self, cp, normal, center_time, exclude_idx, cyl_radius, max_distance, max_window
    ):
        """Nearest-first aggregation that stops when the epochs stop agreeing.

        Signature and return value match
        :meth:`TAM3C2._aggregate_cylinder` so the parent's
        ``calculate_distances`` is reused unchanged.
        """
        bounding_r = np.sqrt(cyl_radius * cyl_radius + max_distance * max_distance)
        nb = _Neighborhood()

        # Running sum / sum of squares of the along-normal projections, so a
        # candidate costs O(points in the new epoch) rather than a rescan of
        # everything accepted so far.
        sum_p = 0.0
        sum_p2 = 0.0
        # (mean, N) per accepted step. The intersection is rebuilt from this
        # history every step rather than carried forward, so that all intervals
        # share ONE sigma: an early, badly underestimated sigma would otherwise
        # freeze an absurdly narrow interval that the next step cannot meet.
        hist = []
        stopped = False
        early = False

        for ei in self._nearest_epoch_order(center_time, exclude_idx, max_window):
            pts, along = self._cylinder_points_for_epoch(
                cp, normal, ei, cyl_radius, max_distance, bounding_r
            )
            if pts is None:
                continue

            # Evaluate this epoch without committing to it yet.
            cand_sum = sum_p + float(along.sum())
            cand_sum2 = sum_p2 + float(np.dot(along, along))
            cand_n = nb.total + len(pts)
            cand_mean = cand_sum / cand_n

            if not hist or cand_n < self.MIN_POINTS_FOR_TEST:
                conflict = False
            else:
                var = max(
                    (cand_sum2 - cand_n * cand_mean * cand_mean) / (cand_n - 1), 0.0
                )
                sigma = np.sqrt(var)
                lo, hi = -np.inf, np.inf
                for m_j, n_j in hist:
                    half_j = self.ici_kappa * sigma / np.sqrt(n_j)
                    lo, hi = max(lo, m_j - half_j), min(hi, m_j + half_j)
                half = self.ici_kappa * sigma / np.sqrt(cand_n)
                lo, hi = max(lo, cand_mean - half), min(hi, cand_mean + half)
                conflict = lo > hi

            if nb.total < self.required_points:
                # Floor phase: completeness wins, so a conflict is recorded but
                # never acted on. History restarts from this step - carrying an
                # already-empty intersection forward would force a stop the
                # instant the floor is cleared.
                if conflict:
                    early = True
                    hist = []
            elif conflict:
                # Adaptive phase: the floor is covered, so the data decide.
                stopped = True
                break

            hist.append((cand_mean, cand_n))
            nb.add(pts, self._epoch_times[ei] - center_time, ei)
            sum_p, sum_p2 = cand_sum, cand_sum2

        self._ici_calls.append((stopped, early))
        return nb.result()

    def calculate_distances(self, epoch1, epoch2, searchtree=None):
        """Delegate to :class:`TAM3C2` and split out the per-side ICI flags.

        The parent loops over corepoints calling ``_aggregate_cylinder`` twice
        each, reference side first, so the flag buffer arrives in that order.
        """
        self._ici_calls = []
        out = super().calculate_distances(epoch1, epoch2, searchtree)

        n_cp = self.corepoints.shape[0]
        flags = np.asarray(self._ici_calls, dtype=bool)
        if flags.shape != (2 * n_cp, 2):
            # Should not happen; keep the diagnostic honest rather than guess.
            logger.warning(
                "TAM3C2ICI: expected %d aggregation calls, got %d - "
                "ICI diagnostics for this target are dropped.",
                2 * n_cp,
                len(self._ici_calls),
            )
            blank = np.zeros(n_cp, dtype=bool)
            stopped_ref = stopped_tgt = early_ref = early_tgt = blank
        else:
            stopped_ref, early_ref = flags[0::2, 0], flags[0::2, 1]
            stopped_tgt, early_tgt = flags[1::2, 0], flags[1::2, 1]

        for key, flag in (("ici_stopped_ref", stopped_ref), ("ici_stopped_tgt", stopped_tgt),
                          ("ici_early_ref", early_ref), ("ici_early_tgt", early_tgt)):
            self._diag[key].append(flag)
        self._ici_calls = []
        return out


__all__ = ["TAM3C2ICI"]
