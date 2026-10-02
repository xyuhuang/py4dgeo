"""Choosing ``space_time_ratio`` for :class:`py4dgeo.TAM3C2`.

Split out of :mod:`py4dgeo.tam3c2` - it is a parameter-selection driver, not part
of the algorithm, and it is the only thing in there that runs TAM3C2 in a loop.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from py4dgeo.epoch import Epoch
from py4dgeo.tam3c2 import TAM3C2
from py4dgeo.util import Py4DGeoError

logger = logging.getLogger("py4dgeo")


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


__all__ = ["sweep_space_time_ratio", "estimate_space_time_ratio"]
