"""方差分量 LoD —— 跨期聚合的探测限修正。

背景
----
M3C2 (Lague et al., 2013) 的探测限是

    LoD95 = 1.96 * (sqrt(s1^2/N1 + s2^2/N2) + e_reg)

它假设邻域里的 N 个点是对同一个量的独立测量。两期比较时成立：期间的系统
偏移由常数 e_reg 承担。跨期聚合后不成立 —— 借来的 k 期各有自己的整体偏移

    p_ij = mu + b_j + eps_ij,   Var(b_j) = sigma_b^2,  Var(eps_ij) = sigma_w^2

b_j 被第 j 期所有点共享，所以同一期内多平均几个点消不掉它。正确的方差是

    Var(m) = sigma_b^2 / k_eff  +  sigma_w^2 / N_eff
             ^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^
             期间：按"期数"缩小     期内：按"点数"缩小

k_eff 与 N_eff 是同一个 Kish (1965) 有效样本量公式，只是一个作用在"期"这一
层、一个在"点"这一层。sigma_w^2 / sigma_b^2 用单因素随机效应模型的矩估计
(Searle et al., 1992) 从邻域点自己算出来 —— 零新参数，不需要任何真值。

适用范围
--------
只对**逐 corepoint 独立求解**的方法有效 (``TAM3C2`` / ``TAM3C2ICI``)：
``keep_neighborhoods=True`` 存下来的邻域**就是**它算 LoD 用的那批点，所以复现
是精确的，``validate`` 会用 C++ 报告的 spread / num_samples 核对这一点。

``TAbaselineOriginal`` / ``FixedWindowAggregation`` 不适用：它们把邻域堆成一朵
点云再让 C++ 内核在上面重新取柱体，实际用的点里混入了别的 corepoint 贡献的点，
复现无法验证。它们直接报 ``uncertainty['lodetection']``，不做校正。

参考
----
Lague, Brodu & Leroux (2013), ISPRS J. Photogramm. Remote Sens. 82:10-26.
Winiwarter, Anders & Höfle (2021), M3C2-EP, ISPRS J. 178:240-258.
Kish (1965), Survey Sampling, Wiley.
Searle, Casella & McCulloch (1992), Variance Components, Wiley.
Cressie (1993), Statistics for Spatial Data, Wiley.  (交叉验证预测方差)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "side_components", "components_tam3c2",
    "collect", "validate", "corrected_lod", "corrected_lod_for",
    "loeo_rho", "sigma_b_summary",
]


# ---------------------------------------------------------------------------
# 核心：一侧邻域的方差分解
# ---------------------------------------------------------------------------
def side_components(p, w, e):
    """把一侧的投影方差按 epoch 分解成期内 / 期间两部分。

    Parameters
    ----------
    p : (n,) 沿法向的投影坐标 [m]
    w : (n,) 权重；``None`` 表示等权
    e : (n,) epoch 标签（整数或 dt 值皆可，只用来分组）

    Returns
    -------
    dict 或 None（点数 < 2 时）

    字段含义（用 M3C2 的说法）
        s2_tot  现公式里那个 s^2 —— 全部点投影后的加权方差
        sw2     各期**内部**方差按权重合并 —— "每期自己量得多散"
        A       各期**均值之间**的加权方差 —— "各期彼此差多少"
        恒等式  s2_tot == sw2_all + A（sw2_all 含单点期，其期内方差为 0）
        k_eff   期这一层的 Kish 有效数
        N_eff   点这一层的 Kish 有效数（与现公式的 N 一致）
        sb2_raw sigma_b^2 的矩估计 = max(0, (A - sw2*d) / c)
                c 修正"A 绕自身均值算所以偏小"，d 扣掉混进 A 的期内噪声
    """
    p = np.asarray(p, float)
    if p.size < 2:
        return None
    w = np.ones(p.size) if w is None else np.asarray(w, float)
    e = np.asarray(e)
    W = float(w.sum())
    if not np.isfinite(W) or W <= 0:
        return None

    uniq = np.unique(e)
    k = int(uniq.size)
    Wj = np.array([w[e == j].sum() for j in uniq], float)
    nj = np.array([(e == j).sum() for j in uniq], float)
    pj = np.array([float(np.sum(w[e == j] * p[e == j]) / Wj[m])
                   for m, j in enumerate(uniq)])
    ss_j = np.array([float(np.sum(w[e == j] * (p[e == j] - pj[m]) ** 2))
                     for m, j in enumerate(uniq)])
    w2j = np.array([float(np.sum(w[e == j] ** 2)) for j in uniq])
    # ML(有偏, 除以 W_j)版本只用于恒等式 s2_tot == sw2_all + A
    s2j_ml = ss_j / Wj
    # sigma_w^2 是总体参数, 要用无偏估计: 分母 W_j - sum(w^2)/W_j,
    # 等权时退化为 n_j - 1。用有偏版会把 sigma_w^2 压低 (n_j-1)/n_j。
    den_j = Wj - w2j / Wj
    s2j_unb = np.where(den_j > 0, ss_j / np.where(den_j > 0, den_j, 1.0), 0.0)
    pbar = float(np.sum(Wj * pj) / W)

    N_eff = W * W / float(np.sum(w * w))
    k_eff = W * W / float(np.sum(Wj * Wj))
    s2_tot = float(np.sum(w * (p - pbar) ** 2) / W)

    f = Wj / W
    A = float(np.sum(f * (pj - pbar) ** 2))
    sw2_all = float(np.sum(f * s2j_ml))
    m2 = den_j > 0                    # 单点期对"期内有多散"没有信息
    sw2 = (float(np.sum(Wj[m2] * s2j_unb[m2]) / np.sum(Wj[m2]))
           if m2.any() else 0.0)

    # E[A] = sigma_b^2 * c + sigma_w^2 * d   (单因素随机效应, 非平衡设计)
    #   c = 1 - sum(f^2)                     A 绕自身均值算所以偏小, 除回来
    #   d = sum(f(1-f) * v_j),  v_j = Var(pbar_j)/sigma_w^2 = sum(w^2)/W_j^2
    #       等权时 v_j = 1/n_j。这一项是"每期均值本身带的噪声"混进 A 的部分。
    v_j = w2j / (Wj * Wj)
    c = 1.0 - float(np.sum(f * f))
    d = float(np.sum(f * (1.0 - f) * v_j))
    sb2_raw = (A - sw2 * d) / c if c > 1e-12 else np.nan   # k=1 -> c=0 -> nan

    return dict(k=k, k_eff=k_eff, N_eff=N_eff, s2_tot=s2_tot, sw2=sw2,
                sw2_all=sw2_all, sb2_raw=sb2_raw, A=A, c=c, d=d, mean=pbar,
                p=p, w=w, e=e, uniq=uniq, Wj=Wj, nj=nj, pj=pj)


# ---------------------------------------------------------------------------
# TAM3C2 / TAM3C2ICI —— 存下来的邻域就是算 LoD 用的那批点
# ---------------------------------------------------------------------------
def components_tam3c2(alg, target=-1):
    """逐 corepoint 的 (ref, tgt) 方差分解。``target`` 选第几次 calculate_distances
    (默认最近一次; 与传给 validate / corrected_lod 的 ``unc`` 必须是同一个 target)。"""
    nb = _neighborhoods(alg, target)
    normals = np.asarray(alg.directions(), float)
    cps = np.asarray(alg.corepoints, float)
    comps = []
    for i, rec in enumerate(nb):
        if rec is None:
            comps.append((None, None))
            continue
        n = normals[i]
        if not np.isfinite(n).all():      # degenerate PCA -> no usable geometry
            comps.append((None, None))
            continue
        out = []
        for side in ("ref", "tgt"):
            pts = rec[f"{side}_pts"]
            if pts is None or len(pts) < 2:
                out.append(None)
                continue
            proj = (np.asarray(pts, float) - cps[i]) @ n
            out.append(side_components(proj, rec.get(f"{side}_w"),
                                       rec[f"{side}_eidx"]))
        comps.append(tuple(out))
    return comps


def collect(alg, verbose=False, target=-1):
    """逐 corepoint 重建邻域并分解方差。

    只支持 :class:`~py4dgeo.TAM3C2` 及其子类里**逐 corepoint 独立求解**的那些
    (TAM3C2 / TAM3C2ICI): 它们 ``keep_neighborhoods=True`` 存下来的邻域就是算
    LoD 用的那批点, 所以复现是精确的。

    TAbaselineOriginal / FixedWindowAggregation 不在此列 —— 它们把邻域堆成一朵
    云再让 C++ 内核重新取柱体, 复现无法验证, 因此不做方差分量校正, 直接用
    ``uncertainty['lodetection']``。(``verbose`` 保留只为兼容旧调用, 不起作用。)
    """
    from py4dgeo.tabaseline import FixedWindowAggregation, TAbaselineOriginal

    if isinstance(alg, (TAbaselineOriginal, FixedWindowAggregation)):
        raise RuntimeError(
            f"{type(alg).__name__}: 方差分量校正只适用于逐 corepoint 求解的 "
            "TAM3C2 / TAM3C2ICI。")
    return components_tam3c2(alg, target)


def _neighborhoods(alg, target=-1):
    nb = getattr(alg, "_neighborhoods", None)
    if not nb:
        raise RuntimeError(
            f"{type(alg).__name__}: 没有存下邻域。构造时(或跑之前)把 "
            "keep_neighborhoods 设为 True。")
    return nb[target]


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def validate(comps, unc, ddof_candidates=(0, 1), tol=1e-6):
    """把复算的 (s, N) 对照方法自己报告的 spread / num_samples。

    只有邻域复现正确 (而且 comps 与 unc 是同一个 target)，s 和 N 才会对上。
    返回 dict，``best_ddof`` 是与报告值吻合的那个约定。
    """
    rep_s = np.column_stack([unc["spread1"], unc["spread2"]]).astype(float)
    rep_n = np.column_stack([unc["num_samples1"], unc["num_samples2"]]).astype(float)
    got_s = np.full_like(rep_s, np.nan)
    got_n = np.full_like(rep_n, np.nan)
    for i, RT in enumerate(comps):
        for j, S in enumerate(RT):
            if S is None:
                continue
            # 报告的 num_samples 是 N_eff, 不是原始点数 —— 等权时两者相等
            # (Kish: n^2/n = n), 加权时 N_eff 是非整数。拿点数去比会让所有
            # 加权方法永远显示 NOT EXACT, 那是口径错配, 不是复现失败。
            got_n[i, j] = S["N_eff"]
            got_s[i, j] = np.sqrt(S["s2_tot"])          # ddof=0
    ok = np.isfinite(rep_s) & np.isfinite(got_s) & np.isfinite(rep_n)
    res = dict(n_compared=int(ok.sum()))
    if not ok.any():
        return res
    # 整数比较对加权方法不成立, 用相对容差
    res["N_match_pct"] = 100.0 * np.mean(
        np.abs(rep_n[ok] - got_n[ok]) <= 1e-9 * np.maximum(np.abs(rep_n[ok]), 1.0))
    best, best_err = None, np.inf
    for ddof in ddof_candidates:
        adj = np.where(got_n > ddof, got_n / np.maximum(got_n - ddof, 1), 1.0)
        s_try = got_s * np.sqrt(adj)
        err = np.nanmedian(np.abs(s_try[ok] - rep_s[ok])
                           / np.maximum(rep_s[ok], 1e-12))
        res[f"s_relerr_ddof{ddof}"] = float(err)
        if err < best_err:
            best, best_err = ddof, err
    res["best_ddof"] = best
    res["s_relerr"] = float(best_err)
    res["exact"] = bool(res.get("N_match_pct", 0) > 99.0 and best_err < tol)
    return res


def sigma_b_summary(comps, pct=(25, 50, 75, 90)):
    v = np.array([S["sb2_raw"] for RT in comps for S in RT
                  if S is not None and np.isfinite(S["sb2_raw"])])
    if v.size == 0:
        return {}
    v = np.sqrt(np.maximum(v, 0.0))
    return {f"sigma_b_p{q}_cm": float(np.percentile(v, q) * 100) for q in pct}


# ---------------------------------------------------------------------------
# 修正后的 LoD
# ---------------------------------------------------------------------------
def _reported_var(S):
    """Var(mean) exactly as TAM3C2._m3c2_and_lod reports it: s^2 (ddof on N_eff) / N."""
    n = S["N_eff"]
    s2 = S["s2_tot"] * (n / (n - 1.0) if n > 1.0 else 1.0)
    return s2 / max(n, 1.0)


def corrected_lod(comps, lod_old, e_reg_old, e_reg_new, z=1.96, reg_inside=True):
    """比值法：把方差分量的修正**比例**施加到方法自己报告的 LoD 上。

    这样不依赖复算的方差与内部实现逐位一致（权重归一化细节等）。k=1 的一侧
    退回原式 —— 因此 plain M3C2 与所有深度=1 的 corepoint 结果逐点不变。

    分母 v_old 必须是方法报告 LoD 时**实际用的**方差: TAM3C2 与 C++ 内核都用
    无偏 spread (s^2 = s2_tot * N/(N-1))。分子里 sw2 / sb2 也是无偏估计, 所以
    分母若用有偏的 s2_tot/N, 比值会系统性偏大 sqrt(N/(N-1)) (N=10 时约 +5%)。
    """
    lod_old = np.asarray(lod_old, float)
    # The constant term to strip and re-add depends on where e_reg sits:
    # z * e_reg when it is inside the factor, plain e_reg when outside.
    c_old = z * e_reg_old if reg_inside else e_reg_old
    c_new = z * e_reg_new if reg_inside else e_reg_new
    lod_new = np.full(len(comps), np.nan)
    ratio = np.full(len(comps), np.nan)
    n_used = n_shrunk = 0
    for i, (R, T) in enumerate(comps):
        if R is None or T is None or not np.isfinite(lod_old[i]):
            continue
        v_old = v_new = 0.0
        applied = False
        for S in (R, T):
            v_o = _reported_var(S)
            if S["k"] >= 2 and np.isfinite(S["sb2_raw"]):
                v_n = max(S["sb2_raw"], 0.0) / S["k_eff"] + S["sw2"] / S["N_eff"]
                applied = True
            else:
                v_n = v_o
            v_old += v_o
            v_new += v_n
        if v_old <= 0 or not np.isfinite(v_new):
            continue
        r = float(np.sqrt(v_new / v_old))
        ratio[i] = r
        lod_new[i] = max(lod_old[i] - c_old, 0.0) * r + c_new
        if applied:
            n_used += 1
            n_shrunk += int(r < 1.0)
    return lod_new, ratio, n_used, n_shrunk


def corrected_lod_for(alg, unc, e_reg, e_reg_new=None, reg_inside=True,
                      require_exact=True, verbose=False, target=-1):
    """collect + validate + corrected_lod in one call.

    Returns ``(lod, info)``. ``lod`` is the variance-component LoD when the
    neighborhood reproduction checks out, and the method's own ``lodetection``
    otherwise - so a caller can apply the correction wherever it is trustworthy
    without hand-wiring the three steps and the guard every time.

    ``info`` carries ``applied`` (bool), the ``validate`` result, and the
    ``ratio`` / ``n_used`` / ``n_shrunk`` diagnostics of :func:`corrected_lod`.

    ``require_exact=False`` applies the correction regardless of what
    :func:`validate` says. Only do that when you have another reason to trust
    the reproduction.

    ``target`` picks which ``calculate_distances`` call of ``alg`` the stored
    neighborhoods come from (default: the latest); ``unc`` must be that same
    target's uncertainty, otherwise :func:`validate` reports a mismatch and the
    method's own LoD is returned.
    """
    lod_old = np.asarray(unc["lodetection"], float)
    comps = collect(alg, target=target)
    v = validate(comps, unc)
    exact = bool(v.get("exact"))
    info = dict(valid=v, exact=exact, applied=False, comps=comps)
    if require_exact and not exact:
        info["lod"] = lod_old
        return lod_old, info
    lod_new, ratio, n_used, n_shrunk = corrected_lod(
        comps, lod_old, e_reg, e_reg if e_reg_new is None else e_reg_new,
        reg_inside=reg_inside)
    # corrected_lod leaves NaN where a side had no components; fall back there
    lod = np.where(np.isfinite(lod_new), lod_new, lod_old)
    info.update(applied=True, ratio=ratio, n_used=n_used, n_shrunk=n_shrunk,
                lod=lod)
    return lod, info


# ---------------------------------------------------------------------------
# 无真值自检：留一期交叉验证
# ---------------------------------------------------------------------------
def loeo_rho(comps, use_vc, max_cp=None, seed=0):
    """藏起一期，用其余期预测它，看标准化残差的 std。

    rho ≈ 1 -> LoD 诚实；rho > 1 -> LoD 低估了这么多倍。
    不使用任何真值，因此用户可以在自己数据上跑。需要一侧借到 >= 3 期。
    """
    idx = np.arange(len(comps))
    if max_cp is not None and idx.size > max_cp:
        idx = np.random.default_rng(seed).choice(idx, max_cp, replace=False)
    zs = []
    for i in idx:
        for S in comps[i]:
            if S is None or S["k"] < 3:
                continue
            Wj, nj, pj = S["Wj"], S["nj"], S["pj"]
            sb2 = max(S["sb2_raw"], 0.0) if np.isfinite(S["sb2_raw"]) else 0.0
            for j in range(pj.size):
                if nj[j] < 2:
                    continue
                keep = np.ones(pj.size, bool)
                keep[j] = False
                Wr, pr = Wj[keep], pj[keep]
                Wt = Wr.sum()
                if Wt <= 0:
                    continue
                m_rest = float(np.sum(Wr * pr) / Wt)
                k_eff_r = Wt * Wt / float(np.sum(Wr * Wr))
                n_rest = float(nj[keep].sum())
                if use_vc:
                    v_j = sb2 + S["sw2"] / nj[j]          # 留出的那期 k=1
                    v_r = sb2 / k_eff_r + S["sw2"] / n_rest
                else:
                    v_j = S["s2_tot"] / nj[j]
                    v_r = S["s2_tot"] / n_rest
                den = np.sqrt(v_j + v_r)
                if den > 0:
                    zs.append((pj[j] - m_rest) / den)
    zs = np.asarray(zs)
    return (float(np.std(zs)), int(zs.size)) if zs.size else (np.nan, 0)
