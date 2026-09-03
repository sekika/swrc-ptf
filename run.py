#!/usr/bin/env python3
"""run.py — reproduce the full computation of a paper currently under submission.

Every analysis program written for the study is included here as a function (each with
its own docstring); ``main()`` calls them in order. Running

    python3 run.py

from this directory reads the GSHP retention data placed in ``data/`` and regenerates:

  result/   computed numerical results (fits, model-support tables, PTF LORO, bootstrap CIs)
  fig/      the seven manuscript figures (SVG)
  table/    numeric tables computed from the data (table1, table3, tableS1, tableS3, tableS4)

The pipeline (functions grouped by stage):

  Stage 1  fit_dualvgch_*      dual-VG-CH (theta_r = 0) fit of every eligible curve
  Stage    vg_fit_aic_*        single-VG (theta_r free) fit; the dual-VG-CH AICc is the one
                               recorded by Stage 1 at fit time (one fit per model), used to
                               classify each curve DVC-type / VG-type
           degeneracy_*        degeneracy-pattern proportions and single-VG equivalence
           identifiability_*   multistart / bootstrap / profile audit (canonical mode order)
           wprofile_*          w1 profile-RMSE example curves (Fig 5a)
           support_*           measurement-conditioned model support (Tables 1, Figs 2-3, 5b)
           ptf_*               support-stratified linear PTFs, all-cell LORO, 2x2, cluster CIs
           downsample_*        within-curve downsampling support test (Fig 4)
  figures  make_figures        render the computed figures from the results
  tables   make_tables         write the numeric tables

Model definitions (unsatfit):
  VG   = 'VG', const=['q=1']            -> free (theta_r, theta_s, alpha, n), k = 4
  DVC  = 'dual-VG-CH', const=['qr=0','q=1'] -> free (theta_s, alpha, w1, n1, n2), theta_r = 0, k = 5

The dual-VG-CH AICc is obtained one-shot: it is recorded at the single Stage-1 dual-VG-CH
fit and reused for the VG-vs-DVC comparison, so each model is fitted once per curve.

Determinism: single-start fits use unsatfit's data-derived get_init(); the multistart,
bootstrap, and downsampling steps use fixed seeds. Results are byte-reproducible up to
BLAS/LAPACK last-bit differences between environments (this affects only unrounded
floats; all rounded/reported values are stable).

Dependencies: see requirements.txt (unsatfit, numpy, pandas, scipy, scikit-learn, matplotlib).
"""

import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# paths (relative to the repository root)
# ---------------------------------------------------------------------------
DATA_CSV = os.path.join("data", "WRC_dataset_surya_et_al_2021_final.csv")
RESULT = "result"
FIG = "fig"
TABLE = "table"
ENCODING = "latin-1"

# stage output directories (under result/)
R_STAGE1 = os.path.join(RESULT, "stage1-dualvgch")
R_VG = os.path.join(RESULT, "vg-fit-aic")
R_DEGEN = os.path.join(RESULT, "degeneracy")
R_IDENT = os.path.join(RESULT, "identifiability")
R_WPROF = os.path.join(RESULT, "wprofile")
R_SUPPORT = os.path.join(RESULT, "support-analysis")
R_PTF = os.path.join(RESULT, "ptf-typed")
R_DOWN = os.path.join(RESULT, "downsample")

# fitting / analysis constants (identical to the study programs)
RHO_P_DEFAULT = 2.65
OC_THRESHOLD = 6.0
MIN_POINTS = 5
BASIC_PTF = ["sand_tot_psa", "clay_tot_psa", "db_od", "depth_mid"]
BASIC_COMPLETE = ["sand_tot_psa", "clay_tot_psa", "db_od"]
NPOINT_BINS = [(5, 5), (6, 6), (7, 7), (8, 10), (11, 15), (16, 30), (31, 10 ** 9)]
HEADRANGE_BINS = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 99)]
VG_TARGETS = [("theta_s_vg", "id"), ("theta_r_vg", "id"), ("alpha_vg", "log10"), ("m_vg", "id")]
DVC_TARGETS = [("theta_s", "id"), ("alpha", "log10"), ("w1", "id"), ("m1", "id"), ("m2", "id")]
BOOT_N = 2000
BOOT_SEED = 20260821
DOWN_TARGETS = [9, 7]
DOWN_SAMPLE_N = 800
DOWN_SEED = 20260821
META_COLS = ["profile_id", "reference", "method", "tex_psda",
             "sand_tot_psa", "silt_tot_psa", "clay_tot_psa",
             "db_od", "db_33", "oc", "porosity", "hzn_top", "hzn_bot",
             "SWCC_classes", "data_flag", "climate_classes"]

try:
    import unsatfit
except ImportError:  # keep the message identical in spirit to the study programs
    unsatfit = None


# ===========================================================================
# common helpers
# ===========================================================================
def sha256(path):
    """Return the hex SHA-256 digest of a file, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_unsatfit():
    """Abort with a helpful message if unsatfit is not importable (needed for fitting)."""
    if unsatfit is None:
        raise SystemExit("ERROR: unsatfit not installed. See requirements.txt "
                         "(python3 -m pip install -r requirements.txt).")


def load_obs_map(raw, keep_saturation=True):
    """Return {layer_id: (heads, thetas)} of sorted observed retention points.

    Rows with missing head or water content are dropped; ``keep_saturation`` keeps the
    saturation point (head == 0), which anchors theta_s, using ``head >= 0`` (else > 0).
    """
    cond = raw["lab_head_m"].notna() & raw["lab_wrc"].notna()
    cond &= (raw["lab_head_m"] >= 0) if keep_saturation else (raw["lab_head_m"] > 0)
    obs = raw[cond]
    return {lid: (g.sort_values("lab_head_m")["lab_head_m"].to_numpy(float),
                  g.sort_values("lab_head_m")["lab_wrc"].to_numpy(float))
            for lid, g in obs.groupby("layer_id")}


def _aicc_of(f):
    """Extract unsatfit's corrected AIC (aicc_ht) as a finite float, else None.

    unsatfit returns aicc_ht = None when N - k - 1 <= 0 (undefined small-sample AICc).
    """
    v = getattr(f, "aicc_ht", None)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ===========================================================================
# Stage 1: dual-VG-CH fit (theta_r = 0)
# ===========================================================================
def particle_density(oc, rho_p_default=RHO_P_DEFAULT, oc_threshold=OC_THRESHOLD):
    """Particle density rho_p [g/cm3]; Ruhlmann et al. (2006) Eq. 12 organic-matter
    correction when organic carbon oc [%] exceeds the threshold, else the default 2.65.

    m_OM = oc/55 (C_OM = 55%); rho_OM = 1.127 + 0.373 m_OM bounded to [1.0, 1.5] (Eq. 11);
    rho_MS = 2.684; rho_p = 1 / (m_OM/rho_OM + (1 - m_OM)/rho_MS) (Eq. 12).
    Returns (rho_p, corrected_flag).
    """
    if oc is None or not np.isfinite(oc) or oc <= oc_threshold:
        return rho_p_default, False
    m_om = oc / 55.0
    rho_om = min(1.5, max(1.0, 1.127 + 0.373 * m_om))
    rho_ms = 2.684
    return 1.0 / (m_om / rho_om + (1.0 - m_om) / rho_ms), True


def theta_s_max(meta, obs_theta_max, rho_p_default=RHO_P_DEFAULT, oc_threshold=OC_THRESHOLD):
    """Determine the theta_s upper bound (box, not a fixed value) for one curve.

    porosity present -> theta_s,max = porosity (theoretical cap, used even if a measured
        theta exceeds it). Else db_od present -> theta_s,max = max(1 - db/rho_p, max theta)
        with rho_p from particle_density(). Else -> {'source': 'no-bound'} (curve excluded).
    """
    por, db = meta.get("porosity"), meta.get("db_od")
    oc = meta.get("oc")
    oc = float(oc) if oc is not None and np.isfinite(oc) else None
    if por is not None and np.isfinite(por) and por > 0:
        return {"theta_s_max": float(por), "source": "porosity", "ts_max_calc": float(por),
                "rho_p_used": None, "oc": oc, "rho_p_corrected": False, "raised_to_obs": False}
    if db is not None and np.isfinite(db) and db > 0:
        rho_p, corrected = particle_density(oc, rho_p_default, oc_threshold)
        calc = 1.0 - db / rho_p
        ts = max(calc, float(obs_theta_max))
        return {"theta_s_max": ts, "source": "db_od", "ts_max_calc": calc, "rho_p_used": rho_p,
                "oc": oc, "rho_p_corrected": corrected, "raised_to_obs": bool(obs_theta_max > calc)}
    return {"source": "no-bound", "oc": oc}


def order_modes(qs, w1, a1, m1, m2):
    """Order the two dual-VG-CH modes so that n1 > n2 (tie -> w1 >= w2); alpha is common.

    Removes the label-switching symmetry (w1, n1, n2) <-> (1-w1, n2, n1). Returns
    (qs, w1, a1, m1, m2, n1, n2) after any swap.
    """
    n1, n2 = 1.0 / (1.0 - m1), 1.0 / (1.0 - m2)
    if (n1 < n2) or (abs(n1 - n2) < 1e-9 and w1 < 0.5):
        w1, m1, m2 = 1.0 - w1, m2, m1
        n1, n2 = n2, n1
    return qs, w1, a1, m1, m2, n1, n2


def fit_one_dvc(h, th, ts_max):
    """Fit dual-VG-CH (const qr=0, q=1) to one curve from unsatfit's get_init() start.

    Returns a result dict (converged flag; fitted theta_s, alpha, w1, m1, m2, n1, n2 after
    mode ordering; rmse, me, r2, aic; the corrected AIC 'aicc' captured at this single fit;
    standard errors if available). 'converged' is False on any optimiser failure or if the
    number of fitted parameters is not 5. The 'aicc' recorded here is the value later used
    for the VG-vs-DVC comparison (one-shot: no second warm-started dual-VG-CH fit).
    """
    f = unsatfit.Fit()
    f.set_model("dual-VG-CH", const=["qr=0", "q=1"])
    f.swrc = (h, th)
    f.b_qs = (0.0, ts_max)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rest = f.get_init()                       # (w1, alpha, m1, m2)
            f.ini = (min(float(np.max(th)), ts_max * 0.999), *rest)
            f.optimize()
    except Exception as exc:  # noqa: BLE001
        return {"converged": False, "message": f"exception: {exc!r}"}
    if not f.success or len(f.fitted) != 5:
        return {"converged": False, "message": str(getattr(f, "message", "no convergence"))}
    qs, w1, a1, m1, m2 = (float(x) for x in f.fitted)
    qs, w1, a1, m1, m2, n1, n2 = order_modes(qs, w1, a1, m1, m2)
    try:
        with np.errstate(all="ignore"):
            me = float(np.mean(np.asarray(f.f_ht(f.fitted, h), float) - th))
    except Exception:  # noqa: BLE001
        me = None
    perr = getattr(f, "perr", None)
    se = {}
    if perr is not None and len(perr) == 5:
        se = {"se_qs": float(perr[0]), "se_w1": float(perr[1]), "se_alpha": float(perr[2]),
              "se_m1": float(perr[3]), "se_m2": float(perr[4])}
    return {"converged": True, "message": str(getattr(f, "message", "ok (dof<=0)")),
            "theta_s": qs, "alpha": a1, "w1": w1, "m1": m1, "m2": m2, "n1": n1, "n2": n2,
            "rmse": float(f.se_ht), "me": me, "r2": float(f.r2_ht), "aic": float(f.aic_ht),
            "aicc": _aicc_of(f), "perr_available": bool(se), **se}


def stage1_dualvgch_fit(input_csv=DATA_CSV, out_dir=R_STAGE1):
    """Stage 1: fit dual-VG-CH to every eligible GSHP curve; write params.csv, exclusions.csv,
    summary.json (counts, input/output SHA-256) to ``out_dir``. Returns the params.csv path.

    Eligibility: >= MIN_POINTS valid points and a settable theta_s upper bound. Saturation
    points (head == 0) are kept. This is the heaviest step (~13.8k curves).
    """
    _require_unsatfit()
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(out_dir, exist_ok=True)
    import csv as _csv

    df = pd.read_csv(input_csv, low_memory=False, encoding=ENCODING)
    in_sha = sha256(input_csv)
    print(f"[stage1] input {input_csv}: rows={len(df)} curves={df['layer_id'].nunique()}", flush=True)

    params_path = os.path.join(out_dir, "params.csv")
    excl_path = os.path.join(out_dir, "exclusions.csv")
    param_fields = [
        "layer_id", "n_points", "n_distinct_heads", "dof", "theta_s_max", "ts_max_source",
        "ts_max_calc", "ts_max_raised_to_obs", "rho_p_used", "oc", "rho_p_corrected",
        "converged", "theta_s", "alpha", "w1", "m1", "m2", "n1", "n2", "rmse", "me", "r2", "aic", "aicc",
        "theta_s_at_ub", "w1_near_bound", "mode_near_step", "obs_exceeds_ts_max",
        "perr_available", "se_qs", "se_w1", "se_alpha", "se_m1", "se_m2", "message"] + META_COLS
    excl_fields = ["layer_id", "reason", "n_points", "has_porosity", "has_db_od"]
    counts = {"total": 0, "fitted": 0, "converged": 0, "nonconverged": 0, "excl_points": 0,
              "excl_nobound": 0, "src_porosity": 0, "src_db_od": 0, "oc_corrected": 0,
              "theta_s_at_ub": 0, "w1_near_bound": 0, "degenerate_heads": 0}

    pf, xf = open(params_path, "w", newline=""), open(excl_path, "w", newline="")
    pw = _csv.DictWriter(pf, fieldnames=param_fields, extrasaction="ignore")
    xw = _csv.DictWriter(xf, fieldnames=excl_fields)
    pw.writeheader(); xw.writeheader()

    for layer_id, sub in df.groupby("layer_id", sort=True):
        counts["total"] += 1
        d = sub[sub["lab_head_m"].notna() & sub["lab_wrc"].notna()]
        d = d[d["lab_head_m"] >= 0].sort_values("lab_head_m")
        n = len(d)
        meta = {c: (sub.iloc[0][c] if c in sub.columns else None)
                for c in set(META_COLS) | {"porosity", "db_od", "oc"}}
        has_por = meta.get("porosity") is not None and np.isfinite(meta.get("porosity")) and meta.get("porosity") > 0
        has_db = meta.get("db_od") is not None and np.isfinite(meta.get("db_od")) and meta.get("db_od") > 0
        if n < MIN_POINTS:
            counts["excl_points"] += 1
            xw.writerow({"layer_id": layer_id, "reason": "min_points", "n_points": n,
                         "has_porosity": has_por, "has_db_od": has_db})
            continue
        h = d["lab_head_m"].to_numpy(float)
        th = d["lab_wrc"].to_numpy(float)
        tb = theta_s_max(meta, float(np.max(th)))
        if tb.get("source") == "no-bound" or tb.get("theta_s_max", 0.0) <= 0:
            counts["excl_nobound"] += 1
            xw.writerow({"layer_id": layer_id, "reason": "no_theta_s_bound", "n_points": n,
                         "has_porosity": has_por, "has_db_od": has_db})
            continue
        counts["fitted"] += 1
        src, ts_max = tb["source"], tb["theta_s_max"]
        counts["src_porosity" if src == "porosity" else "src_db_od"] += 1
        if tb.get("rho_p_corrected"):
            counts["oc_corrected"] += 1
        n_distinct = int(np.unique(h).size)
        if n_distinct < 5:
            counts["degenerate_heads"] += 1
        res = fit_one_dvc(h, th, ts_max)
        row = {"layer_id": layer_id, "n_points": n, "n_distinct_heads": n_distinct, "dof": n - 5,
               "theta_s_max": ts_max, "ts_max_source": src, "ts_max_calc": tb.get("ts_max_calc"),
               "ts_max_raised_to_obs": tb.get("raised_to_obs"), "rho_p_used": tb.get("rho_p_used"),
               "oc": tb.get("oc"), "rho_p_corrected": tb.get("rho_p_corrected"),
               "obs_exceeds_ts_max": bool(np.max(th) > ts_max), **res}
        for c in META_COLS:
            row[c] = meta.get(c)
        if res["converged"]:
            counts["converged"] += 1
            qs = res["theta_s"]
            row["theta_s_at_ub"] = bool(abs(qs - ts_max) < 1e-4 * max(1.0, ts_max))
            row["w1_near_bound"] = bool(res["w1"] < 1e-3 or res["w1"] > 1 - 1e-3)
            row["mode_near_step"] = bool(res["m1"] > 0.999 or res["m2"] < 1e-4)
            counts["theta_s_at_ub"] += int(row["theta_s_at_ub"])
            counts["w1_near_bound"] += int(row["w1_near_bound"])
        else:
            counts["nonconverged"] += 1
        pw.writerow(row)
        if counts["fitted"] % 1000 == 0:
            print(f"  fitted {counts['fitted']} (converged {counts['converged']})", flush=True)
    pf.close(); xf.close()

    out_sha = {"params.csv": sha256(params_path), "exclusions.csv": sha256(excl_path)}
    with open(os.path.join(out_dir, "summary.json"), "w") as sh:
        json.dump({"counts": counts, "outputs_sha256": out_sha, "input_sha256": in_sha},
                  sh, indent=2, default=str); sh.write("\n")
    print(f"[stage1] converged {counts['converged']} / fitted {counts['fitted']} -> {params_path}")
    return params_path


# ===========================================================================
# VG fit + AICc type split
# ===========================================================================
def fit_vg_curve(h, th, ts_max):
    """Fit single VG (theta_r free, const q=1) to one curve; return params + rmse + AICc.

    theta_s / theta_r initial values are set explicitly; alpha and m come from unsatfit's
    data-derived get_init() (which returns (alpha, m) for VG). Returns None on failure.
    """
    f = unsatfit.Fit()
    f.set_model("VG", const=["q=1"])
    f.swrc = (h, th)
    f.b_qs = (0.0, ts_max)
    f.ini = (min(float(np.max(th)), ts_max * 0.99), min(0.05, 0.5 * float(np.min(th))), *f.get_init())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f.optimize()
        except Exception:  # noqa: BLE001
            return None
    if not f.success or len(f.fitted) != 4:
        return None
    qs, qr, a, m = (float(x) for x in f.fitted)
    return dict(theta_s=qs, theta_r=qr, alpha=a, m=m, n=1.0 / (1.0 - m),
                rmse=float(f.se_ht), aicc=_aicc_of(f))


def stage_vg_fit_aic(params_path, input_csv=DATA_CSV, out_dir=R_VG):
    """Fit VG for every Stage-1-converged curve and compare its AICc with the dual-VG-CH AICc
    that Stage 1 recorded at fit time (one-shot: the dual-VG-CH model is not re-fitted here).
    Classify each curve DVC-type (both AICc defined and aicc_dvc < aicc_vg) or VG-type, and
    write vg_params.csv + summary.json. Returns the vg_params.csv path.
    """
    _require_unsatfit()
    np.seterr(all="ignore"); warnings.filterwarnings("ignore")
    os.makedirs(out_dir, exist_ok=True)
    p = pd.read_csv(params_path, low_memory=False)
    p = p[p["converged"] == True].copy()  # noqa: E712
    raw = pd.read_csv(input_csv, low_memory=False, encoding=ENCODING)
    om = load_obs_map(raw, keep_saturation=True)

    rows = []
    for k, (_, r) in enumerate(p.iterrows()):
        lid = r["layer_id"]
        if lid not in om:
            continue
        h, th = om[lid]
        ts_max = float(r["theta_s_max"])
        vg = fit_vg_curve(h, th, ts_max)
        # one-shot: the dual-VG-CH AICc was captured at the single Stage-1 fit (params.csv).
        a_dvc = float(r["aicc"]) if ("aicc" in r and pd.notna(r["aicc"])) else None
        a_vg = vg["aicc"] if vg else None
        is_dvc = (vg is not None and a_vg is not None and a_dvc is not None and a_dvc < a_vg)
        row = dict(layer_id=lid, reference=r.get("reference"), n_points=int(r["n_points"]),
                   vg_converged=vg is not None, aicc_vg=a_vg, aicc_dvc=a_dvc,
                   type="DVC" if is_dvc else "VG")
        if vg is not None:
            row.update(theta_s_vg=vg["theta_s"], theta_r_vg=vg["theta_r"], alpha_vg=vg["alpha"],
                       n_vg=vg["n"], m_vg=vg["m"], vg_rmse=vg["rmse"])
        rows.append(row)
        if (k + 1) % 1000 == 0:
            print(f"  [vg] {k + 1}/{len(p)}", flush=True)

    out = pd.DataFrame(rows)
    dpath = os.path.join(out_dir, "vg_params.csv")
    out.to_csv(dpath, index=False)
    n, n_dvc = len(out), int((out["type"] == "DVC").sum())
    summary = {"n_curves": n, "n_DVC_type": n_dvc, "n_VG_type": n - n_dvc,
               "pct_DVC_type": round(100 * n_dvc / n, 2) if n else None,
               "n_vg_nonconverged": int((~out["vg_converged"]).sum()),
               "outputs_sha256": {"vg_params.csv": sha256(dpath)},
               "inputs_sha256": {"params": sha256(params_path), "raw": sha256(input_csv)}}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str); fh.write("\n")
    print(f"[vg] DVC-type {n_dvc}/{n} ({summary['pct_DVC_type']}%) -> {dpath}")
    return dpath


# ===========================================================================
# degeneracy patterns
# ===========================================================================
def classify_degeneracy(w1, m1, m2, w_eps=0.05, dm_eps=0.02):
    """Classify a dual-VG-CH fit into a degeneracy pattern near the single-VG limits:
    'P2_w1_0' (w1 ~ 0), 'P3_w1_1' (w1 ~ 1), 'P1_n1_n2' (|m1-m2| < dm_eps), else 'bimodal'.
    """
    if w1 < w_eps:
        return "P2_w1_0"
    if w1 > 1 - w_eps:
        return "P3_w1_1"
    if abs(m1 - m2) < dm_eps:
        return "P1_n1_n2"
    return "bimodal"


def single_vg_rmse(h, th, ts_max):
    """Fit a single VG with theta_r = 0 (const qr=0, q=1) to one curve; return its RMSE
    (or None on failure). Used to check that degenerate dual fits equal a single VG.
    """
    f = unsatfit.Fit()
    f.set_model("VG", const=["qr=0", "q=1"])
    f.swrc = (h, th)
    f.b_qs = (0.0, ts_max)
    f.ini = (min(float(np.max(th)), ts_max * 0.999), 1.0, 0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f.optimize()
        except Exception:  # noqa: BLE001
            return None
    return float(f.se_ht) if f.success else None


def stage_degeneracy(params_path, input_csv=DATA_CSV, out_dir=R_DEGEN, sample=800, seed=0):
    """Classify every converged curve into P1/P2/P3/bimodal, report proportions, and verify
    single-VG equivalence on a fixed random sample; write summary.json. Returns its path.
    """
    _require_unsatfit()
    np.seterr(all="ignore"); warnings.filterwarnings("ignore")
    os.makedirs(out_dir, exist_ok=True)
    p = pd.read_csv(params_path, low_memory=False)
    p = p[p["converged"] == True].copy()  # noqa: E712
    N = len(p)
    p["pattern"] = [classify_degeneracy(w, m1, m2) for w, m1, m2 in zip(p["w1"], p["m1"], p["m2"])]
    counts = p["pattern"].value_counts().to_dict()
    proportions = {k: {"n": int(counts.get(k, 0)), "pct": round(100 * counts.get(k, 0) / N, 2)}
                   for k in ["P1_n1_n2", "P2_w1_0", "P3_w1_1", "bimodal"]}
    single_like = sum(counts.get(k, 0) for k in ["P1_n1_n2", "P2_w1_0", "P3_w1_1"])

    raw = pd.read_csv(input_csv, low_memory=False, encoding=ENCODING)
    om = load_obs_map(raw, keep_saturation=True)
    samp = p.sample(min(sample, N), random_state=seed)
    rows = []
    for _, r in samp.iterrows():
        if r["layer_id"] not in om:
            continue
        h, th = om[r["layer_id"]]
        sv = single_vg_rmse(h, th, float(r["theta_s_max"]))
        if sv is None:
            continue
        rows.append((r["pattern"], float(r["rmse"]), sv))
    cmp = pd.DataFrame(rows, columns=["pattern", "dual_rmse", "single_rmse"])
    cmp["improve"] = cmp["single_rmse"] - cmp["dual_rmse"]
    sv_check = {}
    for pat, g in cmp.groupby("pattern"):
        sv_check[pat] = {"n": int(len(g)), "dual_rmse_median": round(float(g["dual_rmse"].median()), 4),
                         "single_rmse_median": round(float(g["single_rmse"].median()), 4),
                         "improve_median": round(float(g["improve"].median()), 4),
                         "frac_dual_eq_single": round(float((g["improve"] < 0.001).mean()), 3)}
    summary = {"n_converged": int(N), "thresholds": {"w1_eps": 0.05, "dm_eps": 0.02},
               "proportions": proportions, "single_like_pct": round(100 * single_like / N, 2),
               "single_vg_check": {"sample_n": int(len(cmp)), "seed": seed, "by_pattern": sv_check,
                                   "overall_frac_dual_eq_single": round(float((cmp["improve"] < 0.001).mean()), 3)},
               "inputs_sha256": {"params": sha256(params_path), "raw": sha256(input_csv)}}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2); fh.write("\n")
    print(f"[degeneracy] non-degenerate (bimodal) "
          f"{100 - summary['single_like_pct']:.2f}% of {N}")
    return os.path.join(out_dir, "summary.json")


# ===========================================================================
# identifiability audit (canonical mode order)
# ===========================================================================
_PERT = {"qs": 0.05, "w1": 0.10, "a1": None, "m1": 0.05, "m2": 0.05}
_NEAR_BEST_SSE = 1e-6


def canon(s):
    """Canonicalise a fitted solution to n1 >= n2 (m1 >= m2), swapping modes and w1 -> 1-w1
    if needed, so dispersion statistics are not inflated by the label-switching symmetry.
    """
    if s["m1"] >= s["m2"]:
        return s
    r = dict(s)
    r["w1"] = 1.0 - s["w1"]
    r["m1"], r["m2"] = s["m2"], s["m1"]
    return r


def _new_dvc_fit(h, th, ts_max, const):
    """Construct an unsatfit dual-VG-CH Fit with the given constraints and data bounds."""
    f = unsatfit.Fit()
    f.set_model("dual-VG-CH", const=const)
    f.swrc = (h, th)
    f.b_qs = (0.0, ts_max)
    return f


def _sse_of(f, h):
    """Return the sum of squared residuals implied by unsatfit's RMSE se_ht over N points."""
    return float(f.se_ht) ** 2 * len(h)


def free_fit(h, th, ts_max, ini):
    """Free dual-VG-CH fit (qr=0, q=1) from a given start; return dict(qs,w1,a1,m1,m2,sse)
    or None if the optimiser fails or does not return 5 parameters.
    """
    f = _new_dvc_fit(h, th, ts_max, ["qr=0", "q=1"])
    f.ini = ini
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.optimize()
    if not f.success or len(f.fitted) != 5:
        return None
    qs, w1, a1, m1, m2 = (float(x) for x in f.fitted)
    return dict(qs=qs, w1=w1, a1=a1, m1=m1, m2=m2, sse=_sse_of(f, h))


def profile_curv(h, th, ts_max, opt):
    """Local profile curvature: for each parameter, fix it a step away from the optimum,
    re-optimise the rest, and record the mean SSE rise. Larger rise => better constrained.
    """
    order = ["qs", "w1", "a1", "m1", "m2"]
    out = {}
    for pname in order:
        ini_others = tuple(opt[q] for q in order if q != pname)
        if pname == "a1":
            targets = [opt["a1"] * 2.0, opt["a1"] / 2.0]
        elif pname == "qs":
            targets = [min(opt["qs"] + _PERT["qs"], ts_max), max(opt["qs"] - _PERT["qs"], 1e-4)]
        else:
            targets = [min(opt[pname] + _PERT[pname], 1 - 1e-6), max(opt[pname] - _PERT[pname], 1e-6)]
        rises = []
        for val in targets:
            f = _new_dvc_fit(h, th, ts_max, ["qr=0", "q=1", f"{pname}={val}"])
            f.ini = ini_others
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f.optimize()
                if f.success:
                    rises.append(max(0.0, _sse_of(f, h) - opt["sse"]))
            except Exception:  # noqa: BLE001
                pass
        out[f"curv_{pname}"] = float(np.mean(rises)) if rises else np.nan
    return out


def multistart(h, th, ts_max, base, n_starts, rng):
    """Multistart free fits from random starts; canonicalise near-best solutions and report
    the SD of each parameter across them (ms_sd_*) as a numerical-stability check.
    """
    thmax = float(np.max(th))
    starts = []
    for _ in range(n_starts):
        r = free_fit(h, th, ts_max, (min(thmax * rng.uniform(0.9, 1.0), ts_max * 0.999),
                                     float(rng.uniform(0.1, 0.9)),
                                     float(base["a1"] * 10 ** rng.uniform(-1, 1)),
                                     float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95))))
        if r:
            starts.append(r)
    if not starts:
        return dict(ms_n=0)
    sse = np.array([s["sse"] for s in starts])
    best = float(np.min(sse))
    near = [canon(s) for s in starts if s["sse"] < best + _NEAR_BEST_SSE]

    def sd(key, log=False):
        v = np.array([(np.log10(s[key]) if log else s[key]) for s in near])
        return float(np.std(v)) if len(v) > 1 else 0.0

    return dict(ms_n=len(starts), ms_best_sse=best, ms_frac_at_best=float(len(near) / len(starts)),
                ms_sd_qs=sd("qs"), ms_sd_log_a1=sd("a1", True), ms_sd_w1=sd("w1"),
                ms_sd_m1=sd("m1"), ms_sd_m2=sd("m2"))


def boot(h, th, ts_max, base, n_boot, rng):
    """Nonparametric bootstrap of the observed points; refit and canonicalise each replicate;
    report the SD of each parameter (boot_sd_*). Skipped implicitly for < 5 usable replicates.
    """
    n = len(h)
    res = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(h[idx]).size < 5:
            continue
        r = free_fit(h[idx], th[idx], ts_max, (base["qs"], base["w1"], base["a1"], base["m1"], base["m2"]))
        if r:
            res.append(canon(r))
    if len(res) < 5:
        return dict(boot_n=len(res))

    def sd(key, log=False):
        v = np.array([(np.log10(s[key]) if log else s[key]) for s in res])
        return float(np.std(v))

    return dict(boot_n=len(res), boot_sd_qs=sd("qs"), boot_sd_log_a1=sd("a1", True),
                boot_sd_w1=sd("w1"), boot_sd_m1=sd("m1"), boot_sd_m2=sd("m2"))


def stratify(p, per_stratum, seed):
    """Stratify converged curves by head-count bin, theta_s-at-bound, extreme w1, and close
    modes, then draw up to ``per_stratum`` curves per stratum (fixed seed). Returns the sample.
    """
    hb = pd.cut(p["n_distinct_heads"], [0, 5, 7, 12, 10 ** 6], labels=["5", "6-7", "8-12", "13+"])
    strata = pd.DataFrame({"head_bin": hb.astype(str),
                           "ts_at_ub": p["theta_s_at_ub"].astype(bool).astype(int),
                           "w1_extreme": ((p["w1"] < 0.05) | (p["w1"] > 0.95)).astype(int),
                           "modes_close": ((p["m1"] - p["m2"]).abs() < 0.02).astype(int)})
    p = p.assign(**{c: strata[c] for c in strata.columns})
    rng = np.random.default_rng(seed)
    picks = [g.sample(min(len(g), per_stratum), random_state=int(rng.integers(0, 1 << 31)))
             for _, g in p.groupby(list(strata.columns))]
    return pd.concat(picks).reset_index(drop=True)


def stage_identifiability(params_path, input_csv=DATA_CSV, out_dir=R_IDENT,
                          per_stratum=30, n_starts=15, n_boot=40, seed=0):
    """Identifiability audit on a stratified subsample: profile curvature, multistart SD,
    and bootstrap SD (all with canonical mode order). Writes identifiability_per_curve.csv
    and summary.json. Returns the per-curve CSV path (input to the support analysis).
    """
    _require_unsatfit()
    np.seterr(all="ignore"); warnings.filterwarnings("ignore")
    os.makedirs(out_dir, exist_ok=True)
    p = pd.read_csv(params_path, low_memory=False)
    p = p[p["converged"] == True].copy()  # noqa: E712
    sample = stratify(p, per_stratum, seed)
    raw = pd.read_csv(input_csv, low_memory=False, encoding=ENCODING)
    obs_map = load_obs_map(raw, keep_saturation=True)
    print(f"[identifiability] subsample {len(sample)} curves", flush=True)

    rng = np.random.default_rng(seed)
    rows = []
    for k, (_, r) in enumerate(sample.iterrows()):
        lid = r["layer_id"]
        if lid not in obs_map:
            continue
        h, th = obs_map[lid]
        ts_max = float(r["theta_s_max"])
        base = dict(qs=float(r["theta_s"]), w1=float(r["w1"]), a1=float(r["alpha"]),
                    m1=float(r["m1"]), m2=float(r["m2"]),
                    sse=float(r["rmse"]) ** 2 * len(h) if np.isfinite(r["rmse"]) else 0.0)
        row = dict(layer_id=lid, reference=r.get("reference"), n_points=int(r["n_points"]),
                   n_distinct_heads=int(r["n_distinct_heads"]), dof=int(r["dof"]),
                   head_bin=r["head_bin"], ts_at_ub=int(r["ts_at_ub"]),
                   w1=base["w1"], m1=base["m1"], m2=base["m2"], alpha=base["a1"],
                   modes_close=int(abs(base["m1"] - base["m2"]) < 0.02),
                   w1_extreme=int(base["w1"] < 0.05 or base["w1"] > 0.95))
        if int(r["dof"]) > 0:
            try:
                row.update(profile_curv(h, th, ts_max, base))
            except Exception:  # noqa: BLE001
                pass
        row.update(multistart(h, th, ts_max, base, n_starts, rng))
        if int(r["n_distinct_heads"]) >= 8:
            row.update(boot(h, th, ts_max, base, n_boot, rng))
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  [identifiability] {k + 1}/{len(sample)}", flush=True)

    per = pd.DataFrame(rows)
    cpath = os.path.join(out_dir, "identifiability_per_curve.csv")
    per.to_csv(cpath, index=False)

    def summ(df, label):
        d = {"label": label, "n": int(len(df))}
        for c in ["ms_frac_at_best", "ms_sd_log_a1", "ms_sd_w1", "ms_sd_m1", "ms_sd_m2",
                  "curv_a1", "curv_w1", "curv_qs", "curv_m1", "curv_m2", "boot_sd_log_a1", "boot_sd_w1"]:
            if c in df:
                d[c + "_median"] = float(np.nanmedian(df[c])) if df[c].notna().any() else None
        return d

    summary = {"n_curves": int(len(per)), "subsample_per_stratum": per_stratum,
               "mode_order": "canonical (n1>=n2) before dispersion", "overall": summ(per, "overall"),
               "by_head_bin": [summ(g, f"head_bin={k}") for k, g in per.groupby("head_bin")],
               "outputs_sha256": {"identifiability_per_curve.csv": sha256(cpath)},
               "inputs_sha256": {"params": sha256(params_path), "raw": sha256(input_csv)}}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str); fh.write("\n")
    print(f"[identifiability] per-curve -> {cpath}")
    return cpath


# ===========================================================================
# w1 profile example (Fig 5a)
# ===========================================================================
def fit_fixed_w1(h, th, ts_max, w1, ini4):
    """Fit dual-VG-CH with w1 fixed (const w1=..., qr=0, q=1), optimising (theta_s, alpha,
    m1, m2) from ini4 = (qs, a1, m1, m2). Returns dict(qs,a1,m1,m2,sse,rmse) or None.
    """
    f = unsatfit.Fit()
    f.set_model("dual-VG-CH", const=["qr=0", "q=1", f"w1={w1}"])
    f.swrc = (h, th)
    f.b_qs = (0.0, ts_max)
    f.ini = ini4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.optimize()
    if not f.success or len(f.fitted) != 4:
        return None
    qs, a1, m1, m2 = (float(x) for x in f.fitted)
    sse = float(f.se_ht) ** 2 * len(h)
    return dict(qs=qs, a1=a1, m1=m1, m2=m2, sse=sse, rmse=(sse / len(h)) ** 0.5)


def profile_one_w1(params, raw, layer_id, n_grid=19):
    """Compute the w1 profile-RMSE envelope for one curve: at each w1 on a grid, keep the
    best of a few starts (immune to path-dependence and to the mode-swap symmetry).
    Returns a dict with the observed points and the (w1, rmse) profile.
    """
    prow = params[params["layer_id"] == layer_id]
    if len(prow) == 0 or not bool(prow.iloc[0]["converged"]):
        raise SystemExit(f"ERROR: layer_id {layer_id} missing or non-converged in params")
    prow = prow.iloc[0]
    ts_max = float(prow["theta_s_max"])
    g = raw[(raw["layer_id"] == layer_id) & raw["lab_head_m"].notna() & raw["lab_wrc"].notna()
            & (raw["lab_head_m"] >= 0)].sort_values("lab_head_m")
    h, th = g["lab_head_m"].to_numpy(float), g["lab_wrc"].to_numpy(float)
    grid = np.linspace(0.05, 0.95, n_grid)
    qs0 = min(float(prow["theta_s"]), ts_max * 0.999)
    m1, m2, a = float(prow["m1"]), float(prow["m2"]), float(prow["alpha"])
    seeds = [(qs0, a, m1, m2), (qs0, a, m2, m1), (qs0, a, 0.5, 0.2), (qs0, a * 3, 0.3, 0.3)]
    profile = []
    for w1 in grid:
        cand = [c for c in (fit_fixed_w1(h, th, ts_max, float(w1), ini) for ini in seeds) if c]
        if cand:
            profile.append(dict(w1=float(w1), rmse=min(cand, key=lambda c: c["sse"])["rmse"]))
    rmses = [p["rmse"] for p in profile]
    return dict(layer_id=str(layer_id), reference=str(prow.get("reference")), n_points=int(len(h)),
                n_distinct_heads=int(prow["n_distinct_heads"]), w1_fit=float(prow["w1"]),
                dm=float(abs(prow["m1"] - prow["m2"])),
                best_rmse=float(prow["rmse"]) if np.isfinite(prow["rmse"]) else None,
                rmse_min=float(min(rmses)), rmse_max=float(max(rmses)),
                observed=[[float(a), float(b)] for a, b in zip(h, th)], profile=profile)


def stage_wprofile(params_path, input_csv=DATA_CSV, out_dir=R_WPROF,
                   layer_ids=("Zalf132", "UNSODA4790"), n_grid=19):
    """Write wprofile_example.json with the w1 profile of two contrasting curves (a VG-like
    flat profile and a two-component curve with a clear minimum). Returns the JSON path.
    """
    _require_unsatfit()
    np.seterr(all="ignore"); warnings.filterwarnings("ignore")
    os.makedirs(out_dir, exist_ok=True)
    params = pd.read_csv(params_path, low_memory=False)
    raw = pd.read_csv(input_csv, low_memory=False, encoding=ENCODING)
    curves = [profile_one_w1(params, raw, lid, n_grid) for lid in layer_ids]
    opath = os.path.join(out_dir, "wprofile_example.json")
    with open(opath, "w") as fh:
        json.dump({"curves": curves,
                   "inputs_sha256": {"params": sha256(params_path), "raw": sha256(input_csv)}},
                  fh, indent=2); fh.write("\n")
    print(f"[wprofile] -> {opath}")
    return opath


# ===========================================================================
# measurement-conditioned model support
# ===========================================================================
def _frac(mask, denom):
    """Return {n, n_denom, pct} of ``mask`` within ``denom`` (both boolean Series)."""
    d = int(denom.sum())
    return {"n": int((mask & denom).sum()), "n_denom": d,
            "pct": (round(100 * (mask & denom).sum() / d, 1) if d else None)}


def stage_support_analysis(params_path, vg_path, input_csv, ident_csv, out_dir=R_SUPPORT):
    """Compute measurement-conditioned model support from the derived fits (no new fitting):
    the three-group counts (Table 1), the Delta-AICc distribution (Fig 2a), support rates by
    number of points / head range / SWCC class / study (Figs 2b, 3), and the identifiability x
    support cross-tabulation (Fig 5b). Writes several CSVs and summary.json. Returns out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    vg = pd.read_csv(vg_path, low_memory=False)
    p = pd.read_csv(params_path, low_memory=False, encoding=ENCODING)
    p = p[p["converged"] == True].copy()  # noqa: E712

    min_n_vg = int(vg.loc[vg["aicc_vg"].notna(), "n_points"].min())
    min_n_dvc = int(vg.loc[vg["aicc_dvc"].notna(), "n_points"].min())
    comparable = vg["aicc_vg"].notna() & vg["aicc_dvc"].notna()
    delta = vg["aicc_dvc"] - vg["aicc_vg"]
    dvc = comparable & (delta < 0)
    vgs = comparable & (delta >= 0)
    unable = ~comparable
    allT = pd.Series(True, index=vg.index)
    groups = {"n_converged": len(vg), "comparable": _frac(comparable, allT),
              "DVC_support": _frac(dvc, allT), "VG_support": _frac(vgs, allT),
              "AICc_uncomparable": _frac(unable, allT),
              "DVC_support_of_comparable_pct": round(100 * dvc.sum() / comparable.sum(), 1),
              "VG_support_of_comparable_pct": round(100 * vgs.sum() / comparable.sum(), 1)}

    p["depth_mid"] = (p["hzn_top"] + p["hzn_bot"]) / 2.0
    pc = p.dropna(subset=BASIC_COMPLETE + ["reference"])
    inpc = vg["layer_id"].isin(set(pc["layer_id"]))
    frame = {"n_frame": int(inpc.sum()), "comparable": _frac(comparable, inpc),
             "DVC_support": _frac(dvc, inpc), "VG_support": _frac(vgs, inpc),
             "AICc_uncomparable": _frac(unable, inpc)}

    d_cmp = delta[comparable].to_numpy(float)
    pd.DataFrame({"layer_id": vg.loc[comparable, "layer_id"].to_numpy(),
                  "n_points": vg.loc[comparable, "n_points"].to_numpy(),
                  "delta_aicc": d_cmp}).to_csv(os.path.join(out_dir, "delta_aicc.csv"), index=False)
    delta_dist = {"n": int(len(d_cmp)), "median": round(float(np.median(d_cmp)), 3),
                  "pct_below_0": round(100 * float(np.mean(d_cmp < 0)), 1),
                  "pct_below_-2": round(100 * float(np.mean(d_cmp < -2)), 1),
                  "pct_above_2": round(100 * float(np.mean(d_cmp > 2)), 1),
                  "q": {q: round(float(np.percentile(d_cmp, q)), 2) for q in (5, 25, 50, 75, 95)}}

    by_np = []
    for lo, hi in NPOINT_BINS:
        m = (vg["n_points"] >= lo) & (vg["n_points"] <= hi)
        c = int((m & comparable).sum())
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10 ** 8 else f"{lo}-{hi}")
        by_np.append({"bin": label, "n": int(m.sum()), "comparable": c,
                      "dvc_support_pct": (round(100 * (m & dvc).sum() / c, 1) if c else None)})

    raw = pd.read_csv(input_csv, low_memory=False, encoding=ENCODING)
    obs = raw[raw["lab_head_m"].notna() & raw["lab_wrc"].notna() & (raw["lab_head_m"] > 0)]
    g = obs.groupby("layer_id")["lab_head_m"]
    hr = np.log10(g.max() / g.min()).rename("head_decades").reset_index()
    vg2 = vg.merge(hr, on="layer_id", how="left")
    by_hr = []
    for lo, hi in HEADRANGE_BINS:
        m = (vg2["head_decades"] >= lo) & (vg2["head_decades"] < hi)
        c = int((m & comparable.values).sum())
        by_hr.append({"decades": f"{lo}-{hi}", "n": int(m.sum()), "comparable": c,
                      "dvc_support_pct": (round(100 * (m.values & dvc.values).sum() / c, 1) if c else None)})

    swcc = p[["layer_id", "SWCC_classes"]].drop_duplicates("layer_id")
    vg3 = vg.merge(swcc, on="layer_id", how="left")
    by_swcc = []
    for cls, sub in vg3.groupby("SWCC_classes"):
        cmp_s, dvc_s = comparable.loc[sub.index], dvc.loc[sub.index]
        c = int(cmp_s.sum())
        by_swcc.append({"class": cls, "n": len(sub), "comparable": c,
                        "dvc_support_pct": (round(100 * dvc_s.sum() / c, 1) if c else None)})

    by_study = []
    for ref, sub in vg.groupby("reference"):
        c = int(comparable.loc[sub.index].sum())
        if c:
            by_study.append({"reference": ref, "n": len(sub), "comparable": c,
                             "dvc_support_pct": round(100 * dvc.loc[sub.index].sum() / c, 1)})
    by_study.sort(key=lambda r: r["dvc_support_pct"])
    study_df = pd.DataFrame(by_study)
    study_df.to_csv(os.path.join(out_dir, "support_by_study.csv"), index=False)
    study_summary = {"n_studies_comparable": len(by_study),
                     "dvc_support_pct_q": {q: round(float(np.percentile(study_df["dvc_support_pct"], q)), 1)
                                           for q in (5, 25, 50, 75, 95)},
                     "n_studies_0pct": int((study_df["dvc_support_pct"] == 0).sum()),
                     "n_studies_ge90pct": int((study_df["dvc_support_pct"] >= 90).sum())}
    pd.DataFrame(by_np).to_csv(os.path.join(out_dir, "support_by_npoints.csv"), index=False)
    pd.DataFrame(by_hr).to_csv(os.path.join(out_dir, "support_by_headrange.csv"), index=False)

    idf = pd.read_csv(ident_csv, low_memory=False)
    idf = idf.merge(vg[["layer_id", "type", "aicc_vg", "aicc_dvc"]], on="layer_id", how="left")
    idf["support"] = np.where(idf["aicc_vg"].notna() & idf["aicc_dvc"].notna(),
                              np.where(idf["aicc_dvc"] < idf["aicc_vg"], "DVC", "VG"), "uncomparable")
    idf["degen"] = [classify_degeneracy(w, a, b) for w, a, b in zip(idf["w1"], idf["m1"], idf["m2"])]
    idf.to_csv(os.path.join(out_dir, "identifiability_support.csv"), index=False)
    id_by_support = {}
    for sup, sub in idf.groupby("support"):
        col = {}
        for metric in ("ms_sd_w1", "boot_sd_w1", "curv_w1"):
            v = pd.to_numeric(sub[metric], errors="coerce").dropna() if metric in sub else pd.Series(dtype=float)
            col[metric] = {"n": int(len(v)), "median": (round(float(v.median()), 4) if len(v) else None),
                           "q25": (round(float(v.quantile(.25)), 4) if len(v) else None),
                           "q75": (round(float(v.quantile(.75)), 4) if len(v) else None)}
        col["degen_mix_pct"] = {k: round(100 * (sub["degen"] == k).mean(), 1)
                                for k in ("P1_n1_n2", "P2_w1_0", "P3_w1_1", "bimodal")}
        col["n"] = int(len(sub))
        id_by_support[sup] = col

    summary = {"aicc_audit": {"k_vg": 4, "k_dvc": 5, "min_n_defined_vg": min_n_vg,
                              "min_n_defined_dvc": min_n_dvc},
               "group_counts_all": groups, "group_counts_ptf_frame": frame,
               "delta_aicc": delta_dist, "support_by_npoints": by_np, "support_by_headrange": by_hr,
               "support_by_swcc": by_swcc, "support_by_study": study_summary,
               "identifiability_by_support": id_by_support,
               "inputs_sha256": {"params": sha256(params_path), "vg": sha256(vg_path),
                                 "raw": sha256(input_csv), "identifiability": sha256(ident_csv)}}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str); fh.write("\n")
    print(f"[support] DVC-support {groups['DVC_support']['pct']}% of converged; "
          f"comparable {groups['comparable']['pct']}% -> {out_dir}")
    return out_dir


# ===========================================================================
# support-stratified PTFs (Stage 2)
# ===========================================================================
def _fwd(y, tr):
    """Forward-transform a target vector: log10 (clipped positive) if tr == 'log10', else identity."""
    return np.log10(np.clip(y, 1e-12, None)) if tr == "log10" else y


def _clip01(x):
    """Clip m-parameters to [1e-4, 1 - 1e-4] so that n = 1/(1-m) stays finite and > 1."""
    return np.clip(x, 1e-4, 1 - 1e-4)


def vg_theta(h, pr):
    """Reconstruct VG theta(h) from predicted parameters (theta_s, theta_r, log10 alpha, m),
    with implementation-range clipping for numerical safety.
    """
    qs = float(np.clip(pr["theta_s_vg"], 1e-3, 1.5))
    qr = float(np.clip(pr["theta_r_vg"], 0.0, qs))
    a = float(np.clip(10 ** pr["alpha_vg"], 1e-4, 1e4))
    m = float(_clip01(pr["m_vg"]))
    n = 1.0 / (1.0 - m)
    ah = np.clip(a * np.asarray(h, float), 1e-12, None)
    with np.errstate(over="ignore", invalid="ignore"):
        return qr + (qs - qr) * (1.0 + ah ** n) ** (-m)


def dvc_theta(h, pr):
    """Reconstruct dual-VG-CH theta(h) from predicted parameters (theta_s, log10 alpha, w1,
    m1, m2) with theta_r = 0 and the common alpha, with implementation-range clipping.
    """
    qs = float(np.clip(pr["theta_s"], 1e-3, 1.5))
    a = float(np.clip(10 ** pr["alpha"], 1e-4, 1e4))
    w1 = float(np.clip(pr["w1"], 0.0, 1.0))
    ah = np.clip(a * np.asarray(h, float), 1e-12, None)

    def sub(m):
        m = float(_clip01(m))
        n = 1.0 / (1.0 - m)
        with np.errstate(over="ignore", invalid="ignore"):
            return (1.0 + ah ** n) ** (-m)

    return qs * (w1 * sub(pr["m1"]) + (1 - w1) * sub(pr["m2"]))


def fit_predict(train_frame, eval_frame, targets):
    """Fit one standardised linear regression per target parameter on ``train_frame`` and
    predict on ``eval_frame`` (indexed by layer_id). Returns a DataFrame of predicted params.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xtr = train_frame[BASIC_PTF].to_numpy(float)
    out = pd.DataFrame(index=eval_frame["layer_id"].to_numpy())
    Xev = eval_frame[BASIC_PTF].to_numpy(float)
    for col, tr in targets:
        y = _fwd(train_frame[col].to_numpy(float), tr)
        out[col] = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, y).predict(Xev)
    return out


def ptf_equations(train_frame, targets):
    """Return the de-standardised linear-regression coefficients and intercept per target
    (predictors in raw units), plus the in-sample R^2, for documentation/reproducibility.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    eqs = {}
    Xtr = train_frame[BASIC_PTF].to_numpy(float)
    for col, tr in targets:
        y = _fwd(train_frame[col].to_numpy(float), tr)
        sc = StandardScaler().fit(Xtr)
        lr = LinearRegression().fit(sc.transform(Xtr), y)
        eqs[col] = {"transform": tr,
                    "intercept": float(lr.intercept_ - np.sum(lr.coef_ * sc.mean_ / sc.scale_)),
                    "coef": {p: float(c) for p, c in zip(BASIC_PTF, lr.coef_ / sc.scale_)},
                    "r2_insample": float(lr.score(sc.transform(Xtr), y))}
    return eqs


def _curve_sq(theta_fn, cols, preds, lid, om):
    """Return (sum of squared theta residuals, count) for one reconstructed curve vs its
    observed points, using the predicted parameters ``preds`` at layer_id ``lid``.
    """
    h, th = om[lid]
    r = np.asarray(theta_fn(h, {c: preds.at[lid, c] for c in cols}), float) - th
    r = r[np.isfinite(r)]
    return float(np.sum(r ** 2)), len(r)


def fit_apply_rmse(f, train_idx, ev_idx, targets, theta_fn, om):
    """Non-cross-validated ('apparent') micro RMSE: fit on the whole training group, apply
    to the evaluation group, and pool squared residuals over all their observed points.
    """
    preds = fit_predict(f.loc[train_idx], f.loc[ev_idx], targets)
    cols = [c for c, _ in targets]
    sq = cnt = 0.0
    for lid in f.loc[ev_idx, "layer_id"]:
        if lid in om:
            s, c = _curve_sq(theta_fn, cols, preds, lid, om)
            sq += s; cnt += c
    return (sq / cnt) ** 0.5 if cnt else np.nan


def loro_per_reference(f, train_idx, ev_idx, targets, theta_fn, om, min_train=10):
    """Study-independent LORO: for each reference in the evaluation group, train on the
    training group with that reference removed and predict its evaluation curves. Returns
    {reference: (sum_sq, count)} pooled per reference.
    """
    cols = [c for c, _ in targets]
    per_ref = {}
    train_refs, ev_refs = f.loc[train_idx, "reference"], f.loc[ev_idx, "reference"]
    for ref in ev_refs.unique():
        tr_idx = train_idx[train_refs != ref]
        ev_ref = ev_idx[ev_refs == ref]
        if len(tr_idx) < min_train or len(ev_ref) == 0:
            continue
        preds = fit_predict(f.loc[tr_idx], f.loc[ev_ref], targets)
        sq = cnt = 0.0
        for lid in f.loc[ev_ref, "layer_id"]:
            if lid in om:
                s, c = _curve_sq(theta_fn, cols, preds, lid, om)
                sq += s; cnt += c
        if cnt:
            per_ref[ref] = (sq, cnt)
    return per_ref


def pooled_rmse(per_ref, refs=None):
    """Pool per-reference (sum_sq, count) into a single micro RMSE over the given references
    (all references in ``per_ref`` if ``refs`` is None).
    """
    if refs is None:
        refs = list(per_ref)
    sq = sum(per_ref[r][0] for r in refs if r in per_ref)
    cnt = sum(per_ref[r][1] for r in refs if r in per_ref)
    return (sq / cnt) ** 0.5 if cnt else np.nan


def boot_ci_diff(per_ref_a, per_ref_b, seed=BOOT_SEED, nb=BOOT_N):
    """Study-clustered bootstrap 95% CI for pooled_rmse(a) - pooled_rmse(b) over the
    references common to both cells (resample references with replacement, percentile CI).
    """
    refs = [r for r in per_ref_a if r in per_ref_b]
    if len(refs) < 3:
        return None
    rng = np.random.default_rng(seed)
    refs = np.array(refs, dtype=object)
    point = pooled_rmse(per_ref_a, refs) - pooled_rmse(per_ref_b, refs)
    diffs = []
    for _ in range(nb):
        rs = refs[rng.choice(len(refs), size=len(refs), replace=True)]
        sqa = cnta = sqb = cntb = 0.0
        for r in rs:
            sqa += per_ref_a[r][0]; cnta += per_ref_a[r][1]
            sqb += per_ref_b[r][0]; cntb += per_ref_b[r][1]
        diffs.append((sqa / cnta) ** 0.5 - (sqb / cntb) ** 0.5)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_refs": len(refs), "point": round(float(point), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_a_worse": round(float(np.mean(np.array(diffs) > 0)), 3)}


def stage_ptf(params_path, vg_path, input_csv, out_dir=R_PTF):
    """Build the five support-stratified linear PTFs (A-E), compute the apparent and
    study-independent LORO micro RMSE for all 15 (PTF x eval-group) cells, the 2x2
    decomposition and study-clustered bootstrap CIs on DVC-supported curves; write
    table.csv, per_reference_loro.csv, equations.json, summary.json. Returns out_dir.
    """
    np.seterr(all="ignore")
    os.makedirs(out_dir, exist_ok=True)
    p = pd.read_csv(params_path, low_memory=False)
    p = p[p["converged"] == True].copy()  # noqa: E712
    p["depth_mid"] = (p["hzn_top"] + p["hzn_bot"]) / 2.0
    vg = pd.read_csv(vg_path, low_memory=False)
    f = p.merge(vg[["layer_id", "type", "vg_converged", "theta_s_vg", "theta_r_vg",
                    "alpha_vg", "n_vg", "m_vg"]], on="layer_id", how="inner")
    f = f[f["vg_converged"] == True]  # noqa: E712
    f = f.dropna(subset=BASIC_PTF + ["reference"]).reset_index(drop=True)
    raw = pd.read_csv(input_csv, low_memory=False, encoding=ENCODING)
    om = load_obs_map(raw, keep_saturation=True)

    groups = {"VG": f.index[f["type"] == "VG"], "DVC": f.index[f["type"] == "DVC"], "ALL": f.index}
    PTF = {"A": ("VG", VG_TARGETS, vg_theta), "B": ("ALL", VG_TARGETS, vg_theta),
           "C": ("DVC", DVC_TARGETS, dvc_theta), "D": ("ALL", DVC_TARGETS, dvc_theta),
           "E": ("DVC", VG_TARGETS, vg_theta)}

    rows, eqs_out, per_ref_store, per_ref_rows = [], {}, {}, []
    for name, (tg, targets, theta_fn) in PTF.items():
        train_idx = groups[tg]
        eqs_out[name] = {"train": tg, "model": ("DVC" if theta_fn is dvc_theta else "VG"),
                         "equations": ptf_equations(f.loc[train_idx], targets)}
        for ev in ["VG", "DVC", "ALL"]:
            ev_idx = groups[ev]
            fit_rmse = fit_apply_rmse(f, train_idx, ev_idx, targets, theta_fn, om)
            per_ref = loro_per_reference(f, train_idx, ev_idx, targets, theta_fn, om)
            per_ref_store[(name, ev)] = per_ref
            loro = pooled_rmse(per_ref)
            rows.append(dict(ptf=name, train=tg, model=("DVC" if theta_fn is dvc_theta else "VG"),
                             eval=ev, fit_rmse_micro=round(float(fit_rmse), 4),
                             loro_rmse_micro=(round(float(loro), 4) if np.isfinite(loro) else None),
                             loro_n_refs=len(per_ref)))
            for ref, (sq, cnt) in per_ref.items():
                per_ref_rows.append({"ptf": name, "eval": ev, "reference": ref, "sum_sq": sq,
                                     "count": cnt, "rmse": round((sq / cnt) ** 0.5, 5)})
        print(f"  [ptf] {name} done", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "table.csv"), index=False)
    pd.DataFrame(per_ref_rows).to_csv(os.path.join(out_dir, "per_reference_loro.csv"), index=False)
    with open(os.path.join(out_dir, "equations.json"), "w") as fh:
        json.dump(eqs_out, fh, indent=2)

    contrasts = {
        "modelform_DVCtrain_E-C": boot_ci_diff(per_ref_store[("E", "DVC")], per_ref_store[("C", "DVC")]),
        "modelform_ALLtrain_B-D": boot_ci_diff(per_ref_store[("B", "DVC")], per_ref_store[("D", "DVC")]),
        "training_DVCmodel_C-D": boot_ci_diff(per_ref_store[("C", "DVC")], per_ref_store[("D", "DVC")]),
        "C-B_on_DVC": boot_ci_diff(per_ref_store[("C", "DVC")], per_ref_store[("B", "DVC")])}
    twobytwo = {"eval": "DVC-supported curves (LORO micro RMSE)",
                "VG_model": {"train_DVCsupport(E)": round(pooled_rmse(per_ref_store[("E", "DVC")]), 4),
                             "train_ALL(B)": round(pooled_rmse(per_ref_store[("B", "DVC")]), 4)},
                "DVC_model": {"train_DVCsupport(C)": round(pooled_rmse(per_ref_store[("C", "DVC")]), 4),
                              "train_ALL(D)": round(pooled_rmse(per_ref_store[("D", "DVC")]), 4)}}
    n, ndvc = len(f), int((f["type"] == "DVC").sum())
    summary = {"n_curves": n, "n_DVC_type": ndvc, "n_VG_type": n - ndvc,
               "pct_DVC_type": round(100 * ndvc / n, 2), "table": rows,
               "two_by_two_DVC_eval": twobytwo, "contrasts": contrasts,
               "boot": {"n": BOOT_N, "seed": BOOT_SEED},
               "inputs_sha256": {"params": sha256(params_path), "vg": sha256(vg_path),
                                 "raw": sha256(input_csv)}}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str); fh.write("\n")
    print(f"[ptf] n={n} DVC-type={ndvc} -> {out_dir}")
    return out_dir


# ===========================================================================
# within-curve downsampling support test (Fig 4)
# ===========================================================================
def thin(h, th, k):
    """Keep k points evenly spaced in the sorted sequence, always including both endpoints."""
    n = len(h)
    if k >= n:
        return h, th
    idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    return h[idx], th[idx]


def aicc_both(h, th, tm, dvc_ini):
    """Fit VG (k=4) and dual-VG-CH (k=5) to the given (possibly thinned) points and return
    {'vg': aicc_vg, 'dvc': aicc_dvc} (each None if undefined/failed).
    """
    out = {}
    for tag, model, const, ini in (
        ("vg", "VG", ["q=1"], (min(float(th.max()), tm * 0.99), 0.05, 1.0, 0.35)),
        ("dvc", "dual-VG-CH", ["qr=0", "q=1"], dvc_ini)):
        f = unsatfit.Fit(); f.set_model(model, const=const)
        f.swrc = (h, th); f.b_qs = (0.0, tm); f.ini = ini
        try:
            f.optimize()
            out[tag] = _aicc_of(f)
        except Exception:  # noqa: BLE001
            out[tag] = None
    return out


def stage_downsample(params_path, vg_path, input_csv, out_dir=R_DOWN, sample=DOWN_SAMPLE_N):
    """Within-curve downsampling test: take curves DVC-supported at full resolution (n>=11),
    thin each to 9 and 7 points (endpoints kept), refit both models, and report the fraction
    that remain DVC-supported. Writes downsample_per_curve.csv and summary.json. Returns out_dir.
    """
    _require_unsatfit()
    np.seterr(all="ignore"); warnings.filterwarnings("ignore")
    os.makedirs(out_dir, exist_ok=True)
    vg = pd.read_csv(vg_path, low_memory=False)
    p = pd.read_csv(params_path, low_memory=False, encoding=ENCODING)
    p = p[p["converged"] == True]  # noqa: E712
    dvc_ini = p.set_index("layer_id")[["theta_s", "w1", "alpha", "m1", "m2"]]
    tsmax = p.set_index("layer_id")["theta_s_max"]
    cand = vg[(vg["n_points"] >= 11) & vg["aicc_vg"].notna() & vg["aicc_dvc"].notna()
              & (vg["aicc_dvc"] < vg["aicc_vg"])]
    cand = cand[cand["layer_id"].isin(dvc_ini.index)]
    rng = np.random.default_rng(DOWN_SEED)
    take = min(sample, len(cand))
    sel = cand.iloc[np.sort(rng.choice(len(cand), size=take, replace=False))]
    raw = pd.read_csv(input_csv, low_memory=False, encoding=ENCODING)
    om = load_obs_map(raw, keep_saturation=True)

    rows = []
    for lid in sel["layer_id"]:
        if lid not in om:
            continue
        h, th = om[lid]
        tm = float(tsmax.get(lid, th.max() * 1.05))
        ini = tuple(float(x) for x in dvc_ini.loc[lid].to_numpy())
        rec = {"layer_id": lid, "n_full": len(h)}
        a_full = aicc_both(h, th, tm, ini)
        rec["dvc_support_full"] = (None if (a_full["vg"] is None or a_full["dvc"] is None)
                                   else int(a_full["dvc"] < a_full["vg"]))
        for k in DOWN_TARGETS:
            hk, thk = thin(h, th, k)
            a = aicc_both(hk, thk, tm, ini)
            rec[f"dvc_support_{k}"] = (None if (a["vg"] is None or a["dvc"] is None)
                                       else int(a["dvc"] < a["vg"]))
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "downsample_per_curve.csv"), index=False)
    base = df[df["dvc_support_full"] == 1]
    support = {"full": {"n": int(len(base)), "dvc_support_pct": 100.0}}
    for k in DOWN_TARGETS:
        col = base[f"dvc_support_{k}"].dropna()
        support[str(k)] = {"n_evaluable": int(len(col)),
                           "dvc_support_pct": round(100 * float(col.mean()), 1) if len(col) else None}
    summary = {"sample_n": len(df), "n_dvc_supported_full": int(len(base)), "seed": DOWN_SEED,
               "targets": DOWN_TARGETS,
               "curve": {"n_full_median": int(df["n_full"].median()),
                         "refit_fidelity_pct": round(100 * float((df["dvc_support_full"] == 1).mean()), 1)},
               "support_retained_by_n": support,
               "inputs_sha256": {"params": sha256(params_path), "vg": sha256(vg_path),
                                 "raw": sha256(input_csv)}}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str); fh.write("\n")
    print(f"[downsample] retained at 9/7 pts: "
          f"{support['9']['dvc_support_pct']}% / {support['7']['dvc_support_pct']}%")
    return out_dir


# ===========================================================================
# figures, rendered from the computed results
# ===========================================================================
def make_figures(result_dir=RESULT, fig_dir=FIG):
    """Render the computed manuscript figures (SVG) from the results in ``result_dir``.

    Presentation only: reads the fixed result JSON/CSV files and writes fig/fig2.svg ..
    fig/fig7.svg. This repository outputs only figures whose content is computed from the
    data. Grayscale, dark legible English labels, no titles inside the figures, no hatching.
    """
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "ptf-generalization"
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    INK, GRAY, FILL, MID, DARKFILL, DARKER, GRID = ("#222222", "#777777", "#E8E8E8", "#C4C4C4",
                                                    "#9A9A9A", "#6E6E6E", "#D0D0D0")
    SUPPORT, PTFV2 = R_SUPPORT, R_PTF
    DOWN, WPROFILE = R_DOWN, os.path.join(R_WPROF, "wprofile_example.json")

    def load(path):
        with open(path) as fh:
            return json.load(fh)

    def style(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK)
        ax.tick_params(colors=INK, labelsize=9); ax.set_axisbelow(True)

    def save(fig, num):
        fig.savefig(os.path.join(fig_dir, f"fig{num}.svg"), format="svg", bbox_inches="tight",
                    facecolor="white", metadata={"Date": None})
        plt.close(fig)

    # Fig 2: Delta-AICc + support vs points
    dd = pd.read_csv(os.path.join(SUPPORT, "delta_aicc.csv"))
    sup = load(os.path.join(SUPPORT, "summary.json"))
    npb = pd.read_csv(os.path.join(SUPPORT, "support_by_npoints.csv"))
    npb = npb[npb["comparable"] > 0]
    fig = plt.figure(figsize=(9.8, 4.4)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.30)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    d = np.clip(dd["delta_aicc"].to_numpy(float), -40, 40); bins = np.linspace(-40, 40, 49)
    a1.hist(d[d < 0], bins=bins, color=DARKFILL, edgecolor=INK, lw=0.4, label="DVC supported")
    a1.hist(d[d >= 0], bins=bins, color=FILL, edgecolor=INK, lw=0.4, label="VG supported")
    a1.axvline(0, color=INK, lw=1.1, ls="--"); da = sup["delta_aicc"]; a1.set_xlim(-40, 40)
    a1.set_xlabel(r"$\Delta$AICc = AICc$_{\mathrm{DVC}}-$AICc$_{\mathrm{VG}}$   ($\leftarrow$ DVC better)", fontsize=9.2)
    a1.set_ylabel("comparable curves", fontsize=9.5)
    a1.text(0.02, 0.96, "(a)  n=%d comparable\nmedian %+.1f;  %.0f%% favour DVC"
            % (da["n"], da["median"], da["pct_below_0"]), transform=a1.transAxes, va="top", fontsize=9, color=INK)
    a1.legend(frameon=False, fontsize=8.5, loc="upper right"); a1.grid(axis="y", color=GRID, lw=0.6); style(a1)
    x = np.arange(len(npb))
    bars = a2.bar(x, npb["dvc_support_pct"], width=0.7, color=FILL, edgecolor=INK, lw=0.8)
    for b, v in zip(bars, npb["dvc_support_pct"]):
        if v >= 50:
            b.set_facecolor(DARKFILL)
    for i, v in enumerate(npb["dvc_support_pct"]):
        a2.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    a2.set_ylim(0, 100); a2.set_xticks(x)
    a2.set_xticklabels([f"{bn}\n({n})" for bn, n in zip(npb["bin"], npb["comparable"])], fontsize=8.5)
    a2.set_xlabel("measured points per curve  (comparable n in parentheses)", fontsize=9)
    a2.set_ylabel("DVC support rate [%]", fontsize=9.5)
    a2.text(0.02, 0.96, "(b)", transform=a2.transAxes, va="top", fontsize=9.5, color=INK, weight="bold")
    a2.grid(axis="y", color=GRID, lw=0.6); style(a2); save(fig, 2)

    # Fig 3: support by measurement design
    hr = pd.read_csv(os.path.join(SUPPORT, "support_by_headrange.csv")); hr = hr[hr["comparable"] > 0]
    study = pd.read_csv(os.path.join(SUPPORT, "support_by_study.csv"))
    fig = plt.figure(figsize=(10.4, 4.3)); gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.42)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    swcc = {r["class"]: r for r in sup["support_by_swcc"]}
    order = ["YWYD", "NWYD", "YWND", "NWND"]
    lab = {"YWYD": "both\nends", "NWYD": "no wet\nend", "YWND": "no dry\nend", "NWND": "neither"}
    vals = [swcc[s]["dvc_support_pct"] if swcc.get(s) else 0 for s in order]
    ns = [swcc[s]["comparable"] if swcc.get(s) else 0 for s in order]
    bars = a.bar(range(4), vals, width=0.66, color=FILL, edgecolor=INK, lw=0.8); bars[0].set_facecolor(DARKFILL)
    for i, v in enumerate(vals):
        a.text(i, (v or 0) + 1.5, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    a.set_xticks(range(4))
    a.set_xticklabels([f"{lab[s]}\n({n})" for s, n in zip(order, ns)], fontsize=8.5); a.set_ylim(0, 100)
    a.set_ylabel("DVC support rate [%]", fontsize=9.5); a.set_xlabel("measured end members  (n)", fontsize=9)
    a.text(0.0, 1.04, "(a) wet/dry completeness", transform=a.transAxes, fontsize=9.5, color=INK)
    a.grid(axis="y", color=GRID, lw=0.6); style(a)
    bars = b.bar(range(len(hr)), hr["dvc_support_pct"], width=0.7, color=FILL, edgecolor=INK, lw=0.8)
    for bar, v in zip(bars, hr["dvc_support_pct"]):
        if v >= 40:
            bar.set_facecolor(DARKFILL)
    for i, v in enumerate(hr["dvc_support_pct"]):
        b.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    b.set_xticks(range(len(hr)))
    b.set_xticklabels([f"{dd2}\n({n})" for dd2, n in zip(hr["decades"], hr["comparable"])], fontsize=8.5); b.set_ylim(0, 100)
    b.set_xlabel("head span [decades]  (n)", fontsize=9)
    b.text(0.0, 1.04, "(b) head-range span", transform=b.transAxes, fontsize=9.5, color=INK)
    b.grid(axis="y", color=GRID, lw=0.6); style(b)
    s = study.sort_values("dvc_support_pct").reset_index(drop=True)
    c.bar(range(len(s)), s["dvc_support_pct"], width=1.0, color=GRAY, edgecolor="none")
    c.set_xlim(-0.5, len(s) - 0.5); c.set_ylim(0, 100); q = sup["support_by_study"]
    c.set_xlabel(f"studies sorted by support ({len(s)} comparable)", fontsize=9.5)
    c.set_ylabel("study support rate [%]", fontsize=9.5)
    c.text(0.0, 1.04, "(c) between-study spread", transform=c.transAxes, fontsize=9.5, color=INK)
    c.text(0.05, 0.94, f"{q['n_studies_0pct']} studies at 0%\nmedian {q['dvc_support_pct_q']['50']:.0f}%",
           transform=c.transAxes, va="top", fontsize=9, color=INK)
    c.grid(axis="y", color=GRID, lw=0.6); style(c); save(fig, 3)

    # Fig 4: downsampling
    dwn = load(os.path.join(DOWN, "summary.json")); sr = dwn["support_retained_by_n"]
    xs = [int(dwn["curve"]["n_full_median"]), 9, 7]
    ys = [sr["full"]["dvc_support_pct"], sr["9"]["dvc_support_pct"], sr["7"]["dvc_support_pct"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.plot(xs, ys, "-o", color=INK, mfc=DARKFILL, mec=INK, ms=9, lw=1.6, zorder=3)
    for x0, y0 in zip(xs, ys):
        ax.annotate(f"{y0:.0f}%", (x0, y0), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=10, color=INK, weight="bold")
    ax.set_xlim(6.3, xs[0] + 0.7); ax.set_ylim(-4, 112); ax.invert_xaxis(); ax.set_xticks(xs)
    ax.set_xlabel("measured points retained  (thinned $\\rightarrow$ fewer)", fontsize=9.5)
    ax.set_ylabel("still DVC supported [%]", fontsize=9.5)
    ax.text(0.03, 0.06, f"n={dwn['n_dvc_supported_full']} curves\nDVC-supported at full resolution\n"
            f"(soil identity fixed; only sampling changed)", transform=ax.transAxes, va="bottom",
            fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6); style(ax); save(fig, 4)

    # Fig 5: identifiability x support
    wp = load(WPROFILE); curves = wp["curves"]; idb = sup["identifiability_by_support"]
    fig = plt.figure(figsize=(9.6, 4.1)); gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.36)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    sty = [("-", "o", INK, "VG-like (flat)"),
           ("--", "s", "white", "two-component (clear minimum)")]
    ymax = 0
    for cv, (ls, mk, mfc, lb) in zip(curves, sty):
        w = [pp["w1"] for pp in cv["profile"]]; rm = [pp["rmse"] for pp in cv["profile"]]
        ymax = max(ymax, max(rm))
        a1.plot(w, rm, ls, marker=mk, color=INK, mfc=mfc, mec=INK, ms=5, lw=1.5, zorder=3, label=lb)
    a1.set_xlim(0, 1); a1.set_ylim(0, ymax * 1.25)
    a1.set_xlabel("mode weight $w_1$ (fixed; re-optimise rest)", fontsize=9.5)
    a1.set_ylabel(r"RMSE [m$^3$ m$^{-3}$]", fontsize=9.5)
    a1.text(0.0, 1.04, "(a) $w_1$ profile RMSE", transform=a1.transAxes, fontsize=9.5, color=INK)
    a1.legend(frameon=False, fontsize=8.5, loc="upper center", handlelength=2.0)
    a1.grid(color=GRID, lw=0.5); style(a1)
    keys = ["bimodal", "P1_n1_n2", "P2_w1_0", "P3_w1_1"]
    klab = ["two-\ncomponent", "$n_1{\\approx}n_2$", "$w_1{\\approx}0$", "$w_1{\\approx}1$"]
    for j, (sk, slab, fc) in enumerate([("DVC", "DVC-supported", DARKFILL), ("VG", "VG-supported", FILL)]):
        mix = idb[sk]["degen_mix_pct"]; vals = [mix[k] for k in keys]
        xx = np.arange(len(keys)) + (j - 0.5) * 0.38
        a2.bar(xx, vals, width=0.38, color=fc, edgecolor=INK, lw=0.7, label=f"{slab} (n={idb[sk]['n']})")
        for xi, v in zip(xx, vals):
            a2.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5, color=INK)
    a2.set_xticks(np.arange(len(keys))); a2.set_xticklabels(klab, fontsize=8.5); a2.set_ylim(0, 100)
    a2.set_ylabel("% of curves in support class", fontsize=9.5)
    a2.text(0.0, 1.04, "(b) degeneracy mix by model support", transform=a2.transAxes, fontsize=9.5, color=INK)
    a2.legend(frameon=False, fontsize=8.5, loc="upper right"); a2.grid(axis="y", color=GRID, lw=0.6)
    style(a2); save(fig, 5)

    # Fig 6: support-conditioned LORO error
    t = pd.read_csv(os.path.join(PTFV2, "table.csv"))
    evlab = {"VG": "not classified as\nDVC-supported", "DVC": "DVC\nsupported", "ALL": "all curves"}
    ev_order = ["VG", "DVC", "ALL"]
    show = [("A", "VG PTF, trained not-DVC-supported", FILL), ("B", "VG PTF, trained all", MID),
            ("C", "DVC PTF, trained DVC-support", DARKFILL), ("D", "DVC PTF, trained all", DARKER)]
    fig, ax = plt.subplots(figsize=(9.2, 4.6)); x = np.arange(len(ev_order)); wid = 0.19
    for j, (ptf, lb, fc) in enumerate(show):
        vals = []
        for ev in ev_order:
            r = t[(t["ptf"] == ptf) & (t["eval"] == ev)]
            vals.append(float(r["loro_rmse_micro"].iloc[0]) if len(r) else np.nan)
        xj = x + (j - 1.5) * wid
        ax.bar(xj, vals, width=wid, color=fc, edgecolor=INK, lw=0.7, label=lb)
        for xi, v in zip(xj, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.001, f"{v:.3f}", ha="center", fontsize=8.5, color=INK, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([evlab[e] for e in ev_order], fontsize=9.5)
    ax.set_xlabel("evaluation group", fontsize=9.5)
    ax.set_ylabel(r"reconstructed $\theta(h)$ LORO RMSE [m$^3$ m$^{-3}$]", fontsize=9.5)
    ax.set_ylim(0, max(t["loro_rmse_micro"].dropna()) * 1.28)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2); ax.grid(axis="y", color=GRID, lw=0.6)
    style(ax); save(fig, 6)

    # Fig 7: 2x2 decomposition + CIs
    psum = load(os.path.join(PTFV2, "summary.json")); bb = psum["two_by_two_DVC_eval"]; con = psum["contrasts"]
    fig = plt.figure(figsize=(10.0, 4.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.42)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    cells = [("VG_model", "train_DVCsupport(E)", "VG /\nDVC-support (E)"),
             ("VG_model", "train_ALL(B)", "VG /\nall (B)"),
             ("DVC_model", "train_DVCsupport(C)", "DVC /\nDVC-support (C)"),
             ("DVC_model", "train_ALL(D)", "DVC /\nall (D)")]
    vals = [bb[m][k] for m, k, _ in cells]
    bars = a1.bar(range(4), vals, width=0.7, color=[FILL, MID, DARKFILL, DARKER], edgecolor=INK, lw=0.8)
    for i, v in enumerate(vals):
        a1.text(i, v + 0.0008, f"{v:.4f}", ha="center", fontsize=8.5, color=INK)
    a1.set_xticks(range(4)); a1.set_xticklabels([c[2] for c in cells], fontsize=8.5); a1.set_ylim(0, max(vals) * 1.22)
    a1.set_ylabel(r"LORO $\theta(h)$ RMSE on DVC-supported [m$^3$ m$^{-3}$]", fontsize=9)
    a1.text(0.0, 1.04, "(a) param. & reconstr. $\\times$ training", transform=a1.transAxes, fontsize=9.5, color=INK)
    a1.grid(axis="y", color=GRID, lw=0.6); style(a1)
    items = [("training_DVCmodel_C-D", "training set\n(C $-$ D, DVC reconstr.)"),
             ("modelform_DVCtrain_E-C", "param. & reconstr.\n(E $-$ C, DVC-support)"),
             ("modelform_ALLtrain_B-D", "param. & reconstr.\n(B $-$ D, all)"),
             ("C-B_on_DVC", "best-vs-baseline\n(C $-$ B)")]
    ys = [i * 1.6 for i in range(len(items))][::-1]
    for y, (key, lb) in zip(ys, items):
        r = con[key]; lo, hi = r["ci95"]
        a2.plot([lo, hi], [y, y], color=INK, lw=2.2, solid_capstyle="round", zorder=2)
        a2.plot(r["point"], y, "o", color=INK, ms=7, zorder=3)
        a2.text(0.99, y + 0.30, lb, transform=a2.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8.5, color=INK)
    a2.axvline(0, color=INK, lw=1.0, ls="--"); a2.set_yticks(ys)
    a2.set_yticklabels([f"n={con[k]['n_refs']}" for k, _ in items], fontsize=8.5)
    a2.set_ylim(-0.9, max(ys) + 1.1)
    a2.set_xlabel(r"study-clustered LORO RMSE difference [m$^3$ m$^{-3}$]", fontsize=9.5)
    a2.text(0.0, 1.04, "(b) contrasts (95% cluster-bootstrap CI)", transform=a2.transAxes, fontsize=9.5, color=INK)
    a2.text(0.02, 0.02, "$\\leftarrow$ first term better", transform=a2.transAxes, fontsize=8.5, color=INK)
    a2.grid(axis="x", color=GRID, lw=0.6); style(a2); save(fig, 7)
    print(f"[figures] fig2-7 -> {fig_dir}")


# ===========================================================================
# numeric tables
# ===========================================================================
def make_tables(result_dir=RESULT, table_dir=TABLE):
    """Write the numeric tables computed from the data: table1 (three-group model-support
    counts), table3 (reconstructed-theta LORO micro RMSE matrix), tableS1 (curve accounting),
    tableS3 (per-study LORO folds for DVC-supported evaluation), tableS4 (apparent RMSE matrix).
    Concept/design tables (Table 2, S2, S5) are not computed and are not emitted here.
    """
    os.makedirs(table_dir, exist_ok=True)
    with open(os.path.join(R_SUPPORT, "summary.json")) as fh:
        sup = json.load(fh)
    with open(os.path.join(R_STAGE1, "summary.json")) as fh:
        st1 = json.load(fh)["counts"]
    ptf_tbl = pd.read_csv(os.path.join(R_PTF, "table.csv"))
    per_ref = pd.read_csv(os.path.join(R_PTF, "per_reference_loro.csv"))

    # table1: three groups x {all converged, predictor-complete, PTF-fit frame}
    g = sup["group_counts_all"]; fr = sup["group_counts_ptf_frame"]
    with open(os.path.join(R_PTF, "summary.json")) as fh:
        ptf_sum = json.load(fh)
    n_ptf = ptf_sum["n_curves"]; n_ptf_dvc = ptf_sum["n_DVC_type"]

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    t1 = pd.DataFrame([
        {"group": "DVC-supported", "all_converged": g["DVC_support"]["n"],
         "predictor_complete": fr["DVC_support"]["n"]},
        {"group": "VG-supported", "all_converged": g["VG_support"]["n"],
         "predictor_complete": fr["VG_support"]["n"]},
        {"group": "AICc-uncomparable", "all_converged": g["AICc_uncomparable"]["n"],
         "predictor_complete": fr["AICc_uncomparable"]["n"]},
        {"group": "comparable_subtotal", "all_converged": g["comparable"]["n"],
         "predictor_complete": fr["comparable"]["n"]},
    ])
    t1.to_csv(os.path.join(table_dir, "table1.csv"), index=False)

    # table3: LORO micro RMSE matrix (PTF x eval group)
    t3 = ptf_tbl.pivot(index="ptf", columns="eval", values="loro_rmse_micro")
    t3 = t3.reindex(index=["A", "B", "C", "D", "E"], columns=["VG", "DVC", "ALL"])
    t3.columns = ["eval_not_DVC_supported", "eval_DVC_supported", "eval_all"]
    t3.to_csv(os.path.join(table_dir, "table3.csv"))

    # tableS1: curve accounting
    total = st1["total"]; excl_pts = st1["excl_points"]; excl_nb = st1["excl_nobound"]
    fitted = st1["fitted"]; converged = st1["converged"]; nonconv = st1["nonconverged"]
    s1 = pd.DataFrame([
        {"stage": "Full GSHP database", "curves": total, "removed": None, "reason": None},
        {"stage": "DVC fitting target", "curves": fitted, "removed": excl_pts + excl_nb,
         "reason": f"valid points < {MIN_POINTS} ({excl_pts}) / no theta_s bound ({excl_nb})"},
        {"stage": "DVC converged", "curves": converged, "removed": nonconv, "reason": "dual-VG-CH non-converged"},
        {"stage": "predictor-complete descriptive set", "curves": sup["group_counts_ptf_frame"]["n_frame"],
         "removed": converged - sup["group_counts_ptf_frame"]["n_frame"], "reason": "sand/clay/bulk-density or VG-table match missing"},
        {"stage": "PTF-fit set", "curves": n_ptf, "removed": sup["group_counts_ptf_frame"]["n_frame"] - n_ptf,
         "reason": "horizon-midpoint depth missing; VG non-converged"},
    ])
    s1.to_csv(os.path.join(table_dir, "tableS1.csv"), index=False)

    # tableS3: per-study LORO folds for PTF C on DVC-supported evaluation
    cdvc = per_ref[(per_ref["ptf"] == "C") & (per_ref["eval"] == "DVC")].copy()
    cdvc = cdvc.rename(columns={"count": "pooled_points", "rmse": "loro_rmse_C"})
    cdvc = cdvc[["reference", "pooled_points", "loro_rmse_C"]].sort_values("pooled_points", ascending=False)
    cdvc.to_csv(os.path.join(table_dir, "tableS3.csv"), index=False)

    # tableS4: apparent (non-cross-validated) micro RMSE matrix
    s4 = ptf_tbl.pivot(index="ptf", columns="eval", values="fit_rmse_micro")
    s4 = s4.reindex(index=["A", "B", "C", "D", "E"], columns=["VG", "DVC", "ALL"])
    s4.columns = ["eval_not_DVC_supported", "eval_DVC_supported", "eval_all"]
    s4.to_csv(os.path.join(table_dir, "tableS4.csv"))
    print(f"[tables] table1, table3, tableS1, tableS3, tableS4 -> {table_dir}")


# ===========================================================================
# orchestration
# ===========================================================================
def main():
    """Run the full pipeline: read data/, fit and analyse every stage, and write result/,
    fig/, table/. The fitting stages require unsatfit and take a while (~13.8k curves).
    """
    if not os.path.exists(DATA_CSV):
        raise SystemExit(f"ERROR: input data not found at {DATA_CSV}. See Readme.md for how to "
                         "obtain the GSHP dataset and place it under data/.")
    for d in (RESULT, FIG, TABLE):
        os.makedirs(d, exist_ok=True)

    params_path = stage1_dualvgch_fit(DATA_CSV, R_STAGE1)
    vg_path = stage_vg_fit_aic(params_path, DATA_CSV, R_VG)
    stage_degeneracy(params_path, DATA_CSV, R_DEGEN)
    ident_csv = stage_identifiability(params_path, DATA_CSV, R_IDENT)
    stage_wprofile(params_path, DATA_CSV, R_WPROF)
    stage_support_analysis(params_path, vg_path, DATA_CSV, ident_csv, R_SUPPORT)
    stage_ptf(params_path, vg_path, DATA_CSV, R_PTF)
    stage_downsample(params_path, vg_path, DATA_CSV, R_DOWN)
    make_figures(RESULT, FIG)
    make_tables(RESULT, TABLE)
    print("\n[run] done: result/, fig/, table/ regenerated.")


if __name__ == "__main__":
    main()
