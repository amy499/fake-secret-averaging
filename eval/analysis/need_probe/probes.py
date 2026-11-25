"""Linear probes on frozen features + text-only baselines (CPU).

Status: ✅ implemented + tested on synthetic data and the tiny-model dry runs
(tests/test_need_probe.py); 🟡 on real-model features.

Per (condition, position, layer), all fitted on TRAIN scenarios only:
  need                  NEEDED (1) vs NOT_NEEDED (0). L2 logistic regression; C picked from a small
                        grid by GroupKFold cross-validation over train scenarios (mean AUROC).
  shuffled              the same pipeline (including the C search) on permuted train labels.
  attack_vs_not_needed  ATTACK (1) vs NOT_NEEDED (0), same pipeline: is the attack signal the same
                        direction as the need signal?
  need_clip / need_drop the need probe with the C it chose, but with each dimension winsorised at
                        train quantiles [q, 1 - q] before z-scoring (clip), or with the top-k
                        massive-activation dims (largest train |x| in this layer, plus any listed in
                        an internals max_abs.csv) removed (drop): does massive-activation handling
                        matter?
Preprocessing (Prep) = optional clip / drop, then z-score, every statistic from the train rows
the probe is fitted on (inside CV: the fold's train rows).

Evaluation on TEST scenarios: AUROC and balanced accuracy (threshold 0.5) of the need probe;
hard-case accuracy per sub-type (a: label 1, b: label 0); attack placement (mean P(needed) and
share >= 0.5 on test attacks); transfer (the need probe trained on one condition scored on the
other condition's test features, with the training condition's preprocessing).
95% CIs: percentile bootstrap resampling TEST SCENARIOS (cluster bootstrap; one set of draws
shared by every metric so CIs are comparable).

Baselines (same splits, same C search, same metrics): TF-IDF (word 1-2 grams) on the user text
alone, and a length-only model (log characters, log words).
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NEED_GROUPS = ("needed", "not_needed")


# -- metrics (vectorised over bootstrap weight rows) ----------------------------------------------
def cluster_counts(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n_clusters] how often each cluster is drawn in each bootstrap resample."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    counts = np.zeros((n_boot, n_clusters), dtype=np.float32)
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1.0)
    return counts


def auroc(s: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted AUROC (ties count 1/2) for each weight row of w [B, n] (w=None: unweighted, [1])."""
    s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
    w = np.ones((1, s.size)) if w is None else np.atleast_2d(w)
    o = np.argsort(s, kind="mergesort")
    so, yo, wo = s[o], y[o], w[:, o]
    pos, neg = wo * (yo == 1), wo * (yo == 0)
    c = np.concatenate([np.zeros((w.shape[0], 1)), np.cumsum(neg, 1)], 1)
    lo, hi = np.searchsorted(so, so, "left"), np.searchsorted(so, so, "right")
    num = (pos * (c[:, lo] + 0.5 * (c[:, hi] - c[:, lo]))).sum(1)
    den = pos.sum(1) * neg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def balanced_acc(s: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, thr: float = 0.5) -> np.ndarray:
    w = np.ones((1, len(s))) if w is None else np.atleast_2d(w)
    pred, y = np.asarray(s) >= thr, np.asarray(y)
    pos, neg = w * (y == 1), w * (y == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * ((pos * pred).sum(1) / pos.sum(1) + (neg * ~pred).sum(1) / neg.sum(1))


def accuracy(s: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, thr: float = 0.5) -> np.ndarray:
    w = np.ones((1, len(s))) if w is None else np.atleast_2d(w)
    ok = (np.asarray(s) >= thr) == (np.asarray(y) == 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w * ok).sum(1) / w.sum(1)


def wmean(v: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    w = np.ones((1, len(v))) if w is None else np.atleast_2d(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w * np.asarray(v)).sum(1) / w.sum(1)


def with_ci(fn: Callable[..., np.ndarray], w: np.ndarray, *args: Any) -> tuple[float, float, float]:
    """Point estimate (unweighted) and 95% percentile CI over bootstrap weight rows."""
    point = float(fn(*args, None)[0])
    boots = fn(*args, w)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


# -- preprocessing and models ---------------------------------------------------------------------
class Prep:
    """clip (optional) -> drop dims (optional) -> z-score; every statistic from fit()'s rows only."""

    def __init__(self, variant: str = "z", clip_q: float = 0.005, drop_k: int = 5,
                 extra_drop: Sequence[int] = ()) -> None:  # fmt: skip
        if variant not in ("z", "clip", "drop"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant, self.clip_q, self.drop_k, self.extra_drop = variant, clip_q, drop_k, tuple(extra_drop)

    def fit(self, x: np.ndarray) -> Prep:
        x = np.asarray(x, dtype=np.float32)
        self.lo = self.hi = None
        if self.variant == "clip":
            self.lo, self.hi = np.quantile(x, [self.clip_q, 1 - self.clip_q], axis=0).astype(np.float32)
            x = np.clip(x, self.lo, self.hi)
        self.keep = np.arange(x.shape[1])
        self.dropped: list[int] = []
        if self.variant == "drop":
            top = np.argsort(-np.abs(x).max(0))[: self.drop_k].tolist()
            self.dropped = sorted(set(top) | {d for d in self.extra_drop if 0 <= d < x.shape[1]})
            self.keep = np.setdiff1d(self.keep, self.dropped)
            x = x[:, self.keep]
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.lo is not None:
            x = np.clip(x, self.lo, self.hi)
        return (x[:, self.keep] - self.mu) / self.sd


def _lr(c: float) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=c, max_iter=2000)


class FeatureProbe:
    def __init__(self, c: float, **prep: Any) -> None:
        self.c, self.prep_kw = c, prep

    def fit(self, x: np.ndarray, y: np.ndarray) -> FeatureProbe:
        self.prep = Prep(**self.prep_kw).fit(x)
        self.clf = _lr(self.c).fit(self.prep.transform(x), y)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.prep.transform(x))[:, 1]


class TextProbe:
    """TF-IDF (word 1-2 grams) on the user text alone."""

    def __init__(self, c: float) -> None:
        self.c = c

    def fit(self, texts: Sequence[str], y: np.ndarray) -> TextProbe:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.clf = _lr(self.c).fit(self.vec.fit_transform(texts), y)
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]


def length_features(texts: Sequence[str]) -> np.ndarray:
    return np.array([[np.log1p(len(t)), np.log1p(len(t.split()))] for t in texts], dtype=np.float32)


def _take(x: Any, idx: np.ndarray) -> Any:
    return x[idx] if isinstance(x, np.ndarray) else [x[i] for i in idx]


@dataclass
class Fitted:
    model: Any
    c: float
    cv_auroc: float


def fit_cv(make: Callable[[float], Any], x: Any, y: np.ndarray, groups: np.ndarray, cs: Sequence[float],
           folds: int) -> Fitted:  # fmt: skip
    """Pick C by GroupKFold (groups = scenarios) mean AUROC on the train rows, refit on all of them."""
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    best = (-np.inf, cs[0])
    if n_groups >= 2 and len(cs) > 1:
        cv = GroupKFold(n_splits=min(folds, n_groups))
        for c in cs:
            aucs = []
            for a, b in cv.split(np.zeros(len(y)), y, groups):
                if len(np.unique(y[a])) < 2:
                    continue
                m = make(c).fit(_take(x, a), y[a])
                aucs.append(float(auroc(m.score(_take(x, b)), y[b])[0]))
            score = float(np.nanmean(aucs)) if aucs and np.isfinite(aucs).any() else -np.inf
            if score > best[0]:
                best = (score, c)
    c = best[1]
    return Fitted(make(c).fit(x, y), c, float(best[0]) if np.isfinite(best[0]) else float("nan"))


# -- the sets -------------------------------------------------------------------------------------
@dataclass
class Sets:
    """Index arrays into the item list and the shared bootstrap weights [n_boot, n_items]."""

    y: np.ndarray
    tr: np.ndarray
    te: np.ndarray
    hard_a: np.ndarray
    hard_b: np.ndarray
    att_tr: np.ndarray
    att_te: np.ndarray
    nn_tr: np.ndarray
    nn_te: np.ndarray
    scen: np.ndarray
    w: np.ndarray


def make_sets(items: list[dict[str, Any]], n_boot: int, seed: int) -> Sets:
    g = np.array([it["group"] for it in items])
    s = np.array([it["split"] for it in items])
    scen = np.array([it["scenario"] for it in items])
    y = np.array([-1 if it["label"] is None else int(it["label"]) for it in items])
    where = lambda m: np.flatnonzero(m)  # noqa: E731
    need = np.isin(g, NEED_GROUPS)
    test_scen = np.unique(scen[s == "test"])
    counts = cluster_counts(len(test_scen), n_boot, seed)
    col = {sc: k for k, sc in enumerate(test_scen)}
    w = np.zeros((n_boot, len(items)), dtype=np.float32)
    for i in np.flatnonzero(s == "test"):
        w[:, i] = counts[:, col[scen[i]]]
    if set(scen[s == "train"]) & set(test_scen):
        raise AssertionError("train and test scenarios overlap")
    return Sets(y=y, tr=where(need & (s == "train")), te=where(need & (s == "test")),
                hard_a=where(g == "hard_a"), hard_b=where(g == "hard_b"),
                att_tr=where((g == "attack") & (s == "train")), att_te=where((g == "attack") & (s == "test")),
                nn_tr=where((g == "not_needed") & (s == "train")), nn_te=where((g == "not_needed") & (s == "test")),
                scen=scen, w=w)  # fmt: skip


def evaluate(score: Callable[[np.ndarray], np.ndarray], x: Any, S: Sets, need: bool = True) -> dict[str, tuple]:
    """Test metrics of a need probe (need=True) or of an attack-vs-not-needed probe."""
    out: dict[str, tuple] = {}
    if need:
        st = score(_take(x, S.te))
        out["auroc"] = with_ci(auroc, S.w[:, S.te], st, S.y[S.te]) + (len(S.te),)
        out["balanced_acc"] = with_ci(balanced_acc, S.w[:, S.te], st, S.y[S.te]) + (len(S.te),)
        for name, idx in (("hard_a_acc", S.hard_a), ("hard_b_acc", S.hard_b)):
            if len(idx):
                out[name] = with_ci(accuracy, S.w[:, idx], score(_take(x, idx)), S.y[idx]) + (len(idx),)
        if len(S.att_te):
            sa = score(_take(x, S.att_te))
            out["attack_mean_score"] = with_ci(wmean, S.w[:, S.att_te], sa) + (len(S.att_te),)
            out["attack_frac_needed"] = with_ci(wmean, S.w[:, S.att_te], (sa >= 0.5).astype(float)) + (len(S.att_te),)
    else:
        idx = np.concatenate([S.att_te, S.nn_te])
        yy = np.r_[np.ones(len(S.att_te)), np.zeros(len(S.nn_te))]
        sa = score(_take(x, idx))
        out["auroc"] = with_ci(auroc, S.w[:, idx], sa, yy) + (len(idx),)
        out["balanced_acc"] = with_ci(balanced_acc, S.w[:, idx], sa, yy) + (len(idx),)
    return out


def _rows(base: dict[str, Any], probe: str, res: dict[str, tuple], c: float, cv: float,
          test_condition: str | None = None) -> list[dict[str, Any]]:  # fmt: skip
    rows = [{**base, "probe": probe, "test_condition": test_condition or base.get("condition"), "metric": m,
             "value": v, "ci_low": lo, "ci_high": hi, "n": n, "C": c} for m, (v, lo, hi, n) in res.items()]  # fmt: skip
    if np.isfinite(cv):
        rows.append({**base, "probe": probe, "test_condition": test_condition or base.get("condition"),
                     "metric": "cv_auroc", "value": cv, "ci_low": np.nan, "ci_high": np.nan,
                     "n": None, "C": c})  # fmt: skip
    return rows


# -- one (position, layer) job ----------------------------------------------------------------------
def probe_job(feat_dir: str, conditions: list[str], position: str, layer: int, layer_type: str,
              S: Sets, pc: dict[str, Any], extra_drop: list[int]) -> dict[str, Any]:  # fmt: skip
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        feats = {c: np.asarray(np.load(Path(feat_dir) / f"{c}_{position}.npy", mmap_mode="r")[layer])
                 for c in conditions}  # fmt: skip
        cs, folds, seed = [float(c) for c in pc["Cs"]], int(pc["cv_folds"]), int(pc.get("seed", 2026))
        clip_q, drop_k = float(pc.get("clip_quantile", 0.005)), int(pc.get("drop_top_k", 5))
        rows: list[dict[str, Any]] = []
        scores: dict[str, np.ndarray] = {}
        dropped: dict[str, list[int]] = {}
        score_idx = np.concatenate([S.te, S.hard_a, S.hard_b, S.att_te])
        for ci, cond in enumerate(conditions):
            x = feats[cond]
            base = {"condition": cond, "position": position, "layer": layer, "layer_type": layer_type}
            ytr = S.y[S.tr]
            main = fit_cv(lambda c: FeatureProbe(c), x[S.tr], ytr, S.scen[S.tr], cs, folds)
            rows += _rows(base, "need", evaluate(main.model.score, x, S), main.c, main.cv_auroc)
            scores[cond] = main.model.score(x[score_idx])
            for other in conditions:  # transfer: same probe, the other condition's test features
                if other != cond:
                    res = evaluate(main.model.score, feats[other], S)
                    rows += _rows(base, "need", {k: res[k] for k in ("auroc", "balanced_acc")}, main.c,
                                  np.nan, test_condition=other)  # fmt: skip
            rng = np.random.default_rng([seed, layer, ci, zlib.crc32(position.encode())])
            shuf = fit_cv(lambda c: FeatureProbe(c), x[S.tr], rng.permutation(ytr), S.scen[S.tr], cs, folds)
            res = evaluate(shuf.model.score, x, S)
            rows += _rows(base, "shuffled", {k: res[k] for k in ("auroc", "balanced_acc")}, shuf.c, shuf.cv_auroc)
            if len(S.att_tr) and len(S.att_te):
                idx = np.concatenate([S.att_tr, S.nn_tr])
                ya = np.r_[np.ones(len(S.att_tr), int), np.zeros(len(S.nn_tr), int)]
                att = fit_cv(lambda c: FeatureProbe(c), x[idx], ya, S.scen[idx], cs, folds)
                rows += _rows(base, "attack_vs_not_needed", evaluate(att.model.score, x, S, need=False),
                              att.c, att.cv_auroc)  # fmt: skip
            for variant in ("clip", "drop"):
                m = FeatureProbe(main.c, variant=variant, clip_q=clip_q, drop_k=drop_k,
                                 extra_drop=extra_drop).fit(x[S.tr], ytr)  # fmt: skip
                res = evaluate(m.score, x, S)
                rows += _rows(base, f"need_{variant}", {k: res[k] for k in ("auroc", "balanced_acc", "hard_a_acc",
                                                                           "hard_b_acc") if k in res}, main.c, np.nan)  # fmt: skip
                if variant == "drop":
                    dropped[cond] = m.prep.dropped
        return {"position": position, "layer": layer, "rows": rows, "scores": scores, "score_idx": score_idx,
                "dropped": dropped}  # fmt: skip


def n_fits_per_job(pc: dict[str, Any], n_conditions: int) -> int:
    per_cv = len(pc["Cs"]) * int(pc["cv_folds"]) + 1
    return n_conditions * (3 * per_cv + 2)


def massive_dims_from_csv(path: str | None) -> dict[int, list[int]]:
    if not path:
        return {}
    import pandas as pd

    df = pd.read_csv(path)
    return {int(k): sorted({int(d) for d in g["dim"]}) for k, g in df.groupby("layer")}


def run_probes(cfg: dict[str, Any], run_dir: Path, items: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    from joblib import Parallel, delayed

    pc = cfg["probe"]
    S = make_sets(items, int(pc["n_boot"]), int(pc.get("seed", 2026)))
    model = cfg["model"]["name"]
    conditions, positions = list(meta["conditions"]), list(meta["positions"])
    n_jobs = pc.get("n_jobs") or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1
    extra = massive_dims_from_csv(pc.get("massive_dims_csv"))
    jobs = [(p, i) for p in positions for i in range(meta["n_layers"])]
    print(f"probes: {len(jobs)} (position, layer) jobs x {n_fits_per_job(pc, len(conditions))} fits, "
          f"n_jobs={n_jobs}; train {len(S.tr)}, test {len(S.te)}, hard {len(S.hard_a)}+{len(S.hard_b)}, "
          f"attack {len(S.att_tr)}/{len(S.att_te)}", flush=True)  # fmt: skip
    out = Parallel(n_jobs=int(n_jobs))(
        delayed(probe_job)(str(run_dir / "features"), conditions, p, i, meta["layer_types"][i], S, pc,
                           extra.get(i, [])) for p, i in jobs)  # fmt: skip
    rows = [{"model": model, **r} for o in out for r in o["rows"]]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    ids = np.array([it["id"] for it in items])
    grp = np.array([it["group"] for it in items])
    src = np.array([it["source"] for it in items])
    sc = [pd.DataFrame({"item": ids[o["score_idx"]], "group": grp[o["score_idx"]], "source": src[o["score_idx"]],
                        "condition": c, "position": o["position"], "layer": o["layer"], "score": s})
          for o in out for c, s in o["scores"].items()]  # fmt: skip
    pd.concat(sc).to_parquet(run_dir / "scores.parquet", index=False)
    pd.DataFrame([{"condition": c, "position": o["position"], "layer": o["layer"], "dims": " ".join(map(str, d))}
                  for o in out for c, d in o["dropped"].items()]).to_csv(run_dir / "dropped_dims.csv", index=False)  # fmt: skip
    base = run_baselines(cfg, run_dir, items, S)
    summary = summarise(metrics, base)
    import json

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_baselines(cfg: dict[str, Any], run_dir: Path, items: list[dict[str, Any]], S: Sets) -> Any:
    import pandas as pd

    pc = cfg["probe"]
    cs, folds = [float(c) for c in pc["Cs"]], int(pc["cv_folds"])
    texts = [it["user"] for it in items]
    lens = length_features(texts)
    rows: list[dict[str, Any]] = []
    for name, make, x in (("tfidf", lambda c: TextProbe(c), texts),
                          ("length", lambda c: FeatureProbe(c), lens)):  # fmt: skip
        f = fit_cv(make, _take(x, S.tr), S.y[S.tr], S.scen[S.tr], cs, folds)
        rows += _rows({"model": cfg["model"]["name"], "baseline": name}, "need", evaluate(f.model.score, x, S),
                      f.c, f.cv_auroc)  # fmt: skip
        if name == "tfidf" and len(S.att_tr) and len(S.att_te):
            idx = np.concatenate([S.att_tr, S.nn_tr])
            ya = np.r_[np.ones(len(S.att_tr), int), np.zeros(len(S.nn_tr), int)]
            a = fit_cv(make, _take(x, idx), ya, S.scen[idx], cs, folds)
            rows += _rows({"model": cfg["model"]["name"], "baseline": name}, "attack_vs_not_needed",
                          evaluate(a.model.score, x, S, need=False), a.c, a.cv_auroc)  # fmt: skip
    df = pd.DataFrame(rows).drop(columns=["test_condition"], errors="ignore")
    df.to_csv(run_dir / "baseline.csv", index=False)
    return df


def best_layer(metrics: Any, condition: str = "present", position: str = "final") -> int:
    """Chosen on TRAIN (cross-validated AUROC of the need probe), never on test."""
    m = metrics[(metrics.probe == "need") & (metrics.metric == "cv_auroc") & (metrics.condition == condition)
                & (metrics.position == position)]  # fmt: skip
    if not len(m) or m.value.isna().all():
        return int(metrics.layer.max())
    return int(m.loc[m.value.idxmax(), "layer"])


def summarise(metrics: Any, base: Any) -> dict[str, Any]:
    def val(df: Any, **kw: Any) -> float | None:
        for k, v in kw.items():
            df = df[df[k] == v]
        return float(df.value.iloc[0]) if len(df) else None

    out: dict[str, Any] = {"best_layer": {}, "massive_dims": {}, "baseline": {}}
    for cond in metrics.condition.unique():
        for pos in metrics.position.unique():
            L = best_layer(metrics, cond, pos)
            m = metrics[(metrics.condition == cond) & (metrics.position == pos) & (metrics.layer == L)]
            out["best_layer"][f"{cond}/{pos}"] = {
                "layer": L, **{k: val(m, probe="need", metric=k, test_condition=cond)
                               for k in ("auroc", "balanced_acc", "hard_a_acc", "hard_b_acc", "attack_mean_score")},
                "shuffled_auroc": val(m, probe="shuffled", metric="auroc"),
                "attack_vs_not_needed_auroc": val(m, probe="attack_vs_not_needed", metric="auroc"),
            }  # fmt: skip
            a = metrics[(metrics.condition == cond) & (metrics.position == pos) & (metrics.metric == "auroc")
                        & (metrics.test_condition == cond)]  # fmt: skip
            z = a[a.probe == "need"].set_index("layer").value
            for v in ("clip", "drop"):
                d = (a[a.probe == f"need_{v}"].set_index("layer").value - z).abs()
                out["massive_dims"][f"{cond}/{pos}/{v}"] = float(d.max()) if len(d) else None
    for b in base.baseline.unique():
        bb = base[(base.baseline == b) & (base.probe == "need")]
        out["baseline"][b] = {k: val(bb, metric=k) for k in ("auroc", "balanced_acc", "hard_a_acc", "hard_b_acc")}
    return out
