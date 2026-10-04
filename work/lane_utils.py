"""Shared, reproducible March contract for the Week 4–7 notebooks.

Only pseudonymized data are used. Page frames/queues stay in ignored parquet/CSV
files; aggregate JSON receipts and figures can be committed.
"""
from pathlib import Path
import hashlib
import json
import os

import duckdb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REVISION = "50cbf7c3909d07be4d1b5906b4d09e882e5acbf2"
REPO_ID = "FlyRank/internship-warehouse"
FEATURES = ["log_impressions", "log_clicks", "ctr_pct", "weighted_position", "active_search_days"]
KEYS = ["client_hash_id", "content_hash_id"]
CONTRACT = {
    "revision": REVISION,
    "source_month": "2026-03",
    "feature_window": ["2026-03-01", "2026-03-14"],
    "decision_date": "2026-03-18",
    "outcome_window": ["2026-03-18", "2026-03-31"],
    "minimum_earlier_impressions": 100,
    "usable_days_per_window": 14,
    "target": "later_impressions < 0.8 * earlier_impressions",
    "features": FEATURES,
    "split": "GroupShuffleSplit(test_size=0.25, random_state=42)",
    "primary_metric": "macro_precision_at_20_full_client_queues",
}
BASELINE = {
    "formula": "log1p(earlier_impressions) * (1 + weak_click_capture)",
    "weak_click_capture": "ctr_pct < 0.5 and 1 <= weighted_position <= 20",
    "boost": 2.0,
    "reason_code": "visible_page_review",
    "action": "review_search_performance",
}

FEATURE_SQL = """
WITH covered AS (
 SELECT *, report_date BETWEEN DATE '2026-03-01' AND DATE '2026-03-14' AS earlier,
           report_date BETWEEN DATE '2026-03-18' AND DATE '2026-03-31' AS later
 FROM march WHERE gsc_data_available IS TRUE
), totals AS (
 SELECT client_hash_id, content_hash_id,
 COUNT(*) FILTER (WHERE earlier AND gsc_impressions IS NOT NULL
                  AND gsc_clicks IS NOT NULL) AS earlier_days,
 COUNT(*) FILTER (WHERE later AND gsc_impressions IS NOT NULL) AS later_days,
 SUM(gsc_impressions) FILTER (WHERE earlier) AS earlier_impressions,
 SUM(gsc_clicks) FILTER (WHERE earlier) AS earlier_clicks,
 SUM(gsc_impressions) FILTER (WHERE later) AS later_impressions,
 COUNT(*) FILTER (WHERE earlier AND gsc_impressions > 0) AS active_search_days,
 SUM(gsc_impressions * gsc_avg_position) FILTER (
   WHERE earlier AND gsc_impressions > 0 AND gsc_avg_position > 0 AND isfinite(gsc_avg_position))
 / NULLIF(SUM(gsc_impressions) FILTER (
   WHERE earlier AND gsc_impressions > 0 AND gsc_avg_position > 0 AND isfinite(gsc_avg_position)), 0)
 AS weighted_position
 FROM covered WHERE earlier OR later GROUP BY client_hash_id, content_hash_id
)
SELECT * FROM totals WHERE earlier_days = 14 AND earlier_impressions >= 100
"""

def fingerprint(frame, columns=None):
    selected = frame[columns] if columns else frame
    values = pd.util.hash_pandas_object(selected, index=False).values.tobytes()
    return hashlib.sha256(values).hexdigest()

def write_receipt(root, name, receipt):
    output = Path(root) / "work/outputs" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output

def load_pages(root):
    root = Path(root)
    cache = root / "work/outputs/march_page_features.parquet"
    manifest_path = cache.with_suffix(".json")
    if cache.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") == CONTRACT:
            pages = pd.read_parquet(cache)
            assert fingerprint(pages) == manifest["data_fingerprint"], "Cached page frame changed."
            return pages
    partition = root / ".venv/warehouse-cache/datasets--FlyRank--internship-warehouse/snapshots" / REVISION
    files = sorted(partition.glob("fact_content_daily_performance/month=2026-03/*.parquet"))
    if not files:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from google.colab import userdata
                token = userdata.get("HF_TOKEN")
            except Exception:
                token = None
        if not token:
            raise RuntimeError("Set HF_TOKEN in the environment or Colab Secrets after accepting dataset access.")
        try:
            partition = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
                allow_patterns=["fact_content_daily_performance/month=2026-03/*.parquet"],
                token=token, cache_dir=str(root / ".venv/warehouse-cache")))
        except Exception:
            raise RuntimeError("March download failed; check dataset access, token permissions, and network.") from None
        finally:
            del token
        files = sorted(partition.glob("fact_content_daily_performance/month=2026-03/*.parquet"))
    assert files
    con = duckdb.connect()
    con.from_parquet([str(p) for p in files], union_by_name=True).create_view("march")
    source = con.sql("""SELECT COUNT(*) AS n, MIN(report_date) AS first_date, MAX(report_date) AS last_date,
        COUNT(*) FILTER (WHERE gsc_data_available IS TRUE) AS available_rows FROM march""").df().iloc[0]
    assert str(source.first_date.date()) == "2026-03-01" and str(source.last_date.date()) == "2026-03-31"
    pages = con.sql(FEATURE_SQL).df().sort_values(KEYS).reset_index(drop=True)
    con.close()
    assert not pages.duplicated(KEYS).any() and len(pages) > 0
    pages["log_impressions"] = np.log1p(pages.earlier_impressions)
    pages["log_clicks"] = np.log1p(pages.earlier_clicks)
    pages["ctr_pct"] = 100 * pages.earlier_clicks / pages.earlier_impressions
    pages["future_decline"] = pd.Series(pd.NA, index=pages.index, dtype="Int64")
    labelled = pages.later_days.eq(14)
    pages.loc[labelled, "future_decline"] = pages.loc[labelled, "later_impressions"].lt(
        0.8 * pages.loc[labelled, "earlier_impressions"]).astype(int)
    assert not np.isinf(pages[FEATURES].to_numpy(dtype=float)).any()
    cache.parent.mkdir(parents=True, exist_ok=True)
    pages.to_parquet(cache, index=False)
    receipt = {
        "contract": CONTRACT, "source_rows": int(source.n), "gsc_rows_available": int(source.available_rows),
        "pages": len(pages), "labelled_pages": int(labelled.sum()), "unknown_targets": int((~labelled).sum()),
        "clients": int(pages.client_hash_id.nunique()), "data_fingerprint": fingerprint(pages),
    }
    write_receipt(root, manifest_path.name, receipt)
    return pages

def labelled_pages(pages):
    return pages.loc[pages.future_decline.notna()].reset_index(drop=True)

def grouped_split(pages):
    y = pages.future_decline.astype(int)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(
        pages[FEATURES], y, pages.client_hash_id))
    assert set(pages.iloc[train].client_hash_id).isdisjoint(set(pages.iloc[test].client_hash_id))
    assert y.iloc[train].nunique() == y.iloc[test].nunique() == 2
    return train, test

def split_fingerprint(pages, train, test):
    return {"train_keys_sha256": fingerprint(pages.iloc[train], KEYS),
            "test_keys_sha256": fingerprint(pages.iloc[test], KEYS),
            "train_pages": len(train), "test_pages": len(test),
            "train_clients": int(pages.iloc[train].client_hash_id.nunique()),
            "test_clients": int(pages.iloc[test].client_hash_id.nunique())}

def rule_score(frame):
    """One fixed demand-and-click-capture rule, using earlier features only."""
    weak_capture = frame.ctr_pct.lt(0.5) & frame.weighted_position.between(1, 20)
    return frame.log_impressions.to_numpy(dtype=float) * (1 + weak_capture.to_numpy(dtype=int))

def rank_frame(frame, scores):
    ranked = frame.copy()
    ranked["score"] = np.asarray(scores, dtype=float)
    assert np.isfinite(ranked.score).all()
    ranked = ranked.sort_values(["client_hash_id", "score", "content_hash_id"], ascending=[True, False, True])
    ranked["client_rank"] = ranked.groupby("client_hash_id").cumcount() + 1
    return ranked

def evaluate(frame, scores, k=20):
    y = frame.future_decline.astype(int)
    ranked = rank_frame(frame, scores)
    selected = ranked.loc[ranked.client_rank.le(k)]
    clients = selected.groupby("client_hash_id").future_decline.agg(["mean", "size"])
    full = clients.loc[clients["size"].eq(k)]
    short = clients.loc[clients["size"].lt(k)]
    base = frame.groupby("client_hash_id").future_decline.mean()
    metric = lambda value: float(value) if np.isfinite(value) else None
    return {
        "roc_auc": metric(roc_auc_score(y, scores)) if y.nunique() == 2 else None,
        "average_precision": metric(average_precision_score(y, scores)) if y.nunique() == 2 else None,
        "macro_precision_at_20": metric(full["mean"].mean()) if len(full) else None,
        "macro_base_rate_same_clients": metric(base.loc[full.index].mean()) if len(full) else None,
        "full_queue_clients": len(full), "short_queue_clients": len(short),
        "short_queue_mean": metric(short["mean"].mean()) if len(short) else None,
        "pages": len(frame), "clients": len(clients), "pooled_base_rate": float(y.mean()),
    }

def make_model():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=1000, random_state=42))

def summary_table(results):
    return pd.DataFrame(results).T[["macro_precision_at_20", "macro_base_rate_same_clients", "roc_auc",
        "average_precision", "full_queue_clients", "short_queue_clients", "pages"]].round(4)
