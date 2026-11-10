"""Shared experiment context: config, data, output folders, helpers."""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ensure_dir, load_config, path
from ..data import FEATURES, TARGET, load_dataset, split_xy
from ..models import available_models, make_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Ctx:
    cfg: dict
    quick: bool = False
    tables: Path = None
    figures: Path = None
    cache: Path = None
    df: pd.DataFrame = None
    X: pd.DataFrame = None
    y: np.ndarray = None
    models: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    @property
    def seed(self) -> int:
        return self.cfg["project"]["seed"]

    def say(self, msg: str) -> None:
        print(msg, flush=True)
        self.log.append(msg)

    def save_table(self, df: pd.DataFrame, name: str) -> Path:
        p = self.tables / f"{name}.csv"
        df.to_csv(p, index=False)
        return p

    def save_json(self, obj, name: str) -> Path:
        p = self.tables / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")
        return p

    def factory(self, name: str, **kw):
        return lambda seed: make_model(name, self.cfg, seed, **kw)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def make_ctx(config_path: str | None = None, quick: bool = False, overrides: dict | None = None) -> Ctx:
    cfg = load_config(config_path, overrides)
    if quick:   # smoke-test sized run
        cfg["cv"].update(n_splits=5, n_repeats=2, n_bootstrap=200)
        cfg["torch"].update(epochs=60, patience=15, paper_epochs=150)
        cfg["explain"].update(n_patients=12, stability_runs=3)
        cfg["faithfulness"].update(n_patients=10)
    ctx = Ctx(cfg=cfg, quick=quick)
    res = path(cfg, "results")
    ctx.tables = ensure_dir(res / "tables")
    ctx.figures = ensure_dir(res / "figures")
    ctx.cache = ensure_dir(res / "cache")
    raw = path(cfg, "data_raw")
    ctx.df = load_dataset(cfg, raw)
    ctx.X, ctx.y = split_xy(ctx.df)
    ctx.y = ctx.y.values
    ctx.models = available_models(cfg["models"]["benchmark"])
    ctx.say(f"[data] source={ctx.df.attrs.get('source')} n={len(ctx.df)} "
            f"(duplicates removed: {ctx.df.attrs.get('duplicates_removed', 0)}) "
            f"prevalence={ctx.y.mean():.3f} sites={sorted(ctx.df.site.unique())}")
    return ctx


def primary_model(ctx: Ctx) -> str:
    """Primary model = configured, else best mean AUC from the benchmark (if it has been run), else logreg."""
    want = ctx.cfg["models"]["primary"]
    if want != "auto":
        return want
    f = ctx.tables / "benchmark_summary.csv"
    if f.exists():
        s = pd.read_csv(f)
        return str(s.sort_values("_auc", ascending=False).iloc[0]["model"])
    return "logreg"


def timer():
    t = time.time()
    return lambda: f"{time.time() - t:.1f}s"
