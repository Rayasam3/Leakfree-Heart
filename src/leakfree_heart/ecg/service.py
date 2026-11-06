"""Inference service for the ECG model used by the web app.

Loads the trained ECG ResNet seeds (and the end-to-end fusion seeds) saved by `run_ecg.py`, re-creates the
temperature and the conformal thresholds from the cached calibration predictions (so the app uses exactly the
numbers of the study), and answers one ECG at a time:

    probability (ECG only), probability (ECG + age/sex), answer-or-defer decision, per-lead explanation.

Explanation = integrated gradients of the ECG logit with respect to the (normalised) signal, baseline = the
training-set mean ECG, averaged over the seeds.  A deletion check (flat-lining the most important leads versus
random leads) is reported for every ECG so a reader can see whether the explanation matters to the model.
This is a model attribution, not a clinical explanation, and it has not been validated by cardiologists.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import data as D
from .nets import TORCH

SAMPLE_PER_CLASS = 4
IG_STEPS = 32


class ECGUnavailable(RuntimeError):
    pass


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class ECGService:
    def __init__(self, root: Path, results_dir: Path, ecfg: dict, seed: int = 42):
        self.root, self.results, self.ecfg, self.seed = Path(root), Path(results_dir), ecfg, seed
        self.cache = self.results / "cache"
        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.alphas = [float(a) for a in ecfg.get("alphas", [0.05, 0.10, 0.15, 0.20])]
        try:
            self._load()
            self.ready = True
        except ECGUnavailable as e:
            self.error = str(e)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if not TORCH:
            raise ECGUnavailable("PyTorch is not installed.")
        import torch

        from ..uncertainty import calibrate_conformal
        from .nets import ECGNet
        from .study import fit_temperature

        sig = self.cache / "signals_100hz.npy"
        preds = self.cache / "ecg_preds.npz"
        seeds = sorted(self.cache.glob("ecg_cnn_seed*.pt"))
        if not (sig.exists() and preds.exists() and seeds):
            raise ECGUnavailable("ECG model files not found. Run:  python -m leakfree_heart run-ecg   (full run, not --quick)")
        if not (self.root / "ptbxl_database.csv").exists():
            raise ECGUnavailable(f"PTB-XL metadata not found in {self.root}. Run:  python -m leakfree_heart download-ptbxl")
        self.meta = D.load_metadata(self.root, self.ecfg.get("min_likelihood", 50))
        self.X = np.load(sig, mmap_mode="r")
        if self.X.shape[0] != len(self.meta):
            raise ECGUnavailable("Cached signals do not match the metadata (was only the --quick run done?). Run the full study.")
        self.split = D.official_split(self.meta, self.seed)
        self.y = self.meta["label"].to_numpy().astype(int)

        z = np.load(preds)
        per: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        for key in z.files:
            c, s, k = key.split("|")
            per.setdefault(c, {}).setdefault(int(s), {})[k] = z[key]
        ens = {c: {k: np.mean([per[c][s][k] for s in per[c]], axis=0) for k in ("cal_a", "cal_b", "test")} for c in per}
        ya, yb, yt = (self.y[self.split[k]] for k in ("cal_a", "cal_b", "test"))
        self.T = {c: fit_temperature(ens[c]["cal_a"], ya) for c in ("ecg_cnn", "fusion_e2e") if c in ens}
        p_cal = _sigmoid(ens["ecg_cnn"]["cal_b"] / self.T["ecg_cnn"])
        self.cal = calibrate_conformal(p_cal, yb)
        self.test_prob = _sigmoid(ens["ecg_cnn"]["test"] / self.T["ecg_cnn"])
        self.test_y = yt

        self.dev = torch.device("cpu")          # the GPU is left to the language model (4 GB)
        self.nets = {"ecg_cnn": [], "fusion_e2e": []}
        for cond in self.nets:
            for f in sorted(self.cache.glob(f"{cond}_seed*.pt")):
                ck = torch.load(f, map_location="cpu", weights_only=False)
                net = ECGNet(tab_dim=2 if ck["tab"] else 0, width=int(self.ecfg.get("width", 32)))
                net.load_state_dict(ck["state"])
                net.eval().to(self.dev)
                self.nets[cond].append((net, torch.as_tensor(np.asarray(ck["mean"]), dtype=torch.float32),
                                        torch.as_tensor(np.asarray(ck["std"]), dtype=torch.float32)))
        self.samples_ids = self._pick_samples()

    def _pick_samples(self) -> list[int]:
        """A fixed, shuffled set of held-out (fold 10) ECGs: normal plus each single diagnostic superclass."""
        rng = np.random.default_rng(self.seed)
        te = self.split["test"]
        sup = self.meta["superclasses"].to_numpy()
        pick: list[int] = []
        for c in ["NORM"] + D.ABNORMAL:
            cand = [i for i in te if sup[i] == {c}]
            if cand:
                pick += [int(i) for i in rng.choice(cand, size=min(SAMPLE_PER_CLASS, len(cand)), replace=False)]
        rng.shuffle(pick)
        return pick

    # ------------------------------------------------------------------ info
    def status(self) -> dict:
        if not self.ready:
            return {"ready": False, "detail": self.error}
        rows = []
        for a in self.alphas:
            dec = self._decide_many(self.test_prob, a)
            ans = dec >= 0
            rows.append({"alpha": a, "confidence_level": round(1 - a, 2), "answered": float(ans.mean()),
                         "accuracy_answered": float(np.mean(dec[ans] == self.test_y[ans])) if ans.any() else None})
        return {"ready": True, "n_seeds": len(self.nets["ecg_cnn"]), "n_test": int(len(self.test_y)),
                "temperature": self.T, "alphas": rows, "sampling_rate_hz": D.FS, "leads": D.LEAD_NAMES,
                "n_samples": D.FS * 10}

    def _decide_many(self, prob: np.ndarray, alpha: float) -> np.ndarray:
        from ..uncertainty import decisions_from_sets, prediction_sets
        return decisions_from_sets(prediction_sets(prob, self.cal.thresholds(alpha, True)))

    def samples(self) -> list[dict]:
        m = self.meta
        return [{"ecg_id": int(m.index[i]), "age": float(m["age_years"].iloc[i]), "sex": int(m["sex_f"].iloc[i])}
                for i in self.samples_ids]

    # ------------------------------------------------------------------ inference
    def _norm(self, sig: np.ndarray, mean, std):
        import torch
        return (torch.as_tensor(sig, dtype=torch.float32)[None] - mean) / std       # (1, T, 12)

    @staticmethod
    def _tab(age: float, sex: int):
        import torch
        return torch.tensor([[(age - 60.0) / 20.0, float(sex)]], dtype=torch.float32)

    def _logit(self, cond: str, sig: np.ndarray, age: float, sex: int, mask_leads: list[int] | None = None) -> float:
        import torch
        outs = []
        with torch.no_grad():
            for net, mean, std in self.nets[cond]:
                x = self._norm(sig, mean, std)
                if mask_leads:
                    x[:, :, mask_leads] = 0.0                               # flat line at the training mean
                outs.append(float(net(x.transpose(1, 2), self._tab(age, sex) if net.tab_dim else None)))
        return float(np.mean(outs))

    def _prob(self, cond: str, logit: float) -> float:
        return float(_sigmoid(logit / self.T[cond]))

    def _ig(self, sig: np.ndarray) -> tuple[np.ndarray, float]:
        """Integrated gradients (midpoint rule) of the ECG-only logit; returns (T, 12) attributions and the
        completeness gap |sum(attr) - (f(x) - f(baseline))|, both averaged over seeds."""
        import torch
        acc = np.zeros(sig.shape, dtype=np.float64)
        gaps = []
        for net, mean, std in self.nets["ecg_cnn"]:
            x = self._norm(sig, mean, std)[0]                               # (T, 12)
            a = ((torch.arange(IG_STEPS, dtype=torch.float32) + 0.5) / IG_STEPS)[:, None, None]
            xs = (a * x[None]).requires_grad_(True)
            with torch.enable_grad():
                out = net(xs.transpose(1, 2))
                out.sum().backward()
            attr = (xs.grad.mean(0) * x).detach().numpy()
            with torch.no_grad():
                f1 = float(net(x[None].transpose(1, 2)))
                f0 = float(net(torch.zeros_like(x)[None].transpose(1, 2)))
            gaps.append(abs(float(attr.sum()) - (f1 - f0)))
            acc += attr
        return acc / len(self.nets["ecg_cnn"]), float(np.mean(gaps))

    def predict(self, signal: np.ndarray | None, ecg_id: int | None, age: float | None, sex: int | None,
                alpha: float = 0.05) -> dict:
        if not self.ready:
            raise ECGUnavailable(self.error or "ECG model not loaded")
        if min(abs(alpha - a) for a in self.alphas) > 1e-9:
            raise ValueError(f"alpha must be one of {self.alphas}")
        recorded = None
        if ecg_id is not None:
            if ecg_id not in self.meta.index:
                raise ValueError("unknown ecg_id")
            i = int(self.meta.index.get_loc(ecg_id))
            if i not in set(self.split["test"].tolist()):
                raise ValueError("Only held-out test ECGs (fold 10) can be selected by id; the others were used for training.")
            signal = np.asarray(self.X[i], dtype=np.float32)
            age, sex = float(self.meta["age_years"].iloc[i]), int(self.meta["sex_f"].iloc[i])
            recorded = {"abnormal": bool(self.y[i]), "superclasses": sorted(self.meta["superclasses"].iloc[i])}
        if signal is None or age is None or sex is None:
            raise ValueError("Provide an ecg_id, or a signal together with age and sex.")
        signal = np.nan_to_num(np.asarray(signal, dtype=np.float32))
        if signal.ndim != 2 or signal.shape[1] != 12 or signal.shape[0] < 10 * D.FS:
            raise ValueError(f"signal must have 12 columns (leads) and at least {10 * D.FS} rows (10 s at {D.FS} Hz)")
        signal = signal[: 10 * D.FS]
        if float(np.abs(signal).max()) > 50:
            raise ValueError("signal amplitude looks wrong; values must be in millivolts")

        with self.lock:
            lg = self._logit("ecg_cnn", signal, age, sex)
            p = self._prob("ecg_cnn", lg)
            dec = int(self._decide_many(np.array([p]), alpha)[0])
            sets = [bool(s) for s in self._set(p, alpha)]
            p_fus = self._prob("fusion_e2e", self._logit("fusion_e2e", signal, age, sex)) if self.nets["fusion_e2e"] else None
            attr, gap = self._ig(signal)
            imp = np.abs(attr).sum(0)
            imp = imp / max(imp.sum(), 1e-12)
            top = [int(i) for i in np.argsort(-imp)[:3]]
            rng = np.random.default_rng(0)
            rand = [self._prob("ecg_cnn", self._logit("ecg_cnn", signal, age, sex, [int(j) for j in rng.choice(12, 3, replace=False)]))
                    for _ in range(20)]
            p_top = self._prob("ecg_cnn", self._logit("ecg_cnn", signal, age, sex, top))

        bins = attr.reshape(100, 10, 12).sum(1)                              # (100, 12): 0.1 s bins
        scale = max(float(np.abs(bins).max()), 1e-12)
        return {
            "probability": p, "probability_with_age_sex": p_fus, "alpha": alpha, "target_coverage": 1 - alpha,
            "decision": {1: "positive", 0: "negative", -1: "defer"}[dec], "conformal_set": [i for i, s in enumerate(sets) if s],
            "age": age, "sex": sex, "recorded": recorded,
            "signal": np.round(signal, 3).T.tolist(), "leads": D.LEAD_NAMES, "fs": D.FS,
            "explanation": {
                "method": "integrated gradients, baseline = training-set mean ECG, averaged over seeds",
                "lead_importance": [round(float(v), 4) for v in imp],
                "relevance": np.round(bins / scale, 3).T.tolist(),            # (12, 100) in [-1, 1]; + pushes toward abnormal
                "completeness_gap": round(gap, 4),
                "deletion_check": {"flatten": [D.LEAD_NAMES[i] for i in top], "probability_full": p,
                                   "probability_without_top_leads": p_top,
                                   "probability_without_random_leads_mean": float(np.mean(rand))},
            },
        }

    def _set(self, p: float, alpha: float):
        from ..uncertainty import prediction_sets
        return prediction_sets(np.array([p]), self.cal.thresholds(alpha, True))[0]
