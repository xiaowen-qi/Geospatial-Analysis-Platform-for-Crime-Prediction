"""
Crime-Prediction Wizard  -  lightweight, CPU-friendly, wizard-style UI
======================================================================
Pages
  1  Crime-data ingest  (load CSV / XLSX, choose columns, filter, aggregate)
  2  Geography & External features  (shapefile + contextual CSV, feature-importance)
  3  Neighbors & Feature engineering  (spatial / cosine neighbors, lag construction)
  4  Model training & Results  (Ridge, ElasticNet, RF, GBT - threaded)

Threading
  Heavy work (neighbor calc, feature build, model training) runs on a
  QThread that emits progress / log / error / done signals so the UI
  never freezes and the user always sees what is happening.
"""
from __future__ import annotations
import sys, os, time, traceback, warnings
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from PyQt5.QtCore import (
    Qt, QDate, QThread, pyqtSignal, pyqtSlot, QTimer
)
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QListWidgetItem, QDateEdit, QGroupBox,
    QTextEdit, QSpinBox, QStackedWidget,
    QProgressBar, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QScrollArea, QTabWidget
)

warnings.filterwarnings("ignore")

# -- optional deps ---------------------------------------------------
try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_validate
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# -- lightweight correlation helpers ---------------------------------
def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    x, y = pd.Series(x).astype(float), pd.Series(y).astype(float)
    mask = x.notna() & y.notna(); x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    xr, yr = x.rank(), y.rank()
    xr = (xr - xr.mean()) / (xr.std(ddof=0) or 1.0)
    yr = (yr - yr.mean()) / (yr.std(ddof=0) or 1.0)
    return float(np.clip((xr * yr).mean(), -1, 1))


def cosine_neighbors(ext_df: pd.DataFrame, ext_key: str,
                     features: List[str], k: int) -> Dict[str, List[str]]:
    """Top-k most similar regions by cosine similarity of their external
    feature vectors. Features are z-scored across regions first so that
    large-valued columns (e.g. POI counts) don't dominate small ones (e.g.
    rates); constant columns become 0 and drop out."""
    edf = ext_df.copy()
    edf[ext_key] = edf[ext_key].astype(str)
    feat_cols = [c for c in features if c in edf.columns and c != ext_key]
    if not feat_cols:
        return {}
    edf = edf.dropna(subset=feat_cols).set_index(ext_key)
    X = edf[feat_cols].values.astype(float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - X.mean(axis=0)) / sd
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T
    idx_names = list(edf.index)
    nbrs: Dict[str, List[str]] = {}
    for i, loc in enumerate(idx_names):
        scores = sim[i].copy()
        scores[i] = -999
        top_k = np.argsort(scores)[::-1][:k]
        nbrs[loc] = [str(idx_names[j]) for j in top_k]
    return nbrs


# ====================================================================
# WORKER  - runs all heavy computation off the UI thread
# ====================================================================
class PipelineWorker(QThread):
    """Runs neighbor computation, feature engineering and model training."""
    progress = pyqtSignal(int)       # 0-100
    log      = pyqtSignal(str)       # status message
    error    = pyqtSignal(str)       # fatal error
    # NB: not called "finished" - that would shadow QThread.finished
    results_ready = pyqtSignal(dict)  # result payload

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.p = params
        self._cancelled = False

    def cancel(self):
        """Request graceful cancellation."""
        self._cancelled = True

    def _emit(self, pct: int, msg: str):
        self.progress.emit(pct)
        self.log.emit(msg)

    def run(self):
        try:
            results = self._run_pipeline()
            if not self._cancelled:
                self.results_ready.emit(results)
        except Exception:
            if not self._cancelled:
                self.error.emit(f"Pipeline failed:\n{traceback.format_exc()}")

    # ----------------------------------------------------------------
    def _run_pipeline(self) -> dict:
        p = self.p
        crime_panel  = p["crime_panel"]
        loc_col      = p["loc_col"]
        target_col   = p["target_col"]
        time_cols    = p["time_cols"]
        shape_gdf    = p.get("shape_gdf")
        shape_key    = p.get("shape_key")
        ext_df       = p.get("ext_df")
        ext_key      = p.get("ext_key")
        chosen_feats = p.get("chosen_features", [])
        seq_len      = p["seq_len"]
        pred_len     = p["pred_len"]
        k_spatial    = p.get("k_spatial", 8)
        k_cosine     = p.get("k_cosine", 5)
        models_sel   = p.get("models", ["Ridge", "RandomForest"])
        cv_folds     = 5

        # --- Intersect locations across all data sources ---
        crime_locs = set(crime_panel[loc_col].astype(str).unique())
        self._emit(1, f"Crime data has {len(crime_locs)} locations.")

        valid_locs = crime_locs.copy()

        if (shape_gdf is not None and shape_key
                and shape_key in shape_gdf.columns):
            shape_locs = set(shape_gdf[shape_key].astype(str).unique())
            self._emit(1, f"Shapefile has {len(shape_locs)} regions.")
            valid_locs = valid_locs & shape_locs
            dropped = crime_locs - shape_locs
            if dropped:
                self._emit(1, f"  {len(dropped)} crime locations "
                              f"not in shapefile (will be excluded).")

        if ext_df is not None and ext_key:
            ext_locs = set(ext_df[ext_key].astype(str).unique())
            self._emit(1, f"External features has {len(ext_locs)} regions.")
            valid_locs = valid_locs & ext_locs
            dropped2 = crime_locs - valid_locs
            if dropped2 and dropped2 != (crime_locs - shape_locs if shape_gdf is not None else set()):
                self._emit(1, f"  Additional locations dropped "
                              f"due to external features mismatch.")

        locations = sorted(valid_locs)
        n_locs = len(locations)
        if n_locs == 0:
            raise ValueError(
                "No locations in common across crime data, "
                "shapefile, and external features. "
                "Check that your region ID columns match.")

        # Filter crime panel to only valid locations
        crime_panel = crime_panel[
            crime_panel[loc_col].astype(str).isin(valid_locs)
        ].copy()

        self._emit(2, f"Using {n_locs} locations (intersection of all sources), "
                      f"seq_len={seq_len}, pred_len={pred_len}")

        # ---- STEP 1: spatial neighbors -------------------------
        spatial_nbrs: Dict[str, List[str]] = {}
        if (shape_gdf is not None and shape_key
                and shape_key in shape_gdf.columns):
            self._emit(5, "Computing spatial neighbors from shapefile ...")
            gdf = shape_gdf.copy()
            gdf[shape_key] = gdf[shape_key].astype(str)
            gdf = gdf.set_index(shape_key)
            for loc in locations:
                if loc not in gdf.index:
                    spatial_nbrs[loc] = []
                    continue
                geom = gdf.loc[loc, "geometry"]
                mask = gdf.geometry.intersects(geom) & (gdf.index != loc)
                nbrs = gdf.index[mask].tolist()
                if len(nbrs) > k_spatial:
                    cent = geom.centroid
                    dists = [(n, gdf.loc[n, "geometry"].centroid.distance(cent))
                             for n in nbrs]
                    dists.sort(key=lambda x: x[1])
                    nbrs = [n for n, _ in dists[:k_spatial]]
                spatial_nbrs[loc] = [str(n) for n in nbrs]
            avg_n = np.mean([len(v) for v in spatial_nbrs.values()])
            self._emit(12, f"Spatial neighbors done: {len(spatial_nbrs)} regions, "
                           f"avg {avg_n:.1f} neighbors each.")
        else:
            self._emit(12, "No shapefile provided - spatial neighbors skipped.")

        # ---- STEP 2: cosine-similarity neighbors ---------------
        cosine_nbrs: Dict[str, List[str]] = {}
        if ext_df is not None and ext_key and chosen_feats:
            self._emit(15, "Computing cosine-similarity neighbors ...")
            cosine_nbrs = cosine_neighbors(ext_df, ext_key, chosen_feats, k_cosine)
            self._emit(22, f"Cosine neighbors done: {len(cosine_nbrs)} regions, "
                           f"top-{k_cosine} each.")
        else:
            self._emit(22, "No features selected - cosine neighbors skipped.")

        # ---- STEP 3: build per-location time series ------------
        self._emit(25, "Building temporal panel ...")
        panel = crime_panel.copy()
        panel[loc_col] = panel[loc_col].astype(str)
        panel = panel.sort_values([loc_col] + time_cols).reset_index(drop=True)

        ts_dict: Dict[str, np.ndarray] = {}
        for loc, grp in panel.groupby(loc_col):
            ts_dict[str(loc)] = grp[target_col].values.astype(float)

        if not ts_dict:
            raise ValueError("No time series data after processing.")
        T = min(len(v) for v in ts_dict.values())
        if T == 0:
            raise ValueError("Time series length is 0.")
        for loc in ts_dict:
            ts_dict[loc] = ts_dict[loc][-T:]

        self._emit(30, f"Common time-series length = {T} across "
                       f"{len(ts_dict)} locations.")

        # ---- STEP 4: feature engineering -----------------------
        self._emit(32, "Engineering lag / rolling / neighbor features ...")

        n_samples = T - seq_len - pred_len + 1
        if n_samples <= 0:
            raise ValueError(
                f"Not enough timesteps ({T}) for seq_len={seq_len} "
                f"+ pred_len={pred_len}. Need at least {seq_len + pred_len}.")
        # n_total includes extra unlabeled samples at the end
        # (up to the very last row) for practical prediction
        n_total = T - seq_len + 1

        def _lag_feats(ts, L):
            feats = {}
            for lag in range(1, L + 1):
                feats[f"crime_{lag}_period_ago"] = ts[L - lag: L - lag + n_total]
            s = pd.Series(ts)
            rm = s.rolling(L, min_periods=1).mean().values
            rx = s.rolling(L, min_periods=1).max().values
            rs = s.rolling(L, min_periods=1).std().values
            slopes = np.zeros(T)
            for t in range(T):
                start = max(0, t - L + 1)
                seg = ts[start:t + 1]
                if len(seg) >= 2:
                    x_ = np.arange(len(seg))
                    slopes[t] = np.polyfit(x_, seg, 1)[0]
            safe = L - 1
            feats[f"avg_crime_last_{L}_periods"]    = rm[safe: safe + n_total]
            feats[f"max_crime_last_{L}_periods"]    = rx[safe: safe + n_total]
            feats[f"crime_variability_{L}_periods"] = rs[safe: safe + n_total]
            feats[f"crime_trend_last_{L}_periods"]  = slopes[safe: safe + n_total]
            return feats

        def _nbr_agg(loc, nbr_dict, prefix):
            label = ("nearby_area" if prefix == "sp"
                     else "similar_area")
            nbrs = nbr_dict.get(loc, [])
            valid = [n for n in nbrs if n in ts_dict]
            if not valid:
                z = np.zeros(n_total)
                return {f"{label}_avg_crime": z,
                        f"{label}_max_crime": z,
                        f"{label}_crime_spread": z}
            stack = np.stack([ts_dict[n] for n in valid], axis=0)
            avg_ts = stack.mean(axis=0)
            max_ts = stack.max(axis=0)
            std_ts = (stack.std(axis=0) if stack.shape[0] > 1
                      else np.zeros(T))
            safe = seq_len - 1
            return {
                f"{label}_avg_crime":    avg_ts[safe: safe + n_total],
                f"{label}_max_crime":    max_ts[safe: safe + n_total],
                f"{label}_crime_spread": std_ts[safe: safe + n_total],
            }

        rows = []
        for idx, loc in enumerate(locations):
            if loc not in ts_dict:
                continue
            ts = ts_dict[loc]
            y_vals = ts[seq_len + pred_len - 1:
                        seq_len + pred_len - 1 + n_samples]

            feats = _lag_feats(ts, seq_len)
            s_agg = _nbr_agg(loc, spatial_nbrs, "sp")
            c_agg = _nbr_agg(loc, cosine_nbrs, "cos")

            for s in range(n_total):
                row = {"__loc": loc, "__time_idx": s}
                for fn, arr in feats.items():
                    row[fn] = arr[s] if s < len(arr) else np.nan
                for fn, arr in s_agg.items():
                    row[fn] = arr[s] if s < len(arr) else np.nan
                for fn, arr in c_agg.items():
                    row[fn] = arr[s] if s < len(arr) else np.nan
                row["__target"] = (y_vals[s] if s < len(y_vals)
                                   else np.nan)
                rows.append(row)

            if (idx + 1) % max(1, n_locs // 10) == 0:
                if self._cancelled:
                    raise InterruptedError("Cancelled by user.")
                pct = 32 + int(28 * (idx + 1) / n_locs)
                self._emit(pct, f"Feature engineering: "
                                f"{idx + 1}/{n_locs} locations ...")

        feat_df_all = pd.DataFrame(rows)
        # Separate labeled samples (for training) from unlabeled
        # latest samples (for practical prediction)
        predict_df = feat_df_all[feat_df_all["__target"].isna()].copy()
        feat_df = feat_df_all.dropna(subset=["__target"])
        self._emit(62, f"Feature matrix: {feat_df.shape[0]} samples x "
                       f"{feat_df.shape[1] - 3} features "
                       f"({len(predict_df)} unlabeled prediction samples).")

        # attach static external features
        if ext_df is not None and ext_key and chosen_feats:
            edf = ext_df.copy()
            edf[ext_key] = edf[ext_key].astype(str)
            keep = [ext_key] + [c for c in chosen_feats
                                if c in edf.columns]
            feat_df = feat_df.merge(edf[keep], left_on="__loc",
                                    right_on=ext_key, how="left")
            if ext_key != "__loc":
                feat_df.drop(columns=[ext_key], inplace=True,
                             errors="ignore")
            predict_df = predict_df.merge(edf[keep], left_on="__loc",
                                          right_on=ext_key, how="left")
            if ext_key != "__loc":
                predict_df.drop(columns=[ext_key], inplace=True,
                                errors="ignore")
            self._emit(65, f"Static features attached "
                           f"({len(chosen_feats)} cols).")

        # ---- STEP 5: train / test split & evaluate models ------
        meta_cols = [c for c in feat_df.columns if c.startswith("__")]
        feature_cols = [c for c in feat_df.columns if c not in meta_cols]

        # Chronological train/test split: keep temporal ordering,
        # hold out the last 20% of time indices as a true test set
        # that NO model ever sees during training or CV.
        time_idx = feat_df["__time_idx"].values
        split_point = int(np.percentile(time_idx, 80))
        train_mask = time_idx <= split_point
        test_mask  = time_idx >  split_point
        if not test_mask.any():
            raise ValueError(
                f"Only {len(np.unique(time_idx))} labelled period(s) - too few "
                f"for a hold-out test set. Use a longer date range or a "
                f"smaller sequence length / horizon.")

        X_all = feat_df[feature_cols].values.astype(float)
        y_all = feat_df["__target"].values.astype(float)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        self._emit(68, f"Train set: {X_train_sc.shape[0]} samples | "
                       f"Test set: {X_test_sc.shape[0]} samples | "
                       f"{X_train_sc.shape[1]} features")

        model_map = {
            "Ridge": lambda: Ridge(alpha=1.0),
            "ElasticNet": lambda: ElasticNet(
                alpha=0.1, l1_ratio=0.5, max_iter=2000),
            "RandomForest": lambda: RandomForestRegressor(
                n_estimators=200, max_depth=12,
                n_jobs=-1, random_state=42),
            "GradientBoosting": lambda: GradientBoostingRegressor(
                n_estimators=150, max_depth=4, subsample=0.8,
                learning_rate=0.1, random_state=42),
        }

        results: Dict[str, dict] = {}
        trained_models: Dict[str, Any] = {}
        # Temporal CV folds: rows are ordered by location, so split on the
        # *time periods* instead of row positions. Each fold trains on
        # earlier periods and validates on the following block, with all
        # locations of a period kept on the same side.
        train_times = time_idx[train_mask]
        uniq_times = np.unique(train_times)
        n_splits = min(cv_folds, max(2, len(y_train) // 50),
                       len(uniq_times) - 1)
        tscv = []
        if n_splits >= 2:
            for tr_t, va_t in TimeSeriesSplit(n_splits=n_splits).split(uniq_times):
                tscv.append((np.flatnonzero(np.isin(train_times, uniq_times[tr_t])),
                             np.flatnonzero(np.isin(train_times, uniq_times[va_t]))))
            self._emit(69, f"Temporal CV: {n_splits} folds over "
                           f"{len(uniq_times)} training periods.")
        else:
            self._emit(69, f"Only {len(uniq_times)} training periods - "
                           f"cross-validation skipped.")
        active = [m for m in models_sel if m in model_map]
        n_models = len(active)

        for mi, mname in enumerate(active):
            if self._cancelled:
                self._emit(pct, "Cancelled by user.")
                break

            pct = 70 + int(25 * mi / max(1, n_models))
            self._emit(pct, f"Training {mname} ({mi + 1}/{n_models}) ...")

            model = model_map[mname]()
            t0 = time.time()

            # Cross-validation on TRAIN set only (one pass, three metrics)
            cv_mae = cv_rmse = cv_r2 = np.array([np.nan])
            try:
                if not tscv:
                    raise ValueError("too few training periods")
                cv = cross_validate(
                    model, X_train_sc, y_train, cv=tscv,
                    scoring=("neg_mean_absolute_error",
                             "neg_mean_squared_error", "r2"))
                cv_mae = -cv["test_neg_mean_absolute_error"]
                cv_rmse = np.sqrt(-cv["test_neg_mean_squared_error"])
                cv_r2 = cv["test_r2"]
            except Exception as e:
                self.log.emit(f"  Warning: CV for {mname} skipped: {e}")

            # Final fit on full train set
            model.fit(X_train_sc, y_train)
            trained_models[mname] = model
            y_pred_train = model.predict(X_train_sc)
            y_pred_test  = model.predict(X_test_sc)
            elapsed = time.time() - t0

            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
            elif hasattr(model, "coef_"):
                fi = np.abs(np.atleast_1d(model.coef_))
            else:
                fi = np.zeros(len(feature_cols))

            fi_pairs = sorted(zip(feature_cols, fi.tolist()),
                              key=lambda x: -x[1])

            results[mname] = {
                "cv_mae_mean":  float(np.nanmean(cv_mae)),
                "cv_mae_std":   float(np.nanstd(cv_mae)),
                "cv_rmse_mean": float(np.nanmean(cv_rmse)),
                "cv_rmse_std":  float(np.nanstd(cv_rmse)),
                "cv_r2_mean":   float(np.nanmean(cv_r2)),
                "cv_r2_std":    float(np.nanstd(cv_r2)),
                "train_mae":    float(mean_absolute_error(
                    y_train, y_pred_train)),
                "train_rmse":   float(np.sqrt(mean_squared_error(
                    y_train, y_pred_train))),
                "train_r2":     float(r2_score(
                    y_train, y_pred_train)),
                "test_mae":     float(mean_absolute_error(
                    y_test, y_pred_test)),
                "test_rmse":    float(np.sqrt(mean_squared_error(
                    y_test, y_pred_test))),
                "test_r2":      float(r2_score(
                    y_test, y_pred_test)),
                "time_sec":     round(elapsed, 2),
                "top_features": fi_pairs[:15],
                "n_train":      len(y_train),
                "n_test":       len(y_test),
                "n_features":   X_train_sc.shape[1],
            }
            self.log.emit(
                f"  {mname}: Val-MAE={np.nanmean(cv_mae):.3f}"
                f"+-{np.nanstd(cv_mae):.3f}  "
                f"Test-MAE={mean_absolute_error(y_test, y_pred_test):.3f}  "
                f"Test-R2={r2_score(y_test, y_pred_test):.3f}  "
                f"time={elapsed:.1f}s")

        # ---- STEP 6: Region risk ranking (latest unlabeled period) ---
        risk_ranking: List[Dict[str, Any]] = []
        best_model_name = ""
        if results:
            finite_maes = {k: v["test_mae"] for k, v in results.items()
                           if np.isfinite(v["test_mae"])}
            if finite_maes:
                best_model_name = min(finite_maes, key=finite_maes.get)
                best_model = trained_models[best_model_name]
                self._emit(96, f"Generating risk ranking with "
                               f"best model ({best_model_name}) ...")

                # Use the unlabeled latest samples (built from
                # the very last row of each location's time series,
                # so no future data is required as a label)
                if not predict_df.empty:
                    latest_tidx = int(predict_df["__time_idx"].max())
                    latest_df = predict_df[
                        predict_df["__time_idx"] == latest_tidx].copy()
                else:
                    latest_df = pd.DataFrame()

                if not latest_df.empty:
                    X_latest = latest_df[feature_cols].values.astype(float)
                    X_latest = np.nan_to_num(
                        X_latest, nan=0.0, posinf=0.0, neginf=0.0)
                    X_latest_sc = scaler.transform(X_latest)
                    preds = best_model.predict(X_latest_sc)

                    for i, (_, row) in enumerate(latest_df.iterrows()):
                        risk_ranking.append({
                            "region": str(row["__loc"]),
                            "predicted_crime": float(preds[i]),
                        })
                    risk_ranking.sort(
                        key=lambda x: -x["predicted_crime"])

        self._emit(100, "Pipeline complete.")
        return {
            "model_results": results,
            "feature_cols":  feature_cols,
            "n_locations":   n_locs,
            "n_train":       len(y_train),
            "n_test":        len(y_test),
            "ts_length":     T,
            "risk_ranking":  risk_ranking,
            "best_model":    best_model_name,
        }


# ====================================================================
# PAGE 1: Crime data ingest
# ====================================================================
class Page1Crime(QWidget):
    def __init__(self, ctx: dict):
        super().__init__()
        self.ctx = ctx
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Step 1  -  Crime Data Ingest")
        title.setProperty("class", "title")
        outer.addWidget(title)

        sub = QLabel(
            "Load your raw crime CSV / XLSX, map columns, "
            "pick a date range and crime categories.")
        sub.setProperty("class", "subtitle")
        sub.setWordWrap(True)
        outer.addWidget(sub)
        outer.addSpacing(6)

        g = QGridLayout()
        r = 0

        self.load_btn = QPushButton("  Load Crime CSV / XLSX  ")
        self.load_btn.clicked.connect(self._load)
        g.addWidget(self.load_btn, r, 0, 1, 4)
        r += 1

        self.loc_combo = QComboBox()
        self.loc_combo.setEnabled(False)
        self.date_combo = QComboBox()
        self.date_combo.setEnabled(False)
        self.cat_combo = QComboBox()
        self.cat_combo.setEnabled(False)

        g.addWidget(QLabel("Location column"), r, 0)
        g.addWidget(self.loc_combo, r, 1)
        g.addWidget(QLabel("Date column"), r, 2)
        g.addWidget(self.date_combo, r, 3)
        r += 1

        self.time_unit = QComboBox()
        self.time_unit.addItems(["Day", "Week"])
        self.time_unit.setEnabled(False)

        g.addWidget(QLabel("Crime-category column"), r, 0)
        g.addWidget(self.cat_combo, r, 1)
        g.addWidget(QLabel("Aggregate by"), r, 2)
        g.addWidget(self.time_unit, r, 3)
        r += 1

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setEnabled(False)
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setEnabled(False)

        g.addWidget(QLabel("Start date"), r, 0)
        g.addWidget(self.start_date, r, 1)
        g.addWidget(QLabel("End date"), r, 2)
        g.addWidget(self.end_date, r, 3)
        r += 1

        g.addWidget(QLabel("Select crime categories:"), r, 0, 1, 4)
        r += 1
        self.cat_list = QListWidget()
        self.cat_list.setSelectionMode(QListWidget.MultiSelection)
        self.cat_list.setMinimumHeight(130)
        g.addWidget(self.cat_list, r, 0, 1, 4)
        g.setRowStretch(r, 1)
        r += 1

        self.process_btn = QPushButton("Process Crime Data")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._process)
        g.addWidget(self.process_btn, r, 0, 1, 4)
        r += 1

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Status messages will appear here...")
        g.addWidget(self.log_area, r, 0, 1, 4)
        g.setRowStretch(r, 1)
        r += 1

        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        outer.addLayout(g, 1)

    # -- callbacks ---------------------------------------------------
    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Crime File", "",
            "Data (*.csv *.xlsx *.xls)")
        if not path:
            return
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to load:\n{e}")
            return
        if df.empty:
            QMessageBox.warning(self, "Empty", "File is empty.")
            return

        self.ctx["crime_df"] = df
        cols = list(df.columns)
        for cb in (self.loc_combo, self.date_combo, self.cat_combo):
            cb.clear()
            cb.addItems(cols)
            cb.setEnabled(True)
        for w in (self.start_date, self.end_date,
                  self.time_unit, self.process_btn):
            w.setEnabled(True)
        self.cat_list.setEnabled(True)

        try:
            self.cat_combo.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.cat_combo.currentIndexChanged.connect(self._refresh_cats)

        try:
            self.date_combo.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.date_combo.currentIndexChanged.connect(
            self._refresh_dates)

        self._refresh_cats()
        self._refresh_dates()
        fname = os.path.basename(path)
        #n_locs_raw = df[self.loc_combo.currentText()].nunique()
        self.log_area.append(
            f"<b>File uploaded:</b> {fname}  "
            f"({len(df):,} rows, {len(cols)} columns)")
        self.log_area.append(
            "Please select column mappings, date range, "
            "and crime categories above, then click "
            "'Process Crime Data'.")

    def _refresh_cats(self):
        df = self.ctx.get("crime_df")
        if df is None:
            return
        col = self.cat_combo.currentText()
        if not col or col not in df.columns:
            return
        vals = sorted(df[col].dropna().astype(str).unique())
        self.cat_list.clear()
        for v in vals:
            it = QListWidgetItem(v)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.cat_list.addItem(it)

    def _refresh_dates(self):
        df = self.ctx.get("crime_df")
        if df is None:
            return
        col = self.date_combo.currentText()
        if not col:
            return
        d = pd.to_datetime(
            df[col], errors="coerce", format="mixed").dropna()
        if d.empty:
            return
        dmin, dmax = d.min().date(), d.max().date()
        self.start_date.setDate(
            QDate(dmin.year, dmin.month, dmin.day))
        self.end_date.setDate(
            QDate(dmax.year, dmax.month, dmax.day))

    def _process(self):
        df = self.ctx.get("crime_df")
        if df is None:
            return
        selected = [
            self.cat_list.item(i).text()
            for i in range(self.cat_list.count())
            if self.cat_list.item(i).checkState() == Qt.Checked]
        if not selected:
            QMessageBox.warning(
                self, "None selected",
                "Tick at least one crime category.")
            return

        loc_col = self.loc_combo.currentText()
        date_col = self.date_combo.currentText()
        cat_col = self.cat_combo.currentText()
        tu = self.time_unit.currentText()

        d_from = pd.to_datetime(self.start_date.date().toPyDate())
        d_to = pd.to_datetime(self.end_date.date().toPyDate())
        if d_from > d_to:
            QMessageBox.warning(self, "Bad range",
                                "Start after end.")
            return

        sub = df[[loc_col, date_col, cat_col]].copy()
        sub[date_col] = pd.to_datetime(
            sub[date_col], errors="coerce", format="mixed")
        sub = sub.dropna(subset=[date_col])

        group = [loc_col, date_col]
        overall = (sub.groupby(group).size()
                   .reset_index(name="overall_crime_count"))
        sel = sub[sub[cat_col].isin(selected)]
        sel_cnt = (sel.groupby(group).size()
                   .reset_index(name="selected_category_crime_count"))
        result = overall.merge(
            sel_cnt, on=group, how="left").fillna(0)

        dr = pd.date_range(d_from, d_to, freq="D")
        mi = pd.MultiIndex.from_product(
            [result[loc_col].unique(), dr], names=group)
        exp = (result.set_index(group).reindex(mi)
               .fillna({"overall_crime_count": 0,
                         "selected_category_crime_count": 0})
               .reset_index())

        exp["year"] = exp[date_col].dt.year
        if tu == "Day":
            exp["month"] = exp[date_col].dt.month
            exp["day"] = exp[date_col].dt.day
            exp.drop(columns=date_col, inplace=True)
            time_cols = ["year", "month", "day"]
        else:
            exp["week"] = (exp[date_col].dt.isocalendar()
                           .week.astype(int))
            exp = (exp.groupby([loc_col, "year", "week"])
                   .agg({"overall_crime_count": "sum",
                          "selected_category_crime_count": "sum"})
                   .reset_index())
            time_cols = ["year", "week"]

        self.ctx["processed_df"] = exp
        self.ctx["loc_col"] = loc_col
        self.ctx["time_cols"] = time_cols
        n_locs = exp[loc_col].nunique()
        n_cats = len(selected)
        d_from_str = d_from.strftime("%Y-%m-%d")
        d_to_str = d_to.strftime("%Y-%m-%d")
        self.log_area.append(
            f"<b>Processing complete:</b>")
        self.log_area.append(
            f"  {n_cats} crime categories selected, "
            f"date range {d_from_str} to {d_to_str}")
        self.log_area.append(
            f"  Aggregated by {tu} -> "
            f"<b>{len(exp):,} rows across {n_locs} locations</b>")
        self.log_area.append(
            f"  Target columns available: "
            f"overall_crime_count, selected_category_crime_count")
        self.log_area.append(
            "You can now proceed to Step 2 (Next).")


# ====================================================================
# PAGE 2: Geography + External features
# ====================================================================
class Page2GeoExt(QWidget):
    def __init__(self, ctx: dict):
        super().__init__()
        self.ctx = ctx
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Step 2  -  Geography & External Features")
        title.setProperty("class", "title")
        outer.addWidget(title)

        sub = QLabel(
            "Load a shapefile for spatial neighbors and an "
            "external-features CSV. Optionally run feature-importance "
            "analysis to decide which features to keep.")
        sub.setProperty("class", "subtitle")
        sub.setWordWrap(True)
        outer.addWidget(sub)
        outer.addSpacing(6)

        g = QGridLayout()
        r = 0

        # -- shapefile -----------------------------------------------
        self.load_shape_btn = QPushButton(
            "  Load Shapefile / GeoJSON / GPKG  ")
        self.load_shape_btn.clicked.connect(self._load_shape)
        g.addWidget(self.load_shape_btn, r, 0, 1, 4)
        r += 1

        g.addWidget(QLabel("Shape region ID"), r, 0)
        self.shape_id = QComboBox()
        self.shape_id.setEnabled(False)
        g.addWidget(self.shape_id, r, 1)
        self.shape_status = QLabel("")
        self.shape_status.setStyleSheet("color:#7f8c8d")
        g.addWidget(self.shape_status, r, 2, 1, 2)
        r += 1

        # -- external csv --------------------------------------------
        self.load_ext_btn = QPushButton(
            "  Load External-Features CSV  ")
        self.load_ext_btn.clicked.connect(self._load_ext)
        g.addWidget(self.load_ext_btn, r, 0, 1, 4)
        r += 1

        g.addWidget(QLabel("External region ID"), r, 0)
        self.ext_id = QComboBox()
        self.ext_id.setEnabled(False)
        g.addWidget(self.ext_id, r, 1)
        self.ext_status = QLabel("")
        self.ext_status.setStyleSheet("color:#7f8c8d")
        g.addWidget(self.ext_status, r, 2, 1, 2)
        r += 1

        # -- analysis row --------------------------------------------
        g.addWidget(QLabel("Target variable"), r, 0)
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        g.addWidget(self.target_combo, r, 1)

        rw = QWidget()
        h = QHBoxLayout(rw)
        h.setContentsMargins(0, 0, 0, 0)
        self.analyze_btn = QPushButton("Analyze Feature Importance")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._analyze)
        h.addWidget(self.analyze_btn)
        g.addWidget(rw, r, 2, 1, 2)
        r += 1

        g.addWidget(QLabel(
            "Select external features "
            "(for cosine neighbors & static input):"), r, 0, 1, 4)
        r += 1
        self.feat_list = QListWidget()
        self.feat_list.setSelectionMode(QListWidget.MultiSelection)
        self.feat_list.setMinimumHeight(140)
        g.addWidget(self.feat_list, r, 0, 1, 4)
        g.setRowStretch(r, 1)
        r += 1

        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        outer.addLayout(g, 1)

    # -- shapefile ---------------------------------------------------
    def _load_shape(self):
        if gpd is None:
            QMessageBox.warning(
                self, "Missing", "geopandas not installed.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Shapefile", "",
            "Vector (*.shp *.geojson *.json *.gpkg)")
        if not path:
            return
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        if gdf.empty or "geometry" not in gdf.columns:
            QMessageBox.warning(self, "Invalid",
                                "No geometry found.")
            return

        self.ctx["shape_gdf_raw"] = gdf
        non_geom = [c for c in gdf.columns if c != "geometry"]
        self.shape_id.clear()
        self.shape_id.addItems(non_geom)
        self.shape_id.setEnabled(True)

        loc = self.ctx.get("loc_col", "")
        if loc in non_geom:
            self.shape_id.setCurrentText(loc)

        try:
            self.shape_id.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.shape_id.currentIndexChanged.connect(
            self._aggregate_shape)
        self._aggregate_shape()
        self.shape_status.setText(
            f"{os.path.basename(path)} ({len(gdf)} rows)")

    def _aggregate_shape(self):
        gdf = self.ctx.get("shape_gdf_raw")
        if gdf is None:
            return
        key = self.shape_id.currentText()
        if not key or key not in gdf.columns:
            return
        try:
            agg = (gdf[[key, "geometry"]]
                   .dropna(subset=["geometry"])
                   .dissolve(by=key, as_index=False))
            self.ctx["shape_gdf"] = agg
            self.ctx["shape_key"] = key
            self.shape_status.setText(
                f"{len(agg)} regions (by '{key}')")
        except Exception:
            self.ctx["shape_gdf"] = gdf
            self.ctx["shape_key"] = key

    # -- external CSV ------------------------------------------------
    def _load_ext(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "External CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        if df.empty:
            QMessageBox.warning(self, "Empty", "CSV is empty.")
            return

        self.ctx["ext_df"] = df
        cols = list(df.columns)
        self.ext_id.clear()
        self.ext_id.addItems(cols)
        self.ext_id.setEnabled(True)

        loc = self.ctx.get("loc_col", "")
        if loc in cols:
            self.ext_id.setCurrentText(loc)

        self.ext_status.setText(
            f"{os.path.basename(path)} "
            f"({len(df)} rows, {len(cols)} cols)")

        # populate feature list (no analysis yet)
        ext_key = self.ext_id.currentText()
        num = [(c, np.nan) for c in cols
               if c != ext_key
               and pd.api.types.is_numeric_dtype(df[c])]
        self._populate_features(num)

        # refresh target combo
        proc = self.ctx.get("processed_df")
        if proc is not None:
            ncols = [c for c in proc.columns
                     if pd.api.types.is_numeric_dtype(proc[c])]
            self.target_combo.clear()
            self.target_combo.addItems(ncols)
            self.target_combo.setEnabled(True)
            self.analyze_btn.setEnabled(True)

    def _populate_features(self, items: List[Tuple[str, float]]):
        self.feat_list.clear()
        for name, imp in items:
            if pd.notna(imp):
                lbl = f"{name}  (importance={imp:.3f})"
            else:
                lbl = name
            it = QListWidgetItem(lbl)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            it.setData(Qt.UserRole, name)
            self.feat_list.addItem(it)

    # -- analysis ----------------------------------------------------
    def _analyze(self):
        proc = self.ctx.get("processed_df")
        ext = self.ctx.get("ext_df")
        if proc is None or ext is None:
            QMessageBox.warning(
                self, "Missing",
                "Complete Step 1 and load External CSV first.")
            return

        loc_col = self.ctx.get("loc_col", "")
        ext_key = self.ext_id.currentText()
        target_col = self.target_combo.currentText()
        if not all([loc_col, ext_key, target_col]):
            return

        tc = self.ctx.get("time_cols", [])
        if tc:
            latest = (proc.sort_values(tc)
                      .dropna(subset=[target_col])
                      .groupby(loc_col, as_index=False).tail(1)
                      [[loc_col, target_col]])
        else:
            latest = (proc.drop_duplicates(
                subset=[loc_col], keep="last")
                [[loc_col, target_col]])
        latest[loc_col] = latest[loc_col].astype(str)

        edf = ext.copy()
        edf[ext_key] = edf[ext_key].astype(str)
        num_cols = [c for c in edf.columns
                    if c != ext_key
                    and pd.api.types.is_numeric_dtype(edf[c])]
        if not num_cols:
            QMessageBox.warning(
                self, "None",
                "No numeric features in external CSV.")
            return

        merged = latest.merge(
            edf[[ext_key] + num_cols],
            left_on=loc_col, right_on=ext_key, how="inner")
        if merged.empty:
            QMessageBox.warning(
                self, "No overlap",
                "No matching region IDs.")
            return
        y = merged[target_col].astype(float)

        imps: List[Tuple[str, float]] = []
        imps = [(c, abs(safe_spearman(merged[c], y)))
                for c in num_cols]

        imps.sort(key=lambda x: (-x[1] if pd.notna(x[1])
                                  else 1e9))
        self._populate_features(imps)

    def get_selected_features(self) -> List[str]:
        return [
            self.feat_list.item(i).data(Qt.UserRole)
            for i in range(self.feat_list.count())
            if self.feat_list.item(i).checkState() == Qt.Checked]


# ====================================================================
# PAGE 3: Neighbor config + feature engineering settings
# ====================================================================
class Page3Config(QWidget):
    def __init__(self, ctx: dict):
        super().__init__()
        self.ctx = ctx
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        title = QLabel(
            "Step 3  -  Neighbors & Feature Engineering")
        title.setProperty("class", "title")
        outer.addWidget(title)

        sub = QLabel(
            "Configure spatial & contextual-similarity neighbors "
            "and the temporal lag window for feature construction.")
        sub.setProperty("class", "subtitle")
        sub.setWordWrap(True)
        outer.addWidget(sub)
        outer.addSpacing(8)

        g = QGridLayout()
        r = 0

        # -- spatial neighbors ---------------------------------------
        gb1 = QGroupBox("Spatial Neighbors (from shapefile)")
        g1 = QGridLayout(gb1)
        g1.addWidget(QLabel("Max spatial neighbors (k):"), 0, 0)
        self.k_spatial = QSpinBox()
        self.k_spatial.setRange(1, 30)
        self.k_spatial.setValue(8)
        self.k_spatial.setToolTip(
            "Cap at k nearest centroids if too many "
            "contiguous neighbors.")
        g1.addWidget(self.k_spatial, 0, 1)
        g1.addWidget(QLabel(
            "Contiguity: queen (shared edge or vertex). "
            "Capped at k nearest centroids if too many."), 1, 0, 1, 2)
        g.addWidget(gb1, r, 0, 1, 4)
        r += 1

        # -- cosine neighbors ----------------------------------------
        gb2 = QGroupBox(
            "Contextual-Similarity Neighbors "
            "(cosine on external features)")
        g2 = QGridLayout(gb2)
        g2.addWidget(QLabel("Top-k cosine neighbors:"), 0, 0)
        self.k_cosine = QSpinBox()
        self.k_cosine.setRange(1, 30)
        self.k_cosine.setValue(5)
        g2.addWidget(self.k_cosine, 0, 1)
        g2.addWidget(QLabel(
            "Uses the external features you selected in Step 2, "
            "standardized (z-score) so each counts equally."), 1, 0, 1, 2)
        g.addWidget(gb2, r, 0, 1, 4)
        r += 1

        # -- temporal lags -------------------------------------------
        gb3 = QGroupBox("Temporal Feature Engineering")
        g3 = QGridLayout(gb3)
        g3.addWidget(QLabel("Sequence length (lag window):"), 0, 0)
        self.seq_len = QSpinBox()
        self.seq_len.setRange(1, 128)
        self.seq_len.setValue(4)
        self.seq_len.setToolTip(
            "Number of past time steps used as lag features "
            "(and rolling-window size).")
        g3.addWidget(self.seq_len, 0, 1)

        g3.addWidget(QLabel("Prediction horizon:"), 1, 0)
        self.pred_len = QSpinBox()
        self.pred_len.setRange(1, 64)
        self.pred_len.setValue(1)
        self.pred_len.setToolTip(
            "How many steps ahead to predict.")
        g3.addWidget(self.pred_len, 1, 1)

        g3.addWidget(QLabel(
            "Features auto-generated per location:"), 2, 0, 1, 2)
        desc = QLabel(
            "  crime_1_period_ago ... crime_k_periods_ago\n"
            "  avg_crime_last_k_periods, max_crime, "
            "variability, trend\n"
            "  nearby_area_avg_crime / max / spread  "
            "(spatial neighbors)\n"
            "  similar_area_avg_crime / max / spread  "
            "(cosine neighbors)\n"
            "  all selected static external features")
        desc.setStyleSheet(
            "color:#7f8c8d; font-size:11px; padding-left:12px")
        g3.addWidget(desc, 3, 0, 1, 2)
        g.addWidget(gb3, r, 0, 1, 4)
        r += 1

        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        outer.addLayout(g)
        outer.addStretch(1)


# ====================================================================
# PAGE 4: Model training & results
# ====================================================================
class Page4Train(QWidget):
    def __init__(self, ctx: dict):
        super().__init__()
        self.ctx = ctx
        self.worker: Optional[PipelineWorker] = None
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Step 4  -  Model Training & Results")
        title.setProperty("class", "title")
        outer.addWidget(title)

        sub = QLabel(
            "Choose one or more lightweight models, then hit "
            "Train. All computation runs in a background thread "
            "so the UI stays responsive.")
        sub.setProperty("class", "subtitle")
        sub.setWordWrap(True)
        outer.addWidget(sub)
        outer.addSpacing(6)

        # -- model checkboxes ----------------------------------------
        gb = QGroupBox("Select Models")
        mg = QHBoxLayout(gb)
        self.model_checks: Dict[str, QCheckBox] = {}
        for name, tip in [
            ("Ridge",
             "L2-regularized linear - fastest, interpretable"),
            ("ElasticNet",
             "L1+L2 - auto feature selection, linear"),
            ("RandomForest",
             "Ensemble of trees - nonlinear, robust"),
            ("GradientBoosting",
             "Boosted trees - often highest accuracy"),
        ]:
            cb = QCheckBox(name)
            cb.setToolTip(tip)
            cb.setChecked(name in (
                "Ridge", "ElasticNet", "RandomForest"))
            self.model_checks[name] = cb
            mg.addWidget(cb)
        outer.addWidget(gb)

        # -- target column -----------------------------------------------
        h = QHBoxLayout()
        h.addWidget(QLabel("Target column:"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(220)
        h.addWidget(self.target_combo)
        h.addStretch(1)
        outer.addLayout(h)

        # -- train / stop buttons + progress --------------------------
        th = QHBoxLayout()
        self.train_btn = QPushButton("  Train Models  ")
        self.train_btn.setProperty("class", "accent")
        self.train_btn.clicked.connect(self._start_training)
        th.addWidget(self.train_btn)

        self.stop_btn = QPushButton("  Stop  ")
        self.stop_btn.setStyleSheet(
            "background:#c0392b; color:white; font-weight:600; "
            "padding:8px 20px; border-radius:4px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        th.addWidget(self.stop_btn)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        th.addWidget(self.progress, 1)
        outer.addLayout(th)

        # -- live log ------------------------------------------------
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        outer.addWidget(self.log)

        # -- results tabs --------------------------------------------
        self.tabs = QTabWidget()

        self.summary_table = QTableWidget()
        self.summary_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.summary_table, "Summary")

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.tabs.addTab(self.detail_text, "Feature Importances")

        self.risk_table = QTableWidget()
        self.risk_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.risk_table, "Region Risk Ranking")

        outer.addWidget(self.tabs, 1)

        # -- export button -------------------------------------------
        eh = QHBoxLayout()
        eh.addStretch(1)
        self.export_btn = QPushButton("  Export Risk Ranking (CSV)  ")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_risk)
        eh.addWidget(self.export_btn)
        outer.addLayout(eh)

        self._risk_data: List[Dict[str, Any]] = []

    def showEvent(self, event):
        super().showEvent(event)
        proc = self.ctx.get("processed_df")
        if proc is not None:
            num = [c for c in proc.columns
                   if pd.api.types.is_numeric_dtype(proc[c])]
            prev = self.target_combo.currentText()
            self.target_combo.clear()
            self.target_combo.addItems(num)
            if prev in num:
                self.target_combo.setCurrentText(prev)

    def _start_training(self):
        if not SKLEARN_OK:
            QMessageBox.critical(
                self, "Missing",
                "scikit-learn is required.\n"
                "pip install scikit-learn")
            return

        # If a worker is still running, warn
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self, "Busy",
                "Training is still running. "
                "Click Stop first, then re-run.")
            return

        proc = self.ctx.get("processed_df")
        if proc is None or proc.empty:
            QMessageBox.warning(
                self, "Missing",
                "Complete Step 1 first.")
            return

        selected_models = [
            n for n, cb in self.model_checks.items()
            if cb.isChecked()]
        if not selected_models:
            QMessageBox.warning(
                self, "None", "Select at least one model.")
            return

        target = self.target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self, "Target", "Choose a target column.")
            return

        p3: Page3Config = self.ctx.get("page3_ref")
        p2: Page2GeoExt = self.ctx.get("page2_ref")
        chosen_feats = p2.get_selected_features() if p2 else []

        params = {
            "crime_panel": proc,
            "loc_col": self.ctx["loc_col"],
            "target_col": target,
            "time_cols": self.ctx["time_cols"],
            "shape_gdf": self.ctx.get("shape_gdf"),
            "shape_key": self.ctx.get("shape_key"),
            "ext_df": self.ctx.get("ext_df"),
            "ext_key": (p2.ext_id.currentText()
                        if p2 else None),
            "chosen_features": chosen_feats,
            "seq_len": p3.seq_len.value() if p3 else 4,
            "pred_len": p3.pred_len.value() if p3 else 1,
            "k_spatial": p3.k_spatial.value() if p3 else 8,
            "k_cosine": p3.k_cosine.value() if p3 else 5,
            "models": selected_models,
        }

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.log.clear()
        self.log.append(
            f"Starting pipeline with models: "
            f"{', '.join(selected_models)} ...")

        self.worker = PipelineWorker(params)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.error.connect(self._on_error)
        self.worker.results_ready.connect(self._on_done)
        self.worker.start()

    def _stop_training(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.log.append(
                "<span style='color:#c0392b'>"
                "<b>Stopping...</b> waiting for current "
                "operation to finish.</span>")
            self.stop_btn.setEnabled(False)
            # Wait briefly, then force-enable rerun
            QTimer.singleShot(500, self._force_reset_buttons)

    def _force_reset_buttons(self):
        """Re-enable Train after a short delay so the user can rerun."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.worker and not self.worker.isRunning():
            self.log.append(
                "<b>Stopped.</b> You can change model "
                "selections and click Train again.")

    @pyqtSlot(int)
    def _on_progress(self, val):
        self.progress.setValue(val)

    @pyqtSlot(str)
    def _on_log(self, msg):
        self.log.append(msg)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(str)
    def _on_error(self, msg):
        self.log.append(
            f"<span style='color:#c0392b'>{msg}</span>")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @pyqtSlot(dict)
    def _on_done(self, results: dict):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        mr = results.get("model_results", {})
        if not mr:
            self.progress.setValue(0)
            self.log.append(
                "<b>No models completed.</b> "
                "Adjust your selections and try again.")
            return

        n_done = len(mr)
        self.progress.setValue(100)
        self.log.append(
            f"<b>{n_done} model(s) trained successfully.</b>"
            f" You can change selections and re-run anytime.")

        # -- summary table -------------------------------------------
        headers = [
            "Model", "Val-MAE", "Val-R2",
            "Test MAE", "Test RMSE", "Test R2",
            "Train MAE", "Train R2", "Time (s)"]
        self.summary_table.setRowCount(len(mr))
        self.summary_table.setColumnCount(len(headers))
        self.summary_table.setHorizontalHeaderLabels(headers)

        finite_test = [v["test_mae"] for v in mr.values()
                       if np.isfinite(v["test_mae"])]
        best_test = min(finite_test) if finite_test else None

        for row, (mname, v) in enumerate(mr.items()):
            vals = [
                mname,
                f"{v['cv_mae_mean']:.3f} +/- {v['cv_mae_std']:.3f}",
                f"{v['cv_r2_mean']:.3f} +/- {v['cv_r2_std']:.3f}",
                f"{v['test_mae']:.3f}",
                f"{v['test_rmse']:.3f}",
                f"{v['test_r2']:.3f}",
                f"{v['train_mae']:.3f}",
                f"{v['train_r2']:.3f}",
                f"{v['time_sec']:.1f}",
            ]
            for col, txt in enumerate(vals):
                item = QTableWidgetItem(str(txt))
                item.setFlags(
                    item.flags() & ~Qt.ItemIsEditable)
                # highlight best test MAE
                if (col == 3 and best_test is not None
                        and v["test_mae"] == best_test):
                    item.setBackground(QColor("#d5f5e3"))
                    item.setToolTip("Best Test-MAE")
                # warn if train >> test (overfitting)
                if (col == 6
                        and v["train_mae"] < v["test_mae"] * 0.5
                        and v["test_mae"] > 0.01):
                    item.setBackground(QColor("#fadbd8"))
                    item.setToolTip(
                        "Possible overfitting: train MAE much "
                        "lower than test MAE")
                self.summary_table.setItem(row, col, item)

        self.summary_table.resizeColumnsToContents()

        # -- feature importance detail -------------------------------
        lines = [
            f"<b>Pipeline info:</b>  "
            f"{results['n_locations']} locations, "
            f"{results['n_train']} train / {results['n_test']} test samples, "
            f"{results['ts_length']} time steps, "
            f"{len(results['feature_cols'])} features"
            f"<br><hr>"]

        for mname, v in mr.items():
            lines.append(
                f"<br><b>{mname}</b>  "
                f"(Test-MAE = {v['test_mae']:.3f}, "
                f"Test-R2 = {v['test_r2']:.3f})")
            lines.append("<table cellpadding='2'>")
            lines.append(
                "<tr><th align='left'>Feature</th>"
                "<th align='right'>Importance</th></tr>")
            top_val = (v["top_features"][0][1]
                       if v["top_features"] else 1e-9)
            if top_val == 0:
                top_val = 1e-9
            for fname, fval in v["top_features"]:
                bar_w = max(1, int(fval / top_val * 150))
                lines.append(
                    f"<tr><td>{fname}</td>"
                    f"<td align='right'>{fval:.4f}</td>"
                    f"<td><div style='background:#3498db;"
                    f"width:{bar_w}px;height:10px;"
                    f"border-radius:2px'></div></td></tr>")
            lines.append("</table>")

        self.detail_text.setHtml("\n".join(lines))

        # -- region risk ranking -------------------------------------
        risk = results.get("risk_ranking", [])
        best_m = results.get("best_model", "")
        self._risk_data = risk

        if risk:
            risk_headers = [
                "Rank", "Region", "Predicted Crime"]
            self.risk_table.setRowCount(len(risk))
            self.risk_table.setColumnCount(len(risk_headers))
            self.risk_table.setHorizontalHeaderLabels(
                risk_headers)

            max_pred = max(r["predicted_crime"] for r in risk) or 1.0

            for row, entry in enumerate(risk):
                rank_item = QTableWidgetItem(str(row + 1))
                rank_item.setFlags(
                    rank_item.flags() & ~Qt.ItemIsEditable)
                rank_item.setTextAlignment(
                    Qt.AlignCenter)
                self.risk_table.setItem(row, 0, rank_item)

                reg_item = QTableWidgetItem(entry["region"])
                reg_item.setFlags(
                    reg_item.flags() & ~Qt.ItemIsEditable)
                self.risk_table.setItem(row, 1, reg_item)

                pred_item = QTableWidgetItem(
                    f"{entry['predicted_crime']:.2f}")
                pred_item.setFlags(
                    pred_item.flags() & ~Qt.ItemIsEditable)
                pred_item.setTextAlignment(
                    Qt.AlignRight | Qt.AlignVCenter)
                # Color gradient: higher prediction = more red
                ratio = entry["predicted_crime"] / max_pred
                if ratio > 0.75:
                    pred_item.setBackground(QColor("#f1948a"))
                    pred_item.setToolTip("High risk")
                elif ratio > 0.5:
                    pred_item.setBackground(QColor("#f9e79f"))
                    pred_item.setToolTip("Moderate risk")
                else:
                    pred_item.setBackground(QColor("#abebc6"))
                    pred_item.setToolTip("Lower risk")
                self.risk_table.setItem(row, 2, pred_item)

            self.risk_table.resizeColumnsToContents()
            hdr = self.risk_table.horizontalHeader()
            hdr.setStretchLastSection(False)
            hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.export_btn.setEnabled(True)
            self.log.append(
                f"<b>Risk ranking:</b> {len(risk)} regions "
                f"ranked by predicted crime (model: {best_m}, "
                f"latest period — no future label needed).")
        else:
            self.risk_table.setRowCount(0)
            self.export_btn.setEnabled(False)

        self.tabs.setCurrentIndex(0)

    def _export_risk(self):
        """Export risk ranking to CSV."""
        if not self._risk_data:
            QMessageBox.warning(
                self, "No data",
                "No risk ranking available. "
                "Train models first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Risk Ranking", "risk_ranking.csv",
            "CSV (*.csv)")
        if not path:
            return

        try:
            df = pd.DataFrame(self._risk_data)
            df.index = range(1, len(df) + 1)
            df.index.name = "rank"
            df.to_csv(path)
            QMessageBox.information(
                self, "Exported",
                f"Risk ranking saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to save:\n{e}")


# ====================================================================
# MAIN WIZARD WINDOW
# ====================================================================
class CrimePredictionWizard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crime Prediction")
        self.resize(1080, 830)

        self.ctx: Dict[str, Any] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)

        # -- step indicator ------------------------------------------
        self.step_bar = QLabel()
        self.step_bar.setAlignment(Qt.AlignCenter)
        self.step_bar.setStyleSheet(
            "font-size:12px; color:#7f8c8d; padding:4px 0")
        root.addWidget(self.step_bar)

        # -- stacked pages -------------------------------------------
        self.stack = QStackedWidget()
        self.page1 = Page1Crime(self.ctx)
        self.page2 = Page2GeoExt(self.ctx)
        self.page3 = Page3Config(self.ctx)
        self.page4 = Page4Train(self.ctx)

        self.ctx["page2_ref"] = self.page2
        self.ctx["page3_ref"] = self.page3

        for pg in (self.page1, self.page2,
                   self.page3, self.page4):
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setWidget(pg)
            self.stack.addWidget(sa)

        root.addWidget(self.stack, 1)

        # -- navigation bar ------------------------------------------
        nav = QHBoxLayout()
        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self._show_help)
        nav.addWidget(self.help_btn)
        nav.addStretch(1)

        self.back_btn = QPushButton("  Back  ")
        self.back_btn.setProperty("class", "nav")
        self.back_btn.clicked.connect(self._go_back)
        nav.addWidget(self.back_btn)

        self.next_btn = QPushButton("  Next  ")
        self.next_btn.setProperty("class", "nav")
        self.next_btn.clicked.connect(self._go_next)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)

        self._update_nav()

    def _go_back(self):
        idx = self.stack.currentIndex()
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
            self._update_nav()

    def _go_next(self):
        idx = self.stack.currentIndex()
        if idx == 0 and self.ctx.get("processed_df") is None:
            QMessageBox.warning(
                self, "Step 1 incomplete",
                "Please load and process the crime data "
                "before continuing.")
            return
        if idx < self.stack.count() - 1:
            self.stack.setCurrentIndex(idx + 1)
            self._update_nav()

    def _update_nav(self):
        idx = self.stack.currentIndex()
        n = self.stack.count()
        self.back_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < n - 1)
        names = ["Crime Data", "Geography & Features",
                 "Neighbors & Lags", "Train & Results"]
        parts = []
        for i, nm in enumerate(names):
            if i == idx:
                parts.append(
                    f"<b style='color:#2c3e50'>[ {nm} ]</b>")
            elif i < idx:
                parts.append(
                    f"<span style='color:#27ae60'>"
                    f"  {nm}  </span>")
            else:
                parts.append(
                    f"<span style='color:#bdc3c7'>"
                    f"  {nm}  </span>")
        self.step_bar.setText("  >  ".join(parts))

    def _show_help(self):
        QMessageBox.information(
            self, "Help",
            "WORKFLOW\n\n"
            "Step 1: Load raw crime CSV/XLSX. Map location, "
            "date, and category columns. Filter by date range "
            "and crime type. Click 'Process Crime Data'.\n\n"
            "Step 2: Load shapefile for spatial "
            "neighbors. Load external-features CSV. Run "
            "feature-importance analysis (Spearman rank "
            "correlation), tick features.\n\n"
            "Step 3: Adjust neighbor and lag-window params.\n\n"
            "Step 4: Pick models, hit Train. Training runs "
            "in a background thread. Results appear in the "
            "Summary table, Feature-Importances tab, and "
            "Region Risk Ranking tab. You can export the risk "
            "ranking to CSV.\n\n"
            "MODELS\n"
            "  Ridge       - fastest, linear, interpretable\n"
            "  ElasticNet  - linear + feature selection\n"
            "  RandomForest - nonlinear, robust\n"
            "  GradientBoosting - highest accuracy\n\n"
            "All models run on CPU. No GPU needed.")


# ====================================================================
# ENTRY POINT
# ====================================================================
def main():
    app = QApplication(sys.argv)
    
    w = CrimePredictionWizard()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()