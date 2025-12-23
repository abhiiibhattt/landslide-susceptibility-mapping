# ============================================================
# Landslide Susceptibility Project
# FINAL GIS PIPELINE
# (Top-3 Models + Stacked)
# Probability | Difference | Uncertainty | Quantile | Jenks
# ============================================================

import matplotlib
matplotlib.use("Agg")

import gc
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path("E:/landslide_project")

DATA_DIR   = BASE_DIR / "data" / "raw"
MAPS_DIR   = BASE_DIR / "maps"
RASTER_DIR = MAPS_DIR / "rasters"

PROB_DIR   = MAPS_DIR / "probability"
DIFF_DIR   = MAPS_DIR / "difference"
UNC_DIR    = MAPS_DIR / "uncertainty"
Q_DIR      = MAPS_DIR / "quantile"
J_DIR      = MAPS_DIR / "jenks"

for d in [PROB_DIR, DIFF_DIR, UNC_DIR, Q_DIR, J_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# FEATURE → RASTER MAPPING
# (MUST MATCH CSV COLUMN NAMES)
# ============================================================

feature_to_raster = {
    "ASPECT": "aspect.tif",
    "dem": "dem.tif",
    "DIS_TO_fau": "DIS_TO_faults.tif",
    "dis_to_riv": "dis_to_rivers.tif",
    "dis_to_roa": "dis_to_road.tif",
    "GEOLOGY": "geology.tif",
    "LANDCOVER": "landcover.tif",
    "ndvi": "ndvi.tif",
    "plan_curv": "plan_curv.tif",
    "profile_cu": "profile_curv.tif",
    "slope": "slope.tif",
    "TWI": "twi.tif"
}

FEATURE_COLS = list(feature_to_raster.keys())

# Check rasters
missing = [v for v in feature_to_raster.values() if not (RASTER_DIR / v).exists()]
if missing:
    raise FileNotFoundError(f"Missing raster files: {missing}")

# ============================================================
# LOAD MODEL RANKING
# ============================================================

rank_df = pd.read_csv(
    BASE_DIR / "results" / "summaries" / "spatial_cv_model_ranking.csv",
    index_col=0
)

TOP_MODELS = rank_df.head(3).index.tolist()
print("Top-3 models:", TOP_MODELS)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(DATA_DIR / "tran_test_26_9.csv")
X = df[FEATURE_COLS]
y = df["Hazard"]

# ============================================================
# MODELS
# ============================================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

base_models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True))
    ]),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    ),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
    "CatBoost": CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function="Logloss", verbose=False, random_state=42
    )
}

# ============================================================
# TRAIN ONLY TOP-3 (CALIBRATED)
# ============================================================

trained = {}
for name in TOP_MODELS:
    print(f"Training calibrated {name}")
    cal = CalibratedClassifierCV(base_models[name], cv=3, method="sigmoid")
    cal.fit(X, y)
    trained[name] = cal

# ============================================================
# STACKED META MODEL
# ============================================================

stack_X = np.column_stack([
    trained[m].predict_proba(X)[:, 1] for m in TOP_MODELS
])

stack_meta = XGBClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42
)
stack_meta.fit(stack_X, y)

# ============================================================
# REFERENCE RASTER
# ============================================================

with rasterio.open(RASTER_DIR / feature_to_raster[FEATURE_COLS[0]]) as ref:
    PROFILE = ref.profile.copy()
    CRS = ref.crs
    TRANSFORM = ref.transform
    HEIGHT, WIDTH = ref.height, ref.width

PROFILE.update(
    dtype="float32",
    count=1,
    nodata=np.nan,
    compress="deflate",
    tiled=True,
    blockxsize=256,
    blockysize=256
)

# ============================================================
# CREATE VRTs
# ============================================================

vrts = []
for f in FEATURE_COLS:
    src = rasterio.open(RASTER_DIR / feature_to_raster[f])
    vrts.append(WarpedVRT(
        src, crs=CRS, transform=TRANSFORM,
        width=WIDTH, height=HEIGHT,
        resampling=Resampling.bilinear
    ))

TILE = 256
BATCH = 150_000

# ============================================================
# PROBABILITY RASTER GENERATOR
# ============================================================

def generate_prob_raster(name, predict_fn):
    out = PROB_DIR / f"{name}_probability.tif"
    print(f"Generating {out.name}")

    with rasterio.open(out, "w", **PROFILE) as dst:
        for y0 in range(0, HEIGHT, TILE):
            for x0 in range(0, WIDTH, TILE):
                win = Window(x0, y0, min(TILE, WIDTH-x0), min(TILE, HEIGHT-y0))

                feats = [
                    np.where(v.read(1, window=win, masked=True).mask,
                             np.nan,
                             v.read(1, window=win))
                    for v in vrts
                ]

                flat = np.column_stack([f.reshape(-1) for f in feats])
                out_arr = np.full(flat.shape[0], np.nan, dtype="float32")

                valid = ~np.isnan(flat).any(axis=1)
                idx = np.where(valid)[0]

                for i in range(0, len(idx), BATCH):
                    sel = idx[i:i+BATCH]
                    out_arr[sel] = predict_fn(flat[sel])

                dst.write(out_arr.reshape(win.height, win.width), 1, window=win)

                del feats, flat, out_arr
                gc.collect()

    return out

# ============================================================
# GENERATE PROBABILITY RASTERS
# ============================================================

prob = {}

for name in TOP_MODELS:
    prob[name] = generate_prob_raster(
        name, lambda Xb, m=name: trained[m].predict_proba(Xb)[:, 1]
    )

prob["Stacked"] = generate_prob_raster(
    "Stacked",
    lambda Xb: stack_meta.predict_proba(
        np.column_stack([trained[m].predict_proba(Xb)[:, 1] for m in TOP_MODELS])
    )[:, 1]
)

# ============================================================
# DIFFERENCE RASTERS (Stacked − Model)
# ============================================================

for name in TOP_MODELS:
    out = DIFF_DIR / f"Stacked_minus_{name}.tif"
    print(f"Difference: {out.name}")

    with rasterio.open(prob["Stacked"]) as s, rasterio.open(prob[name]) as m:
        prof = s.profile.copy()
        with rasterio.open(out, "w", **prof) as dst:
            for y0 in range(0, HEIGHT, TILE):
                for x0 in range(0, WIDTH, TILE):
                    win = Window(x0, y0, min(TILE, WIDTH-x0), min(TILE, HEIGHT-y0))
                    dst.write(s.read(1, window=win) - m.read(1, window=win), 1, window=win)

# ============================================================
# UNCERTAINTY MAP (STD DEV of Top-3)
# ============================================================

uncert = UNC_DIR / "uncertainty_top3_std.tif"
print("Generating uncertainty raster")

with rasterio.open(prob[TOP_MODELS[0]]) as r0:
    prof = r0.profile.copy()
    with rasterio.open(uncert, "w", **prof) as dst:
        for y0 in range(0, HEIGHT, TILE):
            for x0 in range(0, WIDTH, TILE):
                win = Window(x0, y0, min(TILE, WIDTH-x0), min(TILE, HEIGHT-y0))
                arrs = [rasterio.open(prob[m]).read(1, window=win) for m in TOP_MODELS]
                dst.write(np.nanstd(np.stack(arrs), axis=0), 1, window=win)

# ============================================================
# CLASSIFICATION (QUANTILE & JENKS)
# ============================================================

import jenkspy

def classify(src, dst, breaks):
    with rasterio.open(src) as s:
        prof = s.profile.copy()
        prof.update(dtype="uint8", nodata=0)

        with rasterio.open(dst, "w", **prof) as d:
            for y0 in range(0, s.height, TILE):
                for x0 in range(0, s.width, TILE):
                    win = Window(x0, y0, min(TILE, s.width-x0), min(TILE, s.height-y0))
                    a = s.read(1, window=win)
                    cls = np.zeros(a.shape, dtype="uint8")
                    v = ~np.isnan(a)
                    cls[v] = np.digitize(a[v], breaks) + 1
                    d.write(cls, 1, window=win)

for name, tif in prob.items():
    with rasterio.open(tif) as s:
        arr = s.read(1)
        q = np.nanquantile(arr, [0.2, 0.4, 0.6, 0.8])
        j = jenkspy.jenks_breaks(arr[~np.isnan(arr)], nb_class=5)[1:-1]

    classify(tif, Q_DIR / f"{name}_quantile.tif", q)
    classify(tif, J_DIR / f"{name}_jenks.tif", j)

print("\n===== GIS PIPELINE COMPLETED SUCCESSFULLY =====")
