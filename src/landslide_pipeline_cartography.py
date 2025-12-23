# ============================================================
# Landslide Susceptibility – Cartographic GIS Pipeline
# Hillshade + Color Ramps + Boundary Clip + Final Figures
# ============================================================

import matplotlib
matplotlib.use("Agg")

import numpy as np
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.plot import reshape_as_image

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("E:/landslide_project")

MAPS_DIR = BASE_DIR / "maps"
PROB_DIR = MAPS_DIR / "probability"
Q_DIR    = MAPS_DIR / "quantile"
FIG_DIR  = MAPS_DIR / "figures"
BOUNDARY = MAPS_DIR / "boundary" / "study_area_boundary.shp"
DEM_PATH = MAPS_DIR / "rasters" / "dem.tif"

FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# Models to plot
# ============================================================

MODELS = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "SVM",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Stacked"
]

# ============================================================
# Load boundary
# ============================================================

boundary = gpd.read_file(BOUNDARY)

# ============================================================
# Create hillshade from DEM
# ============================================================

def compute_hillshade(dem, azimuth=315, altitude=45):
    x, y = np.gradient(dem)
    slope = np.pi / 2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)

    azimuth = np.deg2rad(azimuth)
    altitude = np.deg2rad(altitude)

    shaded = (
        np.sin(altitude) * np.sin(slope) +
        np.cos(altitude) * np.cos(slope) *
        np.cos(azimuth - aspect)
    )
    return np.clip(shaded, 0, 1)

with rasterio.open(DEM_PATH) as dem_src:
    dem, dem_transform = mask(dem_src, boundary.geometry, crop=True)
    dem = dem[0]

hillshade = compute_hillshade(dem)

# ============================================================
# Professional color ramp (5 classes)
# ============================================================

susceptibility_cmap = ListedColormap([
    "#2c7bb6",  # Very Low
    "#abd9e9",  # Low
    "#ffffbf",  # Moderate
    "#fdae61",  # High
    "#d7191c"   # Very High
])

class_labels = [
    "Very Low", "Low", "Moderate", "High", "Very High"
]

# ============================================================
# Plot function
# ============================================================

def plot_susceptibility(model_name):
    src_path = Q_DIR / f"{model_name}_quantile.tif"

    if not src_path.exists():
        print(f"Skipping {model_name} (missing raster)")
        return

    with rasterio.open(src_path) as src:
        data, transform = mask(src, boundary.geometry, crop=True)
        data = data[0]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Hillshade base
    ax.imshow(
        hillshade,
        cmap="gray",
        extent=[
            transform.c,
            transform.c + transform.a * data.shape[1],
            transform.f + transform.e * data.shape[0],
            transform.f
        ],
        alpha=0.45
    )

    # Susceptibility overlay
    im = ax.imshow(
        data,
        cmap=susceptibility_cmap,
        vmin=1,
        vmax=5,
        alpha=0.75
    )

    # Boundary outline
    boundary.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=1.2
    )

    # Legend
    cbar = plt.colorbar(
        im,
        ax=ax,
        ticks=[1, 2, 3, 4, 5],
        shrink=0.75
    )
    cbar.ax.set_yticklabels(class_labels)
    cbar.set_label("Landslide Susceptibility Class")

    ax.set_title(f"Landslide Susceptibility Map – {model_name}", fontsize=13)
    ax.set_axis_off()

    out_png = FIG_DIR / f"{model_name}_susceptibility_map.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure → {out_png}")

# ============================================================
# Generate figures
# ============================================================

print("\n===== GENERATING FINAL PAPER-READY MAPS =====")

for m in MODELS:
    plot_susceptibility(m)

print("\n✅ ALL FINAL MAPS GENERATED SUCCESSFULLY")

# ============================================================
# Landslide Susceptibility – Advanced Cartography Add-ons
# ============================================================

import numpy as np
from pathlib import Path
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrow
import pandas as pd

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("E:/landslide_project")

MAPS_DIR = BASE_DIR / "maps"
Q_DIR = MAPS_DIR / "quantile"
FIG_DIR = MAPS_DIR / "figures"
STATS_DIR = MAPS_DIR / "stats"
STYLE_DIR = MAPS_DIR / "qgis_styles"
BOUNDARY = MAPS_DIR / "boundary" / "study_area_boundary.shp"

for d in [FIG_DIR, STATS_DIR, STYLE_DIR]:
    d.mkdir(exist_ok=True)

# ============================================================
# Models
# ============================================================

MODELS = [
    "LogisticRegression", "RandomForest", "GradientBoosting", "SVM",
    "XGBoost", "LightGBM", "CatBoost", "Stacked"
]

# ============================================================
# Color ramp
# ============================================================

cmap = ListedColormap([
    "#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"
])

labels = ["Very Low", "Low", "Moderate", "High", "Very High"]

# ============================================================
# Boundary
# ============================================================

boundary = gpd.read_file(BOUNDARY)

# ============================================================
# Helpers
# ============================================================

def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.95, 0.9), xytext=(0.95, 0.75),
        arrowprops=dict(facecolor="black", width=4),
        ha="center", va="center",
        xycoords=ax.transAxes
    )

def add_scale_bar(ax, length_km=5):
    ax.plot([0.1, 0.3], [0.05, 0.05], transform=ax.transAxes, color="black", lw=3)
    ax.text(0.2, 0.01, f"{length_km} km", transform=ax.transAxes, ha="center")

# ============================================================
# 1️⃣ Area-wise statistics (km² per class)
# ============================================================

stats = []

for model in MODELS:
    tif = Q_DIR / f"{model}_quantile.tif"
    if not tif.exists():
        continue

    with rasterio.open(tif) as src:
        data, _ = mask(src, boundary.geometry, crop=True)
        data = data[0]
        pixel_area_km2 = abs(src.res[0] * src.res[1]) / 1e6

    for cls in range(1, 6):
        area = np.sum(data == cls) * pixel_area_km2
        stats.append([model, labels[cls-1], area])

df_stats = pd.DataFrame(stats, columns=["Model", "Class", "Area_km2"])
df_stats.to_csv(STATS_DIR / "susceptibility_area_stats.csv", index=False)

print("✔ Area statistics saved")

# ============================================================
# 2️⃣ Difference map (Stacked − Best Single)
# ============================================================

best_model = "CatBoost"  # from your CV ranking

with rasterio.open(Q_DIR / "Stacked_quantile.tif") as s, \
     rasterio.open(Q_DIR / f"{best_model}_quantile.tif") as b:

    s_data, s_tr = mask(s, boundary.geometry, crop=True)
    b_data, _ = mask(b, boundary.geometry, crop=True)

diff = s_data[0] - b_data[0]

plt.figure(figsize=(7,6))
plt.imshow(diff, cmap="RdBu", vmin=-4, vmax=4)
plt.colorbar(label="Class Difference")
plt.title("Difference Map: Stacked − Best Single")
plt.axis("off")

plt.savefig(FIG_DIR / "difference_stacked_vs_best.png", dpi=300, bbox_inches="tight")
plt.close()

print("✔ Difference map generated")

# ============================================================
# 3️⃣ Model comparison panel (2×4)
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for ax, model in zip(axes, MODELS):
    tif = Q_DIR / f"{model}_quantile.tif"
    if not tif.exists():
        ax.axis("off")
        continue

    with rasterio.open(tif) as src:
        data, tr = mask(src, boundary.geometry, crop=True)
        data = data[0]

    im = ax.imshow(data, cmap=cmap, vmin=1, vmax=5)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_title(model)
    ax.axis("off")

plt.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
plt.savefig(FIG_DIR / "model_comparison_2x4.png", dpi=300, bbox_inches="tight")
plt.close()

print("✔ 2×4 model comparison panel saved")

# ============================================================
# 4️⃣ QGIS style (.qml)
# ============================================================

qml = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis>
  <renderer-v2 type="categorizedSymbol" attr="class">
    <categories>
      <category value="1" label="Very Low" color="#2c7bb6"/>
      <category value="2" label="Low" color="#abd9e9"/>
      <category value="3" label="Moderate" color="#ffffbf"/>
      <category value="4" label="High" color="#fdae61"/>
      <category value="5" label="Very High" color="#d7191c"/>
    </categories>
  </renderer-v2>
</qgis>
"""

(STYLE_DIR / "susceptibility_5class.qml").write_text(qml)
print("✔ QGIS style exported")

print("\n✅ ALL ADVANCED CARTOGRAPHIC PRODUCTS GENERATED")
