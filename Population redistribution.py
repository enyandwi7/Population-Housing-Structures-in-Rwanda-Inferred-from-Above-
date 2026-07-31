# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:01:35 2026

@author: enyan
"""

#DISAGGREGATION OF VILLAGE POPULATION TO BUILDING LEVELS; BIN THEM INTO WORLDPOP AND POPCORN  GRIDS AND COMPARE

import geopandas as gpd
import pandas as pd

# --- Load shapefiles ---
census = gpd.read_file("...\villagepop.shp")
buildings = gpd.read_file("...\WealthMapFinal.pkg", layer="WealthMap")

# --- Ensure projected CRS (so area in m²) ---
if not buildings.crs.is_projected:
    raise ValueError("Please project data to a suitable projected CRS (e.g., UTM).")
buildings = buildings.reset_index(drop=True)
buildings["building_id"] = ["BLD_" + str(i + 1).zfill(6) for i in range(len(buildings))]
# --- Compute building area ---
buildings["area_m2"] = buildings.geometry.area

# --- Compute building centroid (for spatial join) ---
building_points = buildings.copy()
building_points["geometry"] = building_points.geometry.centroid

# --- Spatial join: assign each building point to census polygon ---
bld_join = gpd.sjoin(building_points, census, how="left", predicate="within")

# --- Assign default population based on area ---
#po derive from NISR maps at sector level and ranges from 3 to 5.5
bld_join["pop_default"] = bld_join.apply(lambda row: row["po"] if row["area_m2"] >= 45 else 0, axis=1)
# --- Compute totals per census polygon ---
group = bld_join.groupby("key").agg(
    total_area=("area_m2", "sum"),
    total_default=("pop_default", "sum")
).reset_index()

# --- Merge totals into census polygons ---
census = census.merge(group, on="key", how="left").fillna(0)
census.columns
# --- Compute residual population ---
census["pop_residual"] = census["pop"] - census["total_default"]
census.loc[census["pop_residual"] < 0, "pop_residual"] = 0  # avoid negative

# --- Bring residuals & total area back to building points ---
bld_join = bld_join.merge(
    census[["key", "pop_residual", "total_area"]],
    on="key", how="left"
)

# --- Compute area-weighted redistribution ---
bld_join["pop_area_share"] = bld_join["pop_residual"] * (bld_join["area_m2"] / bld_join["total_area"])

# --- Final building population ---
bld_join["POP_ALLOC"] = bld_join["pop_default"] + bld_join["pop_area_share"]

# --- (Optional) attach population back to building polygons ---
buildings = buildings.merge(bld_join[["building_id", "POP_ALLOC"]], on="building_id", how="left")

# --- Export ---
buildings.to_file("buildings_with_pop.shp")

###############################################################################
buildings = gpd.read_file("...\Buildings_with_villagepop.shp")
import numpy as np
# === 1. Assign class labels ===; THESE ARE FROM OUR PREDICTION
def assign_class(row):
    if row["DN"] in [102, 103, 201]:
        return "low"
    elif row["DN"] in [172, 217, 23]:
        return "high"
    else:
        return np.nan

buildings["class"] = buildings.apply(assign_class, axis=1)

# === 2. Convert building polygons to centroid points ===
# (So we can easily check which grid cell each building falls into)
buildings_points = buildings.copy()
buildings_points["geometry"] = buildings_points.geometry.centroid

# === 3. Read grid shapefile ===
grids = gpd.read_file("...\WorldpopGrid.shp")

# Ensure CRS (coordinate reference system) matches
if buildings_points.crs != grids.crs:
    buildings_points = buildings_points.to_crs(grids.crs)

# === 4. Spatial join: allocate each building point to a grid cell ===
buildings_joined = gpd.sjoin(buildings_points, grids, how="inner", predicate="within")

# === 5. Compute per-grid aggregations ===
# total population per grid cell
agg = (
    buildings_joined.groupby("ID")  # replace with the actual ID column in your grid shapefile
    .agg(
        POPCELL=("POP_ALLOC", "sum"),                 # total population in the cell
        LOW_COUNT=("class", lambda x: (x == "low").sum()),
        HIGH_COUNT=("class", lambda x: (x == "high").sum()),
        TOTAL_BLD=("class", "count")
    )
    .reset_index()
)

# === 6. Compute % share of each class per cell ===
agg["low"] = agg["LOW_COUNT"] / agg["TOTAL_BLD"] * 100
agg["high"] = agg["HIGH_COUNT"] / agg["TOTAL_BLD"] * 100

# === 7. Drop cells with no buildings (if any slipped through)
agg = agg[agg["TOTAL_BLD"] > 0]

# === 8. Merge back with grid geometries ===
grids_agg = grids.merge(agg, on="ID", how="left")

# === 9. Drop empty cells (those without any buildings)
grids_agg = grids_agg.dropna(subset=["POPCELL"])

# === 10. Save the final gridded layer ===
grids_agg.to_file("...\gridded_village_population_building2grids_village.shp")

#EVALUATE#####################################################################
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# --- Step 1: Read shapefiles ---
shp1 = gpd.read_file("...\gridded_village_population_building2grids_village.shp")
shp2 = gpd.read_file("...\rwa_pop_2022_CN_100m_R2025A_v1.shp")   

# --- Step 2: Ensure both have same CRS ---
if shp1.crs != shp2.crs:
    shp2 = shp2.to_crs(shp1.crs)

# --- Step 3: Spatial join (keep all from shp1) ---
joined = gpd.sjoin(shp1, shp2[['value', 'geometry']], how='left', predicate='intersects')
joined.columns

# --- Step 4: Rename columns for clarity ---
joined = joined.rename(columns={'POPCELL': 'actual', 'value': 'pred'})
joined.columns

#selected = gpd.sjoin(gdf1, aoi, how="inner", predicate="intersects")
# --- Step 5: Replace NaN with 0 ---
joined['actual'] = joined['actual'].fillna(0)
joined['pred']   = joined['pred'].fillna(0)
y_true = joined['actual']
y_pred = joined['pred']

# --- Step 6: Compute evaluation metrics ---
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
wmape = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100

# --- Step 7: Print results ---
print("📊 Spatial Evaluation Metrics (keeping all records from shapefile 1)")
print(f"R²     : {r2:.4f}")
print(f"MAE    : {mae:.4f}")
print(f"RMSE   : {rmse:.4f}")
print(f"MAPE   : {mape:.2f}%")
print(f"WMAPE  : {wmape:.2f}%")


