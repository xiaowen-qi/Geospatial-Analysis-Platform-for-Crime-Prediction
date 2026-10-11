"""
Generate a small, fully synthetic demo dataset so both UIs can be tried
end-to-end without real (and often confidential) police data.

Creates, next to this script:
  demo_regions/demo_regions.shp   square grid cells clipped to one county
                                  (field REGION_ID) - stands in for police beats
  demo_incidents.csv              one row per synthetic incident
                                  (incident_id, REGION_ID, date, offense)

The incident counts are random (Poisson, seeded) with a mild link to the
bundled Census poverty rate, so models have *some* signal to find. They do
NOT represent real crime.

Usage:
    python examples/make_demo_data.py [--county-fips 045] [--cell-km 3]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BG_SHP = REPO_ROOT / "data" / "census" / "tl_2020_24_bg" / "tl_2020_24_bg.shp"
POVERTY_CSV = (REPO_ROOT / "data" / "census" / "ACSDT5Y2020.B17017"
               / "ACSDT5Y2020.B17017-Data.csv")
OFFENSES = ["Burglary", "Theft", "Assault", "Vandalism"]


def make_regions(county_fips: str, cell_km: float) -> gpd.GeoDataFrame:
    bg = gpd.read_file(BG_SHP)
    bg = bg[bg["COUNTYFP"] == county_fips].to_crs(epsg=26985)  # NAD83 / Maryland (m)
    if bg.empty:
        raise SystemExit(f"No block groups found for county FIPS {county_fips}")
    county = bg.union_all() if hasattr(bg, "union_all") else bg.unary_union
    minx, miny, maxx, maxy = county.bounds
    step = cell_km * 1000
    cells = [box(x, y, x + step, y + step)
             for x in np.arange(minx, maxx, step)
             for y in np.arange(miny, maxy, step)]
    grid = gpd.GeoDataFrame(geometry=cells, crs=bg.crs)
    grid["geometry"] = grid.intersection(county)
    grid = grid[~grid.is_empty & (grid.area > 0.2 * step * step)].reset_index(drop=True)
    grid["REGION_ID"] = [f"R{i + 1:03d}" for i in range(len(grid))]
    return grid[["REGION_ID", "geometry"]]


def region_poverty(regions: gpd.GeoDataFrame, county_fips: str) -> pd.Series:
    """Area-weighted household poverty rate per region (used only to shape rates)."""
    bg = gpd.read_file(BG_SHP)
    bg = bg[bg["COUNTYFP"] == county_fips].to_crs(regions.crs)
    pov = pd.read_csv(POVERTY_CSV, skiprows=[1], dtype={"GEO_ID": str})
    pov["GEOID"] = pov["GEO_ID"].str[-12:]
    bg = bg.merge(pov[["GEOID", "B17017_001E", "B17017_002E"]], on="GEOID", how="left")
    bg["bg_area"] = bg.area
    ov = gpd.overlay(regions, bg, how="intersection")
    w = ov.area / ov["bg_area"]
    ov["tot"] = w * ov["B17017_001E"]
    ov["below"] = w * ov["B17017_002E"]
    agg = ov.groupby("REGION_ID")[["tot", "below"]].sum()
    return (agg["below"] / agg["tot"].replace(0, np.nan)).reindex(regions["REGION_ID"]).fillna(0)


def make_incidents(regions, poverty, start, end, seed):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    # base daily rate per region: lognormal noise x poverty effect
    base = 0.08 * np.exp(rng.normal(0, 0.6, len(regions))) * (1 + 4 * poverty.values)
    weekday = np.array([0.9, 0.9, 0.95, 1.0, 1.15, 1.3, 1.1])  # Mon..Sun
    season = 1 + 0.2 * np.sin(2 * np.pi * (days.dayofyear - 100) / 365.25)
    rows = []
    for r, rid in enumerate(regions["REGION_ID"]):
        lam = base[r] * weekday[days.dayofweek] * season
        counts = rng.poisson(lam)
        for d, c in zip(days[counts > 0], counts[counts > 0]):
            for off in rng.choice(OFFENSES, size=c, p=[0.25, 0.45, 0.15, 0.15]):
                rows.append((rid, d.strftime("%Y-%m-%d"), off))
    df = pd.DataFrame(rows, columns=["REGION_ID", "date", "offense"])
    df = df.sort_values(["date", "REGION_ID"]).reset_index(drop=True)
    df.insert(0, "incident_id", np.arange(1, len(df) + 1))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--county-fips", default="045", help="Maryland county FIPS (045 = Wicomico)")
    ap.add_argument("--cell-km", type=float, default=3.0, help="grid cell size in km")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    regions = make_regions(args.county_fips, args.cell_km)
    out_dir = HERE / "demo_regions"
    out_dir.mkdir(exist_ok=True)
    regions.to_file(out_dir / "demo_regions.shp")
    print(f"Wrote {len(regions)} regions -> {out_dir / 'demo_regions.shp'}")

    pov = region_poverty(regions, args.county_fips)
    inc = make_incidents(regions, pov, args.start, args.end, args.seed)
    inc.to_csv(HERE / "demo_incidents.csv", index=False)
    print(f"Wrote {len(inc):,} incidents -> {HERE / 'demo_incidents.csv'}")


if __name__ == "__main__":
    main()
