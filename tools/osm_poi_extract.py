"""
Extract POI data from an OSM PBF file → GeoPackage (.gpkg)
with one-hot category columns ready for the alignment UI.

Usage:
    python osm_poi_extract.py <path_to.osm.pbf> [output.gpkg]

Requires:  pyrosm, geopandas, pandas
"""
import sys
import pandas as pd
import geopandas as gpd
from pyrosm import OSM


def classify_poi(row):
    a = row.get("amenity")
    s = row.get("shop")
    l = row.get("leisure")
    t = row.get("tourism")
    b = row.get("building")
    if a in ("restaurant", "cafe", "bar", "pub", "nightclub", "fast_food"):
        return "Eating & Drinking"
    if s == "money_lender":
        return "Check Cashing"
    if a == "fuel":
        return "Gas Stations"
    if s in ("supermarket", "convenience"):
        return "Grocery"
    if s == "alcohol":
        return "Liquor Stores"
    if t in ("hotel", "motel"):
        return "Hotels/Motels"
    if l == "park":
        return "Parks"
    if b == "apartments":
        return "Apartments"
    if s == "mall":
        return "Shopping Malls"
    if s in ("department_store", "variety_store", "hardware", "doityourself") or a == "pharmacy":
        return "Business Retailers"
    # Large Retail: wholesale OR retail building > 2000 m²
    area = row.get("area_sqm", 0) or 0
    if s == "wholesale" or (b == "retail" and area > 2000):
        return "Large Retail"
    return None


def main(pbf_path: str, out_path: str = "poi_dataset.gpkg"):
    print(f"Reading: {pbf_path}")
    osm = OSM(pbf_path)
    custom_tags = {
        "amenity": True, "shop": True, "tourism": True,
        "leisure": True, "building": ["apartments", "retail"],
        "railway": ["station"], "public_transport": ["stop_area"],
    }
    pois = osm.get_pois(custom_filter=custom_tags)
    pois = pois.copy()  # defragment the DataFrame
    print(f"  Raw features: {len(pois)}")

    # Compute area in square metres using a projected CRS.
    # EPSG:6487 = NAD83(2011) / Maryland, whose unit is the metre, so .area
    # is already in m² (no ft² conversion; EPSG:6488 is the ftUS variant).
    pois_proj = pois.to_crs(epsg=6487)
    pois["area_sqm"] = pois_proj.area

    # Classify
    pois["category"] = pois.apply(classify_poi, axis=1)
    dataset = pois.dropna(subset=["category"]).copy()
    print(f"  Classified features: {len(dataset)}")

    # One-hot encode categories
    dummies = pd.get_dummies(dataset["category"], prefix="category").astype(int)
    dataset = pd.concat([dataset, dummies], axis=1)

    # Keep useful columns only
    addr_cols = [c for c in dataset.columns if c.startswith("addr:")]
    cat_cols = [c for c in dataset.columns if c.startswith("category_")]
    keep = addr_cols + ["name", "geometry"] + sorted(cat_cols)
    keep = [c for c in keep if c in dataset.columns]
    dataset = dataset[keep].copy()

    # Set CRS to NAD83 (EPSG:4269)
    if dataset.crs is None:
        dataset = dataset.set_crs(epsg=4326)
    dataset = dataset.to_crs(epsg=4269)

    dataset.to_file(out_path, driver="GPKG")
    print(f"  Saved: {out_path}  ({len(dataset)} rows, CRS={dataset.crs})")
    print(f"  Categories: {', '.join(c.replace('category_', '') for c in cat_cols)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python osm_poi_extract.py <file.osm.pbf> [output.gpkg]")
        sys.exit(1)
    pbf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "poi_dataset.gpkg"
    main(pbf, out)