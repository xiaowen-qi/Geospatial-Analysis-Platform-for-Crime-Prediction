"""
Census / POI Contextual-Feature Alignment UI
============================================
Re-distributes Census block-group statistics (race, unemployment, poverty)
and optional POI counts onto a user-supplied set of regions (e.g. police
beats), and writes per-region CSV / GeoJSON / interactive HTML outputs.

Run:
    python census_alignment_ui.py [--output-dir DIR] [--block-groups SHP] ...
See README.md for details.
"""
import argparse
import sys
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QLabel, QComboBox, QTextEdit, QVBoxLayout, QWidget, QMessageBox,
    QHBoxLayout, QCheckBox, QGroupBox, QScrollArea,
)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QTextCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Default data locations (relative to this repository)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CENSUS_DIR = REPO_ROOT / "data" / "census"
DEFAULT_BLOCK_GROUPS = CENSUS_DIR / "tl_2020_24_bg" / "tl_2020_24_bg.shp"
DEFAULT_RACE_CSV = CENSUS_DIR / "DECENNIALPL2020.P1" / "DECENNIALPL2020.P1-Data.csv"
DEFAULT_UNEMPLOYMENT_CSV = CENSUS_DIR / "ACSDT5Y2020.B23025" / "ACSDT5Y2020.B23025-Data.csv"
DEFAULT_POVERTY_CSV = CENSUS_DIR / "ACSDT5Y2020.B17017" / "ACSDT5Y2020.B17017-Data.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "census_alignment"


# ---------------------------------------------------------------------------
# Helper: parse Census NAME field → block_group / census_tract / county
# ---------------------------------------------------------------------------
def parse_location_from_name(name_str: str):
    if pd.isna(name_str):
        return pd.Series([None, None, None])

    parts = [p.strip() for p in str(name_str).split(",")]
    block_group = None
    census_tract = None
    county = None

    if len(parts) > 0 and parts[0].startswith("Block Group"):
        m = re.search(r"Block Group\s+(\d+)", parts[0])
        if m:
            block_group = m.group(1)

    if len(parts) > 1 and parts[1].startswith("Census Tract"):
        m = re.search(r"Census Tract\s+([\d\.]+)", parts[1])
        if m:
            tract_num = m.group(1)
            if "." in tract_num:
                main, suffix = tract_num.split(".")
            else:
                main, suffix = tract_num, "00"
            try:
                census_tract = f"{int(main):04d}{int(suffix):02d}"
            except ValueError:
                census_tract = None

    if len(parts) > 2:
        county = parts[2]

    return pd.Series([block_group, census_tract, county])


# ═══════════════════════════════════════════════════════════════════════════
#  Main UI
# ═══════════════════════════════════════════════════════════════════════════
class AlignmentWorker(QThread):
    """Runs ShapeComparerUI._alignment_job off the UI thread so the window
    stays responsive and the log updates live."""
    succeeded = pyqtSignal(object)   # path of the combined HTML map, or None
    failed = pyqtSignal(str)         # error message

    def __init__(self, ui, settings: dict):
        super().__init__(ui)
        self.ui = ui
        self.settings = settings

    def run(self):
        try:
            self.succeeded.emit(self.ui._alignment_job(**self.settings))
        except Exception as e:
            self.failed.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════
class ShapeComparerUI(QMainWindow):

    # Requests that touch widgets. Emitted from the alignment worker they
    # are queued to the UI thread (Qt widgets / matplotlib are not
    # thread-safe); emitted from the UI thread they run immediately.
    _log_requested = pyqtSignal(str)
    _plot_requested = pyqtSignal(object, str, str)   # gdf, value_col, title
    _warning_requested = pyqtSignal(str, str)        # title, text

    # ------------------------------------------------------------------
    # Safe logger (thread-safe)
    # ------------------------------------------------------------------
    def log(self, text: str):
        self._log_requested.emit(str(text))

    @pyqtSlot(str)
    def _append_log(self, text: str):
        try:
            if not hasattr(self, "log_text"):
                self.log_text = ""
            self.log_text += text + "\n"
            if hasattr(self, "log_area") and self.log_area is not None:
                # append (instead of re-setting the whole text, which
                # would jump the view back to the top)
                self.log_area.moveCursor(QTextCursor.End)
                self.log_area.insertPlainText(text + "\n")
            else:
                print(text)
        except Exception:
            print(text)

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------
    def __init__(self,
                 block_groups_path=DEFAULT_BLOCK_GROUPS,
                 race_csv_path=DEFAULT_RACE_CSV,
                 unemployment_csv_path=DEFAULT_UNEMPLOYMENT_CSV,
                 poverty_csv_path=DEFAULT_POVERTY_CSV,
                 output_dir=DEFAULT_OUTPUT_DIR):
        super().__init__()
        self._log_requested.connect(self._append_log)
        self._plot_requested.connect(self._draw_choropleth)
        self._warning_requested.connect(
            lambda title, text: QMessageBox.warning(self, title, text))
        self._worker = None
        self.setWindowTitle("Contextual Feature Alignment (Census + POI)")
        self.setGeometry(100, 100, 1200, 720)

        # ── Census input paths & output folder ──────────────────────
        self.default_shp_path = str(block_groups_path)
        self.default_csv_path = str(race_csv_path)
        self.unemployment_csv_path = str(unemployment_csv_path)
        self.poverty_csv_path = str(poverty_csv_path)
        self.output_dir = Path(output_dir)

        # ── Data stores ─────────────────────────────────────────────
        self.custom_gdf = None
        if not Path(self.default_shp_path).exists():
            raise FileNotFoundError(
                f"Census block-group shapefile not found:\n{self.default_shp_path}\n\n"
                "Pass --block-groups <path.shp> or restore data/census/.")
        self.default_gdf = gpd.read_file(self.default_shp_path)
        self._ensure_bg_geoid()
        self.county_names = {}            # 5-digit FIPS -> "Wicomico County"
        self.detected_county_label = ""   # used in plot titles

        self.default_csv_df = None        # Race
        self.unemployment_csv_df = None   # Unemployment
        self.poverty_csv_df = None        # Poverty

        self.poi_gdf = None               # POI data (mixed geom types, reprojected to default CRS)
        self.poi_category_cols = []       # detected category_* columns
        self.poi_checkboxes = []          # QCheckBox list for POI categories

        self.log_text = ""

        # ── Build UI ────────────────────────────────────────────────
        self._build_ui()

        # ── Initial data load ───────────────────────────────────────
        self.plot_geometry(self.default_gdf, title="Default Census Block Groups")
        self.load_and_process_race_csv()
        self.load_and_process_unemployment_csv()
        self.load_and_process_poverty_csv()
        self.log(f"Outputs will be written to: {self.output_dir}")

    def _ensure_bg_geoid(self):
        """Make sure the block-group layer has a 12-digit GEOID column."""
        bg = self.default_gdf
        for alt in ("GEOID", "GEOID20", "GEOID10"):
            if alt in bg.columns:
                bg["GEOID"] = bg[alt].astype(str).str.strip()
                return
        parts = ("STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE")
        if all(c in bg.columns for c in parts):
            bg["GEOID"] = bg[list(parts)].astype(str).agg("".join, axis=1)
            return
        raise ValueError("Block-group shapefile has no GEOID column "
                         "(or STATEFP/COUNTYFP/TRACTCE/BLKGRPCE to build one).")

    def _out(self, filename: str) -> str:
        """Return the full output path for *filename* (creates the folder)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return str(self.output_dir / filename)

    # ==================================================================
    #  UI construction
    # ==================================================================
    def _build_ui(self):
        # ── Custom shapefile upload ──
        self.upload_btn = QPushButton("Upload Custom Shapefile")
        self.upload_btn.clicked.connect(self.load_shapefile)

        self.location_col_dropdown = QComboBox()
        self.geometry_col_dropdown = QComboBox()
        self.location_col_dropdown.setEnabled(False)
        self.geometry_col_dropdown.setEnabled(False)

        # ── Census feature checkboxes ──
        self.census_group = QGroupBox("Census Features")
        census_lay = QVBoxLayout()
        self.race_checkbox = QCheckBox("Race (Black Ratio)")
        self.unemp_checkbox = QCheckBox("Unemployment Rate")
        self.poverty_checkbox = QCheckBox("Poverty Rate")
        for cb in (self.race_checkbox, self.unemp_checkbox, self.poverty_checkbox):
            cb.stateChanged.connect(self._on_any_feature_toggled)
            census_lay.addWidget(cb)

        self.county_label = QLabel("Counties covered (auto-detected, info only)")
        self.county_dropdown = QComboBox()
        self.county_dropdown.setToolTip(
            "Detected from your regions' geometry. Census values are matched "
            "per block group, so every county listed here is used automatically.")
        self.county_dropdown.addItem("Upload a custom shapefile to detect")
        self.county_dropdown.setEnabled(False)
        census_lay.addWidget(self.county_label)
        census_lay.addWidget(self.county_dropdown)
        self.census_group.setLayout(census_lay)

        # ── POI section ──
        self.poi_group = QGroupBox("POI Features")
        poi_outer = QVBoxLayout()
        self.poi_upload_btn = QPushButton("Upload POI Data (.gpkg / .geojson / .shp)")
        self.poi_upload_btn.clicked.connect(self.load_poi_data)
        poi_outer.addWidget(self.poi_upload_btn)

        self.poi_scroll_area = QScrollArea()
        self.poi_scroll_area.setWidgetResizable(True)
        self.poi_scroll_widget = QWidget()
        self.poi_scroll_layout = QVBoxLayout()
        self.poi_scroll_widget.setLayout(self.poi_scroll_layout)
        self.poi_scroll_area.setWidget(self.poi_scroll_widget)
        self.poi_scroll_area.setMaximumHeight(180)
        poi_outer.addWidget(self.poi_scroll_area)
        self.poi_group.setLayout(poi_outer)

        # ── Run / Log / Help ──
        self.run_btn = QPushButton("Run Alignment")
        self.run_btn.clicked.connect(self.run_alignment)
        self.run_btn.setEnabled(False)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        # auto-follow: whenever the content grows (or the widget is laid
        # out / resized), jump to the newest line
        log_sb = self.log_area.verticalScrollBar()
        log_sb.rangeChanged.connect(lambda _lo, hi: log_sb.setValue(hi))

        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self.show_help)

        # ── Canvas ──
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # ── Assemble left panel ──
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.upload_btn)
        left_layout.addWidget(QLabel("Select Location Column"))
        left_layout.addWidget(self.location_col_dropdown)
        left_layout.addWidget(QLabel("Select Geometry Column"))
        left_layout.addWidget(self.geometry_col_dropdown)
        left_layout.addWidget(self.census_group)
        left_layout.addWidget(self.poi_group)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(QLabel("Log"))
        left_layout.addWidget(self.log_area)
        left_layout.addWidget(self.help_btn)

        main_layout = QHBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, 3)
        main_layout.addWidget(self.canvas, 5)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ==================================================================
    #  CSV load & processing  (Census)
    # ==================================================================
    def load_and_process_race_csv(self):
        try:
            df = pd.read_csv(self.default_csv_path)
            if len(df) > 0:
                df = df.iloc[1:].reset_index(drop=True)
            if "NAME" not in df.columns:
                raise ValueError("Race CSV missing 'NAME' column.")

            rename = {}
            if "P1_001N" in df.columns:
                rename["P1_001N"] = "total population"
            if "P1_004N" in df.columns:
                rename["P1_004N"] = "black alone"
            if rename:
                df = df.rename(columns=rename)

            parsed = df["NAME"].apply(parse_location_from_name)
            parsed.columns = ["block_group", "census_tract", "county"]
            df = pd.concat([df, parsed], axis=1)
            df["block_group"] = df["block_group"].astype(str).str.strip()
            df["census_tract"] = df["census_tract"].astype(str).str.strip()
            for col in ("total population", "black alone"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = self._add_geoid(df, "Race")
            self.default_csv_df = df
            n_counties = self._register_county_names(df)
            self.log(f"Loaded Race CSV  ({len(df)} rows, {n_counties} counties)")
        except Exception as e:
            self.log(f"Warning – Race CSV: {e}")

    def load_and_process_unemployment_csv(self):
        try:
            df = pd.read_csv(self.unemployment_csv_path)
            if "NAME" not in df.columns:
                raise ValueError("Unemployment CSV missing 'NAME'.")
            df = df.rename(columns={"B23025_002E": "in labor force",
                                    "B23025_005E": "unemployed"})
            parsed = df["NAME"].apply(parse_location_from_name)
            parsed.columns = ["block_group", "census_tract", "county"]
            df = pd.concat([df, parsed], axis=1)
            df["block_group"] = df["block_group"].astype(str).str.strip()
            df["census_tract"] = df["census_tract"].astype(str).str.strip()
            for col in ("in labor force", "unemployed"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = self._add_geoid(df, "Unemployment")
            self.unemployment_csv_df = df
            self._register_county_names(df)
            self.log(f"Loaded Unemployment CSV ({len(df)} rows)")
        except Exception as e:
            self.log(f"Warning – Unemployment CSV: {e}")

    def load_and_process_poverty_csv(self):
        try:
            df = pd.read_csv(self.poverty_csv_path)
            if "NAME" not in df.columns:
                raise ValueError("Poverty CSV missing 'NAME'.")
            df = df.rename(columns={"B17017_001E": "total",
                                    "B17017_002E": "below poverty level"})
            parsed = df["NAME"].apply(parse_location_from_name)
            parsed.columns = ["block_group", "census_tract", "county"]
            df = pd.concat([df, parsed], axis=1)
            df["block_group"] = df["block_group"].astype(str).str.strip()
            df["census_tract"] = df["census_tract"].astype(str).str.strip()
            for col in ("total", "below poverty level"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = self._add_geoid(df, "Poverty")
            self.poverty_csv_df = df
            self._register_county_names(df)
            self.log(f"Loaded Poverty CSV ({len(df)} rows)")
        except Exception as e:
            self.log(f"Warning – Poverty CSV: {e}")

    # ── GEOID / county helpers ──
    @staticmethod
    def _add_geoid(df, label):
        """Add the 12-digit block-group GEOID (last 12 chars of GEO_ID,
        e.g. '1500000US240010001001' -> '240010001001'). Census rows are
        joined to block groups on this key."""
        if "GEO_ID" not in df.columns:
            raise ValueError(f"{label} CSV missing 'GEO_ID' column.")
        df["GEOID"] = df["GEO_ID"].astype(str).str.strip().str[-12:]
        return df.drop_duplicates(subset="GEOID", keep="first")

    def _register_county_names(self, df):
        """Remember county names keyed by 5-digit state+county FIPS
        (first 5 digits of GEOID). Used only for display."""
        rows = df.dropna(subset=["county"])
        rows = rows[rows["GEOID"].str.fullmatch(r"\d{12}")]
        self.county_names.update(
            zip(rows["GEOID"].str[:5], rows["county"].astype(str).str.strip()))
        return rows["GEOID"].str[:5].nunique()

    def _detect_counties(self):
        """Fill the (informational) county dropdown with the counties that
        the uploaded regions cover, by share of the regions' area.
        Census values are joined per block group (GEOID), so every county
        a region overlaps contributes automatically - nothing to select."""
        self.county_dropdown.clear()
        self.detected_county_label = ""
        try:
            bg = self.default_gdf[["GEOID", "geometry"]]
            regions = self.custom_gdf[[self.custom_gdf.geometry.name]]
            _, cand = bg.sindex.query(regions.geometry, predicate="intersects")
            if len(cand) == 0:
                self.county_dropdown.addItem("No Census block groups overlap your regions")
                self.log("Warning: your regions do not overlap any Census block group.")
                return
            bg = bg.iloc[np.unique(cand)]
            crs = regions.estimate_utm_crs()
            ov = gpd.overlay(regions.to_crs(crs), bg.to_crs(crs),
                             how="intersection", keep_geom_type=True)
            area = ov.geometry.area.groupby(ov["GEOID"].str[:5]).sum()
            share = (area / area.sum()).sort_values(ascending=False)
        except Exception as e:
            self.county_dropdown.addItem("County detection failed (see log)")
            self.log(f"County detection failed: {e}")
            return

        # boundary slivers (<0.01% of area) are still used in the
        # calculation but not listed, to keep the list readable
        shown = share[share >= 1e-4]
        for fips, sh in shown.items():
            name = self.county_names.get(fips, f"County FIPS {fips}")
            self.county_dropdown.addItem(f"{name}  ({sh:.2%} of area)")
        names = [self.county_names.get(f, f"County FIPS {f}") for f in shown.index]
        self.detected_county_label = (names[0] if len(names) == 1
                                      else f"{len(names)} counties")
        self.log(f"Detected counties: {', '.join(names)}")
        if len(share) > len(shown):
            self.log(f"  (+{len(share) - len(shown)} county/counties touched only "
                     f"by boundary slivers < 0.01% of area)")

    # ==================================================================
    #  POI data load  (GeoPackage / GeoJSON / Shapefile)
    # ==================================================================
    def load_poi_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open POI Data", "",
            "Geospatial Files (*.gpkg *.geojson *.json *.shp);;All Files (*)"
        )
        if not path:
            return
        try:
            gdf = gpd.read_file(path)
            self.log(f"Loaded POI file: {path}  ({len(gdf)} features, CRS={gdf.crs})")

            # ── Report geometry types (no centroid conversion) ──
            geom_types = gdf.geometry.geom_type.value_counts()
            for gt, cnt in geom_types.items():
                self.log(f"  {gt}: {cnt} features")
            n_poly = (~gdf.geometry.geom_type.isin(("Point", "MultiPoint"))).sum()
            if n_poly:
                self.log(f"  → {n_poly} polygon/multipolygon features will use "
                         f"area-weighted overlay (not centroids).")

            if gdf.empty:
                QMessageBox.warning(self, "Empty", "POI file has no features.")
                return

            # ── Re-project to match the census (default) CRS ──
            target_crs = self.default_gdf.crs
            if gdf.crs is None:
                self.log(f"POI has no CRS — assuming EPSG:4326 and reprojecting → {target_crs}")
                gdf = gdf.set_crs(epsg=4326)
            if gdf.crs != target_crs:
                self.log(f"Reprojecting POI from {gdf.crs} → {target_crs}")
                gdf = gdf.to_crs(target_crs)

            # ── Detect category_* columns ──
            cat_cols = [c for c in gdf.columns if c.startswith("category_")]
            if not cat_cols:
                QMessageBox.warning(
                    self, "No Categories",
                    "No columns starting with 'category_' found in the POI data.\n"
                    "Use tools/osm_poi_extract.py to produce a compatible .gpkg."
                )
                return

            for c in cat_cols:
                gdf[c] = pd.to_numeric(gdf[c], errors="coerce").fillna(0).astype(int)

            self.poi_gdf = gdf
            self.poi_category_cols = cat_cols
            self._populate_poi_checkboxes(cat_cols)

            self.log(f"Detected {len(cat_cols)} POI categories: "
                     f"{', '.join(c.replace('category_', '') for c in cat_cols)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load POI data:\n{e}")
            self.log(f"Error loading POI: {e}")

    def _populate_poi_checkboxes(self, cat_cols):
        for cb in self.poi_checkboxes:
            self.poi_scroll_layout.removeWidget(cb)
            cb.deleteLater()
        self.poi_checkboxes = []

        for col in sorted(cat_cols):
            label = col.replace("category_", "")
            cb = QCheckBox(label)
            cb.setProperty("poi_col", col)
            cb.stateChanged.connect(self._on_any_feature_toggled)
            self.poi_scroll_layout.addWidget(cb)
            self.poi_checkboxes.append(cb)

    # ==================================================================
    #  Feature-toggle helpers
    # ==================================================================
    def _any_census_checked(self):
        return (self.race_checkbox.isChecked()
                or self.unemp_checkbox.isChecked()
                or self.poverty_checkbox.isChecked())

    def _checked_poi_cols(self):
        return [cb.property("poi_col") for cb in self.poi_checkboxes if cb.isChecked()]

    def _any_feature_checked(self):
        return self._any_census_checked() or bool(self._checked_poi_cols())

    def _on_any_feature_toggled(self):
        pass  # county dropdown is informational; nothing to toggle

    # ==================================================================
    #  Custom shapefile upload
    # ==================================================================
    def load_shapefile(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Shapefile", "", "Shapefiles (*.shp)")
        if not file_path:
            return
        try:
            custom_gdf = gpd.read_file(file_path)
            self.log(f"Loaded custom shapefile: {file_path}  (CRS={custom_gdf.crs})")

            # Re-project to match census CRS
            if custom_gdf.crs != self.default_gdf.crs:
                self.log(f"Re-projecting from {custom_gdf.crs} → {self.default_gdf.crs}")
                custom_gdf = custom_gdf.to_crs(self.default_gdf.crs)

            self.custom_gdf = custom_gdf
            self._detect_counties()
            self.county_dropdown.setEnabled(True)

            self.location_col_dropdown.clear()
            self.geometry_col_dropdown.clear()
            columns = list(self.custom_gdf.columns)
            self.location_col_dropdown.addItems(columns)
            self.geometry_col_dropdown.addItems(columns)
            self.location_col_dropdown.setEnabled(True)
            self.geometry_col_dropdown.setEnabled(True)
            self.run_btn.setEnabled(True)

            self.plot_geometry(self.custom_gdf, title="Custom Shapefile")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load shapefile:\n{e}")

    # ==================================================================
    #  RUN ALIGNMENT
    # ==================================================================
    def run_alignment(self):
        """UI thread: validate, snapshot the settings, start the worker."""
        if self._worker is not None and self._worker.isRunning():
            return
        if self.custom_gdf is None:
            QMessageBox.warning(self, "Warning", "Upload a custom shapefile first.")
            return

        loc_col = self.location_col_dropdown.currentText()
        geom_col = self.geometry_col_dropdown.currentText()
        if not loc_col or not geom_col:
            QMessageBox.warning(self, "Warning", "Select location and geometry columns.")
            return

        census_on = self._any_census_checked()
        poi_cols = self._checked_poi_cols()

        if not census_on and not poi_cols:
            QMessageBox.warning(self, "Warning",
                                "Select at least one feature (Census or POI).")
            return

        # snapshot widget state - the worker must not read widgets
        settings = dict(
            loc_col=loc_col, geom_col=geom_col, poi_cols=poi_cols,
            race=self.race_checkbox.isChecked(),
            unemp=self.unemp_checkbox.isChecked(),
            poverty=self.poverty_checkbox.isChecked(),
        )
        self.log("Running alignment ...")
        self._set_running(True)
        self._worker = AlignmentWorker(self, settings)
        self._worker.succeeded.connect(self._on_alignment_succeeded)
        self._worker.failed.connect(self._on_alignment_failed)
        self._worker.finished.connect(lambda: self._set_running(False))
        self._worker.start()

    def _set_running(self, running: bool):
        """Lock the inputs while the worker runs (Log and Help stay usable)."""
        for w in (self.upload_btn, self.location_col_dropdown,
                  self.geometry_col_dropdown, self.census_group,
                  self.poi_group, self.run_btn):
            w.setEnabled(not running)
        self.run_btn.setText("Running ..." if running else "Run Alignment")

    @pyqtSlot(object)
    def _on_alignment_succeeded(self, map_path):
        msg = "Alignment completed — outputs saved."
        if map_path:
            msg += f"\n\nInteractive map saved:\n{map_path}"
        QMessageBox.information(self, "Success", msg)

    @pyqtSlot(str)
    def _on_alignment_failed(self, err):
        QMessageBox.critical(self, "Error", f"Alignment failed:\n{err}")
        self.log(f"Error: {err}")

    def closeEvent(self, event):
        # a QThread must not be destroyed while running
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Please wait",
                                    "Alignment is still running.")
            event.ignore()
            return
        super().closeEvent(event)

    def _alignment_job(self, loc_col, geom_col, poi_cols,
                       race, unemp, poverty):
        """Worker thread: the alignment itself (no direct widget access;
        log / plot go through queued signals). Returns the combined map
        path (or None)."""
        census_on = race or unemp or poverty
        county_label = self.detected_county_label or "detected counties"

        # ── Dissolve custom regions ──
        self.custom_gdf = self.custom_gdf.set_geometry(geom_col)
        self.custom_gdf = self.custom_gdf.dissolve(by=loc_col).reset_index()
        # Region IDs are handled as text everywhere downstream (POI summary,
        # combined table); numeric IDs would otherwise break the merges.
        self.custom_gdf[loc_col] = self.custom_gdf[loc_col].astype(str)

        layers_for_combo = []
        combined_df = pd.DataFrame(
            {loc_col: self.custom_gdf[loc_col].astype(str).values}
        )

        # ============================================================
        # A) Census alignment  (area-overlap distribution)
        # ============================================================
        align_df = None
        if census_on:
            align_df = self._build_census_overlap(loc_col)
            align_df.to_csv(self._out("geometry_alignment.csv"), index=False)
            self.log("Saved geometry_alignment.csv")

            if race:
                s, g = self._align_race(align_df, loc_col, county_label)
                if s is not None:
                    combined_df = combined_df.merge(
                        s[[loc_col, "black ratio"]].rename(
                            columns={"black ratio": "black_rate"}),
                        on=loc_col, how="left")
                    layers_for_combo.append({
                        "name": "Race (Black Ratio)", "gdf": g,
                        "value_col": "black ratio",
                        "tooltip_extra": ["black_population",
                                          "overall_population"],
                    })

            if unemp:
                s, g = self._align_unemployment(align_df, loc_col,
                                                 county_label)
                if s is not None:
                    combined_df = combined_df.merge(
                        s[[loc_col, "unemployment rate"]].rename(
                            columns={"unemployment rate": "unemployment_rate"}),
                        on=loc_col, how="left")
                    layers_for_combo.append({
                        "name": "Unemployment Rate", "gdf": g,
                        "value_col": "unemployment rate",
                        "tooltip_extra": ["labor_force", "unemployed"],
                    })

            if poverty:
                s, g = self._align_poverty(align_df, loc_col,
                                            county_label)
                if s is not None:
                    combined_df = combined_df.merge(
                        s[[loc_col, "poverty rate"]].rename(
                            columns={"poverty rate": "poverty_rate"}),
                        on=loc_col, how="left")
                    layers_for_combo.append({
                        "name": "Poverty Rate", "gdf": g,
                        "value_col": "poverty rate",
                        "tooltip_extra": ["total", "below_poverty"],
                    })

        # ============================================================
        # B) POI alignment  (point-in-polygon counts)
        # ============================================================
        if poi_cols:
            poi_summary = self._align_poi(loc_col, poi_cols)
            if poi_summary is not None and not poi_summary.empty:
                merge_cols = [loc_col] + [c for c in poi_summary.columns
                                          if c != loc_col]
                combined_df = combined_df.merge(
                    poi_summary[merge_cols], on=loc_col, how="left")

                for col in poi_cols:
                    count_col = col.replace("category_", "") + "_count"
                    if count_col not in poi_summary.columns:
                        continue
                    poi_mapped = self.custom_gdf.merge(
                        poi_summary[[loc_col, count_col]],
                        on=loc_col, how="left")
                    poi_mapped[count_col] = poi_mapped[count_col].fillna(0)

                    mx = poi_mapped[count_col].max()
                    norm_col = count_col + "_norm"
                    poi_mapped[norm_col] = (
                        poi_mapped[count_col] / mx if mx > 0 else 0.0
                    )

                    layers_for_combo.append({
                        "name": f"POI: {col.replace('category_', '')}",
                        "gdf": poi_mapped,
                        "value_col": norm_col,
                        "tooltip_extra": [count_col],
                    })

        # ============================================================
        # C) Save combined CSV
        # ============================================================
        keep = [loc_col] + [c for c in combined_df.columns if c != loc_col]
        combined_df = combined_df[keep].sort_values(loc_col)
        combined_df.to_csv(self._out("alignment_combined_by_location.csv"), index=False)
        self.log("Saved alignment_combined_by_location.csv")

        # ============================================================
        # D) Combined interactive map
        # ============================================================
        map_path = None
        if layers_for_combo:
            map_path = self.export_interactive_combo_map(
                base_gdf=self.custom_gdf,
                layers=layers_for_combo,
                out_path=self._out("context_layers_map.html"),
                loc_field=loc_col,
            )

        return map_path

    # ==================================================================
    #  Census overlap matrix
    # ==================================================================
    def _build_census_overlap(self, loc_col):
        census = self.default_gdf
        custom = self.custom_gdf
        rows = []
        # Spatial index returns exactly the block groups for which
        # g1.intersects(g2) is True, without testing all ~4k of them.
        sindex = census.sindex
        for i, g1 in enumerate(custom.geometry):
            for j in sorted(sindex.query(g1, predicate="intersects")):
                g2 = census.geometry.iloc[j]
                inter = g1.intersection(g2)
                if inter.is_empty:
                    continue
                area_pb = inter.area / g1.area if g1.area else 0.0
                area_bg = inter.area / g2.area if g2.area else 0.0
                rows.append({
                    "police_beats": custom.iloc[i][loc_col],
                    "tract": str(census.iloc[j]["TRACTCE"]).strip(),
                    "block_group": str(census.iloc[j]["BLKGRPCE"]).strip(),
                    "GEOID": str(census.iloc[j]["GEOID"]).strip(),
                    "intersected_area_pb": area_pb,
                    "intersected_area_bg": area_bg,
                    "geometry_pb": g1,
                })
        return pd.DataFrame(rows)

    # ==================================================================
    #  Census feature branches
    # ==================================================================
    def _align_race(self, align_df, loc_col, county_label):
        if self.default_csv_df is None:
            raise ValueError("Race CSV not loaded.")
        for c in ("total population", "black alone"):
            if c not in self.default_csv_df.columns:
                raise ValueError(f"Race CSV missing '{c}'")

        # join on the unique 12-digit block-group GEOID (all counties)
        filt = self.default_csv_df[["GEOID", "total population", "black alone"]]
        merged = align_df.merge(filt, on="GEOID", how="left")

        merged["total_pop_piece"] = merged["intersected_area_bg"] * merged["total population"]
        merged["black_pop_piece"] = merged["intersected_area_bg"] * merged["black alone"]

        summary = (merged.groupby("police_beats", as_index=False)
                   .agg(overall_population=("total_pop_piece", "sum"),
                        black_population=("black_pop_piece", "sum")))
        summary["black ratio"] = (
            summary["black_population"] /
            summary["overall_population"].replace(0, np.nan)
        ).fillna(0.0)
        summary = summary.rename(columns={"police_beats": loc_col})

        mapped = self.custom_gdf.merge(
            summary[[loc_col, "overall_population",
                     "black_population", "black ratio"]],
            on=loc_col, how="left")
        mapped["black ratio"] = mapped["black ratio"].fillna(0.0)

        merged.to_csv(self._out("alignment_race_detailed.csv"), index=False)
        summary.to_csv(self._out("alignment_race_by_location.csv"), index=False)
        mapped.to_file(self._out("alignment_race_map.geojson"), driver="GeoJSON")
        self.log("Race outputs saved.")

        self.plot_choropleth(mapped, "black ratio",
                             f"Black Ratio by {loc_col} ({county_label})")
        self.export_interactive_map(
            mapped, "black ratio", self._out("black_ratio_map.html"),
            label_field=loc_col,
            extra_fields=["black_population", "overall_population"])
        return summary, mapped

    def _align_unemployment(self, align_df, loc_col, county_label):
        if self.unemployment_csv_df is None:
            raise ValueError("Unemployment CSV not loaded.")
        for c in ("in labor force", "unemployed"):
            if c not in self.unemployment_csv_df.columns:
                raise ValueError(f"Unemployment CSV missing '{c}'")

        filt = self.unemployment_csv_df[["GEOID", "in labor force", "unemployed"]]
        merged = align_df.merge(filt, on="GEOID", how="left")

        merged["labor_force_piece"] = merged["intersected_area_bg"] * merged["in labor force"]
        merged["unemployed_piece"] = merged["intersected_area_bg"] * merged["unemployed"]

        summary = (merged.groupby("police_beats", as_index=False)
                   .agg(labor_force=("labor_force_piece", "sum"),
                        unemployed=("unemployed_piece", "sum")))
        summary["unemployment rate"] = (
            summary["unemployed"] /
            summary["labor_force"].replace(0, np.nan)
        ).fillna(0.0)
        summary = summary.rename(columns={"police_beats": loc_col})

        mapped = self.custom_gdf.merge(
            summary[[loc_col, "labor_force", "unemployed",
                     "unemployment rate"]],
            on=loc_col, how="left")
        mapped["unemployment rate"] = mapped["unemployment rate"].fillna(0.0)

        merged.to_csv(self._out("alignment_unemployment_detailed.csv"), index=False)
        summary.to_csv(self._out("alignment_unemployment_by_location.csv"), index=False)
        mapped.to_file(self._out("alignment_unemployment_map.geojson"), driver="GeoJSON")
        self.log("Unemployment outputs saved.")

        self.plot_choropleth(mapped, "unemployment rate",
                             f"Unemployment Rate by {loc_col} ({county_label})")
        self.export_interactive_map(
            mapped, "unemployment rate", self._out("unemployment_rate_map.html"),
            label_field=loc_col,
            extra_fields=["labor_force", "unemployed"])
        return summary, mapped

    def _align_poverty(self, align_df, loc_col, county_label):
        if self.poverty_csv_df is None:
            raise ValueError("Poverty CSV not loaded.")
        for c in ("total", "below poverty level"):
            if c not in self.poverty_csv_df.columns:
                raise ValueError(f"Poverty CSV missing '{c}'")

        filt = self.poverty_csv_df[["GEOID", "total", "below poverty level"]]
        merged = align_df.merge(filt, on="GEOID", how="left")

        merged["total_piece"] = merged["intersected_area_bg"] * merged["total"]
        merged["below_poverty_piece"] = merged["intersected_area_bg"] * merged["below poverty level"]

        summary = (merged.groupby("police_beats", as_index=False)
                   .agg(total=("total_piece", "sum"),
                        below_poverty=("below_poverty_piece", "sum")))
        summary["poverty rate"] = (
            summary["below_poverty"] /
            summary["total"].replace(0, np.nan)
        ).fillna(0.0)
        summary = summary.rename(columns={"police_beats": loc_col})

        mapped = self.custom_gdf.merge(
            summary[[loc_col, "total", "below_poverty", "poverty rate"]],
            on=loc_col, how="left")
        mapped["poverty rate"] = mapped["poverty rate"].fillna(0.0)

        merged.to_csv(self._out("alignment_poverty_detailed.csv"), index=False)
        summary.to_csv(self._out("alignment_poverty_by_location.csv"), index=False)
        mapped.to_file(self._out("alignment_poverty_map.geojson"), driver="GeoJSON")
        self.log("Poverty outputs saved.")

        self.plot_choropleth(mapped, "poverty rate",
                             f"Poverty Rate by {loc_col} ({county_label})")
        self.export_interactive_map(
            mapped, "poverty rate", self._out("poverty_rate_map.html"),
            label_field=loc_col,
            extra_fields=["total", "below_poverty"])
        return summary, mapped

    # ==================================================================
    #  POI alignment  (hybrid: point-in-polygon + area-weighted overlay)
    # ==================================================================
    def _align_poi(self, loc_col, poi_cols):
        if self.poi_gdf is None:
            self._warning_requested.emit("Warning", "No POI data loaded.")
            return None

        # POI was already reprojected to default CRS at load time.
        # Custom shapefile was also reprojected at load time.
        # Both match self.default_gdf.crs — verify & fix just in case.
        poi = self.poi_gdf.copy()
        target_crs = self.custom_gdf.crs
        if poi.crs != target_crs:
            self.log(f"POI CRS {poi.crs} != custom CRS {target_crs} — reprojecting.")
            poi = poi.to_crs(target_crs)

        regions = self.custom_gdf[[loc_col, "geometry"]].copy()

        # ── Split POI into points and polygons ──
        is_point = poi.geometry.geom_type.isin(("Point", "MultiPoint"))
        poi_points = poi.loc[is_point].copy()
        poi_polys = poi.loc[~is_point].copy()

        self.log(f"POI split: {len(poi_points)} point(s), "
                 f"{len(poi_polys)} polygon(s)")

        # ── A) Points: standard spatial join ──
        point_records = []
        if not poi_points.empty:
            joined_pts = gpd.sjoin(poi_points, regions,
                                   how="inner", predicate="within")
            self.log(f"  Point sjoin: {len(joined_pts)} matched "
                     f"out of {len(poi_points)}")
            for region_id, grp in joined_pts.groupby(loc_col):
                row = {loc_col: region_id}
                for col in poi_cols:
                    nice = col.replace("category_", "") + "_count"
                    row[nice] = float(grp[col].sum())
                point_records.append(row)

        # ── B) Polygons: area-weighted overlay ──
        poly_records = []
        if not poi_polys.empty:
            # Fix any invalid geometries before overlay
            try:
                poi_polys["geometry"] = poi_polys.geometry.buffer(0)
                regions["geometry"] = regions.geometry.buffer(0)
            except Exception:
                pass

            # Add a unique POI id for grouping after overlay
            poi_polys = poi_polys.reset_index(drop=True)
            poi_polys["_poi_uid"] = range(len(poi_polys))

            # Compute each POI polygon's own area before overlay.
            # Areas are measured in a projected (metre) CRS - the data's
            # CRS is geographic (degrees), where .area is not meaningful.
            area_crs = regions.estimate_utm_crs()
            poi_polys["_poi_area"] = poi_polys.geometry.to_crs(area_crs).area

            # Overlay: find every intersection of POI polygons × regions
            try:
                ov = gpd.overlay(
                    poi_polys[["_poi_uid", "_poi_area", "geometry"] + poi_cols],
                    regions,
                    how="intersection",
                )
            except Exception as ex:
                self.log(f"  Polygon overlay failed ({ex}) — "
                         f"falling back to centroid sjoin.")
                # Fallback: use centroids if overlay fails
                poi_polys_c = poi_polys.copy()
                poi_polys_c["geometry"] = poi_polys_c.geometry.centroid
                joined_fb = gpd.sjoin(poi_polys_c, regions,
                                      how="inner", predicate="within")
                for region_id, grp in joined_fb.groupby(loc_col):
                    row = {loc_col: region_id}
                    for col in poi_cols:
                        nice = col.replace("category_", "") + "_count"
                        row[nice] = float(grp[col].sum())
                    poly_records.append(row)
                ov = None

            if ov is not None and not ov.empty:
                # fraction = intersection area / original POI area
                ov["_frac"] = (ov.geometry.to_crs(area_crs).area
                               / ov["_poi_area"].replace(0, np.nan))
                ov["_frac"] = ov["_frac"].clip(0, 1).fillna(0)

                n_matched = ov["_poi_uid"].nunique()
                self.log(f"  Polygon overlay: {n_matched} POI(s) matched "
                         f"({len(ov)} intersection pieces)")

                for region_id, grp in ov.groupby(loc_col):
                    row = {loc_col: region_id}
                    for col in poi_cols:
                        nice = col.replace("category_", "") + "_count"
                        # Weighted count: each POI contributes its overlap fraction
                        row[nice] = float((grp[col] * grp["_frac"]).sum())
                    poly_records.append(row)

        # ── C) Combine point + polygon counts per region ──
        all_records = point_records + poly_records
        if not all_records:
            self.log("No POI features matched any region.")
            # Return empty summary with all regions
            summary = pd.DataFrame(
                {loc_col: self.custom_gdf[loc_col].astype(str).values})
            for col in poi_cols:
                nice = col.replace("category_", "") + "_count"
                summary[nice] = 0
            summary.to_csv(self._out("alignment_poi_by_location.csv"), index=False)
            return summary

        raw = pd.DataFrame(all_records)
        count_cols = [c for c in raw.columns if c.endswith("_count")]
        # Sum across the two strategies (points + polys may share regions)
        summary = raw.groupby(loc_col, as_index=False)[count_cols].sum()
        summary[loc_col] = summary[loc_col].astype(str)

        all_regions = pd.DataFrame(
            {loc_col: self.custom_gdf[loc_col].astype(str).values})
        summary = all_regions.merge(summary, on=loc_col, how="left")
        summary[count_cols] = summary[count_cols].fillna(0)

        # Round to 2 decimals (area-weighted polygon POIs give fractional counts)
        for c in count_cols:
            summary[c] = summary[c].round(2)

        summary.to_csv(self._out("alignment_poi_by_location.csv"), index=False)
        self.log(f"POI summary saved ({len(summary)} regions, "
                 f"{len(count_cols)} categories)")

        if count_cols:
            first_col = count_cols[0]
            poi_mapped = self.custom_gdf.merge(
                summary[[loc_col, first_col]], on=loc_col, how="left")
            poi_mapped[first_col] = poi_mapped[first_col].fillna(0)
            self.plot_choropleth(poi_mapped, first_col,
                                 f"{first_col} by {loc_col}")
        return summary

    # ==================================================================
    #  Plotting helpers
    # ==================================================================
    def plot_geometry(self, gdf_or_df, title="Map"):
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_title(title)
            if isinstance(gdf_or_df, pd.DataFrame) and "geometry_pb" in gdf_or_df.columns:
                plot_gdf = gpd.GeoDataFrame(
                    gdf_or_df, geometry="geometry_pb", crs=self.default_gdf.crs)
                plot_gdf.plot(ax=ax, edgecolor="blue", alpha=0.5)
            elif isinstance(gdf_or_df, gpd.GeoDataFrame):
                gdf_or_df.plot(ax=ax, edgecolor="black", alpha=0.6)
            ax.set_axis_off()
            self.canvas.draw()
        except Exception as e:
            self.log(f"Plot error: {e}")

    def plot_choropleth(self, gdf, value_col, title="Choropleth"):
        """Thread-safe: drawing always happens on the UI thread."""
        self._plot_requested.emit(gdf.copy(), value_col, title)

    @pyqtSlot(object, str, str)
    def _draw_choropleth(self, gdf, value_col, title="Choropleth"):
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_title(title)
            gdf.plot(column=value_col, ax=ax, legend=True,
                     alpha=0.9, edgecolor="white", linewidth=0.5)
            ax.set_axis_off()
            self.canvas.draw()
        except Exception as e:
            self.log(f"Choropleth error: {e}")

    # ==================================================================
    #  Folium helper: Esri light-gray basemap
    #  (no API key needed; CARTO now requires a key and OSM's own tile
    #   servers block maps opened from a local file)
    # ==================================================================
    ESRI_TILES = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "Canvas/{layer}/MapServer/tile/{{z}}/{{y}}/{{x}}")
    ESRI_ATTR = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"

    def _make_folium_map(self, center, zoom=12):
        import folium
        m = folium.Map(location=center, zoom_start=zoom, tiles=None)
        folium.TileLayer(
            tiles=self.ESRI_TILES.format(layer="World_Light_Gray_Base"),
            attr=self.ESRI_ATTR, name="Esri Light Gray",
            max_native_zoom=16, max_zoom=19,   # tiles exist up to z16
        ).add_to(m)
        # place / road labels (tiles render beneath the data polygons,
        # visible through their semi-transparent fill)
        folium.TileLayer(
            tiles=self.ESRI_TILES.format(layer="World_Light_Gray_Reference"),
            attr=self.ESRI_ATTR, name="Labels", overlay=True, control=False,
            max_native_zoom=16, max_zoom=19,
        ).add_to(m)
        return m

    # ==================================================================
    #  Interactive map — single layer
    # ==================================================================
    def export_interactive_map(self, mapped_gdf, value_col, out_path,
                               label_field=None, extra_fields=None):
        import folium
        from folium.features import GeoJson, GeoJsonTooltip

        if extra_fields is None:
            extra_fields = []
        if mapped_gdf is None or mapped_gdf.empty:
            return
        if value_col not in mapped_gdf.columns:
            return

        try:
            gdf = mapped_gdf.copy()
            if gdf.crs is None:
                gdf = gdf.set_crs(self.default_gdf.crs, allow_override=True)
            gdf = gdf.to_crs(epsg=4326)
            try:
                gdf["geometry"] = gdf.geometry.buffer(0)
            except Exception:
                pass
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
            if gdf.empty:
                return

            wanted = {value_col}
            for c in extra_fields:
                if c in gdf.columns:
                    wanted.add(c)
            if label_field and label_field in gdf.columns:
                wanted.add(label_field)

            cols = [c for c in gdf.columns
                    if c != gdf.geometry.name and c in wanted]
            gdf_props = gdf[cols].copy()
            for c in gdf_props.columns:
                if pd.api.types.is_datetime64_any_dtype(gdf_props[c]):
                    gdf_props[c] = gdf_props[c].astype(str)
                elif pd.api.types.is_numeric_dtype(gdf_props[c]):
                    gdf_props[c] = pd.to_numeric(
                        gdf_props[c], errors="coerce").astype(float)
                else:
                    gdf_props[c] = gdf_props[c].astype(str)
            gdf_props = gdf_props.reset_index(drop=True)
            gdf_props = gpd.GeoDataFrame(gdf_props, geometry=gdf.geometry,
                                          crs=gdf.crs)

            minx, miny, maxx, maxy = gdf_props.total_bounds
            center = [(miny + maxy) / 2, (minx + maxx) / 2]
            m = self._make_folium_map(center)

            def style_fn(feature):
                try:
                    v = float(feature["properties"].get(value_col, 0) or 0)
                except Exception:
                    v = 0.0
                v = max(0.0, min(1.0, v))
                shade = int(round(v * 255))
                return {"fillColor": f"#{shade:02x}00{255 - shade:02x}",
                        "color": "white", "weight": 0.5, "fillOpacity": 0.8}

            tt_f, tt_a = [], []
            if label_field and label_field in gdf_props.columns:
                tt_f.append(label_field); tt_a.append("Location:")
            tt_f.append(value_col); tt_a.append(f"{value_col}:")
            for c in extra_fields:
                if c in gdf_props.columns:
                    tt_f.append(c); tt_a.append(c + ":")

            GeoJson(
                data=gdf_props.to_json(), name=value_col,
                style_function=style_fn,
                tooltip=GeoJsonTooltip(fields=tt_f, aliases=tt_a,
                                       localize=True),
            ).add_to(m)
            folium.LayerControl().add_to(m)
            m.save(out_path)
            self.log(f"Map saved: {out_path}")
        except Exception as ex:
            self.log(f"Interactive map error: {ex}")

    # ==================================================================
    #  Interactive map — combined multi-layer
    # ==================================================================
    def export_interactive_combo_map(self, base_gdf, layers, out_path,
                                     loc_field=None):
        import folium
        from folium.features import GeoJson, GeoJsonTooltip

        try:
            if base_gdf is None or base_gdf.empty:
                return

            bg = base_gdf.copy()
            if bg.crs is None:
                bg = bg.set_crs(self.default_gdf.crs, allow_override=True)
            bg = bg.to_crs(epsg=4326)
            try:
                bg["geometry"] = bg.geometry.buffer(0)
            except Exception:
                pass
            bg = bg[~bg.geometry.is_empty & bg.geometry.notna()].copy()

            keep = []
            if loc_field and loc_field in bg.columns:
                keep.append(loc_field)
            bg_props = gpd.GeoDataFrame(bg[keep].copy(),
                                         geometry=bg.geometry, crs=bg.crs)
            for c in keep:
                bg_props[c] = bg_props[c].astype(str)
            bg_props = bg_props.reset_index(drop=True)

            minx, miny, maxx, maxy = bg_props.total_bounds
            center = [(miny + maxy) / 2, (minx + maxx) / 2]
            m = self._make_folium_map(center)

            folium.GeoJson(
                data=bg_props.to_json(),
                name="Basemap (custom regions)",
                style_function=lambda f: {
                    "fillColor": "#00000000", "color": "black",
                    "weight": 1.0, "fillOpacity": 0.0},
            ).add_to(m)

            def _add(layer):
                gdf = layer["gdf"].copy()
                val = layer["value_col"]
                name = layer.get("name", val)
                extra = [c for c in layer.get("tooltip_extra", [])
                         if c in gdf.columns]

                if gdf.crs is None:
                    gdf = gdf.set_crs(self.default_gdf.crs,
                                       allow_override=True)
                gdf = gdf.to_crs(epsg=4326)
                try:
                    gdf["geometry"] = gdf.geometry.buffer(0)
                except Exception:
                    pass
                gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
                if gdf.empty:
                    return

                wanted = {val}
                if loc_field and loc_field in gdf.columns:
                    wanted.add(loc_field)
                for c in extra:
                    wanted.add(c)
                cols = [c for c in gdf.columns
                        if c != gdf.geometry.name and c in wanted]
                gp = gdf[cols].copy()
                for c in gp.columns:
                    if pd.api.types.is_datetime64_any_dtype(gp[c]):
                        gp[c] = gp[c].astype(str)
                    elif pd.api.types.is_numeric_dtype(gp[c]):
                        gp[c] = pd.to_numeric(
                            gp[c], errors="coerce").astype(float)
                    else:
                        gp[c] = gp[c].astype(str)
                gp = gp.reset_index(drop=True)
                gp = gpd.GeoDataFrame(gp, geometry=gdf.geometry, crs=gdf.crs)

                def style_fn(feature):
                    try:
                        v = float(feature["properties"].get(val, 0) or 0)
                    except Exception:
                        v = 0.0
                    v = max(0.0, min(1.0, v))
                    shade = int(round(v * 255))
                    return {"fillColor": f"#{shade:02x}00{255 - shade:02x}",
                            "color": "white", "weight": 0.5,
                            "fillOpacity": 0.8}

                tt_f, tt_a = [], []
                if loc_field and loc_field in gp.columns:
                    tt_f.append(loc_field); tt_a.append("Location:")
                tt_f.append(val); tt_a.append(f"{val}:")
                for c in extra:
                    if c in gp.columns:
                        tt_f.append(c); tt_a.append(c + ":")

                GeoJson(
                    data=gp.to_json(), name=name,
                    style_function=style_fn,
                    tooltip=GeoJsonTooltip(fields=tt_f, aliases=tt_a,
                                           localize=True),
                ).add_to(m)

            for layer in layers:
                _add(layer)

            folium.LayerControl().add_to(m)
            m.save(out_path)
            self.log(f"Combined map saved: {out_path}")
            return out_path
        except Exception as ex:
            self.log(f"Combo map error: {ex}")
            return None

    # ==================================================================
    #  Help
    # ==================================================================
    def show_help(self):
        QMessageBox.information(
            self, "Help",
            "Workflow\n"
            "────────\n"
            "1. Upload your custom shapefile → pick location & geometry columns.\n"
            "2. (Optional) Check Census features (Race / Unemployment / Poverty).\n"
            "   The counties your regions cover are detected automatically\n"
            "   and listed for information.\n"
            "3. (Optional) Upload a POI dataset (.gpkg / .geojson / .shp)\n"
            "   whose columns start with 'category_'.  Check desired categories.\n"
            "   Use tools/osm_poi_extract.py to generate a compatible .gpkg from .osm.pbf.\n"
            "4. Click 'Run Alignment'.\n\n"
            "How it works\n"
            "────────────\n"
            "• Census features use area-overlap between your regions and census\n"
            "  block groups to distribute population counts proportionally.\n"
            "  Block groups are matched by GEOID, so regions spanning several\n"
            "  counties are handled automatically.\n"
            "• POI point features use a point-in-polygon spatial join — each POI\n"
            "  point is assigned to whichever custom region contains it.\n"
            "• POI polygon/multipolygon features (parks, malls, buildings) use an\n"
            "  area-weighted overlay — if a park spans two regions, its count is\n"
            "  split proportionally by overlap area (same logic as census).\n"
            "• All three datasets (census, custom shapefile, POI) are automatically\n"
            "  reprojected to the same CRS before alignment.\n\n"
            f"Outputs  (written to {self.output_dir})\n"
            "───────\n"
            "• geometry_alignment.csv — census overlap matrix.\n"
            "• alignment_race_*.csv / .geojson / .html\n"
            "• alignment_unemployment_*.csv / .geojson / .html\n"
            "• alignment_poverty_*.csv / .geojson / .html\n"
            "• alignment_poi_by_location.csv — POI counts per region.\n"
            "• alignment_combined_by_location.csv — all selected features.\n"
            "• context_layers_map.html — interactive map with layer toggles.\n"
        )


# ═══════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Align Census block-group and POI features to custom regions.")
    ap.add_argument("--block-groups", default=str(DEFAULT_BLOCK_GROUPS),
                    help="TIGER/Line block-group shapefile (default: bundled Maryland 2020).")
    ap.add_argument("--race-csv", default=str(DEFAULT_RACE_CSV),
                    help="Decennial P1 table CSV (data.census.gov export).")
    ap.add_argument("--unemployment-csv", default=str(DEFAULT_UNEMPLOYMENT_CSV),
                    help="ACS B23025 table CSV.")
    ap.add_argument("--poverty-csv", default=str(DEFAULT_POVERTY_CSV),
                    help="ACS B17017 table CSV.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="Folder for generated CSV / GeoJSON / HTML files.")
    # parse_known_args leaves Qt's own options (e.g. -platform) untouched
    args, _ = ap.parse_known_args(argv)
    return args


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    try:
        window = ShapeComparerUI(
            block_groups_path=args.block_groups,
            race_csv_path=args.race_csv,
            unemployment_csv_path=args.unemployment_csv,
            poverty_csv_path=args.poverty_csv,
            output_dir=args.output_dir,
        )
    except Exception as e:
        QMessageBox.critical(None, "Startup error", str(e))
        sys.exit(1)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
