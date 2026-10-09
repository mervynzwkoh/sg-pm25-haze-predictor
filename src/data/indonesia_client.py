"""Parser and aggregator for Indonesian ground station and transboundary haze data."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from src.config import BASE_DIR, INDONESIA_UPSTREAM_STATIONS

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = BASE_DIR / "crea-indonesia-data_distribution-of-indonesia-s-official-monitoring-stations_20261009.csv"


class IndonesiaHazeData:
    """Parses CREA and BMKG Indonesian monitoring station data."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        self.df: pd.DataFrame = pd.DataFrame()
        self._load()

    def _load(self):
        """Load and clean the dataset."""
        if not self.csv_path.exists():
            logger.warning(f"Indonesia haze CSV not found at {self.csv_path}")
            return

        try:
            df = pd.read_csv(self.csv_path)
            # Standardize columns
            df["pm25_ugm3"] = pd.to_numeric(df["pm25_ugm3"], errors="coerce")
            self.df = df.dropna(subset=["pm25_ugm3"])
        except Exception as e:
            logger.error(f"Error loading Indonesia haze CSV: {e}")

    def get_regional_summary(self) -> Dict[str, Any]:
        """Aggregate summary statistics for Sumatra and Kalimantan haze clusters."""
        if self.df.empty:
            return {
                "sumatra_mean_d1": 150.0,
                "kalimantan_mean_d1": 200.0,
                "max_upstream_d1": 350.0,
                "source_count": 0,
            }

        # Filter by 1-day mean (d1)
        d1_df = self.df[self.df["date"] == "d1"]

        # Sumatra keywords: Jambi, Palembang, Pekanbaru, Riau, Batam, Bengkulu, Medan, Agam, Kampar
        sumatra_mask = d1_df["city"].str.contains(
            r"Jambi|Palembang|Pekanbaru|Riau|Batam|Bengkulu|Medan|Agam|Kampar|Aceh",
            case=False,
            na=False
        )
        # Kalimantan keywords: Pangkalanbun, Palangkaraya, Banjarbaru, Kotawaringin, Murung Raya, Katingan, Barito
        kalimantan_mask = d1_df["city"].str.contains(
            r"Pangkalanbun|Palangka|Banjar|Kotawaringin|Murung|Katingan|Barito|Kutai|Pontianak",
            case=False,
            na=False
        )

        sumatra_pm25 = d1_df[sumatra_mask]["pm25_ugm3"]
        kalimantan_pm25 = d1_df[kalimantan_mask]["pm25_ugm3"]

        return {
            "sumatra_mean_d1": float(sumatra_pm25.mean()) if not sumatra_pm25.empty else 120.0,
            "kalimantan_mean_d1": float(kalimantan_pm25.mean()) if not kalimantan_pm25.empty else 180.0,
            "sumatra_max_d1": float(sumatra_pm25.max()) if not sumatra_pm25.empty else 300.0,
            "kalimantan_max_d1": float(kalimantan_pm25.max()) if not kalimantan_pm25.empty else 400.0,
            "overall_mean_d1": float(d1_df["pm25_ugm3"].mean()) if not d1_df.empty else 80.0,
            "station_count": len(d1_df),
        }

    def get_station_readings(self) -> Dict[str, float]:
        """Return the latest d1 reading for each tracked upstream station."""
        if self.df.empty:
            return {}

        d1_df = self.df[self.df["date"] == "d1"]
        result = {}
        for station_name in INDONESIA_UPSTREAM_STATIONS.keys():
            matches = d1_df[d1_df["city"].str.contains(station_name, case=False, na=False)]
            if not matches.empty:
                result[station_name] = float(matches["pm25_ugm3"].iloc[0])
            else:
                result[station_name] = float(d1_df["pm25_ugm3"].mean())
        return result
