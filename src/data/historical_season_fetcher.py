"""Historical multi-year haze season and baseline dataset fetcher.

Harvests 2022-2026 Southwest Monsoon haze episodes and clean inter-monsoon baseline periods
from Open-Meteo CAMS reanalysis, Open-Meteo weather archives, and Singapore ground data.
Caches processed seasons on disk under data/raw/historical_seasons/ to avoid re-fetching.
"""
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import requests

from src.config import DATA_DIR, RAW_DATA_DIR, SG_REGIONS, SG_CENTER

logger = logging.getLogger(__name__)

SEASONS_DIR = RAW_DATA_DIR / "historical_seasons"
SEASONS_DIR.mkdir(parents=True, exist_ok=True)

MULTI_YEAR_CSV = RAW_DATA_DIR / "multi_year_haze_seasons_2022_2026.csv"

# Pre-defined target episodes:
# Captures acute haze episodes (August-October) and clean inter-monsoon baselines (December-January)
SEASON_WINDOWS = [
    {"name": "2022_haze_season", "start": "2022-08-01", "end": "2022-10-31", "type": "haze_season"},
    {"name": "2022_2023_baseline", "start": "2022-12-01", "end": "2023-01-31", "type": "baseline"},
    {"name": "2023_haze_season", "start": "2023-08-01", "end": "2023-10-31", "type": "haze_season"},
    {"name": "2023_2024_baseline", "start": "2023-12-01", "end": "2024-01-31", "type": "baseline"},
    {"name": "2024_haze_season", "start": "2024-08-01", "end": "2024-10-31", "type": "haze_season"},
    {"name": "2024_2025_baseline", "start": "2024-12-01", "end": "2025-01-31", "type": "baseline"},
    {"name": "2025_haze_season", "start": "2025-08-01", "end": "2025-10-31", "type": "haze_season"},
    {"name": "2026_haze_season", "start": "2026-08-01", "end": "2026-10-09", "type": "haze_season"},
]


class HistoricalSeasonFetcher:
    """Manages downloading, aligning, and caching seasonal air quality & meteorological history."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_weather_archive(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch surface weather variables for Singapore center."""
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": SG_CENTER["lat"],
            "longitude": SG_CENTER["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure",
            "timezone": "Asia/Singapore",
        }
        try:
            r = self.session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json().get("hourly", {})
            df = pd.DataFrame(data)
            if "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize("Asia/Singapore")
                df = df.drop(columns=["time"])
            return df
        except Exception as e:
            logger.error(f"Weather archive fetch failed for {start_date} to {end_date}: {e}")
            return pd.DataFrame()

    def fetch_upstream_palembang_archive(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch upstream smoke corridor tracers from Palembang, South Sumatra."""
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": -2.8986,
            "longitude": 104.7014,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "pm2_5,carbon_monoxide,aerosol_optical_depth",
            "timezone": "Asia/Singapore",
        }
        try:
            r = self.session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json().get("hourly", {})
            df = pd.DataFrame(data)
            if "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize("Asia/Singapore")
                df = df.drop(columns=["time"])
            df = df.rename(columns={
                "pm2_5": "palembang_pm25",
                "carbon_monoxide": "palembang_co",
                "aerosol_optical_depth": "palembang_aod",
            })
            return df
        except Exception as e:
            logger.error(f"Palembang archive fetch failed for {start_date} to {end_date}: {e}")
            return pd.DataFrame()

    def fetch_singapore_regional_pm25(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch regional PM2.5 across Singapore's five planning regions."""
        base_df = pd.DataFrame()
        for reg_key, info in SG_REGIONS.items():
            url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            params = {
                "latitude": info["lat"],
                "longitude": info["lon"],
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "pm2_5",
                "timezone": "Asia/Singapore",
            }
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json().get("hourly", {})
                    reg_df = pd.DataFrame(data)
                    if "time" in reg_df.columns:
                        reg_df["timestamp"] = pd.to_datetime(reg_df["time"]).dt.tz_localize("Asia/Singapore")
                        reg_df = reg_df.drop(columns=["time"])
                    reg_df = reg_df.rename(columns={"pm2_5": f"pm25_{reg_key}"})
                    if base_df.empty:
                        base_df = reg_df
                    else:
                        base_df = pd.merge(base_df, reg_df, on="timestamp", how="outer")
                time.sleep(0.15)
            except Exception as e:
                logger.warning(f"Failed to fetch {reg_key} for {start_date} to {end_date}: {e}")

        return base_df

    def fetch_and_cache_season(self, season_info: Dict[str, str]) -> pd.DataFrame:
        """Fetch or load cached season data."""
        season_name = season_info["name"]
        cache_path = SEASONS_DIR / f"{season_name}.csv"

        if cache_path.exists():
            logger.info(f"Loading cached season {season_name} from {cache_path}")
            df = pd.read_csv(cache_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        start_date = season_info["start"]
        end_date = season_info["end"]
        logger.info(f"Fetching season {season_name} ({start_date} to {end_date})...")

        w_df = self.fetch_weather_archive(start_date, end_date)
        aq_df = self.fetch_singapore_regional_pm25(start_date, end_date)
        sm_df = self.fetch_upstream_palembang_archive(start_date, end_date)

        if w_df.empty:
            logger.error(f"Failed to obtain weather for season {season_name}")
            return pd.DataFrame()

        df_merged = w_df.copy()
        if not aq_df.empty:
            df_merged = pd.merge(df_merged, aq_df, on="timestamp", how="left")
        if not sm_df.empty:
            df_merged = pd.merge(df_merged, sm_df, on="timestamp", how="left")

        df_merged["season_name"] = season_name
        df_merged["season_type"] = season_info["type"]

        # If 2026 season, overlay ground truth readings from sg_pm25_haze_season_2026.csv if available
        gt_2026_file = RAW_DATA_DIR / "sg_pm25_haze_season_2026.csv"
        if season_name == "2026_haze_season" and gt_2026_file.exists():
            try:
                df_gt = pd.read_csv(gt_2026_file)
                df_gt["timestamp"] = pd.to_datetime(df_gt["timestamp"])
                for reg in SG_REGIONS.keys():
                    col = f"pm25_{reg}"
                    gt_map = dict(zip(df_gt["timestamp"], df_gt[col]))
                    df_merged[col] = df_merged["timestamp"].map(gt_map).fillna(df_merged[col])
                logger.info("Successfully overlaid 2026 ground truth readings onto 2026 season.")
            except Exception as e:
                logger.warning(f"Could not overlay 2026 ground truth: {e}")

        # Ensure all regional PM2.5 columns exist
        for reg in SG_REGIONS.keys():
            col = f"pm25_{reg}"
            if col not in df_merged.columns:
                df_merged[col] = 25.0
            else:
                df_merged[col] = df_merged[col].ffill().bfill().fillna(25.0)

        df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
        df_merged.to_csv(cache_path, index=False)
        logger.info(f"Season {season_name} saved ({len(df_merged)} rows) to {cache_path}")
        return df_merged

    def build_multi_year_dataset(self, force_refresh: bool = False) -> pd.DataFrame:
        """Build and merge all target historical seasons into a single multi-year dataset."""
        if MULTI_YEAR_CSV.exists() and not force_refresh:
            logger.info(f"Multi-year dataset already exists at {MULTI_YEAR_CSV}")
            df = pd.read_csv(MULTI_YEAR_CSV)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        dfs = []
        for s_info in SEASON_WINDOWS:
            season_df = self.fetch_and_cache_season(s_info)
            if not season_df.empty:
                dfs.append(season_df)
            time.sleep(0.5)

        if not dfs:
            raise RuntimeError("No historical seasonal data could be fetched.")

        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        combined_df.to_csv(MULTI_YEAR_CSV, index=False)
        logger.info(f"Successfully assembled multi-year dataset: {len(combined_df)} rows at {MULTI_YEAR_CSV}")
        return combined_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fetcher = HistoricalSeasonFetcher()
    df = fetcher.build_multi_year_dataset()
    print(f"Total rows: {len(df)}")
    print("Seasons captured:\n", df["season_name"].value_counts())
