from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TASK_CLASSIFICATION = "rain_or_not"
TASK_REGRESSION = "precipitation_fall"

CLASSIFICATION_TARGET = "target_will_rain_in_7d"
REGRESSION_TARGET = "target_precip_3d_next"

DEFAULT_CLASSIFICATION_THRESHOLD = 0.35

LOCATION_NAME = "Sydney, Australia"
LATITUDE = -33.8678
LONGITUDE = 151.2073
TIMEZONE = "Australia/Sydney"
SYDNEY_TZ = ZoneInfo(TIMEZONE)

START_DATE = "2000-01-01"
CURRENT_WEATHER_FILE = "sydney_daily_current.csv"
