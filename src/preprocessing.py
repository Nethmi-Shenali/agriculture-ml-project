"""
Data preprocessing for Sri Lanka cereal production prediction.
Supports both wide-format CSV and World Bank long-format (Indicator Name, Year, Value).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "Fertilizer consumption",
    "Arable land",
    "Average precipitation in depth",
    "Employment in agriculture",
    "Land under cereal production",
    "Rural population",
    "Crop production index",
    "Food production index",
]
TARGET_COLUMN = "Cereal production"

# Map World Bank indicator names (long format) to our column names
WORLD_BANK_INDICATOR_MAP = {
    "Fertilizer consumption (kilograms per hectare of arable land)": "Fertilizer consumption",
    "Arable land (hectares)": "Arable land",
    "Average precipitation in depth (mm per year)": "Average precipitation in depth",
    "Employment in agriculture (% of total employment) (modeled ILO estimate)": "Employment in agriculture",
    "Land under cereal production (hectares)": "Land under cereal production",
    "Rural population": "Rural population",
    "Crop production index (2014-2016 = 100)": "Crop production index",
    "Food production index (2014-2016 = 100)": "Food production index",
    "Cereal production (metric tons)": "Cereal production",
}


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    If file is World Bank long-format (Country, Year, Indicator Name, Value), pivot to wide format.
    """
    skip = [1] if _has_comment_row(path) else None
    df = pd.read_csv(path, skiprows=skip, encoding="utf-8")
    # Drop rows that are clearly not data (e.g. comment rows)
    if "Value" in df.columns and df["Value"].dtype == object:
        df = df[pd.to_numeric(df["Value"], errors="coerce").notna()]
        df["Value"] = pd.to_numeric(df["Value"])

    if "Indicator Name" in df.columns and "Year" in df.columns and "Value" in df.columns:
        return _pivot_world_bank_long(df)
    return df


def _has_comment_row(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            next(f)
            second = next(f)
            return second.strip().startswith("#")
    except Exception:
        return False


def _pivot_world_bank_long(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format World Bank data to one row per Year with our column names."""
    needed = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    name_to_short = {k: v for k, v in WORLD_BANK_INDICATOR_MAP.items() if v in needed}
    df = df[df["Indicator Name"].isin(name_to_short)].copy()
    df["ShortName"] = df["Indicator Name"].map(name_to_short)
    pivot = df.pivot_table(index="Year", columns="ShortName", values="Value", aggfunc="first")
    pivot = pivot.reset_index()
    return pivot


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Clean data, select features, scale, and split.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔥 Convert back to DataFrame to preserve feature names
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler, FEATURE_COLUMNS