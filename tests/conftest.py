from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_weather(start: str, days: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic but realistic daily weather series (subtropical-ish)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    doy = dates.dayofyear.to_numpy()
    season = np.sin((doy - 105) / 365 * 2 * np.pi)
    tmin = 14 + 6 * season + rng.normal(0, 2, days)
    tmax = tmin + 9 + rng.normal(0, 1.5, days)
    rain_prob = 0.25 + 0.15 * season
    prcp = np.where(rng.random(days) < rain_prob, rng.gamma(2.0, 6.0, days), 0.0)
    et0 = np.clip(3.5 + 1.5 * season + rng.normal(0, 0.6, days), 0.3, None)
    return pd.DataFrame(
        {
            "MinTemp": tmin,
            "MaxTemp": tmax,
            "Precipitation": prcp,
            "ReferenceET": et0,
            "Date": dates,
        }
    )


@pytest.fixture(scope="session")
def weather_df() -> pd.DataFrame:
    return make_weather("2019/05/01", 720, seed=42)
