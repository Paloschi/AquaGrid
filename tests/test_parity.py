"""Parity tests: AquaGrid compiled kernels vs AquaCrop-OSPy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacrop import AquaCropModel, Crop, InitialWaterContent, Soil

from aquagrid.engine.run import run_grid_arrays
from aquagrid.kernels import constants as C
from aquagrid.params import (
    co2_concentration_for_year,
    crop_params_array,
    initial_water_content,
    soil_params,
)


def run_reference(weather_df: pd.DataFrame, crop_name: str, soil_name: str,
                  planting: str):
    """Run original aquacrop from planting date to end of weather."""
    start = pd.Timestamp(planting)
    end = weather_df.Date.iloc[-1]
    crop = Crop(crop_name, planting_date=start.strftime("%m/%d"))
    model = AquaCropModel(
        sim_start_time=start.strftime("%Y/%m/%d"),
        sim_end_time=end.strftime("%Y/%m/%d"),
        weather_df=weather_df,
        soil=Soil(soil_name),
        crop=crop,
        initial_water_content=InitialWaterContent(value=["FC"]),
    )
    model.run_model(till_termination=True)
    return model


def run_grid_kernel(weather_df: pd.DataFrame, crop_name: str, soil_name: str,
                    planting: str):
    """Run AquaGrid kernels for a single pixel starting at ``planting``."""
    start = pd.Timestamp(planting)
    plant_idx = int((weather_df.Date == start).idxmax())
    # weather from planting onward, mirroring the reference clock
    sub = weather_df.iloc[plant_idx:].reset_index(drop=True)
    nt = len(sub)

    crop = Crop(crop_name, planting_date=start.strftime("%m/%d"))
    soil = Soil(soil_name)
    co2 = co2_concentration_for_year(start.year)
    cp = crop_params_array(crop, co2_conc=co2)
    sp, prof = soil_params(soil, zmax=crop.Zmax)
    thini = initial_water_content(soil, "FC")

    fin, daily = run_grid_arrays(
        tmin=sub.MinTemp.to_numpy()[:, None],
        tmax=sub.MaxTemp.to_numpy()[:, None],
        prcp=sub.Precipitation.to_numpy()[:, None],
        et0=sub.ReferenceET.to_numpy()[:, None],
        plant_idx=np.array([0], np.int64),
        cp=cp, sp=sp, profile=prof, th_init=thini,
        save_daily=True, parallel=False,
    )
    return fin, daily, nt


CASES = [
    ("Maize", "SandyLoam", "2019/05/15"),      # CalendarType 1
    ("MaizeGDD", "SandyLoam", "2019/05/15"),   # CalendarType 2
    ("SoybeanGDD", "ClayLoam", "2019/06/01"),  # CalendarType 2, other soil
    ("Maize", "Clay", "2019/10/01"),           # different season window
]


@pytest.mark.parametrize("crop_name,soil_name,planting", CASES)
def test_parity_vs_aquacrop(weather_df, crop_name, soil_name, planting):
    model = run_reference(weather_df, crop_name, soil_name, planting)
    fin, daily, nt = run_grid_kernel(weather_df, crop_name, soil_name, planting)

    assert fin[0, C.OF_STATUS] == C.STATUS_OK

    # --- daily series (reference rows are relative to sim start = planting)
    growth = model._outputs.crop_growth
    n_days = int(fin[0, C.OF_DAY_END]) + 1
    ref_cc = growth["canopy_cover"].to_numpy()[:n_days]
    ref_biomass = growth["biomass"].to_numpy()[:n_days]
    ref_zroot = growth["z_root"].to_numpy()[:n_days]
    ref_gddcum = growth["gdd_cum"].to_numpy()[:n_days]

    got_cc = daily[C.OD_CANOPY_COVER, :n_days, 0]
    got_biomass = daily[C.OD_BIOMASS, :n_days, 0]
    got_zroot = daily[C.OD_Z_ROOT, :n_days, 0]
    got_gddcum = daily[C.OD_GDD_CUM, :n_days, 0]

    np.testing.assert_allclose(got_gddcum, ref_gddcum, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(got_zroot, ref_zroot, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_cc, ref_cc, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_biomass, ref_biomass, rtol=1e-12, atol=1e-10)

    # --- final yield
    stats = model._outputs.final_stats
    ref_dry = float(stats["Dry yield (tonne/ha)"].iloc[0])
    ref_fresh = float(stats["Fresh yield (tonne/ha)"].iloc[0])
    assert fin[0, C.OF_DRY_YIELD] == pytest.approx(ref_dry, rel=1e-12)
    assert fin[0, C.OF_FRESH_YIELD] == pytest.approx(ref_fresh, rel=1e-12)
