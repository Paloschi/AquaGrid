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
                  planting: str, *, crop=None, soil=None, initial_wc=None):
    """Run original aquacrop from planting date to end of weather.

    ``crop`` / ``soil`` override the named presets so a raster profile or a
    cube-adjusted parameter can be checked against the same model.
    ``initial_wc`` defaults to field capacity on layer 1, which covers a
    preset soil (one layer). A multi-layer soil must list every layer.
    """
    start = pd.Timestamp(planting)
    end = weather_df.Date.iloc[-1]
    if crop is None:
        crop = Crop(crop_name, planting_date=start.strftime("%m/%d"))
    if soil is None:
        soil = Soil(soil_name)
    if initial_wc is None:
        initial_wc = InitialWaterContent(value=["FC"])
    model = AquaCropModel(
        sim_start_time=start.strftime("%Y/%m/%d"),
        sim_end_time=end.strftime("%Y/%m/%d"),
        weather_df=weather_df,
        soil=soil,
        crop=crop,
        initial_water_content=initial_wc,
    )
    model.run_model(till_termination=True)
    return model


def assert_ospy_parity(model, fin, daily, pixel: int = 0) -> None:
    """Every published final and daily field matches AquaCrop-OSPy at 1e-12."""
    assert fin[pixel, C.OF_STATUS] == C.STATUS_OK

    growth = model._outputs.crop_growth
    flux = model._outputs.water_flux
    n_days = int(fin[pixel, C.OF_DAY_END]) + 1
    last = n_days - 1

    daily_pairs = (
        (C.OD_GDD_CUM, growth["gdd_cum"].to_numpy()[:n_days], 1e-10),
        (C.OD_Z_ROOT, growth["z_root"].to_numpy()[:n_days], 1e-12),
        (C.OD_CANOPY_COVER, growth["canopy_cover"].to_numpy()[:n_days], 1e-12),
        (C.OD_BIOMASS, growth["biomass"].to_numpy()[:n_days], 1e-10),
        (C.OD_ES, flux["Es"].to_numpy()[:n_days], 1e-12),
        (C.OD_TR, flux["Tr"].to_numpy()[:n_days], 1e-12),
        (C.OD_WR, flux["Wr"].to_numpy()[:n_days], 1e-12),
    )
    for od, ref, atol in daily_pairs:
        np.testing.assert_allclose(
            daily[od, :n_days, pixel], ref, rtol=1e-12, atol=atol)

    stats = model._outputs.final_stats
    finals = (
        (C.OF_DRY_YIELD, float(stats["Dry yield (tonne/ha)"].iloc[0]), 1e-12),
        (C.OF_FRESH_YIELD, float(stats["Fresh yield (tonne/ha)"].iloc[0]), 1e-12),
        (C.OF_YIELD_POT, float(stats["Yield potential (tonne/ha)"].iloc[0]), 1e-12),
        (C.OF_BIOMASS, float(growth["biomass"].iloc[last]), 1e-10),
        (C.OF_BIOMASS_NS, float(growth["biomass_ns"].iloc[last]), 1e-10),
        (C.OF_HI_ADJ, float(growth["harvest_index_adj"].iloc[last]), 1e-12),
    )
    for of, ref, atol in finals:
        assert fin[pixel, of] == pytest.approx(ref, rel=1e-12, abs=atol)
    # Season length in days after planting, published as dap_end.
    assert int(fin[pixel, C.OF_DAP_END]) == int(growth["dap"].iloc[last])


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

    assert_ospy_parity(model, fin, daily)
