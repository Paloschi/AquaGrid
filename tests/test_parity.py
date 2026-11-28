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
    default_field_params,
    default_irrigation_params,
    field_params_array,
    initial_water_content,
    irrigation_params_array,
    soil_params,
)


def _ks_cold(crop, gdd, growing_season):
    """Cold-stress coefficient on transpiration, as AquaCrop-OSPy computes it."""
    if not growing_season:
        return np.nan
    if crop.TrColdStress == 0:
        return 1.0
    if gdd >= crop.GDD_up:
        return 1.0
    if gdd <= crop.GDD_lo:
        return 0.0
    ks_up = 1.0
    ks_lo = 0.02
    fshape = (-1) * np.log(
        ((ks_lo * ks_up) - 0.98 * ks_lo) / (0.98 * (ks_up - ks_lo)))
    gdd_rel = (gdd - crop.GDD_lo) / (crop.GDD_up - crop.GDD_lo)
    ks = (ks_up * ks_lo) / (ks_lo + (ks_up - ks_lo) * np.exp(-fshape * gdd_rel))
    return float(ks - ks_lo * (1 - gdd_rel))


class _StressTrace:
    """Record the stress coefficients AquaCrop-OSPy applies each day.

    Water-stress coefficients are not one vector: expansion and senescence
    come from the canopy step, stomatal stress from transpiration, and
    pollination stress from the harvest-index step after extraction.
    """

    def __init__(self):
        self.ksw_exp = []
        self.ksw_sen = []
        self.ksw_sto = []
        self.ksw_pol = []
        self.kst_polh = []
        self.kst_polc = []
        self.ks_cold = []
        self._canopy = []
        self._sto = []
        self._pol = []
        self._heat = []
        self._saved = {}

    def install(self):
        import aquacrop.solution.canopy_cover as cc
        import aquacrop.solution.harvest_index as hi
        import aquacrop.solution.transpiration as tr
        import aquacrop.timestep.run_single_timestep as step

        self._saved = {
            "cc_ws": cc.water_stress,
            "tr_ws": tr.water_stress,
            "hi_ws": hi.water_stress,
            "hi_ts": hi.temperature_stress,
            "step_cc": step.canopy_cover,
            "step_tr": step.transpiration,
            "step_hi": step.harvest_index,
        }
        trace = self

        def cc_ws(*args, **kwargs):
            out = trace._saved["cc_ws"](*args, **kwargs)
            beta = kwargs["beta"] if "beta" in kwargs else args[-1]
            if beta:
                trace._canopy.append(out)
            return out

        def canopy(*args, **kwargs):
            n = len(trace._canopy)
            out = trace._saved["step_cc"](*args, **kwargs)
            added = trace._canopy[n:]
            if added:
                trace.ksw_exp.append(float(added[0][0]))
                trace.ksw_sen.append(float(added[0][2]))
            else:
                trace.ksw_exp.append(np.nan)
                trace.ksw_sen.append(np.nan)
            return out

        def tr_ws(*args, **kwargs):
            out = trace._saved["tr_ws"](*args, **kwargs)
            trace._sto.append(float(out[1]))
            return out

        def transpiration(*args, **kwargs):
            n = len(trace._sto)
            crop = kwargs.get("Crop", args[3])
            growing = kwargs.get("growing_season", args[9])
            gdd = kwargs.get("gdd", args[10])
            out = trace._saved["step_tr"](*args, **kwargs)
            if len(trace._sto) == n:
                trace.ksw_sto.append(np.nan)
            else:
                trace.ksw_sto.append(trace._sto[-1])
            trace.ks_cold.append(_ks_cold(crop, gdd, growing))
            return out

        def hi_ws(*args, **kwargs):
            out = trace._saved["hi_ws"](*args, **kwargs)
            trace._pol.append(float(out[3]))
            return out

        def hi_ts(*args, **kwargs):
            out = trace._saved["hi_ts"](*args, **kwargs)
            trace._heat.append(out)
            return out

        def harvest(*args, **kwargs):
            np_ = len(trace._pol)
            nh = len(trace._heat)
            out = trace._saved["step_hi"](*args, **kwargs)
            if len(trace._pol) == np_:
                trace.ksw_pol.append(np.nan)
            else:
                trace.ksw_pol.append(trace._pol[-1])
            if len(trace._heat) == nh:
                trace.kst_polh.append(np.nan)
                trace.kst_polc.append(np.nan)
            else:
                heat, cold = trace._heat[-1]
                trace.kst_polh.append(float(heat))
                trace.kst_polc.append(float(cold))
            return out

        cc.water_stress = cc_ws
        tr.water_stress = tr_ws
        hi.water_stress = hi_ws
        hi.temperature_stress = hi_ts
        step.canopy_cover = canopy
        step.transpiration = transpiration
        step.harvest_index = harvest

    def restore(self):
        import aquacrop.solution.canopy_cover as cc
        import aquacrop.solution.harvest_index as hi
        import aquacrop.solution.transpiration as tr
        import aquacrop.timestep.run_single_timestep as step

        cc.water_stress = self._saved["cc_ws"]
        tr.water_stress = self._saved["tr_ws"]
        hi.water_stress = self._saved["hi_ws"]
        hi.temperature_stress = self._saved["hi_ts"]
        step.canopy_cover = self._saved["step_cc"]
        step.transpiration = self._saved["step_tr"]
        step.harvest_index = self._saved["step_hi"]


def run_reference(weather_df: pd.DataFrame, crop_name: str, soil_name: str,
                  planting: str, *, crop=None, soil=None, initial_wc=None,
                  irrigation=None, field=None):
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
        irrigation_management=irrigation,
        field_management=field,
    )
    trace = _StressTrace()
    trace.install()
    try:
        model.run_model(till_termination=True)
    finally:
        trace.restore()
    model._stress = trace
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
        (C.OD_IRR, flux["IrrDay"].to_numpy()[:n_days], 1e-12),
        (C.OD_SURFACE_STORAGE, flux["surface_storage"].to_numpy()[:n_days], 1e-12),
        (C.OD_RUNOFF, flux["Runoff"].to_numpy()[:n_days], 1e-12),
        (C.OD_KSW_EXP, np.asarray(model._stress.ksw_exp[:n_days]), 1e-12),
        (C.OD_KSW_SEN, np.asarray(model._stress.ksw_sen[:n_days]), 1e-12),
        (C.OD_KSW_STO, np.asarray(model._stress.ksw_sto[:n_days]), 1e-12),
        (C.OD_KSW_POL, np.asarray(model._stress.ksw_pol[:n_days]), 1e-12),
        (C.OD_KST_POLH, np.asarray(model._stress.kst_polh[:n_days]), 1e-12),
        (C.OD_KST_POLC, np.asarray(model._stress.kst_polc[:n_days]), 1e-12),
        (C.OD_KS_COLD, np.asarray(model._stress.ks_cold[:n_days]), 1e-12),
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
        (C.OF_IRR_CUM, float(stats["Seasonal irrigation (mm)"].iloc[0]), 1e-12),
    )
    for of, ref, atol in finals:
        assert fin[pixel, of] == pytest.approx(ref, rel=1e-12, abs=atol)
    # Season length in days after planting, published as dap_end.
    assert int(fin[pixel, C.OF_DAP_END]) == int(growth["dap"].iloc[last])


def run_grid_kernel(weather_df: pd.DataFrame, crop_name: str, soil_name: str,
                    planting: str, *, irrigation=None, schedule=None,
                    initial_wc: str = "FC", field=None, backend: str = "cpu"):
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
    thini = initial_water_content(soil, initial_wc)
    if irrigation is None:
        ip = default_irrigation_params()
    else:
        ip = irrigation_params_array(irrigation)
    if field is None:
        fp = default_field_params()
    else:
        fp = field_params_array(field)

    fin, daily = run_grid_arrays(
        tmin=sub.MinTemp.to_numpy()[:, None],
        tmax=sub.MaxTemp.to_numpy()[:, None],
        prcp=sub.Precipitation.to_numpy()[:, None],
        et0=sub.ReferenceET.to_numpy()[:, None],
        plant_idx=np.array([0], np.int64),
        cp=cp, sp=sp, profile=prof, th_init=thini,
        ip=ip, fp=fp, schedule=schedule,
        save_daily=True, parallel=False, backend=backend,
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


def _events_frame(weather_df, planting, events):
    """Schedule array indexed from planting, plus the OSPy date frame."""
    start = pd.Timestamp(planting)
    plant_idx = int(np.flatnonzero(weather_df.Date == start)[0])
    sub = weather_df.iloc[plant_idx:].reset_index(drop=True)
    schedule = np.zeros((len(sub), 1), np.float64)
    rows = []
    for date, depth in events:
        ts = pd.Timestamp(date)
        hit = np.flatnonzero(sub.Date == ts)
        assert hit.size == 1
        schedule[int(hit[0]), 0] = depth
        rows.append({"Date": ts, "Depth": depth})
    return schedule, pd.DataFrame(rows)


IRRIGATION_CASES = [
    ("Maize", "SandyLoam", "2019/05/15", 1,
     dict(SMT=[60, 70, 80, 50], AppEff=90, MaxIrr=15, WetSurf=80,
          MaxIrrSeason=400)),
    ("MaizeGDD", "SandyLoam", "2019/05/15", 2,
     dict(IrrInterval=10, AppEff=85, MaxIrr=20)),
    ("SoybeanGDD", "ClayLoam", "2019/06/01", 4,
     dict(NetIrrSMT=65)),
    ("Maize", "Clay", "2019/10/01", 5,
     dict(depth=4, MaxIrr=10, WetSurf=50, AppEff=70)),
]


@pytest.mark.parametrize(
    "crop_name,soil_name,planting,method,options", IRRIGATION_CASES)
def test_irrigation_parity(weather_df, crop_name, soil_name, planting,
                           method, options):
    from aquacrop import IrrigationManagement

    irrigation = IrrigationManagement(method, **options)
    model = run_reference(
        weather_df, crop_name, soil_name, planting, irrigation=irrigation)
    fin, daily, _nt = run_grid_kernel(
        weather_df, crop_name, soil_name, planting, irrigation=irrigation)
    assert_ospy_parity(model, fin, daily)
    assert fin[0, C.OF_IRR_CUM] > 0


def test_schedule_irrigation_parity(weather_df):
    from aquacrop import IrrigationManagement

    planting = "2019/05/15"
    events = [("2019-06-10", 20.0), ("2019-07-20", 15.0), ("2019-08-15", 25.0)]
    schedule, frame = _events_frame(weather_df, planting, events)
    irrigation = IrrigationManagement(
        3, Schedule=frame, MaxIrr=40, AppEff=80, WetSurf=70)
    model = run_reference(
        weather_df, "Maize", "SandyLoam", planting, irrigation=irrigation)
    fin, daily, _nt = run_grid_kernel(
        weather_df, "Maize", "SandyLoam", planting,
        irrigation=irrigation, schedule=schedule)
    assert_ospy_parity(model, fin, daily)
    assert fin[0, C.OF_IRR_CUM] == pytest.approx(20 + 15 + 25)


def test_net_irrigation_day1_pre_irrigation(weather_df):
    """Method 4 fills a dry profile up to NetIrrSMT before drainage.

    Field capacity sits above the target, so the refill only happens
    when the season starts at wilting point.
    """
    from aquacrop import InitialWaterContent, IrrigationManagement

    planting = "2019/06/01"
    irrigation = IrrigationManagement(4, NetIrrSMT=70)
    initial = InitialWaterContent(value=["WP"])
    model = run_reference(
        weather_df, "SoybeanGDD", "ClayLoam", planting,
        irrigation=irrigation, initial_wc=initial)
    fin, daily, _nt = run_grid_kernel(
        weather_df, "SoybeanGDD", "ClayLoam", planting,
        irrigation=irrigation, initial_wc="WP")
    assert_ospy_parity(model, fin, daily)
    assert daily[C.OD_IRR, 0, 0] > 1.0
    assert fin[0, C.OF_IRR_CUM] > daily[C.OD_IRR, 0, 0]


FIELD_CASES = [
    ("bunds", "Clay", dict(bunds=True, z_bund=0.005, bund_water=40)),
    ("mulches", "SandyLoam", dict(mulches=True, mulch_pct=80, f_mulch=0.8)),
    ("sr_inhb", "SandyLoam", dict(sr_inhb=True)),
    ("cn_up", "SandyLoam", dict(curve_number_adj_pct=20)),
    ("cn_down", "SandyLoam", dict(curve_number_adj_pct=-20)),
]


@pytest.mark.parametrize("name,soil_name,options", FIELD_CASES)
def test_field_management_parity(weather_df, name, soil_name, options):
    """Bunds, mulches, inhibited runoff and both curve-number shifts."""
    from aquacrop import FieldMngt

    planting = "2019/05/15"
    field = FieldMngt(**options)
    model = run_reference(
        weather_df, "Maize", soil_name, planting, field=field)
    fin, daily, _nt = run_grid_kernel(
        weather_df, "Maize", soil_name, planting, field=field)
    assert_ospy_parity(model, fin, daily)

    _fin0, bare, _nt = run_grid_kernel(
        weather_df, "Maize", soil_name, planting)
    managed = np.nansum(daily[C.OD_RUNOFF, :, 0])
    unmanaged = np.nansum(bare[C.OD_RUNOFF, :, 0])
    flux = model._outputs.water_flux
    if name == "bunds":
        assert float(flux["surface_storage"].max()) > 0.0
        assert float(flux["Runoff"].sum()) > 0.0
    elif name == "mulches":
        assert not np.allclose(
            daily[C.OD_ES, :, 0], bare[C.OD_ES, :, 0], equal_nan=True)
    elif name == "sr_inhb":
        assert managed == 0.0
        assert unmanaged > 0.0
    elif name == "cn_up":
        assert managed > unmanaged
    else:
        assert managed < unmanaged
