"""Flatten aquacrop ``Crop``/``Soil`` objects into kernel parameter arrays.

Reuses the ``aquacrop`` package (Apache-2.0) for its crop database, soil
pedotransfer functions and default parameters; only the flattening and the
static (weather-independent) parts of ``compute_crop_calendar`` /
``compute_variables`` / ``reset_initial_conditions`` are replicated here.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from aquagrid.kernels import constants as C


def prepare_crop(crop) -> None:
    """Fill in the static calendar values on an aquacrop ``Crop`` object,
    mirroring ``aquacrop.initialize.compute_crop_calendar`` (the parts that
    do not depend on weather)."""
    if getattr(crop, "SwitchGDD", 0) == 1:
        raise NotImplementedError(
            "SwitchGDD (calendar->GDD conversion) is not supported; "
            "provide the crop directly in GDD mode (CalendarType=2)")

    if crop.CalendarType == 1:
        if crop.Determinant == 1:
            crop.CanopyDevEndCD = round(crop.HIstartCD + (crop.FloweringCD / 2))
        else:
            crop.CanopyDevEndCD = crop.SenescenceCD
        crop.Canopy10PctCD = round(
            crop.EmergenceCD + (np.log(0.1 / crop.CC0) / crop.CGC_CD))
        crop.MaxCanopyCD = round(
            crop.EmergenceCD
            + (np.log((0.25 * crop.CCx * crop.CCx / crop.CC0)
                      / (crop.CCx - (0.98 * crop.CCx))) / crop.CGC_CD))
        crop.HIendCD = crop.HIstartCD + crop.YldFormCD

        # duplicate calendar values into the "GDD" slots (as aquacrop does)
        crop.Emergence = crop.EmergenceCD
        crop.Canopy10Pct = crop.Canopy10PctCD
        crop.MaxRooting = crop.MaxRootingCD
        crop.Senescence = crop.SenescenceCD
        crop.Maturity = crop.MaturityCD
        crop.MaxCanopy = crop.MaxCanopyCD
        crop.CanopyDevEnd = crop.CanopyDevEndCD
        crop.HIstart = crop.HIstartCD
        crop.HIend = crop.HIendCD
        crop.YldForm = crop.YldFormCD
        if crop.CropType == 3:
            crop.FloweringEndCD = crop.HIstartCD + crop.FloweringCD
        else:
            crop.FloweringEnd = -999
            crop.FloweringEndCD = -999
            crop.FloweringCD = -999
        crop.CDC = crop.CDC_CD
        crop.CGC = crop.CGC_CD
    elif crop.CalendarType == 2:
        if crop.Determinant == 1:
            crop.CanopyDevEnd = round(crop.HIstart + (crop.Flowering / 2))
        else:
            crop.CanopyDevEnd = crop.Senescence
        crop.Canopy10Pct = round(
            crop.Emergence + (np.log(0.1 / crop.CC0) / crop.CGC))
        crop.MaxCanopy = round(
            crop.Emergence
            + (np.log((0.25 * crop.CCx * crop.CCx / crop.CC0)
                      / (crop.CCx - (0.98 * crop.CCx))) / crop.CGC))
        crop.HIend = crop.HIstart + crop.YldForm
        if crop.CropType == 3:
            crop.FloweringEnd = crop.HIstart + crop.Flowering
        else:
            crop.FloweringEnd = -999
    else:
        raise ValueError(f"unknown CalendarType: {crop.CalendarType}")


def compute_fco2(crop, co2_conc: float, co2_ref: float) -> float:
    """WP adjustment factor for CO2, mirroring
    ``aquacrop.timestep.reset_initial_conditions``."""
    fco2old = np.nan
    if co2_conc <= 550:
        if co2_conc <= co2_ref:
            fw = 0.0
        elif co2_conc >= 550:
            fw = 1.0
        else:
            fw = 1 - ((550 - co2_conc) / (550 - co2_ref))
        fco2old = (co2_conc / co2_ref) / (
            1
            + (co2_conc - co2_ref)
            * ((1 - fw) * crop.bsted
               + fw * ((crop.bsted * crop.fsink)
                       + (crop.bface * (1 - crop.fsink))))
        )

    fco2new = np.nan
    if co2_conc > co2_ref:
        fshape = -4.61824 - 3.43831 * crop.fsink - 5.32587 * crop.fsink ** 2
        if co2_conc >= 2000:
            fco2new = 1.58
        else:
            co2rel = (co2_conc - co2_ref) / (2000 - co2_ref)
            fco2new = 1 + 0.58 * ((np.exp(co2rel * fshape) - 1)
                                  / (np.exp(fshape) - 1))

    if co2_conc <= co2_ref:
        fco2 = fco2old
    else:
        fco2 = fco2new
        if (co2_conc <= 550) and (fco2old < fco2new):
            fco2 = fco2old

    if crop.WP >= 40:
        ftype = 0.0
    elif crop.WP <= 20:
        ftype = 1.0
    else:
        ftype = (40 - crop.WP) / (40 - 20)

    return 1 + ftype * (fco2 - 1)


def co2_concentration_for_year(year: int) -> float:
    """Interpolated CO2 concentration (ppm) from aquacrop's CO2 dataset."""
    from aquacrop.entities.co2 import CO2

    co2 = CO2()
    data = co2.co2_data
    return float(np.interp(year, data.year, data.ppm))


def crop_params_array(crop, co2_conc: float, co2_ref: float = 369.41,
                      prepared: bool = False) -> np.ndarray:
    """Flatten an aquacrop ``Crop`` into the kernel parameter array."""
    if not prepared:
        prepare_crop(crop)
    fco2 = compute_fco2(crop, co2_conc, co2_ref)

    cp = np.zeros(C.CP_N, dtype=np.float64)
    cp[C.CP_GDD_METHOD] = crop.GDDmethod
    cp[C.CP_T_UPP] = crop.Tupp
    cp[C.CP_T_BASE] = crop.Tbase

    cp[C.CP_ZMIN] = crop.Zmin
    cp[C.CP_ZMAX] = crop.Zmax
    cp[C.CP_PCT_ZMIN] = crop.PctZmin
    cp[C.CP_FSHAPE_R] = crop.fshape_r
    cp[C.CP_FSHAPE_EX] = crop.fshape_ex
    cp[C.CP_SX_TOP] = crop.SxTop
    cp[C.CP_SX_BOT] = crop.SxBot

    cp[C.CP_P_UP0] = crop.p_up[0]
    cp[C.CP_P_UP1] = crop.p_up[1]
    cp[C.CP_P_UP2] = crop.p_up[2]
    cp[C.CP_P_UP3] = crop.p_up[3]
    cp[C.CP_P_LO0] = crop.p_lo[0]
    cp[C.CP_P_LO1] = crop.p_lo[1]
    cp[C.CP_P_LO2] = crop.p_lo[2]
    cp[C.CP_P_LO3] = crop.p_lo[3]
    cp[C.CP_FSHAPE_W0] = crop.fshape_w[0]
    cp[C.CP_FSHAPE_W1] = crop.fshape_w[1]
    cp[C.CP_FSHAPE_W2] = crop.fshape_w[2]
    cp[C.CP_FSHAPE_W3] = crop.fshape_w[3]
    cp[C.CP_ET_ADJ] = crop.ETadj
    cp[C.CP_BETA] = crop.beta

    cp[C.CP_GERM_THR] = crop.GermThr
    cp[C.CP_PLANT_METHOD] = crop.PlantMethod

    cp[C.CP_CALENDAR_TYPE] = crop.CalendarType
    cp[C.CP_EMERGENCE] = crop.Emergence
    cp[C.CP_MAX_ROOTING] = crop.MaxRooting
    cp[C.CP_SENESCENCE] = crop.Senescence
    cp[C.CP_MATURITY] = crop.Maturity
    cp[C.CP_HI_START] = crop.HIstart
    cp[C.CP_FLOWERING] = crop.Flowering
    cp[C.CP_YLD_FORM] = crop.YldForm
    cp[C.CP_CANOPY_DEV_END] = crop.CanopyDevEnd
    cp[C.CP_CANOPY_10PCT] = crop.Canopy10Pct
    cp[C.CP_MAX_CANOPY] = crop.MaxCanopy
    cp[C.CP_HI_END] = crop.HIend
    cp[C.CP_FLOWERING_END] = crop.FloweringEnd

    cp[C.CP_CC0] = crop.CC0
    cp[C.CP_CGC] = crop.CGC
    cp[C.CP_CDC] = crop.CDC
    cp[C.CP_CCX] = crop.CCx

    cp[C.CP_KCB] = crop.Kcb
    cp[C.CP_FAGE] = crop.fage
    cp[C.CP_A_TR] = crop.a_Tr
    cp[C.CP_TR_COLD_STRESS] = crop.TrColdStress
    cp[C.CP_GDD_UP] = crop.GDD_up
    cp[C.CP_GDD_LO] = crop.GDD_lo
    cp[C.CP_LAG_AER] = crop.LagAer
    cp[C.CP_AER] = crop.Aer

    cp[C.CP_HI_INI] = crop.HIini
    cp[C.CP_HI0] = crop.HI0
    cp[C.CP_CROP_TYPE] = crop.CropType
    cp[C.CP_DETERMINANT] = crop.Determinant
    cp[C.CP_WP] = crop.WP
    cp[C.CP_WPY] = crop.WPy
    cp[C.CP_F_CO2] = fco2
    cp[C.CP_DHI_PRE] = crop.dHI_pre
    cp[C.CP_CC_MIN] = crop.CCmin
    cp[C.CP_EXC] = crop.exc
    cp[C.CP_DHI0] = crop.dHI0
    cp[C.CP_A_HI] = crop.a_HI
    cp[C.CP_B_HI] = crop.b_HI
    cp[C.CP_YLD_WC] = crop.YldWC

    cp[C.CP_POL_HEAT_STRESS] = crop.PolHeatStress
    cp[C.CP_TMAX_LO] = crop.Tmax_lo
    cp[C.CP_TMAX_UP] = crop.Tmax_up
    cp[C.CP_FSHAPE_B] = crop.fshape_b
    cp[C.CP_POL_COLD_STRESS] = crop.PolColdStress
    cp[C.CP_TMIN_UP] = crop.Tmin_up
    cp[C.CP_TMIN_LO] = crop.Tmin_lo

    cp[C.CP_CO2_CONC] = co2_conc
    cp[C.CP_CO2_REF] = co2_ref

    cp[C.CP_EMERGENCE_CD] = crop.EmergenceCD
    cp[C.CP_MAX_ROOTING_CD] = crop.MaxRootingCD
    cp[C.CP_SENESCENCE_CD] = crop.SenescenceCD
    cp[C.CP_MATURITY_CD] = crop.MaturityCD
    cp[C.CP_HI_START_CD] = crop.HIstartCD
    cp[C.CP_FLOWERING_CD] = crop.FloweringCD
    cp[C.CP_YLD_FORM_CD] = crop.YldFormCD
    cp[C.CP_CGC_CD] = crop.CGC_CD
    cp[C.CP_CDC_CD] = crop.CDC_CD
    cp[C.CP_MAX_CANOPY_CD] = crop.MaxCanopyCD
    cp[C.CP_CANOPY_DEV_END_CD] = crop.CanopyDevEndCD
    cp[C.CP_HI_END_CD] = crop.HIendCD

    return cp


# Cube layer name -> kernel slot. Length-4 crop vectors are split.
CROP_PARAM_INDEX: dict[str, int] = {
    "GDDmethod": C.CP_GDD_METHOD,
    "Tupp": C.CP_T_UPP,
    "Tbase": C.CP_T_BASE,
    "Zmin": C.CP_ZMIN,
    "Zmax": C.CP_ZMAX,
    "PctZmin": C.CP_PCT_ZMIN,
    "fshape_r": C.CP_FSHAPE_R,
    "fshape_ex": C.CP_FSHAPE_EX,
    "SxTop": C.CP_SX_TOP,
    "SxBot": C.CP_SX_BOT,
    "p_up1": C.CP_P_UP0,
    "p_up2": C.CP_P_UP1,
    "p_up3": C.CP_P_UP2,
    "p_up4": C.CP_P_UP3,
    "p_lo1": C.CP_P_LO0,
    "p_lo2": C.CP_P_LO1,
    "p_lo3": C.CP_P_LO2,
    "p_lo4": C.CP_P_LO3,
    "fshape_w1": C.CP_FSHAPE_W0,
    "fshape_w2": C.CP_FSHAPE_W1,
    "fshape_w3": C.CP_FSHAPE_W2,
    "fshape_w4": C.CP_FSHAPE_W3,
    "ETadj": C.CP_ET_ADJ,
    "beta": C.CP_BETA,
    "GermThr": C.CP_GERM_THR,
    "PlantMethod": C.CP_PLANT_METHOD,
    "CalendarType": C.CP_CALENDAR_TYPE,
    "Emergence": C.CP_EMERGENCE,
    "MaxRooting": C.CP_MAX_ROOTING,
    "Senescence": C.CP_SENESCENCE,
    "Maturity": C.CP_MATURITY,
    "HIstart": C.CP_HI_START,
    "Flowering": C.CP_FLOWERING,
    "YldForm": C.CP_YLD_FORM,
    "CanopyDevEnd": C.CP_CANOPY_DEV_END,
    "Canopy10Pct": C.CP_CANOPY_10PCT,
    "MaxCanopy": C.CP_MAX_CANOPY,
    "HIend": C.CP_HI_END,
    "FloweringEnd": C.CP_FLOWERING_END,
    "CC0": C.CP_CC0,
    "CGC": C.CP_CGC,
    "CDC": C.CP_CDC,
    "CCx": C.CP_CCX,
    "Kcb": C.CP_KCB,
    "fage": C.CP_FAGE,
    "a_Tr": C.CP_A_TR,
    "TrColdStress": C.CP_TR_COLD_STRESS,
    "GDD_up": C.CP_GDD_UP,
    "GDD_lo": C.CP_GDD_LO,
    "LagAer": C.CP_LAG_AER,
    "Aer": C.CP_AER,
    "HIini": C.CP_HI_INI,
    "HI0": C.CP_HI0,
    "CropType": C.CP_CROP_TYPE,
    "Determinant": C.CP_DETERMINANT,
    "WP": C.CP_WP,
    "WPy": C.CP_WPY,
    "dHI_pre": C.CP_DHI_PRE,
    "CCmin": C.CP_CC_MIN,
    "exc": C.CP_EXC,
    "dHI0": C.CP_DHI0,
    "a_HI": C.CP_A_HI,
    "b_HI": C.CP_B_HI,
    "YldWC": C.CP_YLD_WC,
    "PolHeatStress": C.CP_POL_HEAT_STRESS,
    "Tmax_lo": C.CP_TMAX_LO,
    "Tmax_up": C.CP_TMAX_UP,
    "fshape_b": C.CP_FSHAPE_B,
    "PolColdStress": C.CP_POL_COLD_STRESS,
    "Tmin_up": C.CP_TMIN_UP,
    "Tmin_lo": C.CP_TMIN_LO,
}

# GDD stages scaled with Maturity so a shorter cycle stays internally consistent.
PHENOLOGY_WITH_MATURITY = (
    "Emergence", "Flowering", "Senescence", "MaxRooting", "HIstart", "YldForm",
    "CanopyDevEnd", "Canopy10Pct", "MaxCanopy", "HIend", "FloweringEnd",
)

# Calendar-day crops (CalendarType == 1) end the season from these slots.
_CALENDAR_CD_SLOT = {
    "Maturity": C.CP_MATURITY_CD,
    "MaxCanopy": C.CP_MAX_CANOPY_CD,
    "CanopyDevEnd": C.CP_CANOPY_DEV_END_CD,
    "HIstart": C.CP_HI_START_CD,
    "HIend": C.CP_HI_END_CD,
    "YldForm": C.CP_YLD_FORM_CD,
    "Flowering": C.CP_FLOWERING_CD,
}


def apply_crop_cube(base: np.ndarray, cube: xr.DataArray) -> np.ndarray:
    """Broadcast ``base`` ``(CP_N,)`` over a ``(param, y, x)`` cube.

    Missing layers and NaN cells keep the base value. For a calendar-day
    pixel, an override of ``Maturity``, ``MaxCanopy``, ``CanopyDevEnd``,
    ``HIstart``, ``HIend``, ``YldForm``, or ``Flowering`` is copied into the
    matching ``*CD`` slot. Returns ``(y, x, CP_N)``.
    """
    if set(cube.dims) != {"param", "y", "x"}:
        raise ValueError(
            f"crop cube dims must be (param, y, x), got {cube.dims}")
    cube = cube.transpose("param", "y", "x")
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    out = np.broadcast_to(base, (ny, nx, base.shape[0])).copy()
    names = [str(name) for name in cube.param.values]
    unknown = [name for name in names if name not in CROP_PARAM_INDEX]
    if unknown:
        raise ValueError(f"unknown crop parameter(s): {unknown}")
    finite = {}
    for name in names:
        layer = np.asarray(cube.sel(param=name).values, np.float64)
        good = np.isfinite(layer)
        out[:, :, CROP_PARAM_INDEX[name]][good] = layer[good]
        finite[name] = good
    calendar = out[:, :, C.CP_CALENDAR_TYPE] == 1
    for name, good in finite.items():
        cd = _CALENDAR_CD_SLOT.get(name)
        if cd is None:
            continue
        copy = good & calendar
        out[:, :, cd][copy] = out[:, :, CROP_PARAM_INDEX[name]][copy]
    return out


def soil_params(soil, zmax: float | None = None
                ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Flatten an aquacrop ``Soil`` into (scalar array, profile arrays),
    applying the adjustments from ``compute_variables``.

    If ``zmax`` (crop maximum rooting depth, m) is given, the compartment
    discretisation is first deepened exactly like aquacrop's
    ``read_model_parameters``: while the profile is shallower than
    ``zmax + 0.1``, thicken (bottom-up) the first compartment with
    ``dz < 0.25`` by 0.1 m. Mutates ``soil`` in place, as aquacrop does.
    """
    if zmax is not None:
        while soil.zSoil < zmax + 0.1:
            for i in soil.profile.index[::-1]:
                if soil.profile.loc[i, "dz"] < 0.25:
                    soil.profile.loc[i, "dz"] += 0.1
                    soil.fill_nan()
                    break

    prof = soil.profile

    rew = soil.rew
    if soil.adj_rew == 0:
        rew = round(
            1000
            * (float(prof.th_fc.iloc[0]) - float(prof.th_dry.iloc[0]))
            * soil.evap_z_surf,
            2,
        )

    cn = soil.cn
    if soil.calc_cn == 1:
        ksat0 = float(prof.Ksat.iloc[0])
        if ksat0 > 864:
            cn = 46
        elif ksat0 > 347:
            cn = 61
        elif ksat0 > 36:
            cn = 72
        elif ksat0 > 0:
            cn = 77

    sp = np.zeros(C.SP_N, dtype=np.float64)
    sp[C.SP_CN] = cn
    sp[C.SP_ADJ_CN] = soil.adj_cn
    sp[C.SP_Z_CN] = soil.z_cn
    sp[C.SP_Z_GERM] = soil.z_germ
    sp[C.SP_Z_TOP] = soil.z_top
    sp[C.SP_EVAP_Z_MIN] = soil.evap_z_min
    sp[C.SP_EVAP_Z_MAX] = soil.evap_z_max
    sp[C.SP_REW] = rew
    sp[C.SP_KEX] = soil.kex
    sp[C.SP_FWCC] = soil.fwcc
    sp[C.SP_F_WREL_EXP] = soil.f_wrel_exp
    sp[C.SP_F_EVAP] = soil.f_evap

    arrays = {
        "dz": prof.dz.to_numpy(np.float64),
        "dzsum": prof.dzsum.to_numpy(np.float64),
        "th_fc": prof.th_fc.to_numpy(np.float64),
        "th_s": prof.th_s.to_numpy(np.float64),
        "th_wp": prof.th_wp.to_numpy(np.float64),
        "th_dry": prof.th_dry.to_numpy(np.float64),
        "ksat": prof.Ksat.to_numpy(np.float64),
        "tau": prof.tau.to_numpy(np.float64),
        "layer": prof.Layer.to_numpy(np.float64),
        "penetrability": prof.penetrability.to_numpy(np.float64),
    }
    return sp, arrays


def initial_water_content(soil, kind: str = "FC") -> np.ndarray:
    """Initial volumetric water content per compartment."""
    prof = soil.profile
    kind = kind.upper()
    if kind == "FC":
        return prof.th_fc.to_numpy(np.float64).copy()
    if kind == "WP":
        return prof.th_wp.to_numpy(np.float64).copy()
    if kind == "SAT":
        return prof.th_s.to_numpy(np.float64).copy()
    raise ValueError(f"unsupported initial water content: {kind!r}")
