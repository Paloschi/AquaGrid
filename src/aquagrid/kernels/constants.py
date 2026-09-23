"""Index constants for the flat parameter/output arrays used by the kernels.

Crop parameters are packed into a 1-D float64 array (``cp``), soil scalar
parameters into a per-profile float64 array (``sp``), and final outputs into
a (npixel, NF) float64 array. Keeping everything in flat arrays keeps kernel
signatures identical between the CPU (njit) and GPU (cuda.jit) backends.
"""

# ---------------------------------------------------------------- crop params
_i = iter(range(200))


def _n() -> int:
    return next(_i)


# GDD / temperature
CP_GDD_METHOD = _n()
CP_T_UPP = _n()
CP_T_BASE = _n()

# Roots
CP_ZMIN = _n()
CP_ZMAX = _n()
CP_PCT_ZMIN = _n()
CP_FSHAPE_R = _n()
CP_FSHAPE_EX = _n()
CP_SX_TOP = _n()
CP_SX_BOT = _n()

# Water stress thresholds
CP_P_UP0 = _n()
CP_P_UP1 = _n()
CP_P_UP2 = _n()
CP_P_UP3 = _n()
CP_P_LO0 = _n()
CP_P_LO1 = _n()
CP_P_LO2 = _n()
CP_P_LO3 = _n()
CP_FSHAPE_W0 = _n()
CP_FSHAPE_W1 = _n()
CP_FSHAPE_W2 = _n()
CP_FSHAPE_W3 = _n()
CP_ET_ADJ = _n()
CP_BETA = _n()

# Germination / planting
CP_GERM_THR = _n()
CP_PLANT_METHOD = _n()

# Calendar (GDD mode values; CD-mode duplicates further below)
CP_CALENDAR_TYPE = _n()
CP_EMERGENCE = _n()
CP_MAX_ROOTING = _n()
CP_SENESCENCE = _n()
CP_MATURITY = _n()
CP_HI_START = _n()
CP_FLOWERING = _n()
CP_YLD_FORM = _n()
CP_CANOPY_DEV_END = _n()
CP_CANOPY_10PCT = _n()
CP_MAX_CANOPY = _n()
CP_HI_END = _n()
CP_FLOWERING_END = _n()

# Canopy
CP_CC0 = _n()
CP_CGC = _n()
CP_CDC = _n()
CP_CCX = _n()

# Transpiration
CP_KCB = _n()
CP_FAGE = _n()
CP_A_TR = _n()
CP_TR_COLD_STRESS = _n()
CP_GDD_UP = _n()
CP_GDD_LO = _n()
CP_LAG_AER = _n()
CP_AER = _n()

# Harvest index / biomass
CP_HI_INI = _n()
CP_HI0 = _n()
CP_CROP_TYPE = _n()
CP_DETERMINANT = _n()
CP_WP = _n()
CP_WPY = _n()
CP_F_CO2 = _n()
CP_DHI_PRE = _n()
CP_CC_MIN = _n()
CP_EXC = _n()
CP_DHI0 = _n()
CP_A_HI = _n()
CP_B_HI = _n()
CP_YLD_WC = _n()

# Temperature stress (pollination)
CP_POL_HEAT_STRESS = _n()
CP_TMAX_LO = _n()
CP_TMAX_UP = _n()
CP_FSHAPE_B = _n()
CP_POL_COLD_STRESS = _n()
CP_TMIN_UP = _n()
CP_TMIN_LO = _n()

# CO2
CP_CO2_CONC = _n()
CP_CO2_REF = _n()

# Calendar-day mode duplicates (used when CalendarType == 1)
CP_EMERGENCE_CD = _n()
CP_MAX_ROOTING_CD = _n()
CP_SENESCENCE_CD = _n()
CP_MATURITY_CD = _n()
CP_HI_START_CD = _n()
CP_FLOWERING_CD = _n()
CP_YLD_FORM_CD = _n()
CP_CGC_CD = _n()
CP_CDC_CD = _n()
CP_MAX_CANOPY_CD = _n()
CP_CANOPY_DEV_END_CD = _n()
CP_HI_END_CD = _n()

CP_N = next(_i)

# ---------------------------------------------------------------- soil params
_j = iter(range(50))


def _m() -> int:
    return next(_j)


SP_CN = _m()
SP_ADJ_CN = _m()
SP_Z_CN = _m()
SP_Z_GERM = _m()
SP_Z_TOP = _m()
SP_EVAP_Z_MIN = _m()
SP_EVAP_Z_MAX = _m()
SP_REW = _m()
SP_KEX = _m()
SP_FWCC = _m()
SP_F_WREL_EXP = _m()
SP_F_EVAP = _m()

SP_N = next(_j)

# ------------------------------------------------------------- final outputs
_k = iter(range(50))


def _o() -> int:
    return next(_k)


OF_STATUS = _o()          # see STATUS_* below
OF_DRY_YIELD = _o()       # tonne/ha
OF_FRESH_YIELD = _o()     # tonne/ha
OF_YIELD_POT = _o()       # tonne/ha
OF_BIOMASS = _o()         # g/m2
OF_BIOMASS_NS = _o()      # g/m2
OF_HI_ADJ = _o()          # fraction
OF_DAP_END = _o()         # days after planting at season end
OF_DAY_END = _o()         # time index (0-based) of season end
OF_CROP_MATURE = _o()     # 1.0 if crop reached maturity
OF_CROP_DEAD = _o()       # 1.0 if canopy died
OF_GERMINATED = _o()      # 1.0 if crop germinated
OF_MATURITY_CD = _o()     # maturity in calendar days (per-pixel calendar)

OF_N = next(_k)

# ------------------------------------------------------------- daily outputs
_d = iter(range(50))


def _p() -> int:
    return next(_d)


OD_CANOPY_COVER = _p()
OD_BIOMASS = _p()
OD_Z_ROOT = _p()
OD_GDD_CUM = _p()
OD_ES = _p()
OD_TR = _p()
OD_WR = _p()

OD_N = next(_d)

# ------------------------------------------------------------------ statuses
STATUS_OK = 0.0
STATUS_NOT_SIMULATED = 1.0    # masked pixel / no sowing date
STATUS_NO_MATURITY_GDD = 2.0  # not enough GDDs in weather series to mature
STATUS_TRUNCATED = 3.0        # weather series ended before maturity
