"""Backend-neutral AquaCrop kernels operating on per-pixel state.

Vendored and adapted from AquaCrop-OSPy (https://github.com/aquacropos/aquacrop,
Apache-2.0). Each function is a faithful scalar port of the corresponding
``aquacrop.solution`` module, rewritten to:

* operate on one pixel (state passed in/out as scalars + 1-D compartment views),
* avoid allocations, numpy vector ops and strings (CUDA-compatible subset),
* read crop/soil parameters from flat float64 arrays indexed by the
  constants in :mod:`aquagrid.kernels.constants`.

Scope (phase 1): rainfed simulations, no groundwater table, no irrigation,
no field management (bunds/mulches), CO2 adjustment precomputed on the
Python side. Calendar per pixel is computed inside the kernel from the
pixel's own temperature series (GDD mode) starting at its sowing date.

The module is compiled twice by :mod:`aquagrid.kernels.loader`: with
``numba.njit`` for the CPU backend and ``numba.cuda.jit(device=True)`` for
the GPU backend. The ``kernel`` decorator is injected by the loader; the
fallback below keeps the module importable (and testable) in pure Python.
"""

import math

from aquagrid.kernels.constants import (
    CP_GDD_METHOD, CP_T_UPP, CP_T_BASE,
    CP_ZMIN, CP_ZMAX, CP_PCT_ZMIN, CP_FSHAPE_R, CP_FSHAPE_EX,
    CP_SX_TOP, CP_SX_BOT,
    CP_P_UP0, CP_P_UP1, CP_P_UP2, CP_P_UP3,
    CP_P_LO0, CP_P_LO1, CP_P_LO2, CP_P_LO3,
    CP_FSHAPE_W0, CP_FSHAPE_W1, CP_FSHAPE_W2, CP_FSHAPE_W3,
    CP_ET_ADJ, CP_BETA,
    CP_GERM_THR, CP_PLANT_METHOD,
    CP_CALENDAR_TYPE, CP_EMERGENCE, CP_MAX_ROOTING, CP_SENESCENCE,
    CP_MATURITY, CP_HI_START, CP_FLOWERING, CP_YLD_FORM,
    CP_CANOPY_DEV_END, CP_CANOPY_10PCT, CP_MAX_CANOPY, CP_HI_END,
    CP_FLOWERING_END,
    CP_CC0, CP_CGC, CP_CDC, CP_CCX,
    CP_KCB, CP_FAGE, CP_A_TR, CP_TR_COLD_STRESS, CP_GDD_UP, CP_GDD_LO,
    CP_LAG_AER, CP_AER,
    CP_HI_INI, CP_HI0, CP_CROP_TYPE, CP_DETERMINANT, CP_WP, CP_WPY,
    CP_F_CO2, CP_DHI_PRE, CP_CC_MIN, CP_EXC, CP_DHI0, CP_A_HI, CP_B_HI,
    CP_YLD_WC,
    CP_POL_HEAT_STRESS, CP_TMAX_LO, CP_TMAX_UP, CP_FSHAPE_B,
    CP_POL_COLD_STRESS, CP_TMIN_UP, CP_TMIN_LO,
    CP_CO2_CONC, CP_CO2_REF,
    CP_EMERGENCE_CD, CP_MAX_ROOTING_CD, CP_SENESCENCE_CD, CP_MATURITY_CD,
    CP_HI_START_CD, CP_FLOWERING_CD, CP_YLD_FORM_CD, CP_CGC_CD, CP_CDC_CD,
    CP_MAX_CANOPY_CD, CP_CANOPY_DEV_END_CD, CP_HI_END_CD,
    SP_CN, SP_ADJ_CN, SP_Z_CN, SP_Z_GERM, SP_Z_TOP,
    SP_EVAP_Z_MIN, SP_EVAP_Z_MAX, SP_REW, SP_KEX, SP_FWCC,
    SP_F_WREL_EXP, SP_F_EVAP,
    OF_STATUS, OF_DRY_YIELD, OF_FRESH_YIELD, OF_YIELD_POT, OF_BIOMASS,
    OF_BIOMASS_NS, OF_HI_ADJ, OF_DAP_END, OF_DAY_END, OF_CROP_MATURE,
    OF_CROP_DEAD, OF_GERMINATED, OF_MATURITY_CD,
    OD_CANOPY_COVER, OD_BIOMASS, OD_Z_ROOT, OD_GDD_CUM, OD_ES, OD_TR, OD_WR,
    STATUS_OK, STATUS_NO_MATURITY_GDD, STATUS_TRUNCATED,
)

if "kernel" not in globals():  # plain import: no-op decorator (pure Python)
    def kernel(f):
        return f


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

@kernel
def round_half_even(x):
    """Round to nearest integer, ties to even (matches python/numpy round)."""
    f = math.floor(x)
    d = x - f
    if d > 0.5:
        return f + 1.0
    if d < 0.5:
        return f
    half = f * 0.5
    if half == math.floor(half):  # f is even
        return f
    return f + 1.0


@kernel
def fmin(a, b):
    """Two-scalar min (builtin min(a, b) is not typed on the CUDA target)."""
    return a if a < b else b


@kernel
def fmax(a, b):
    """Two-scalar max (builtin max(a, b) is not typed on the CUDA target)."""
    return a if a > b else b


@kernel
def roundn(x, n):
    """Round ``x`` to ``n`` decimals, ties to even."""
    s = 10.0 ** n
    return round_half_even(x * s) / s


@kernel
def first_ge(arr, n, x):
    """First index i in [0, n) with arr[i] >= x (n-1 if none)."""
    for i in range(n):
        if arr[i] >= x:
            return i
    return n - 1


@kernel
def count_lt(arr, n, x):
    """Number of elements arr[i] < x for i in [0, n)."""
    c = 0
    for i in range(n):
        if arr[i] < x:
            c += 1
    return c


@kernel
def layer_first_comp(layer, ncomp, li):
    """First compartment index belonging to soil layer ``li``."""
    for i in range(ncomp):
        if layer[i] == li:
            return i
    return ncomp - 1


@kernel
def layer_dz_sum(layer, dz, ncomp, li):
    """Total thickness of soil layer ``li``."""
    s = 0.0
    for i in range(ncomp):
        if layer[i] == li:
            s += dz[i]
    return s


# --------------------------------------------------------------------------
# growing degree days                       (solution/growing_degree_day.py)
# --------------------------------------------------------------------------

@kernel
def growing_degree_day(gdd_method, t_upp, t_base, temp_max, temp_min):
    if gdd_method == 1:
        tmean = (temp_max + temp_min) / 2
        tmean = fmin(tmean, t_upp)
        tmean = fmax(tmean, t_base)
        gdd = tmean - t_base
    elif gdd_method == 2:
        tmax = fmin(temp_max, t_upp)
        tmax = fmax(tmax, t_base)
        tmin = fmin(temp_min, t_upp)
        tmin = fmax(tmin, t_base)
        tmean = (tmax + tmin) / 2
        gdd = tmean - t_base
    else:  # method 3
        tmax = fmin(temp_max, t_upp)
        tmax = fmax(tmax, t_base)
        tmin = fmin(temp_min, t_upp)
        tmean = (tmax + tmin) / 2
        tmean = fmax(tmean, t_base)
        gdd = tmean - t_base
    return gdd


# --------------------------------------------------------------------------
# drainage                                            (solution/drainage.py)
# --------------------------------------------------------------------------

@kernel
def drainage(ncomp, dz, dzsum, th_fc, th_s, tau, ksat, th_fc_adj,
             th, thnew, flux_out):
    """Redistribute stored soil water. Updates ``th`` in place, fills
    ``flux_out``; returns total deep percolation (mm)."""
    drainsum = 0.0
    for ii in range(ncomp):
        cth_fc = th_fc[ii]
        cth_s = th_s[ii]
        ctau = tau[ii]
        cdz = dz[ii]
        cdzsum = dzsum[ii]
        cksat = ksat[ii]

        # drainage ability of compartment ii
        if th[ii] <= th_fc_adj[ii]:
            dthdt = 0.0
        elif th[ii] >= cth_s:
            dthdt = ctau * (cth_s - cth_fc)
            if (th[ii] - dthdt) < th_fc_adj[ii]:
                dthdt = th[ii] - th_fc_adj[ii]
        else:
            dthdt = (
                ctau
                * (cth_s - cth_fc)
                * ((math.exp(th[ii] - cth_fc) - 1) / (math.exp(cth_s - cth_fc) - 1))
            )
            if (th[ii] - dthdt) < th_fc_adj[ii]:
                dthdt = th[ii] - th_fc_adj[ii]

        # drainage from compartment ii (mm)
        draincomp = dthdt * cdz * 1000

        # check drainage ability against cumulative drainage from above
        excess = 0.0
        prethick = cdzsum - cdz
        drainmax = dthdt * 1000 * prethick
        drainability = drainsum <= drainmax

        if drainability:
            thnew[ii] = th[ii] - dthdt
            drainsum = drainsum + draincomp
            if drainsum > cksat:
                excess = excess + drainsum - cksat
                drainsum = cksat
        else:
            dthdt = drainsum / (1000 * prethick)
            # theta (thX) needed for drainage ability equal to cum. drainage
            if dthdt <= 0:
                thx = th_fc_adj[ii]
            elif ctau > 0:
                a = 1 + (
                    (dthdt * (math.exp(cth_s - cth_fc) - 1)) / (ctau * (cth_s - cth_fc))
                )
                thx = cth_fc + math.log(a)
                if thx < th_fc_adj[ii]:
                    thx = th_fc_adj[ii]
            else:
                thx = cth_s + 0.01

            if thx <= cth_s:
                thnew[ii] = th[ii] + (drainsum / (1000 * cdz))
                if thnew[ii] > thx:
                    # cumulative drainage is drainage difference between
                    # theta_x and new theta plus drainage ability at theta_x
                    drainsum = (thnew[ii] - thx) * 1000 * cdz
                    # drainage ability for thX
                    if thx <= th_fc_adj[ii]:
                        dthdt = 0.0
                    elif thx >= cth_s:
                        dthdt = ctau * (cth_s - cth_fc)
                        if (thx - dthdt) < th_fc_adj[ii]:
                            dthdt = thx - th_fc_adj[ii]
                    else:
                        dthdt = (
                            ctau
                            * (cth_s - cth_fc)
                            * ((math.exp(thx - cth_fc) - 1)
                               / (math.exp(cth_s - cth_fc) - 1))
                        )
                        if (thx - dthdt) < th_fc_adj[ii]:
                            dthdt = thx - th_fc_adj[ii]
                    drainsum = drainsum + (dthdt * 1000 * cdz)
                    if drainsum > cksat:
                        excess = excess + drainsum - cksat
                        drainsum = cksat
                    thnew[ii] = thx - dthdt
                elif thnew[ii] > th_fc_adj[ii]:
                    # drainage ability for updated water content
                    if thnew[ii] <= th_fc_adj[ii]:
                        dthdt = 0.0
                    elif thnew[ii] >= cth_s:
                        dthdt = ctau * (cth_s - cth_fc)
                        if (thnew[ii] - dthdt) < th_fc_adj[ii]:
                            dthdt = thnew[ii] - th_fc_adj[ii]
                    else:
                        dthdt = (
                            ctau
                            * (cth_s - cth_fc)
                            * ((math.exp(thnew[ii] - cth_fc) - 1)
                               / (math.exp(cth_s - cth_fc) - 1))
                        )
                        if (thnew[ii] - dthdt) < th_fc_adj[ii]:
                            dthdt = thnew[ii] - th_fc_adj[ii]
                    thnew[ii] = thnew[ii] - dthdt
                    drainsum = dthdt * 1000 * cdz
                    if drainsum > cksat:
                        excess = excess + drainsum - cksat
                        drainsum = cksat
                else:
                    drainsum = 0.0
            else:  # thx > cth_s
                thnew[ii] = th[ii] + (drainsum / (1000 * cdz))
                if thnew[ii] <= cth_s:
                    if thnew[ii] > th_fc_adj[ii]:
                        # new drainage ability
                        if thnew[ii] <= th_fc_adj[ii]:
                            dthdt = 0.0
                        elif thnew[ii] >= cth_s:
                            dthdt = ctau * (cth_s - cth_fc)
                            if (thnew[ii] - dthdt) < th_fc_adj[ii]:
                                dthdt = thnew[ii] - th_fc_adj[ii]
                        else:
                            dthdt = (
                                ctau
                                * (cth_s - cth_fc)
                                * ((math.exp(thnew[ii] - cth_fc) - 1)
                                   / (math.exp(cth_s - cth_fc) - 1))
                            )
                            if (thnew[ii] - dthdt) < th_fc_adj[ii]:
                                dthdt = thnew[ii] - th_fc_adj[ii]
                        thnew[ii] = thnew[ii] - dthdt
                        drainsum = dthdt * 1000 * cdz
                        if drainsum > cksat:
                            excess = excess + drainsum - cksat
                            drainsum = cksat
                    else:
                        drainsum = 0.0
                else:  # thnew[ii] > cth_s
                    excess = (thnew[ii] - cth_s) * 1000 * cdz
                    # drainage ability for updated water content
                    if thnew[ii] <= th_fc_adj[ii]:
                        dthdt = 0.0
                    elif thnew[ii] >= cth_s:
                        dthdt = ctau * (cth_s - cth_fc)
                        if (thnew[ii] - dthdt) < th_fc_adj[ii]:
                            dthdt = thnew[ii] - th_fc_adj[ii]
                    else:
                        dthdt = (
                            ctau
                            * (cth_s - cth_fc)
                            * ((math.exp(thnew[ii] - cth_fc) - 1)
                               / (math.exp(cth_s - cth_fc) - 1))
                        )
                        if (thnew[ii] - dthdt) < th_fc_adj[ii]:
                            dthdt = thnew[ii] - th_fc_adj[ii]
                    thnew[ii] = cth_s - dthdt
                    draincomp = dthdt * 1000 * cdz
                    drainmax = dthdt * 1000 * prethick
                    if drainmax > excess:
                        drainmax = excess
                    excess = excess - drainmax
                    drainsum = draincomp + drainmax
                    if drainsum > cksat:
                        excess = excess + drainsum - cksat
                        drainsum = cksat

        flux_out[ii] = drainsum

        # redistribute excess into compartments above
        if excess > 0:
            precomp = ii + 1
            while (excess > 0) and (precomp != 0):
                precomp = precomp - 1
                if precomp < ii:
                    flux_out[precomp] = flux_out[precomp] - excess
                thnew[precomp] = thnew[precomp] + (excess / (1000 * dz[precomp]))
                if thnew[precomp] > th_s[precomp]:
                    excess = (thnew[precomp] - th_s[precomp]) * 1000 * dz[precomp]
                    thnew[precomp] = th_s[precomp]
                else:
                    excess = 0.0

    for ii in range(ncomp):
        th[ii] = thnew[ii]
    return drainsum


# --------------------------------------------------------------------------
# rainfall partition                         (solution/rainfall_partition.py)
# --------------------------------------------------------------------------

@kernel
def rainfall_partition(prcp, th, ncomp, dz, dzsum, th_fc, th_wp,
                       soil_cn, adj_cn, z_cn, day_submerged):
    """Partition rainfall into runoff and infiltration (rainfed: no bunds,
    surface runoff not inhibited). Returns (runoff, infl, day_submerged)."""
    day_submerged = 0.0
    cn = soil_cn
    if adj_cn == 1:  # adjust cn for antecedent moisture
        cn_bot = round_half_even(
            1.4 * (math.exp(-14 * math.log(10.0)))
            + (0.507 * cn)
            - (0.00374 * cn ** 2)
            + (0.0000867 * cn ** 3)
        )
        cn_top = round_half_even(
            5.6 * (math.exp(-14 * math.log(10.0)))
            + (2.33 * cn)
            - (0.0209 * cn ** 2)
            + (0.000076 * cn ** 3)
        )
        comp_sto = fmin(first_ge(dzsum, ncomp, z_cn) + 1, ncomp)

        # weighting factors and relative wetness of top soil (single pass)
        xx = 0.0
        wet_top = 0.0
        for ii in range(comp_sto):
            cdzsum = dzsum[ii]
            if cdzsum > z_cn:
                cdzsum = z_cn
            wx = 1.016 * (1 - math.exp(-4.16 * (cdzsum / z_cn)))
            wrel = wx - xx
            if wrel < 0:
                wrel = 0.0
            elif wrel > 1:
                wrel = 1.0
            xx = wx
            thii = fmax(th_wp[ii], th[ii])
            wet_top = wet_top + (
                wrel * ((thii - th_wp[ii]) / (th_fc[ii] - th_wp[ii]))
            )

        if wet_top > 1:
            wet_top = 1.0
        elif wet_top < 0:
            wet_top = 0.0
        cn = round_half_even(cn_bot + (cn_top - cn_bot) * wet_top)

    # partition rainfall into runoff and infiltration (mm)
    s = (25400 / cn) - 254
    term = prcp - ((5 / 100) * s)
    if term <= 0:
        runoff = 0.0
        infl = prcp
    else:
        runoff = (term ** 2) / (prcp + (1 - (5 / 100)) * s)
        infl = prcp - runoff

    return runoff, infl, day_submerged


# --------------------------------------------------------------------------
# infiltration                                     (solution/infiltration.py)
# --------------------------------------------------------------------------

@kernel
def infiltration(ncomp, dz, th_fc, th_s, tau, ksat, th_fc_adj,
                 th, flux_out, infl, surface_storage, deep_perc0, runoff0):
    """Infiltrate incoming water (rainfed: no bunds, no irrigation).
    Updates ``th`` and ``flux_out`` in place.
    Returns (surface_storage, deep_perc, runoff_tot, infl)."""
    init_surface_storage = surface_storage * 1.0

    infl = fmax(infl, 0.0)

    # no bunds on field
    if infl > ksat[0]:
        to_store = ksat[0]
        runoff_ini = infl - ksat[0]
    else:
        to_store = infl
        runoff_ini = 0.0
    surface_storage = 0.0
    runoff_ini = runoff_ini + init_surface_storage

    ii = -1
    runoff = 0.0
    if to_store > 0:
        while (to_store > 0) and (ii < ncomp - 1):
            ii = ii + 1
            # saturated drainage ability
            dthdt_s = tau[ii] * (th_s[ii] - th_fc[ii])
            # drainage factor
            factor = ksat[ii] / (dthdt_s * 1000 * dz[ii])
            # drainage ability required
            dthdt0 = to_store / (1000 * dz[ii])

            if dthdt0 < dthdt_s:
                # water content needed to meet drainage dthdt0
                if dthdt0 <= 0:
                    theta0 = th_fc_adj[ii]
                else:
                    a = 1 + (
                        (dthdt0 * (math.exp(th_s[ii] - th_fc[ii]) - 1))
                        / (tau[ii] * (th_s[ii] - th_fc[ii]))
                    )
                    theta0 = th_fc[ii] + math.log(a)
                if theta0 > th_s[ii]:
                    theta0 = th_s[ii]
                elif theta0 <= th_fc_adj[ii]:
                    theta0 = th_fc_adj[ii]
                    dthdt0 = 0.0
            else:
                theta0 = th_s[ii]
                dthdt0 = dthdt_s

            # maximum water flow through compartment ii
            drainmax = factor * dthdt0 * 1000 * dz[ii]
            drainage_ = drainmax + flux_out[ii]
            if drainage_ > ksat[ii]:
                drainmax = ksat[ii] - flux_out[ii]

            # difference between threshold and current water content
            diff = theta0 - th[ii]
            if diff > 0:
                th[ii] = th[ii] + (to_store / (1000 * dz[ii]))
                if th[ii] > theta0:
                    to_store = (th[ii] - theta0) * 1000 * dz[ii]
                    th[ii] = theta0
                else:
                    to_store = 0.0

            # update outflow from current compartment
            flux_out[ii] = flux_out[ii] + to_store

            # back-up of water into compartments above
            excess = to_store - drainmax
            if excess < 0:
                excess = 0.0
            to_store = to_store - excess

            if excess > 0:
                precomp = ii + 1
                while (excess > 0) and (precomp != 0):
                    precomp = precomp - 1
                    flux_out[precomp] = flux_out[precomp] - excess
                    th[precomp] = th[precomp] + (excess / (dz[precomp] * 1000))
                    if th[precomp] > th_s[precomp]:
                        excess = (th[precomp] - th_s[precomp]) * 1000 * dz[precomp]
                        th[precomp] = th_s[precomp]
                    else:
                        excess = 0.0
                if excess > 0:
                    runoff = runoff + excess

        deep_perc = to_store
    else:
        deep_perc = 0.0
        runoff = 0.0

    runoff = runoff + runoff_ini

    deep_perc = deep_perc + deep_perc0
    infl = infl - runoff
    runoff_tot = runoff + runoff0
    return surface_storage, deep_perc, runoff_tot, infl


# --------------------------------------------------------------------------
# germination                                       (solution/germination.py)
# --------------------------------------------------------------------------

@kernel
def germination(th, ncomp, dz, dzsum, th_fc, th_wp, z_germ, germ_thr,
                plant_method, gdd, growing_season,
                germinated, protected_seed, delayed_cds, delayed_gdds):
    """Check germination. Returns (germinated, protected_seed,
    delayed_cds, delayed_gdds)."""
    if growing_season:
        if not germinated:
            comp_sto = first_ge(dzsum, ncomp, z_germ)
            wr = 0.0
            wr_fc = 0.0
            wr_wp = 0.0
            for ii in range(comp_sto + 1):
                if dzsum[ii] > z_germ:
                    factor = 1 - ((dzsum[ii] - z_germ) / dz[ii])
                else:
                    factor = 1.0
                wr = wr + roundn(factor * 1000 * th[ii] * dz[ii], 3)
                wr_fc = wr_fc + roundn(factor * 1000 * th_fc[ii] * dz[ii], 3)
                wr_wp = wr_wp + roundn(factor * 1000 * th_wp[ii] * dz[ii], 3)
            if wr < 0:
                wr = 0.0
            wc_prop = 1 - ((wr_fc - wr) / (wr_fc - wr_wp))
            if wc_prop >= germ_thr:
                germinated = True
                if plant_method:
                    protected_seed = True
                else:
                    protected_seed = False
            else:
                delayed_cds = delayed_cds + 1
                delayed_gdds = delayed_gdds + gdd
                protected_seed = False
    else:
        germinated = False
        protected_seed = False
        delayed_cds = 0.0
        delayed_gdds = 0.0
    return germinated, protected_seed, delayed_cds, delayed_gdds


# --------------------------------------------------------------------------
# root development                             (solution/root_development.py)
# --------------------------------------------------------------------------

@kernel
def root_development(cp, ncomp, nlayer, dz, dzsum, th_fc, th_wp,
                     layer, penetrability,
                     dap, z_root, delayed_cds, gdd_cum, delayed_gdds,
                     tr_ratio, th, cc, cc_ns, germinated, r_cor, t_pot,
                     gdd, growing_season):
    """Root zone expansion. Returns (z_root, r_cor)."""
    zroot_init = z_root * 1.0

    if growing_season:
        if dap == 1:
            z_root = cp[CP_ZMIN] * 1.0
            zroot_init = cp[CP_ZMIN] * 1.0

        if cp[CP_CALENDAR_TYPE] == 1:
            t_adj = dap - delayed_cds
        else:
            t_adj = gdd_cum - delayed_gdds

        zini = cp[CP_ZMIN] * (cp[CP_PCT_ZMIN] / 100)
        t0 = round_half_even(cp[CP_EMERGENCE] / 2)
        tmax = cp[CP_MAX_ROOTING]
        if cp[CP_CALENDAR_TYPE] == 1:
            t_old = t_adj - 1
        else:
            t_old = t_adj - gdd

        # potential root depth on previous day
        if t_old >= tmax:
            zr_old = cp[CP_ZMAX]
        elif t_old <= t0:
            zr_old = zini
        else:
            x = (t_old - t0) / (tmax - t0)
            zr_old = zini + (cp[CP_ZMAX] - zini) * (x ** (1 / cp[CP_FSHAPE_R]))
        if zr_old < cp[CP_ZMIN]:
            zr_old = cp[CP_ZMIN]

        # potential root depth on current day
        if t_adj >= tmax:
            zr = cp[CP_ZMAX]
        elif t_adj <= t0:
            zr = zini
        else:
            x = (t_adj - t0) / (tmax - t0)
            zr = zini + (cp[CP_ZMAX] - zini) * (x ** (1 / cp[CP_FSHAPE_R]))
        if zr < cp[CP_ZMIN]:
            zr = cp[CP_ZMIN]

        zr_pot = zr
        d_zr = zr - zr_old

        # adjust expansion for restrictive soil horizons
        if zr > cp[CP_ZMIN]:
            layeri = 1
            zsoil = layer_dz_sum(layer, dz, ncomp, layeri)
            while (roundn(zsoil, 2) <= cp[CP_ZMIN]) and (layeri < nlayer):
                layeri = layeri + 1
                zsoil = zsoil + layer_dz_sum(layer, dz, ncomp, layeri)
            layer_comp = layer_first_comp(layer, ncomp, layeri)
            zr_adj = cp[CP_ZMIN]
            zr_remain = zr - cp[CP_ZMIN]
            delta_z = zsoil - cp[CP_ZMIN]
            end_prof = False
            zr_out = zr
            while not end_prof:
                zr_test = zr_adj + (zr_remain * (penetrability[layer_comp] / 100))
                if (
                    (layeri == nlayer)
                    or (penetrability[layer_comp] == 0)
                    or (zr_test <= zsoil)
                ):
                    zr_out = zr_test
                    end_prof = True
                else:
                    zr_adj = zsoil
                    zr_remain = zr_remain - (delta_z / (penetrability[layer_comp] / 100))
                    layeri = layeri + 1
                    layer_comp = layer_first_comp(layer, ncomp, layeri)
                    soil_layer_dz = layer_dz_sum(layer, dz, ncomp, layeri)
                    zsoil = zsoil + soil_layer_dz
                    delta_z = soil_layer_dz

            zr = fmax(zr_out, zr_old)
            d_zr = zr - zr_old

        # adjust rate of expansion for stomatal water stress
        if tr_ratio < 0.9999:
            if cp[CP_FSHAPE_EX] >= 0:
                d_zr = d_zr * tr_ratio
            else:
                f_adj = (math.exp(tr_ratio * cp[CP_FSHAPE_EX]) - 1) / (
                    math.exp(cp[CP_FSHAPE_EX]) - 1)
                d_zr = d_zr * f_adj

        # adjust for dry soil at expansion front
        if d_zr > 0.001:
            p_zexp = cp[CP_P_UP1] + ((1 - cp[CP_P_UP1]) / 2)
            zi_tmp = zroot_init + d_zr
            idx = first_ge(dzsum, ncomp, zi_tmp)
            taw_prof = th_fc[idx] - th_wp[idx]
            th_thr = th_fc[idx] - (p_zexp * taw_prof)
            if th[idx] < th_thr:
                if th[idx] <= th_wp[idx]:
                    d_zr = 0.0
                else:
                    w_rel = (th_fc[idx] - th[idx]) / taw_prof
                    d_rel = 1 - ((1 - w_rel) / (1 - p_zexp))
                    ks = 1 - (
                        (math.exp(d_rel * cp[CP_FSHAPE_W1]) - 1)
                        / (math.exp(cp[CP_FSHAPE_W1]) - 1)
                    )
                    d_zr = d_zr * ks

        # adjust for early senescence
        if (cc <= 0) and (cc_ns > 0.5):
            d_zr = 0.0

        # roots cannot expand if crop has not germinated
        if not germinated:
            d_zr = 0.0

        z_root = zroot_init + d_zr

        # adjust root density if deepening restricted
        if z_root < zr_pot:
            r_cor = (
                2 * (zr_pot / z_root) * ((cp[CP_SX_TOP] + cp[CP_SX_BOT]) / 2)
                - cp[CP_SX_TOP]
            ) / cp[CP_SX_BOT]
            if t_pot > 0:
                r_cor = r_cor * tr_ratio
                if r_cor < 1:
                    r_cor = 1.0
        else:
            r_cor = 1.0
    else:
        z_root = 0.0

    return z_root, r_cor


# --------------------------------------------------------------------------
# root zone water                               (solution/root_zone_water.py)
# --------------------------------------------------------------------------

@kernel
def root_zone_water(z_root, th, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry,
                    z_top, zmin, aer):
    """Returns (wr_act, dr_zt, dr_rz, taw_zt, taw_rz,
    thrz_act, thrz_s, thrz_fc, thrz_wp, thrz_dry, thrz_aer)."""
    rootdepth = roundn(fmax(z_root, zmin), 2)
    comp_sto = first_ge(dzsum, ncomp, rootdepth)

    wr_act = 0.0
    wr_s = 0.0
    wr_fc = 0.0
    wr_wp = 0.0
    wr_dry = 0.0
    wr_aer = 0.0
    for ii in range(comp_sto + 1):
        if dzsum[ii] > rootdepth:
            factor = 1 - ((dzsum[ii] - rootdepth) / dz[ii])
        else:
            factor = 1.0
        wr_act = wr_act + roundn(factor * 1000 * th[ii] * dz[ii], 2)
        wr_s = wr_s + roundn(factor * 1000 * th_s[ii] * dz[ii], 2)
        wr_fc = wr_fc + roundn(factor * 1000 * th_fc[ii] * dz[ii], 2)
        wr_wp = wr_wp + roundn(factor * 1000 * th_wp[ii] * dz[ii], 2)
        wr_dry = wr_dry + roundn(factor * 1000 * th_dry[ii] * dz[ii], 2)
        wr_aer = wr_aer + roundn(
            factor * 1000 * (th_s[ii] - (aer / 100)) * dz[ii], 2)

    if wr_act < 0:
        wr_act = 0.0

    taw_rz = fmax(wr_fc - wr_wp, 0.0)
    dr_rz = fmin(wr_fc - wr_act, taw_rz)

    thrz_act = wr_act / (rootdepth * 1000)
    thrz_s = wr_s / (rootdepth * 1000)
    thrz_fc = wr_fc / (rootdepth * 1000)
    thrz_wp = wr_wp / (rootdepth * 1000)
    thrz_dry = wr_dry / (rootdepth * 1000)
    thrz_aer = wr_aer / (rootdepth * 1000)

    # top soil
    if rootdepth > z_top:
        ztopdepth = roundn(z_top, 2)
        comp_sto_t = 0
        for ii in range(ncomp):
            if dzsum[ii] <= ztopdepth:
                comp_sto_t += 1
        wr_act_zt = 0.0
        wr_fc_zt = 0.0
        wr_wp_zt = 0.0
        for ii in range(comp_sto_t):
            if dzsum[ii] > ztopdepth:
                factor = 1 - ((dzsum[ii] - ztopdepth) / dz[ii])
            else:
                factor = 1.0
            wr_act_zt = wr_act_zt + (factor * 1000 * th[ii] * dz[ii])
            wr_fc_zt = wr_fc_zt + (factor * 1000 * th_fc[ii] * dz[ii])
            wr_wp_zt = wr_wp_zt + (factor * 1000 * th_wp[ii] * dz[ii])
        if wr_act_zt < 0:
            wr_act_zt = 0.0
        taw_zt = fmax(wr_fc_zt - wr_wp_zt, 0.0)
        dr_zt = fmin(wr_fc_zt - wr_act_zt, taw_zt)
    else:
        dr_zt = dr_rz
        taw_zt = taw_rz

    return (wr_act, dr_zt, dr_rz, taw_zt, taw_rz,
            thrz_act, thrz_s, thrz_fc, thrz_wp, thrz_dry, thrz_aer)


# --------------------------------------------------------------------------
# water stress                                     (solution/water_stress.py)
# --------------------------------------------------------------------------

@kernel
def water_stress(cp, t_early_sen, dr, taw, et0, beta):
    """Water stress coefficients.
    Returns (ksw_exp, ksw_sto, ksw_sen, ksw_pol, ksw_sto_lin)."""
    p_up0 = cp[CP_P_UP0]
    p_up1 = cp[CP_P_UP1]
    p_up2 = cp[CP_P_UP2]
    p_up3 = cp[CP_P_UP3]
    p_lo0 = cp[CP_P_LO0]
    p_lo1 = cp[CP_P_LO1]
    p_lo2 = cp[CP_P_LO2]
    p_lo3 = cp[CP_P_LO3]

    if cp[CP_ET_ADJ] == 1:
        # adjust stress thresholds for et0 (not for pollination)
        p_up0 = p_up0 + (0.04 * (5 - et0)) * (math.log10(10 - 9 * p_up0))
        p_lo0 = p_lo0 + (0.04 * (5 - et0)) * (math.log10(10 - 9 * p_lo0))
        p_up1 = p_up1 + (0.04 * (5 - et0)) * (math.log10(10 - 9 * p_up1))
        p_lo1 = p_lo1 + (0.04 * (5 - et0)) * (math.log10(10 - 9 * p_lo1))
        p_up2 = p_up2 + (0.04 * (5 - et0)) * (math.log10(10 - 9 * p_up2))
        p_lo2 = p_lo2 + (0.04 * (5 - et0)) * (math.log10(10 - 9 * p_lo2))

    # adjust senescence threshold if early senescence triggered
    if beta and (t_early_sen > 0):
        p_up2 = p_up2 * (1 - cp[CP_BETA] / 100)

    p_up0 = fmin(fmax(p_up0, 0.0), 1.0)
    p_up1 = fmin(fmax(p_up1, 0.0), 1.0)
    p_up2 = fmin(fmax(p_up2, 0.0), 1.0)
    p_up3 = fmin(fmax(p_up3, 0.0), 1.0)
    p_lo0 = fmin(fmax(p_lo0, 0.0), 1.0)
    p_lo1 = fmin(fmax(p_lo1, 0.0), 1.0)
    p_lo2 = fmin(fmax(p_lo2, 0.0), 1.0)
    p_lo3 = fmin(fmax(p_lo3, 0.0), 1.0)

    # relative depletion
    if dr <= (p_up0 * taw):
        drel0 = 0.0
    elif dr < (p_lo0 * taw):
        drel0 = 1 - ((p_lo0 - (dr / taw)) / (p_lo0 - p_up0))
    else:
        drel0 = 1.0
    if dr <= (p_up1 * taw):
        drel1 = 0.0
    elif dr < (p_lo1 * taw):
        drel1 = 1 - ((p_lo1 - (dr / taw)) / (p_lo1 - p_up1))
    else:
        drel1 = 1.0
    if dr <= (p_up2 * taw):
        drel2 = 0.0
    elif dr < (p_lo2 * taw):
        drel2 = 1 - ((p_lo2 - (dr / taw)) / (p_lo2 - p_up2))
    else:
        drel2 = 1.0
    if dr <= (p_up3 * taw):
        drel3 = 0.0
    elif dr < (p_lo3 * taw):
        drel3 = 1 - ((p_lo3 - (dr / taw)) / (p_lo3 - p_up3))
    else:
        drel3 = 1.0

    # stress coefficients
    ksw_exp = 1 - ((math.exp(drel0 * cp[CP_FSHAPE_W0]) - 1)
                   / (math.exp(cp[CP_FSHAPE_W0]) - 1))
    ksw_sto = 1 - ((math.exp(drel1 * cp[CP_FSHAPE_W1]) - 1)
                   / (math.exp(cp[CP_FSHAPE_W1]) - 1))
    ksw_sen = 1 - ((math.exp(drel2 * cp[CP_FSHAPE_W2]) - 1)
                   / (math.exp(cp[CP_FSHAPE_W2]) - 1))
    ksw_pol = 1 - drel3
    ksw_sto_lin = 1 - drel1
    return ksw_exp, ksw_sto, ksw_sen, ksw_pol, ksw_sto_lin


# --------------------------------------------------------------------------
# canopy development helpers    (solution/cc_development.py & related)
# --------------------------------------------------------------------------

@kernel
def cc_development(cco, ccx, cgc, cdc, dt, mode_decline, ccx0):
    """Canopy cover development (mode_decline: 0=growth, 1=decline)."""
    if mode_decline == 0:
        cc = cco * math.exp(cgc * dt)
        if cc > (ccx / 2):
            cc = ccx - 0.25 * (ccx / cco) * ccx * math.exp(-cgc * dt)
        if cc > ccx:
            cc = ccx
    else:
        if ccx < 0.001:
            cc = 0.0
        else:
            cc = ccx * (
                1
                - 0.05
                * (math.exp(dt * cdc * 3.33 * ((ccx + 2.29) / (ccx0 + 2.29))
                            / (ccx + 2.29)) - 1)
            )
    if cc > 1:
        cc = 1.0
    elif cc < 0:
        cc = 0.0
    return cc


@kernel
def cc_required_time(cc_prev, cco, ccx, cgc, cdc, mode_cdc):
    """Time to reach cc_prev given CGC (mode_cdc=0) or CDC (mode_cdc=1)."""
    if mode_cdc == 0:
        if cc_prev <= (ccx / 2):
            cgcx = math.log(cc_prev / cco)
        else:
            cgcx = math.log((0.25 * ccx * ccx / cco) / (ccx - cc_prev))
        t_req = cgcx / cgc
    else:
        t_req = (math.log(1 + (1 - cc_prev / ccx) / 0.05)) / (cdc / ccx)
    return t_req


@kernel
def adjust_ccx(cc_prev, cco, ccx, cgc, cdc, dt, tsum, canopy_dev_end, crop_ccx):
    """Adjust CCx for changes in CGC due to water stress."""
    tcc_tmp = cc_required_time(cc_prev, cco, ccx, cgc, cdc, 0)
    if tcc_tmp > 0:
        tcc_tmp = tcc_tmp + (canopy_dev_end - tsum) + dt
        ccx_adj = cc_development(cco, ccx, cgc, cdc, tcc_tmp, 0, crop_ccx)
    else:
        ccx_adj = 0.0
    return ccx_adj


@kernel
def update_ccx_cdc(cc_prev, cdc, ccx, dt):
    """Update CCx/CDC for rewatering in late season."""
    ccx_adj = cc_prev / (
        1 - 0.05 * (math.exp(dt * ((cdc * 3.33) / (ccx + 2.29))) - 1))
    cdc_adj = cdc * ((ccx_adj + 2.29) / (ccx + 2.29))
    return ccx_adj, cdc_adj


# --------------------------------------------------------------------------
# canopy cover                                     (solution/canopy_cover.py)
# --------------------------------------------------------------------------

@kernel
def canopy_cover(cp, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry, z_top,
                 th, z_root, gdd, gdd_cum, dap, delayed_cds, delayed_gdds,
                 et0, growing_season,
                 canopy_dev_end, senescence, maturity,
                 cc, cc_ns, cc_adj, cc_adj_ns, ccx_act, ccx_act_ns,
                 ccx_w, ccx_w_ns, ccx_early_sen, cc0_adj, protected_seed,
                 premat_senes, t_early_sen, crop_dead):
    """Simulate canopy growth/decline.

    Returns (cc, cc_ns, cc_adj, cc_adj_ns, ccx_act, ccx_act_ns, ccx_w,
    ccx_w_ns, ccx_early_sen, cc0_adj, cc_prev, protected_seed, premat_senes,
    t_early_sen, crop_dead).

    ``canopy_dev_end``/``senescence``/``maturity`` are passed explicitly so
    the caller can supply the per-pixel effective values.
    """
    init_cc_ns = cc_ns
    init_cc = cc
    init_protected_seed = protected_seed
    init_ccx_act = ccx_act
    init_crop_dead = crop_dead
    init_t_early_sen = t_early_sen
    init_ccx_w = ccx_w

    cc_prev = init_cc

    if growing_season:
        # root zone water content
        (_wr, dr_zt, dr_rz, taw_zt, taw_rz,
         _a, _s, _f, _w, _d, _ae) = root_zone_water(
            z_root, th, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry,
            z_top, cp[CP_ZMIN], cp[CP_AER])

        if (dr_rz / taw_rz) <= (dr_zt / taw_zt):
            dr = dr_rz
            taw = taw_rz
        else:
            dr = dr_zt
            taw = taw_zt

        ksw_exp, ksw_sto, ksw_sen, ksw_pol, ksw_sto_lin = water_stress(
            cp, t_early_sen, dr, taw, et0, True)

        # canopy cover growth time
        if cp[CP_CALENDAR_TYPE] == 1:
            dt_cc = 1.0
            t_cc_adj = dap - delayed_cds
        else:
            dt_cc = gdd
            t_cc_adj = gdd_cum - delayed_gdds

        # -------------------- potential canopy development --------------
        if (t_cc_adj < cp[CP_EMERGENCE]) or (round_half_even(t_cc_adj) > maturity):
            cc_ns = 0.0
        elif t_cc_adj < canopy_dev_end:
            if init_cc_ns <= cp[CP_CC0]:
                cc_ns = cp[CP_CC0] * math.exp(cp[CP_CGC] * dt_cc)
            else:
                tmp_t_cc = t_cc_adj - cp[CP_EMERGENCE]
                cc_ns = cc_development(
                    cp[CP_CC0], 0.98 * cp[CP_CCX], cp[CP_CGC], cp[CP_CDC],
                    tmp_t_cc, 0, cp[CP_CCX])
            ccx_act_ns = cc_ns
        elif t_cc_adj > canopy_dev_end:
            ccx_w_ns = ccx_act_ns
            if t_cc_adj < senescence:
                cc_ns = init_cc_ns
                ccx_act_ns = cc_ns
            else:
                tmp_t_cc = t_cc_adj - senescence
                cc_ns = cc_development(
                    cp[CP_CC0], ccx_act_ns, cp[CP_CGC], cp[CP_CDC],
                    tmp_t_cc, 1, ccx_act_ns)

        # -------------------- actual canopy development ------------------
        if (t_cc_adj < cp[CP_EMERGENCE]) or (round_half_even(t_cc_adj) > maturity):
            cc = 0.0
            cc0_adj = cp[CP_CC0]
        elif t_cc_adj < canopy_dev_end:
            if init_cc <= cc0_adj or (
                init_protected_seed and (init_cc <= (1.25 * cc0_adj))
            ):
                # very small canopy or protected seedling
                if init_protected_seed:
                    tmp_t_cc = t_cc_adj - cp[CP_EMERGENCE]
                    cc = cc_development(
                        cp[CP_CC0], cp[CP_CCX], cp[CP_CGC], cp[CP_CDC],
                        tmp_t_cc, 0, cp[CP_CCX])
                    if cc > (1.25 * cc0_adj):
                        protected_seed = False
                else:
                    cc = cc0_adj * math.exp(cp[CP_CGC] * dt_cc)
            else:
                # canopy growing
                if init_cc < (0.9799 * cp[CP_CCX]):
                    cgc_adj = cp[CP_CGC] * ksw_exp
                    if cgc_adj > 0:
                        ccx_adj_v = adjust_ccx(
                            init_cc, cc0_adj, cp[CP_CCX], cgc_adj, cp[CP_CDC],
                            dt_cc, t_cc_adj, canopy_dev_end, cp[CP_CCX])
                        if ccx_adj_v < 0:
                            cc = init_cc
                        elif abs(init_cc - (0.9799 * cp[CP_CCX])) < 0.001:
                            tmp_t_cc = t_cc_adj - cp[CP_EMERGENCE]
                            cc = cc_development(
                                cp[CP_CC0], cp[CP_CCX], cp[CP_CGC], cp[CP_CDC],
                                tmp_t_cc, 0, cp[CP_CCX])
                        else:
                            t_req = cc_required_time(
                                init_cc, cc0_adj, ccx_adj_v, cgc_adj,
                                cp[CP_CDC], 0)
                            if t_req > 0:
                                tmp_t_cc = t_req + dt_cc
                                cc = cc_development(
                                    cc0_adj, ccx_adj_v, cgc_adj, cp[CP_CDC],
                                    tmp_t_cc, 0, cp[CP_CCX])
                            else:
                                cc = init_cc
                    else:
                        cc = init_cc
                        if cc > cc0_adj:
                            cc0_adj = cp[CP_CC0]
                        else:
                            cc0_adj = cc
                else:
                    # approaching maximum size
                    tmp_t_cc = t_cc_adj - cp[CP_EMERGENCE]
                    cc = cc_development(
                        cp[CP_CC0], cp[CP_CCX], cp[CP_CGC], cp[CP_CDC],
                        tmp_t_cc, 0, cp[CP_CCX])
                    cc0_adj = cp[CP_CC0]

            if cc > init_ccx_act:
                ccx_act = cc

        elif t_cc_adj > canopy_dev_end:
            if t_cc_adj < senescence:
                cc = init_cc
                if cc > init_ccx_act:
                    ccx_act = cc
            else:
                cdc_adj = cp[CP_CDC] * ((ccx_act + 2.29) / (cp[CP_CCX] + 2.29))
                tmp_t_cc = t_cc_adj - senescence
                cc = cc_development(
                    cc0_adj, ccx_act, cp[CP_CGC], cdc_adj,
                    tmp_t_cc, 1, ccx_act)

            if (cc < 0.001) and (not init_crop_dead):
                cc = 0.0
                crop_dead = True

        # ---------------- canopy senescence due to water stress ----------
        if t_cc_adj >= cp[CP_EMERGENCE]:
            if (t_cc_adj < senescence) or (init_t_early_sen > 0):
                if (ksw_sen < 1) and (not init_protected_seed):
                    # early canopy senescence
                    premat_senes = True
                    if init_t_early_sen == 0:
                        ccx_early_sen = init_cc
                    t_early_sen = init_t_early_sen + dt_cc

                    ksw_exp2, ksw_sto2, ksw_sen2, ksw_pol2, ksw_sto_lin2 = (
                        water_stress(cp, t_early_sen, dr, taw, et0, False))

                    if ksw_sen2 > 0.99999:
                        cdc_adj = 0.0001
                    else:
                        cdc_adj = (1 - (ksw_sen2 ** 8)) * cp[CP_CDC]

                    if ccx_early_sen < 0.001:
                        cc_sen = 0.0
                    else:
                        t_req = (math.log(
                            1 + (1 - init_cc / ccx_early_sen) / 0.05)) / (
                            (cdc_adj * 3.33) / (ccx_early_sen + 2.29))
                        tmp_t_cc = t_req + dt_cc
                        cc_sen = ccx_early_sen * (
                            1 - 0.05 * (
                                math.exp(tmp_t_cc * ((cdc_adj * 3.33)
                                         / (ccx_early_sen + 2.29))) - 1)
                        )
                        if cc_sen < 0:
                            cc_sen = 0.0

                    if t_cc_adj < senescence:
                        if cc_sen > cp[CP_CCX]:
                            cc_sen = cp[CP_CCX]
                        cc = cc_sen
                        if cc > init_cc:
                            cc = init_cc
                        ccx_act = cc
                        if cc < cp[CP_CC0]:
                            cc0_adj = cc
                        else:
                            cc0_adj = cp[CP_CC0]
                    else:
                        if cc_sen < cc:
                            cc = cc_sen

                    if (cc < 0.001) and (not init_crop_dead):
                        cc = 0.0
                        crop_dead = True
                else:
                    # no water stress
                    premat_senes = False
                    if (t_cc_adj > senescence) and (init_t_early_sen > 0):
                        # rewatering of canopy in late season
                        tmp_t_cc = t_cc_adj - dt_cc - senescence
                        ccx_adj_v, cdc_adj = update_ccx_cdc(
                            init_cc, cp[CP_CDC], cp[CP_CCX], tmp_t_cc)
                        ccx_act = ccx_adj_v
                        tmp_t_cc = t_cc_adj - senescence
                        cc = cc_development(
                            cc0_adj, ccx_adj_v, cp[CP_CGC], cdc_adj,
                            tmp_t_cc, 1, ccx_adj_v)
                        if (cc < 0.001) and (not init_crop_dead):
                            cc = 0.0
                            crop_dead = True
                    t_early_sen = 0.0

                # adjust CCx for effects of withered canopy
                if cc > init_ccx_w:
                    ccx_w = cc

        # ------------- micro-advective adjustments -----------------------
        if cc_ns < cc:
            cc_ns = cc
            if t_cc_adj < canopy_dev_end:
                ccx_act_ns = cc_ns

        cc_adj = (1.72 * cc) - (cc ** 2) + (0.3 * (cc ** 3))
        cc_adj_ns = ((1.72 * cc_ns) - (cc_ns ** 2) + (0.3 * (cc_ns ** 3)))
    else:
        cc = 0.0
        cc_adj = 0.0
        cc_ns = 0.0
        cc_adj_ns = 0.0
        ccx_w = 0.0
        ccx_act = 0.0
        ccx_w_ns = 0.0
        ccx_act_ns = 0.0

    return (cc, cc_ns, cc_adj, cc_adj_ns, ccx_act, ccx_act_ns, ccx_w,
            ccx_w_ns, ccx_early_sen, cc0_adj, cc_prev, protected_seed,
            premat_senes, t_early_sen, crop_dead)


# --------------------------------------------------------------------------
# soil evaporation           (solution/soil_evaporation.py + evap layer w.c.)
# --------------------------------------------------------------------------

@kernel
def evap_layer_water_content(th, evap_z, ncomp, dz, dzsum,
                             th_s, th_fc, th_wp, th_dry):
    """Water contents in the evaporation layer.
    Returns (w_sat, w_fc, w_wp, w_dry, w_act)."""
    comp_sto = fmin(count_lt(dzsum, ncomp, evap_z) + 1, ncomp)
    w_sat = 0.0
    w_fc = 0.0
    w_wp = 0.0
    w_dry = 0.0
    w_act = 0.0
    for ii in range(comp_sto):
        if dzsum[ii] > evap_z:
            factor = 1 - ((dzsum[ii] - evap_z) / dz[ii])
        else:
            factor = 1.0
        w_act += factor * 1000 * th[ii] * dz[ii]
        w_sat += factor * 1000 * th_s[ii] * dz[ii]
        w_fc += factor * 1000 * th_fc[ii] * dz[ii]
        w_wp += factor * 1000 * th_wp[ii] * dz[ii]
        w_dry += factor * 1000 * th_dry[ii] * dz[ii]
    if w_act < 0:
        w_act = 0.0
    return w_sat, w_fc, w_wp, w_dry, w_act


@kernel
def soil_evaporation(sp, cp, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry,
                     evap_time_steps, day_idx, dap, delayed_cds, gdd_cum,
                     delayed_gdds, senescence,
                     ccx_w, cc_adj, ccx_act, cc, premat_senes,
                     surface_storage, w_surf, evap_z, stage2, w_stage2,
                     th, et0, infl, prcp, growing_season):
    """Daily soil evaporation (rainfed: no irrigation/mulches/bunds).
    Updates ``th`` in place.
    Returns (e_pot, stage2, w_stage2, w_surf, surface_storage, evap_z,
    es_act)."""
    # prepare stage 2 (first day of simulation == first day of season here,
    # matching aquacrop with off_season=False)
    if (day_idx == 0) or (dap == 1):
        w_surf = 0.0
        evap_z = sp[SP_EVAP_Z_MIN]
        stage2 = True
        w_sat, w_fc, w_wp, w_dry, w_act = evap_layer_water_content(
            th, evap_z, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry)
        w_stage2 = roundn(
            (w_act - (w_fc - sp[SP_REW])) / (w_sat - (w_fc - sp[SP_REW])), 2)
        if w_stage2 < 0:
            w_stage2 = 0.0

    # prepare stage 1 when rainfall occurs
    if prcp > 0:
        if infl > 0:
            w_surf = infl
            if w_surf > sp[SP_REW]:
                w_surf = sp[SP_REW]
            w_stage2 = 0.0
            evap_z = sp[SP_EVAP_Z_MIN]
            stage2 = False

    # potential soil evaporation rate (mm/day)
    if growing_season:
        if cp[CP_CALENDAR_TYPE] == 1:
            t_adj = dap - delayed_cds
        else:
            t_adj = gdd_cum - delayed_gdds

        es_pot_max = sp[SP_KEX] * et0 * (1 - ccx_w * (sp[SP_FWCC] / 100))
        es_pot = sp[SP_KEX] * (1 - cc_adj) * et0

        if (t_adj > senescence) and (ccx_act > 0):
            if cc > (ccx_act / 2):
                if cc > ccx_act:
                    mult = 0.0
                else:
                    mult = (ccx_act - cc) / (ccx_act / 2)
            else:
                mult = 1.0
            es_pot = es_pot * (1 - ccx_act * (sp[SP_FWCC] / 100) * mult)
            ccx_act_adj = ((1.72 * ccx_act) - (ccx_act ** 2)
                           + 0.3 * (ccx_act ** 3))
            es_pot_min = sp[SP_KEX] * (1 - ccx_act_adj) * et0
            if es_pot_min < 0:
                es_pot_min = 0.0
            if es_pot < es_pot_min:
                es_pot = es_pot_min
            elif es_pot > es_pot_max:
                es_pot = es_pot_max

        if premat_senes:
            if es_pot > es_pot_max:
                es_pot = es_pot_max
    else:
        es_pot = sp[SP_KEX] * et0

    # (no mulches, no irrigation-wetting adjustments in rainfed scope)

    # surface evaporation
    es_act = 0.0
    if surface_storage > 0:
        if surface_storage > es_pot:
            es_act = es_pot
            surface_storage = surface_storage - es_act
        else:
            es_act = surface_storage
            surface_storage = 0.0
            w_surf = sp[SP_REW]
            w_stage2 = 0.0
            evap_z = sp[SP_EVAP_Z_MIN]
            stage2 = False

    # stage 1 evaporation
    to_extract = es_pot - es_act
    extract_pot_stg1 = fmin(to_extract, w_surf)
    if extract_pot_stg1 > 0:
        comp_sto = fmin(count_lt(dzsum, ncomp, sp[SP_EVAP_Z_MIN]) + 1, ncomp - 1)
        comp = -1
        while (extract_pot_stg1 > 0) and (comp < comp_sto):
            comp = comp + 1
            if dzsum[comp] > sp[SP_EVAP_Z_MIN]:
                factor = 1 - ((dzsum[comp] - sp[SP_EVAP_Z_MIN]) / dz[comp])
            else:
                factor = 1.0
            w_dry_c = 1000 * th_dry[comp] * dz[comp]
            w = 1000 * th[comp] * dz[comp]
            av_w = (w - w_dry_c) * factor
            if av_w < 0:
                av_w = 0.0
            if av_w >= extract_pot_stg1:
                es_act = es_act + extract_pot_stg1
                w = w - extract_pot_stg1
                to_extract = to_extract - extract_pot_stg1
                extract_pot_stg1 = 0.0
            else:
                es_act = es_act + av_w
                extract_pot_stg1 = extract_pot_stg1 - av_w
                to_extract = to_extract - av_w
                w = w - av_w
            th[comp] = w / (1000 * dz[comp])

        w_surf = w_surf - es_act
        if (w_surf < 0) or (extract_pot_stg1 > 0.0001):
            w_surf = 0.0

        if w_surf < 0.0001:
            w_sat, w_fc, w_wp, w_dry, w_act = evap_layer_water_content(
                th, evap_z, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry)
            w_stage2 = roundn(
                (w_act - (w_fc - sp[SP_REW])) / (w_sat - (w_fc - sp[SP_REW])),
                2)
            if w_stage2 < 0:
                w_stage2 = 0.0

    # stage 2 evaporation
    if to_extract > 0:
        stage2 = True
        edt = to_extract / evap_time_steps
        for _jj in range(evap_time_steps):
            w_sat, w_fc, w_wp, w_dry, w_act = evap_layer_water_content(
                th, evap_z, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry)
            w_upper = (w_stage2 * (w_sat - (w_fc - sp[SP_REW]))
                       + (w_fc - sp[SP_REW]))
            w_lower = w_dry
            w_rel = (w_act - w_lower) / (w_upper - w_lower)
            if sp[SP_EVAP_Z_MAX] > sp[SP_EVAP_Z_MIN]:
                w_check = sp[SP_F_WREL_EXP] * (
                    (sp[SP_EVAP_Z_MAX] - evap_z)
                    / (sp[SP_EVAP_Z_MAX] - sp[SP_EVAP_Z_MIN]))
                while (w_rel < w_check) and (evap_z < sp[SP_EVAP_Z_MAX]):
                    evap_z = evap_z + 0.001
                    w_sat, w_fc, w_wp, w_dry, w_act = evap_layer_water_content(
                        th, evap_z, ncomp, dz, dzsum, th_s, th_fc, th_wp,
                        th_dry)
                    w_upper = (w_stage2 * (w_sat - (w_fc - sp[SP_REW]))
                               + (w_fc - sp[SP_REW]))
                    w_lower = w_dry
                    w_rel = (w_act - w_lower) / (w_upper - w_lower)
                    w_check = sp[SP_F_WREL_EXP] * (
                        (sp[SP_EVAP_Z_MAX] - evap_z)
                        / (sp[SP_EVAP_Z_MAX] - sp[SP_EVAP_Z_MIN]))

            kr = ((math.exp(sp[SP_F_EVAP] * w_rel) - 1)
                  / (math.exp(sp[SP_F_EVAP]) - 1))
            if kr > 1:
                kr = 1.0

            to_extract_stg2 = kr * edt

            comp_sto = fmin(count_lt(dzsum, ncomp, evap_z) + 1, ncomp - 1)
            comp = -1
            while (to_extract_stg2 > 0) and (comp < comp_sto):
                comp = comp + 1
                if dzsum[comp] > evap_z:
                    factor = 1 - ((dzsum[comp] - evap_z) / dz[comp])
                else:
                    factor = 1.0
                w_dry_c = 1000 * th_dry[comp] * dz[comp]
                w = 1000 * th[comp] * dz[comp]
                av_w = (w - w_dry_c) * factor
                if av_w >= to_extract_stg2:
                    es_act = es_act + to_extract_stg2
                    w = w - to_extract_stg2
                    to_extract = to_extract - to_extract_stg2
                    to_extract_stg2 = 0.0
                else:
                    es_act = es_act + av_w
                    w = w - av_w
                    to_extract_stg2 = to_extract_stg2 - av_w
                    to_extract = to_extract - av_w
                th[comp] = w / (1000 * dz[comp])

    e_pot = es_pot
    return e_pot, stage2, w_stage2, w_surf, surface_storage, evap_z, es_act


# --------------------------------------------------------------------------
# aeration stress                               (solution/aeration_stress.py)
# --------------------------------------------------------------------------

@kernel
def aeration_stress(aer_days, lag_aer, thrz_act, thrz_s, thrz_aer):
    """Aeration stress coefficient. Returns (ksa_aer, aer_days)."""
    if thrz_act > thrz_aer:
        if aer_days < lag_aer:
            stress = 1 - ((thrz_s - thrz_act) / (thrz_s - thrz_aer))
            ksa_aer = 1 - ((aer_days / 3) * stress)
        else:
            ksa_aer = (thrz_s - thrz_act) / (thrz_s - thrz_aer)
        aer_days = aer_days + 1
        if aer_days > lag_aer:
            aer_days = lag_aer
    else:
        ksa_aer = 1.0
        aer_days = 0.0
    return ksa_aer, aer_days


# --------------------------------------------------------------------------
# transpiration                                   (solution/transpiration.py)
# --------------------------------------------------------------------------

@kernel
def transpiration(cp, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry, z_top,
                  th, aer_days_comp, max_canopy_cd,
                  dap, delayed_cds, age_days, age_days_ns, ccx_w_ns, ccx_w,
                  cc_adj, cc_adj_ns, cc_ns, cc, cc_prev, z_root, r_cor,
                  t_early_sen, surface_storage, day_submerged, aer_days,
                  tr_ratio, et0, gdd, growing_season):
    """Crop transpiration (rainfed: irrigation method 0).

    ``max_canopy_cd`` is the per-pixel MaxCanopyCD (calendar days).
    Updates ``th`` and ``aer_days_comp`` in place.
    Returns (tr_act, tr_pot_ns, tr_pot0, age_days, age_days_ns,
    day_submerged, surface_storage, aer_days, tr_ratio, cc, t_pot)."""
    if growing_season:
        # 1. potential transpiration, no prior water stress
        dap_adj = dap - delayed_cds
        if dap_adj > max_canopy_cd:
            age_days_ns = dap_adj - max_canopy_cd
        if age_days_ns > 5:
            kcb_ns = (cp[CP_KCB]
                      - ((age_days_ns - 5) * (cp[CP_FAGE] / 100)) * ccx_w_ns)
        else:
            kcb_ns = cp[CP_KCB]

        co2_conc = cp[CP_CO2_CONC]
        co2_ref = cp[CP_CO2_REF]
        if co2_conc > co2_ref:
            kcb_ns = kcb_ns * (1 - 0.05 * ((co2_conc - co2_ref)
                                           / (550 - co2_ref)))
        tr_pot_ns = kcb_ns * cc_adj_ns * et0
        if cc_ns < ccx_w_ns:
            if (ccx_w_ns > 0.001) and (cc_ns > 0.001):
                tr_pot_ns = tr_pot_ns * ((cc_ns / ccx_w_ns) ** cp[CP_A_TR])

        # 2. potential prior water stress / delayed development
        dap_adj = dap - delayed_cds
        if dap_adj > max_canopy_cd:
            age_days = dap_adj - max_canopy_cd
        if age_days > 5:
            kcb = cp[CP_KCB] - ((age_days - 5) * (cp[CP_FAGE] / 100)) * ccx_w
        else:
            kcb = cp[CP_KCB]
        if co2_conc > co2_ref:
            kcb = kcb * (1 - 0.05 * ((co2_conc - co2_ref) / (550 - co2_ref)))
        tr_pot0 = kcb * cc_adj * et0
        if cc < ccx_w:
            if (ccx_w > 0.001) and (cc > 0.001):
                tr_pot0 = tr_pot0 * ((cc / ccx_w) ** cp[CP_A_TR])

        # 3. cold stress effects
        if cp[CP_TR_COLD_STRESS] == 0:
            ks_cold = 1.0
        else:
            if gdd >= cp[CP_GDD_UP]:
                ks_cold = 1.0
            elif gdd <= cp[CP_GDD_LO]:
                ks_cold = 0.0
            else:
                ks_tr_up = 1.0
                ks_tr_lo = 0.02
                fshapeb = (-1) * (
                    math.log(((ks_tr_lo * ks_tr_up) - 0.98 * ks_tr_lo)
                             / (0.98 * (ks_tr_up - ks_tr_lo))))
                gdd_rel = ((gdd - cp[CP_GDD_LO])
                           / (cp[CP_GDD_UP] - cp[CP_GDD_LO]))
                ks_cold = (ks_tr_up * ks_tr_lo) / (
                    ks_tr_lo + (ks_tr_up - ks_tr_lo)
                    * math.exp(-fshapeb * gdd_rel))
                ks_cold = ks_cold - ks_tr_lo * (1 - gdd_rel)
        tr_pot0 = tr_pot0 * ks_cold
        tr_pot_ns = tr_pot_ns * ks_cold

        # surface layer transpiration
        if (surface_storage > 0) and (day_submerged < cp[CP_LAG_AER]):
            day_submerged = day_submerged + 1
            for ii in range(ncomp):
                aer_days_comp[ii] = aer_days_comp[ii] + 1
                if aer_days_comp[ii] > cp[CP_LAG_AER]:
                    aer_days_comp[ii] = cp[CP_LAG_AER]
            f_sub = 1 - (day_submerged / cp[CP_LAG_AER])
            if surface_storage > (f_sub * tr_pot0):
                surface_storage = surface_storage - (f_sub * tr_pot0)
                tr_act0 = f_sub * tr_pot0
            else:
                tr_act0 = 0.0
            if tr_act0 < (f_sub * tr_pot0):
                tr_pot = (f_sub * tr_pot0) - tr_act0
            else:
                tr_pot = 0.0
        else:
            tr_pot = tr_pot0
            tr_act0 = 0.0

        # update potential root zone transpiration for water stress
        (_wr, dr_zt, dr_rz, taw_zt, taw_rz,
         thrz_act, thrz_s, thrz_fc, thrz_wp, thrz_dry, thrz_aer) = (
            root_zone_water(z_root, th, ncomp, dz, dzsum, th_s, th_fc,
                            th_wp, th_dry, z_top, cp[CP_ZMIN], cp[CP_AER]))
        if (dr_rz / taw_rz) <= (dr_zt / taw_zt):
            dr = dr_rz
            taw = taw_rz
        else:
            dr = dr_zt
            taw = taw_zt

        ksw_exp, ksw_sto, ksw_sen, ksw_pol, ksw_sto_lin = water_stress(
            cp, t_early_sen, dr, taw, et0, True)

        ksa_aer, aer_days = aeration_stress(
            aer_days, cp[CP_LAG_AER], thrz_act, thrz_s, thrz_aer)
        ks = fmin(ksw_sto_lin, ksa_aer)
        tr_pot = tr_pot * ks

        # compartments covered by the root zone
        rootdepth = roundn(fmax(z_root, cp[CP_ZMIN]), 2)
        comp_sto = fmin(count_lt(dzsum, ncomp, rootdepth) + 1, ncomp)

        # extract water (root fraction and sink term computed on the fly)
        to_extract = tr_pot
        comp = -1
        tr_act = 0.0
        sx_comp_bot = cp[CP_SX_TOP]
        p_up_sto = cp[CP_P_UP1]
        if cp[CP_ET_ADJ] == 1:
            p_up_sto = (cp[CP_P_UP1]
                        + (0.04 * (5 - et0))
                        * (math.log10(10 - 9 * cp[CP_P_UP1])))
        while (to_extract > 0) and (comp < comp_sto - 1):
            comp = comp + 1

            # fraction of compartment covered by root zone
            if dzsum[comp] > rootdepth:
                root_fact = 1 - ((dzsum[comp] - rootdepth) / dz[comp])
            else:
                root_fact = 1.0

            # maximum sink term (declines linearly with depth)
            sx_comp_top = sx_comp_bot
            if dzsum[comp] <= rootdepth:
                sx_comp_bot = cp[CP_SX_BOT] * r_cor + (
                    (cp[CP_SX_TOP] - cp[CP_SX_BOT] * r_cor)
                    * ((rootdepth - dzsum[comp]) / rootdepth))
            else:
                sx_comp_bot = cp[CP_SX_BOT] * r_cor
            sx_comp = (sx_comp_top + sx_comp_bot) / 2

            # taw for compartment
            th_taw = th_fc[comp] - th_wp[comp]
            th_crit = th_fc[comp] - (th_taw * p_up_sto)

            # soil water stress
            if th[comp] >= th_crit:
                ks_comp = 1.0
            elif th[comp] > th_wp[comp]:
                w_rel = ((th_fc[comp] - th[comp])
                         / (th_fc[comp] - th_wp[comp]))
                p_rel = ((w_rel - cp[CP_P_UP1])
                         / (cp[CP_P_LO1] - cp[CP_P_UP1]))
                if p_rel <= 0:
                    ks_comp = 1.0
                elif p_rel >= 1:
                    ks_comp = 0.0
                else:
                    ks_comp = 1 - (
                        (math.exp(p_rel * cp[CP_FSHAPE_W1]) - 1)
                        / (math.exp(cp[CP_FSHAPE_W1]) - 1))
                if ks_comp > 1:
                    ks_comp = 1.0
                elif ks_comp < 0:
                    ks_comp = 0.0
            else:
                ks_comp = 0.0

            # aeration stress
            if day_submerged >= cp[CP_LAG_AER]:
                aer_comp = 0.0
            elif th[comp] > (th_s[comp] - (cp[CP_AER] / 100)):
                aer_days_comp[comp] = aer_days_comp[comp] + 1
                if aer_days_comp[comp] >= cp[CP_LAG_AER]:
                    aer_days_comp[comp] = cp[CP_LAG_AER]
                    f_aer = 0.0
                else:
                    f_aer = 1.0
                aer_comp = (th_s[comp] - th[comp]) / (
                    th_s[comp] - (th_s[comp] - (cp[CP_AER] / 100)))
                if aer_comp < 0:
                    aer_comp = 0.0
                aer_comp = ((f_aer + (aer_days_comp[comp] - 1) * aer_comp)
                            / (f_aer + aer_days_comp[comp] - 1))
            else:
                aer_comp = 1.0
                aer_days_comp[comp] = 0.0

            # extract water
            th_to_extract = (to_extract / 1000) / dz[comp]
            if ks_comp == aer_comp:
                sink = ks_comp * sx_comp * root_fact
            else:
                sink = fmin(ks_comp, aer_comp) * sx_comp * root_fact

            if th_to_extract < sink:
                sink = th_to_extract

            if (th[comp] - sink) < th_dry[comp]:
                sink = th[comp] - th_dry[comp]
                if sink < 0:
                    sink = 0.0

            th[comp] = th[comp] - sink
            to_extract = to_extract - (sink * 1000 * dz[comp])
            tr_act = tr_act + (sink * 1000 * dz[comp])

        # add any surface transpiration
        tr_act = tr_act + tr_act0

        # feedback with canopy cover development
        if ((cc - cc_prev) > 0.005) and (tr_act == 0):
            cc = cc_prev

        # update transpiration ratio
        if tr_pot0 > 0:
            if tr_act < tr_pot0:
                tr_ratio = tr_act / tr_pot0
            else:
                tr_ratio = 1.0
        else:
            tr_ratio = 1.0
        if tr_ratio < 0:
            tr_ratio = 0.0
        elif tr_ratio > 1:
            tr_ratio = 1.0
    else:
        tr_act = 0.0
        tr_pot0 = 0.0
        tr_pot_ns = 0.0

    t_pot = tr_pot0
    return (tr_act, tr_pot_ns, tr_pot0, age_days, age_days_ns,
            day_submerged, surface_storage, aer_days, tr_ratio, cc, t_pot)


# --------------------------------------------------------------------------
# reference harvest index                     (solution/HIref_current_day.py)
# --------------------------------------------------------------------------

@kernel
def hi_ref_current_day(cp, hi_start_cd, yld_form_cd, higc, t_lin_switch,
                       d_hi_linear, hi_ref, hi_final, dap, delayed_cds,
                       yield_form, pct_lag_phase, cc, ccx_w, growing_season):
    """Reference harvest index on current day.
    Returns (hi_ref, yield_form, pct_lag_phase, hi_final)."""
    if growing_season:
        t_adj = dap - delayed_cds
        if t_adj > hi_start_cd:
            yield_form = True
        else:
            yield_form = False

        hi_t = dap - delayed_cds - hi_start_cd - 1

        if hi_t <= 0:
            hi_ref = 0.0
            pct_lag_phase = 0.0
        else:
            crop_type = cp[CP_CROP_TYPE]
            if (crop_type == 1) or (crop_type == 2):
                pct_lag_phase = 100.0
                hi_ref = (cp[CP_HI_INI] * cp[CP_HI0]) / (
                    cp[CP_HI_INI] + (cp[CP_HI0] - cp[CP_HI_INI])
                    * math.exp(-higc * hi_t))
                if hi_ref >= (0.9799 * cp[CP_HI0]):
                    hi_ref = cp[CP_HI0]
            else:  # crop_type == 3
                if hi_t < t_lin_switch:
                    pct_lag_phase = 100 * (hi_t / t_lin_switch)
                    hi_ref = (cp[CP_HI_INI] * cp[CP_HI0]) / (
                        cp[CP_HI_INI] + (cp[CP_HI0] - cp[CP_HI_INI])
                        * math.exp(-higc * hi_t))
                else:
                    pct_lag_phase = 100.0
                    hi_ref = (cp[CP_HI_INI] * cp[CP_HI0]) / (
                        cp[CP_HI_INI] + (cp[CP_HI0] - cp[CP_HI_INI])
                        * math.exp(-higc * t_lin_switch))
                    hi_ref = hi_ref + (d_hi_linear * (hi_t - t_lin_switch))

            # limit hi_ref
            if hi_ref > cp[CP_HI0]:
                hi_ref = cp[CP_HI0]
            elif hi_ref <= (cp[CP_HI_INI] + 0.004):
                hi_ref = 0.0
            elif (cp[CP_HI0] - hi_ref) < 0.004:
                hi_ref = cp[CP_HI0]

            # adjust hi_ref for inadequate photosynthesis
            if ((hi_final == cp[CP_HI0]) and (hi_t <= yld_form_cd)
                    and (cc <= 0.05) and (ccx_w > 0) and (cc < ccx_w)
                    and (cp[CP_CROP_TYPE] == 2 or cp[CP_CROP_TYPE] == 3)):
                hi_final = hi_ref

            if hi_ref > hi_final:
                hi_ref = hi_final
    else:
        hi_ref = 0.0

    return hi_ref, yield_form, pct_lag_phase, hi_final


# --------------------------------------------------------------------------
# biomass accumulation                     (solution/biomass_accumulation.py)
# --------------------------------------------------------------------------

@kernel
def biomass_accumulation(cp, hi_start_cd, yld_form_cd, dap, delayed_cds,
                         hi_ref, pct_lag_phase, b, b_ns, tr, tr_pot, et0,
                         growing_season):
    """Biomass accumulation. Returns (b, b_ns)."""
    if growing_season:
        hi_t = dap - delayed_cds - hi_start_cd - 1
        crop_type = cp[CP_CROP_TYPE]
        if ((crop_type == 2) or (crop_type == 3)) and (hi_ref > 0):
            if cp[CP_DETERMINANT] == 1:
                fswitch = pct_lag_phase / 100
            else:
                if hi_t < (yld_form_cd / 3):
                    fswitch = hi_t / (yld_form_cd / 3)
                else:
                    fswitch = 1.0
            wp_adj = cp[CP_WP] * (1 - (1 - cp[CP_WPY] / 100) * fswitch)
        else:
            wp_adj = cp[CP_WP]

        wp_adj = wp_adj * cp[CP_F_CO2]

        db_ns = wp_adj * (tr_pot / et0)
        db = wp_adj * (tr / et0)
        if db != db:  # nan check
            db = 0.0

        b = b + db
        b_ns = b_ns + db_ns
    else:
        b = 0.0
        b_ns = 0.0
    return b, b_ns


# --------------------------------------------------------------------------
# temperature stress                         (solution/temperature_stress.py)
# --------------------------------------------------------------------------

@kernel
def temperature_stress(cp, temp_max, temp_min):
    """Temperature stress coefficients for pollination.
    Returns (kst_polh, kst_polc)."""
    ks_pol_up = 1.0
    ks_pol_lo = 0.001

    if cp[CP_POL_HEAT_STRESS] == 0:
        kst_polh = 1.0
    else:
        if temp_max <= cp[CP_TMAX_LO]:
            kst_polh = 1.0
        elif temp_max >= cp[CP_TMAX_UP]:
            kst_polh = 0.0
        else:
            t_rel = ((temp_max - cp[CP_TMAX_LO])
                     / (cp[CP_TMAX_UP] - cp[CP_TMAX_LO]))
            kst_polh = (ks_pol_up * ks_pol_lo) / (
                ks_pol_lo + (ks_pol_up - ks_pol_lo)
                * math.exp(-cp[CP_FSHAPE_B] * (1 - t_rel)))

    if cp[CP_POL_COLD_STRESS] == 0:
        kst_polc = 1.0
    else:
        if temp_min >= cp[CP_TMIN_UP]:
            kst_polc = 1.0
        elif temp_min <= cp[CP_TMIN_LO]:
            kst_polc = 0.0
        else:
            t_rel = ((cp[CP_TMIN_UP] - temp_min)
                     / (cp[CP_TMIN_UP] - cp[CP_TMIN_LO]))
            kst_polc = (ks_pol_up * ks_pol_lo) / (
                ks_pol_lo + (ks_pol_up - ks_pol_lo)
                * math.exp(-cp[CP_FSHAPE_B] * (1 - t_rel)))

    return kst_polh, kst_polc


# --------------------------------------------------------------------------
# harvest index adjustments                (solution/HIadj_*.py)
# --------------------------------------------------------------------------

@kernel
def hi_adj_pre_anthesis(b, b_ns, cc, dhi_pre):
    """Adjustment to HI for pre-anthesis water stress. Returns f_pre."""
    if dhi_pre > 0:
        br = b / b_ns
        br_range = math.log(dhi_pre) / 5.62
        br_upp = 1.0
        br_low = 1 - br_range
        br_top = br_upp - (br_range / 3)

        ratio_low = (br - br_low) / (br_top - br_low)
        ratio_upp = (br - br_top) / (br_upp - br_top)

        if (br >= br_low) and (br < br_top):
            f_pre = 1 + (((1 + math.sin((1.5 - ratio_low) * math.pi)) / 2)
                         * (dhi_pre / 100))
        elif (br > br_top) and (br <= br_upp):
            f_pre = 1 + (((1 + math.sin((0.5 + ratio_upp) * math.pi)) / 2)
                         * (dhi_pre / 100))
        else:
            f_pre = 1.0
    else:
        f_pre = 1.0

    if cc <= 0.01:
        f_pre = 0.0
    return f_pre


@kernel
def hi_adj_pollination(cc, f_pol, flowering_cd, cc_min, exc,
                       ksw_pol, kst_polh, kst_polc, hi_t):
    """Adjustment to HI for pollination failure. Returns f_pol."""
    if hi_t == 0:
        frac_flow = 0.0
    else:
        t1 = hi_t - 1
        if t1 == 0:
            f1 = 0.0
        else:
            t1_pct = 100 * (t1 / flowering_cd)
            if t1_pct > 100:
                t1_pct = 100.0
            f1 = (0.00558 * math.exp(0.63 * math.log(t1_pct))
                  - (0.000969 * t1_pct) - 0.00383)
        if f1 < 0:
            f1 = 0.0

        t2 = hi_t
        if t2 == 0:
            f2 = 0.0
        else:
            t2_pct = 100 * (t2 / flowering_cd)
            if t2_pct > 100:
                t2_pct = 100.0
            f2 = (0.00558 * math.exp(0.63 * math.log(t2_pct))
                  - (0.000969 * t2_pct) - 0.00383)
        if f2 < 0:
            f2 = 0.0

        if abs(f1 - f2) < 0.0000001:
            f = 0.0
        else:
            f = 100 * ((f1 + f2) / 2) / flowering_cd
        frac_flow = f

    if cc < cc_min:
        d_fpol = 0.0
    else:
        ks = fmin(fmin(ksw_pol, kst_polc), kst_polh)
        d_fpol = ks * frac_flow * (1 + (exc / 100))

    f_pol = f_pol + d_fpol
    if f_pol > 1:
        f_pol = 1.0
    return f_pol


@kernel
def hi_adj_post_anthesis(cp, canopy_dev_end_cd, hi_start_cd, hi_end_cd,
                         yld_form_cd, delayed_cds, s_cor1, s_cor2, dap,
                         f_pre, cc, fpost_upp, fpost_dwn, ksw_exp, ksw_sto):
    """Adjustment to HI for post-anthesis water stress.
    Returns (s_cor1, s_cor2, fpost_upp, fpost_dwn, f_post)."""
    # 1. adjustment for leaf expansion
    tmax1 = canopy_dev_end_cd - hi_start_cd
    dap_adj = dap - delayed_cds
    if ((dap_adj <= (canopy_dev_end_cd + 1)) and (tmax1 > 0)
            and (f_pre > 0.99) and (cc > 0.001) and (cp[CP_A_HI] > 0)):
        d_cor = 1 + (1 - ksw_exp) / cp[CP_A_HI]
        s_cor1 = s_cor1 + (d_cor / tmax1)
        day_cor = dap_adj - 1 - hi_start_cd
        fpost_upp = (tmax1 / day_cor) * s_cor1

    # 2. adjustment for stomatal closure
    tmax2 = yld_form_cd
    dap_adj = dap - delayed_cds
    if ((dap_adj <= (hi_end_cd + 1)) and (tmax2 > 0)
            and (f_pre > 0.99) and (cc > 0.001) and (cp[CP_B_HI] > 0)):
        d_cor = (ksw_sto ** 0.1) * (1 - (1 - ksw_sto) / cp[CP_B_HI])
        s_cor2 = s_cor2 + (d_cor / tmax2)
        day_cor = dap_adj - 1 - hi_start_cd
        fpost_dwn = (tmax2 / day_cor) * s_cor2

    # total multiplier
    if (tmax1 == 0) and (tmax2 == 0):
        f_post = 1.0
    else:
        if tmax2 == 0:
            f_post = fpost_upp
        else:
            if tmax1 == 0:
                f_post = fpost_dwn
            elif tmax1 <= tmax2:
                f_post = fpost_dwn * (
                    ((tmax1 * fpost_upp) + (tmax2 - tmax1)) / tmax2)
            else:
                f_post = fpost_upp * (
                    ((tmax2 * fpost_dwn) + (tmax1 - tmax2)) / tmax1)

    return s_cor1, s_cor2, fpost_upp, fpost_dwn, f_post


# --------------------------------------------------------------------------
# harvest index                                   (solution/harvest_index.py)
# --------------------------------------------------------------------------

@kernel
def harvest_index(cp, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry, z_top,
                  th, z_root, t_early_sen,
                  hi_start_cd, hi_end_cd, yld_form_cd, flowering_cd,
                  canopy_dev_end_cd,
                  dap, delayed_cds, hi_ref, yield_form, cc, b, b_ns,
                  hi, hi_adj, pre_adj, f_pre, f_pol, s_cor1, s_cor2,
                  fpost_upp, fpost_dwn, f_post,
                  et0, temp_max, temp_min, growing_season):
    """Harvest index build-up.
    Returns (hi, hi_adj, pre_adj, f_pre, f_pol, s_cor1, s_cor2,
    fpost_upp, fpost_dwn, f_post)."""
    init_hi = hi
    init_hi_adj = hi_adj
    init_pre_adj = pre_adj

    if growing_season:
        (_wr, dr_zt, dr_rz, taw_zt, taw_rz,
         _a, _s, _f, _w, _d, _ae) = root_zone_water(
            z_root, th, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry,
            z_top, cp[CP_ZMIN], cp[CP_AER])
        if (dr_rz / taw_rz) <= (dr_zt / taw_zt):
            dr = dr_rz
            taw = taw_rz
        else:
            dr = dr_zt
            taw = taw_zt

        ksw_exp, ksw_sto, ksw_sen, ksw_pol, ksw_sto_lin = water_stress(
            cp, t_early_sen, dr, taw, et0, True)

        kst_polh, kst_polc = temperature_stress(cp, temp_max, temp_min)

        hi_i = hi_ref
        hi_t = dap - delayed_cds - hi_start_cd - 1

        if yield_form and (hi_t >= 0):
            crop_type = cp[CP_CROP_TYPE]
            if (crop_type == 2) or (crop_type == 3):
                # adjustment for water stress before anthesis
                if not init_pre_adj:
                    pre_adj = True
                    f_pre = hi_adj_pre_anthesis(b, b_ns, cc, cp[CP_DHI_PRE])

                if crop_type == 3:
                    if (hi_t > 0) and (hi_t <= flowering_cd):
                        f_pol = hi_adj_pollination(
                            cc, f_pol, flowering_cd, cp[CP_CC_MIN],
                            cp[CP_EXC], ksw_pol, kst_polh, kst_polc, hi_t)
                    hi_max = f_pol * cp[CP_HI0]
                else:
                    hi_max = cp[CP_HI0]

                # adjustments for post-anthesis water stress
                if hi_t > 0:
                    (s_cor1, s_cor2, fpost_upp, fpost_dwn, f_post) = (
                        hi_adj_post_anthesis(
                            cp, canopy_dev_end_cd, hi_start_cd, hi_end_cd,
                            yld_form_cd, delayed_cds, s_cor1, s_cor2, dap,
                            f_pre, cc, fpost_upp, fpost_dwn,
                            ksw_exp, ksw_sto))

                hi_mult = f_pre * f_post
                if hi_mult > 1 + (cp[CP_DHI0] / 100):
                    hi_mult = 1 + (cp[CP_DHI0] / 100)

                if hi_max >= hi_i:
                    hi_adj_v = hi_mult * hi_i
                else:
                    hi_adj_v = hi_mult * hi_max
            else:  # crop_type == 1
                hi_adj_v = hi_i
        else:
            hi_i = init_hi
            hi_adj_v = init_hi_adj

        hi = hi_i
        hi_adj = hi_adj_v
    else:
        hi = 0.0
        hi_adj = 0.0

    return (hi, hi_adj, pre_adj, f_pre, f_pol, s_cor1, s_cor2,
            fpost_upp, fpost_dwn, f_post)


# --------------------------------------------------------------------------
# per-pixel crop calendar    (initialize/compute_crop_calendar.py + HIGC)
# --------------------------------------------------------------------------

@kernel
def calculate_higc(yld_form_cd, hi0, hi_ini):
    """Harvest index growth coefficient (iterative)."""
    t_hi = yld_form_cd
    higc = 0.001
    hi_est = 0.0
    while hi_est <= (0.98 * hi0):
        higc = higc + 0.001
        hi_est = (hi_ini * hi0) / (
            hi_ini + (hi0 - hi_ini) * math.exp(-higc * t_hi))
    if hi_est >= hi0:
        higc = higc - 0.001
    return higc


@kernel
def calculate_hi_linear(yld_form_cd, hi_ini, hi0, higc):
    """Linear switch point and rate for fruit/grain HI build-up.
    Returns (t_lin_switch, d_hi_linear)."""
    ti = 0.0
    tmax = yld_form_cd
    hi_est = 0.0
    hi_prev = hi_ini
    while (hi_est <= hi0) and (ti < tmax):
        ti = ti + 1
        hi_new = (hi_ini * hi0) / (
            hi_ini + (hi0 - hi_ini) * math.exp(-higc * ti))
        hi_est = hi_new + (tmax - ti) * (hi_new - hi_prev)
        hi_prev = hi_new
    t_switch = ti - 1

    if t_switch > 0:
        hi_est = (hi_ini * hi0) / (
            hi_ini + (hi0 - hi_ini) * math.exp(-higc * t_switch))
    else:
        hi_est = 0.0
    d_hi_lin = (hi0 - hi_est) / (tmax - t_switch)
    return t_switch, d_hi_lin


@kernel
def compute_calendar_cds(cp, tmin2d, tmax2d, p, plant_idx, nt):
    """Per-pixel GDD-to-calendar-day conversions (CalendarType == 2),
    replicating aquacrop's reset_initial_conditions.

    Returns (ok, maturity_cd, max_canopy_cd, canopy_dev_end_cd,
    hi_start_cd, hi_end_cd, yld_form_cd, flowering_cd)."""
    if cp[CP_CALENDAR_TYPE] == 1:
        # all calendar-day values are weather-independent and prepared on
        # the Python side (see aquagrid.params)
        return (True, cp[CP_MATURITY_CD], cp[CP_MAX_CANOPY_CD],
                cp[CP_CANOPY_DEV_END_CD], cp[CP_HI_START_CD],
                cp[CP_HI_END_CD], cp[CP_YLD_FORM_CD], cp[CP_FLOWERING_CD])

    gdd_method = cp[CP_GDD_METHOD]
    t_base = cp[CP_T_BASE]
    t_upp = cp[CP_T_UPP]

    maturity = cp[CP_MATURITY]
    max_canopy = cp[CP_MAX_CANOPY]
    canopy_dev_end = cp[CP_CANOPY_DEV_END]
    hi_start = cp[CP_HI_START]
    hi_end = cp[CP_HI_END]
    flowering_end = cp[CP_FLOWERING_END]

    gdd_cum = 0.0
    maturity_cd = 0.0
    max_canopy_cd = 0.0
    canopy_dev_end_cd = 0.0
    hi_start_cd = 0.0
    hi_end_cd = 0.0
    flowering_end_cd = 0.0

    cd = 0.0
    for tt in range(plant_idx, nt):
        cd = cd + 1
        gdd = growing_degree_day(
            gdd_method, t_upp, t_base, tmax2d[tt, p], tmin2d[tt, p])
        gdd_cum = gdd_cum + gdd
        if (maturity_cd == 0) and (gdd_cum > maturity):
            maturity_cd = cd
        if (max_canopy_cd == 0) and (gdd_cum > max_canopy):
            max_canopy_cd = cd
        if (canopy_dev_end_cd == 0) and (gdd_cum > canopy_dev_end):
            canopy_dev_end_cd = cd
        if (hi_start_cd == 0) and (gdd_cum > hi_start):
            hi_start_cd = cd
        if (hi_end_cd == 0) and (gdd_cum > hi_end):
            hi_end_cd = cd
        if (flowering_end_cd == 0) and (gdd_cum > flowering_end):
            flowering_end_cd = cd

    # replicate pandas argmax semantics: never-crossed -> index 0 -> CD 1
    if max_canopy_cd == 0:
        max_canopy_cd = 1.0
    if canopy_dev_end_cd == 0:
        canopy_dev_end_cd = 1.0
    if hi_start_cd == 0:
        hi_start_cd = 1.0
    if hi_end_cd == 0:
        hi_end_cd = 1.0
    if flowering_end_cd == 0:
        flowering_end_cd = 1.0

    ok = True
    if (maturity_cd == 0) or (gdd_cum <= maturity) or (maturity_cd >= 365):
        ok = False
        maturity_cd = 1.0

    yld_form_cd = hi_end_cd - hi_start_cd
    if cp[CP_CROP_TYPE] == 3:
        flowering_cd = flowering_end_cd - hi_start_cd
    else:
        flowering_cd = -999.0

    return (ok, maturity_cd, max_canopy_cd, canopy_dev_end_cd,
            hi_start_cd, hi_end_cd, yld_form_cd, flowering_cd)


# --------------------------------------------------------------------------
# full season for one pixel      (orchestration of timestep/run_single_...)
# --------------------------------------------------------------------------

@kernel
def run_pixel(p, plant_idx, nt, tmin2d, tmax2d, prcp2d, et02d,
              cp, sp, ncomp, nlayer,
              dz, dzsum, th_fc, th_s, th_wp, th_dry, ksat, tau,
              layer, penetrability, th_init,
              th, thnew, flux_out, aer_days_comp,
              evap_time_steps, max_season_days,
              out_final, save_daily, out_daily):
    """Run a full rainfed AquaCrop season for pixel ``p``.

    Weather arrays are (time, npixel); profile arrays are 1-D per-profile
    views; ``th``/``thnew``/``flux_out``/``aer_days_comp`` are 1-D per-pixel
    scratch views. Results are written to ``out_final[p, :]`` and,
    if ``save_daily``, to ``out_daily[:, t, p]``.
    """
    # ----- per-pixel crop calendar ------------------------------------
    (cal_ok, maturity_cd, max_canopy_cd, canopy_dev_end_cd,
     hi_start_cd, hi_end_cd, yld_form_cd, flowering_cd) = (
        compute_calendar_cds(cp, tmin2d, tmax2d, p, plant_idx, nt))

    if not cal_ok:
        out_final[p, OF_STATUS] = STATUS_NO_MATURITY_GDD
        return

    higc = calculate_higc(yld_form_cd, cp[CP_HI0], cp[CP_HI_INI])
    if cp[CP_CROP_TYPE] == 3:
        t_lin_switch, d_hi_linear = calculate_hi_linear(
            yld_form_cd, cp[CP_HI_INI], cp[CP_HI0], higc)
    else:
        t_lin_switch = 0.0
        d_hi_linear = 0.0

    # effective phenology params (calendar-day mode uses CD values for the
    # "GDD" slots; the caller prepares cp accordingly, mirroring aquacrop)
    senescence = cp[CP_SENESCENCE]
    maturity_gdd = cp[CP_MATURITY]
    canopy_dev_end = cp[CP_CANOPY_DEV_END]

    # ----- initial conditions (season start; off_season=False) ---------
    for ii in range(ncomp):
        th[ii] = th_init[ii]
        aer_days_comp[ii] = 0.0

    dap = 0.0
    gdd_cum = 0.0
    delayed_cds = 0.0
    delayed_gdds = 0.0
    age_days = 0.0
    age_days_ns = 0.0
    aer_days = 0.0
    t_early_sen = 0.0
    day_submerged = 0.0
    e_pot = 0.0
    t_pot = 0.0

    pre_adj = False
    crop_mature = False
    crop_dead = False
    germinated = False
    premat_senes = False
    yield_form = False
    stage2 = False
    protected_seed = False

    f_pre = 1.0
    f_post = 1.0
    fpost_dwn = 1.0
    fpost_upp = 1.0
    f_pol = 0.0
    s_cor1 = 0.0
    s_cor2 = 0.0
    hi_ref = 0.0
    hi_final = cp[CP_HI0]
    pct_lag_phase = 0.0

    tr_ratio = 1.0
    r_cor = 1.0

    cc = 0.0
    cc_adj = 0.0
    cc_ns = 0.0
    cc_adj_ns = 0.0
    biomass = 0.0
    biomass_ns = 0.0
    yield_pot = 0.0
    hi = 0.0
    hi_adj = 0.0
    ccx_act = 0.0
    ccx_act_ns = 0.0
    ccx_w = 0.0
    ccx_w_ns = 0.0
    ccx_early_sen = 0.0
    cc_prev = 0.0
    cc0_adj = 0.0
    dry_yield = 0.0
    fresh_yield = 0.0

    z_root = 0.0
    surface_storage = 0.0
    w_surf = 0.0
    evap_z = 0.0
    w_stage2 = 0.0

    status = STATUS_TRUNCATED
    end_day = nt - 1

    # ----- daily loop ---------------------------------------------------
    for t in range(plant_idx, nt):
        day_idx = t - plant_idx
        temp_min = tmin2d[t, p]
        temp_max = tmax2d[t, p]
        prcp = prcp2d[t, p]
        et0 = et02d[t, p]

        growing_season = True  # (mature/dead pixels break out below)

        # increment time counters
        dap = dap + 1
        gdd = growing_degree_day(
            cp[CP_GDD_METHOD], cp[CP_T_UPP], cp[CP_T_BASE],
            temp_max, temp_min)
        gdd_cum = gdd_cum + gdd

        # 2. root development
        z_root, r_cor = root_development(
            cp, ncomp, nlayer, dz, dzsum, th_fc, th_wp, layer, penetrability,
            dap, z_root, delayed_cds, gdd_cum, delayed_gdds, tr_ratio, th,
            cc, cc_ns, germinated, r_cor, t_pot, gdd, growing_season)

        # 4. drainage (th_fc_adj == th_fc: no groundwater table)
        deep_perc = drainage(
            ncomp, dz, dzsum, th_fc, th_s, tau, ksat, th_fc, th, thnew,
            flux_out)

        # 5. surface runoff
        runoff, infl, day_submerged = rainfall_partition(
            prcp, th, ncomp, dz, dzsum, th_fc, th_wp,
            sp[SP_CN], sp[SP_ADJ_CN], sp[SP_Z_CN], day_submerged)

        # 7. infiltration
        surface_storage, deep_perc, runoff, infl = infiltration(
            ncomp, dz, th_fc, th_s, tau, ksat, th_fc,
            th, flux_out, infl, surface_storage, deep_perc, runoff)

        # 9. germination
        germinated, protected_seed, delayed_cds, delayed_gdds = germination(
            th, ncomp, dz, dzsum, th_fc, th_wp, sp[SP_Z_GERM],
            cp[CP_GERM_THR], cp[CP_PLANT_METHOD] == 1, gdd, growing_season,
            germinated, protected_seed, delayed_cds, delayed_gdds)

        # 11. canopy cover development
        (cc, cc_ns, cc_adj, cc_adj_ns, ccx_act, ccx_act_ns, ccx_w,
         ccx_w_ns, ccx_early_sen, cc0_adj, cc_prev, protected_seed,
         premat_senes, t_early_sen, crop_dead) = canopy_cover(
            cp, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry, sp[SP_Z_TOP],
            th, z_root, gdd, gdd_cum, dap, delayed_cds, delayed_gdds,
            et0, growing_season, canopy_dev_end, senescence, maturity_gdd,
            cc, cc_ns, cc_adj, cc_adj_ns, ccx_act, ccx_act_ns,
            ccx_w, ccx_w_ns, ccx_early_sen, cc0_adj, protected_seed,
            premat_senes, t_early_sen, crop_dead)

        # 12. soil evaporation
        (e_pot, stage2, w_stage2, w_surf, surface_storage, evap_z,
         es_act) = soil_evaporation(
            sp, cp, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry,
            evap_time_steps, day_idx, dap, delayed_cds, gdd_cum,
            delayed_gdds, senescence, ccx_w, cc_adj, ccx_act, cc,
            premat_senes, surface_storage, w_surf, evap_z, stage2,
            w_stage2, th, et0, infl, prcp, growing_season)

        # 13. crop transpiration
        (tr_act, tr_pot_ns, tr_pot0, age_days, age_days_ns, day_submerged,
         surface_storage, aer_days, tr_ratio, cc, t_pot) = transpiration(
            cp, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry, sp[SP_Z_TOP],
            th, aer_days_comp, max_canopy_cd,
            dap, delayed_cds, age_days, age_days_ns,
            ccx_w_ns, ccx_w, cc_adj, cc_adj_ns, cc_ns, cc, cc_prev, z_root,
            r_cor, t_early_sen, surface_storage, day_submerged, aer_days,
            tr_ratio, et0, gdd, growing_season)

        # 15. reference harvest index
        hi_ref, yield_form, pct_lag_phase, hi_final = hi_ref_current_day(
            cp, hi_start_cd, yld_form_cd, higc, t_lin_switch, d_hi_linear,
            hi_ref, hi_final, dap, delayed_cds, yield_form, pct_lag_phase,
            cc, ccx_w, growing_season)

        # 16. biomass accumulation
        biomass, biomass_ns = biomass_accumulation(
            cp, hi_start_cd, yld_form_cd, dap, delayed_cds, hi_ref,
            pct_lag_phase, biomass, biomass_ns, tr_act, tr_pot_ns, et0,
            growing_season)

        # 17. harvest index
        (hi, hi_adj, pre_adj, f_pre, f_pol, s_cor1, s_cor2,
         fpost_upp, fpost_dwn, f_post) = harvest_index(
            cp, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry, sp[SP_Z_TOP],
            th, z_root, t_early_sen, hi_start_cd, hi_end_cd, yld_form_cd,
            flowering_cd, canopy_dev_end_cd, dap, delayed_cds, hi_ref,
            yield_form, cc, biomass, biomass_ns, hi, hi_adj, pre_adj,
            f_pre, f_pol, s_cor1, s_cor2, fpost_upp, fpost_dwn, f_post,
            et0, temp_max, temp_min, growing_season)

        # 18./19. yields
        yield_pot = (biomass_ns / 100) * hi
        dry_yield = (biomass / 100) * hi_adj
        fresh_yield = dry_yield / (cp[CP_YLD_WC] / 100)

        # maturity check
        if ((cp[CP_CALENDAR_TYPE] == 1) and (dap >= maturity_cd)) or (
                (cp[CP_CALENDAR_TYPE] == 2) and (gdd_cum >= maturity_gdd)):
            crop_mature = True

        # daily outputs
        if save_daily:
            (wr, _d1, _d2, _t1, _t2,
             _a, _s, _f, _w, _dd, _ae) = root_zone_water(
                z_root, th, ncomp, dz, dzsum, th_s, th_fc, th_wp, th_dry,
                sp[SP_Z_TOP], cp[CP_ZMIN], cp[CP_AER])
            out_daily[OD_CANOPY_COVER, t, p] = cc
            out_daily[OD_BIOMASS, t, p] = biomass
            out_daily[OD_Z_ROOT, t, p] = z_root
            out_daily[OD_GDD_CUM, t, p] = gdd_cum
            out_daily[OD_ES, t, p] = es_act
            out_daily[OD_TR, t, p] = tr_act
            out_daily[OD_WR, t, p] = wr

        # end of season?
        if crop_mature or crop_dead or (dap >= max_season_days):
            status = STATUS_OK
            end_day = t
            break

    # ----- final outputs -------------------------------------------------
    out_final[p, OF_STATUS] = status
    out_final[p, OF_DRY_YIELD] = dry_yield
    out_final[p, OF_FRESH_YIELD] = fresh_yield
    out_final[p, OF_YIELD_POT] = yield_pot
    out_final[p, OF_BIOMASS] = biomass
    out_final[p, OF_BIOMASS_NS] = biomass_ns
    out_final[p, OF_HI_ADJ] = hi_adj
    out_final[p, OF_DAP_END] = dap
    out_final[p, OF_DAY_END] = end_day
    out_final[p, OF_CROP_MATURE] = 1.0 if crop_mature else 0.0
    out_final[p, OF_CROP_DEAD] = 1.0 if crop_dead else 0.0
    out_final[p, OF_GERMINATED] = 1.0 if germinated else 0.0
    out_final[p, OF_MATURITY_CD] = maturity_cd
