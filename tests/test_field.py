"""Bunds: rainfall stays on the field and ponds up to the bund height."""

from __future__ import annotations

import numpy as np
import pytest
from aquacrop import FieldMngt
from aquacrop.entities.soilProfile import SoilProfile
from aquacrop.solution.infiltration import infiltration as ospy_infiltration
from aquacrop.solution.rainfall_partition import (
    rainfall_partition as ospy_rainfall_partition,
)

from aquagrid.kernels import constants as C
from aquagrid.kernels.impl import (
    infiltration,
    initial_surface_storage,
    rainfall_partition,
)
from aquagrid.params import field_params_array


def _profile():
    prof = SoilProfile(1)
    prof.dz[:] = 0.1
    prof.dzsum[:] = 0.1
    prof.th_wp[:] = 0.1
    prof.th_fc[:] = 0.3
    prof.th_s[:] = 0.5
    prof.Ksat[:] = 10.0
    prof.tau[:] = 0.5
    return prof


def test_field_params_store_bund_height_in_mm():
    fp = field_params_array(FieldMngt(bunds=True, z_bund=0.2, bund_water=30))
    assert fp[C.FP_BUNDS] == 1.0
    assert fp[C.FP_Z_BUND] == 200.0
    assert fp[C.FP_BUND_WATER] == 30.0


def test_initial_bund_water_is_capped_at_the_bund_height():
    assert initial_surface_storage(1.0, 20.0, 50.0) == 20.0
    assert initial_surface_storage(1.0, 20.0, 5.0) == 5.0
    assert initial_surface_storage(0.0, 20.0, 50.0) == 0.0
    assert initial_surface_storage(1.0, 0.0, 50.0) == 0.0


def test_bunds_produce_no_curve_number_runoff():
    prof = _profile()
    th = np.array([0.2])
    runoff, infl, submerged = rainfall_partition(
        50.0, th, 1, prof.dz, prof.dzsum, prof.th_fc, prof.th_wp,
        77.0, 0.0, 0.3, 4.0, 1.0, 20.0)
    ospy_runoff, ospy_infl, ospy_sub = ospy_rainfall_partition(
        50.0, th, 4, False, True, 20.0, 0.0, 77.0, 0.0, 0.3, 1, prof)
    assert runoff == 0.0
    assert infl == 50.0
    assert submerged == 4.0
    assert (runoff, infl, submerged) == (ospy_runoff, ospy_infl, ospy_sub)


def test_no_bunds_keeps_the_curve_number_split():
    prof = _profile()
    th = np.array([0.2])
    runoff, infl, submerged = rainfall_partition(
        50.0, th, 1, prof.dz, prof.dzsum.copy(), prof.th_fc, prof.th_wp,
        77.0, 0.0, 0.3, 4.0, 0.0, 0.0)
    ospy_runoff, ospy_infl, ospy_sub = ospy_rainfall_partition(
        50.0, th, 4, False, False, 0.0, 0.0, 77.0, 0.0, 0.3, 1, prof)
    assert submerged == 0.0
    assert runoff == pytest.approx(ospy_runoff)
    assert infl == pytest.approx(ospy_infl)
    assert ospy_sub == 0.0
    assert runoff > 0.0


def _infiltrate(surface, infl, z_bund, bunds):
    prof = _profile()
    th = np.array([0.2])
    flux = np.zeros(1)
    th_fc = prof.th_fc.copy()
    ospy_th, ospy_ss, _dp, ospy_ro, ospy_infl, _flux = ospy_infiltration(
        prof, surface, th_fc.copy(), th.copy(), infl, 0.0, 100.0,
        bunds, z_bund, flux.copy(), 0.0, 0.0, True)
    ss, dp, ro, infl_out = infiltration(
        1, prof.dz, prof.th_fc, prof.th_s, prof.tau, prof.Ksat, th_fc,
        th, flux, infl, surface, 0.0, 0.0, 0.0, 100.0, True,
        1.0 if bunds else 0.0, z_bund)
    return (ss, ro, infl_out, th), (ospy_ss, ospy_ro, ospy_infl, ospy_th)


def test_infiltration_ponds_up_to_the_bund_and_sheds_the_excess():
    got, ref = _infiltrate(0.0, 40.0, 8.0, True)
    assert got[0] == pytest.approx(ref[0])
    assert got[0] == pytest.approx(8.0)
    assert got[1] == pytest.approx(ref[1])
    assert got[1] > 0.0
    np.testing.assert_allclose(got[3], ref[3])


def test_infiltration_without_bunds_still_runs_off_above_ksat():
    got, ref = _infiltrate(12.0, 40.0, 8.0, False)
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(ref[1])
    assert got[1] > 12.0
