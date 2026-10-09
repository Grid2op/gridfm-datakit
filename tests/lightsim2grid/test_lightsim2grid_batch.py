"""The batched lightsim2grid power flows must match one power flow per perturbation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matpowercaseframes import CaseFrames

from gridfm_datakit import lightsim2grid as l2g
from gridfm_datakit.network import Network
from gridfm_datakit.perturbations.topology_perturbation import (
    NMinusKGenerator,
    RandomComponentDropGenerator,
)
from gridfm_datakit.utils.idx_bus import PD

pytestmark = pytest.mark.skipif(
    not l2g.is_lightsim2grid_available(),
    reason="lightsim2grid is not installed. Install with: pip install gridfm-datakit[lightsim2grid]",
)

_GRID = Path(__file__).parents[1] / "powsybl" / "grids" / "ieee14.m"
_TOL = 1e-8


def _load_net() -> Network:
    # same as load_net_from_file, without the PowerModels correction (which needs Julia)
    frames = CaseFrames(str(_GRID))
    mpc = {
        key: getattr(frames, key).values
        if isinstance(getattr(frames, key), pd.DataFrame)
        else getattr(frames, key)
        for key in frames._attributes
    }
    return Network(mpc)


def _single(p: Network, dc: bool):
    converted = l2g.to_lightsim2grid(p)
    try:
        return l2g.run_ls_pf(
            converted.ls_net,
            p,
            converted.mapping_l2g,
            dc=dc,
            as_arrays=True,
        )["solution"]
    except ValueError:
        return None


def _assert_batch_matches_single(net: Network, perturbations: list) -> None:
    converted = l2g.update_lightsim2grid(net)
    res_ac, res_dc = l2g.run_ls_pf_batch(converted, net, perturbations, True)
    assert len(res_ac) == len(res_dc) == len(perturbations)
    for p, batched_ac, batched_dc in zip(perturbations, res_ac, res_dc):
        single_ac, single_dc = _single(p, False), _single(p, True)
        assert (single_ac is None) == (batched_ac is None)
        assert (single_dc is None) == (batched_dc is None)
        if single_ac is not None:
            for a, b in zip(single_ac["arrays"], batched_ac["solution"]["arrays"]):
                np.testing.assert_allclose(a, b, rtol=0, atol=_TOL)
            ybus = single_ac["Ybus"].tocoo()
            rows, cols, values = batched_ac["solution"]["Ybus"]
            assert np.array_equal(ybus.row, rows) and np.array_equal(ybus.col, cols)
            np.testing.assert_allclose(ybus.data, values, rtol=0, atol=_TOL)
        if single_dc is not None:
            a, b = single_dc["arrays"], batched_dc["solution"]["arrays"]
            np.testing.assert_allclose(
                a[0][:, [0, 2]], b[0][:, [0, 2]], rtol=0, atol=_TOL
            )
            np.testing.assert_allclose(a[1][:, 0], b[1][:, 0], rtol=0, atol=_TOL)
            np.testing.assert_allclose(a[2][:, 1], b[2][:, 1], rtol=0, atol=_TOL)


def test_batch_matches_one_power_flow_per_outage():
    net = _load_net()
    perturbations = list(NMinusKGenerator(1, net).generate(net))
    assert len(perturbations) > 5
    _assert_batch_matches_single(net, perturbations)


def test_batch_matches_with_generator_and_multiple_outages():
    net = _load_net()
    np.random.seed(0)
    generator = RandomComponentDropGenerator(25, 3, net, ["branch", "gen"])
    perturbations = list(generator.generate(net))
    assert any(
        (p.gens[:, 7] <= 0).sum() > (net.gens[:, 7] <= 0).sum() for p in perturbations
    ), "the test should include generator outages"
    _assert_batch_matches_single(net, perturbations)


def test_batch_refuses_anything_but_outages():
    net = _load_net()
    changed = net.copy_for_perturbation()
    changed.buses[:, PD] *= 1.1
    converted = l2g.update_lightsim2grid(net)
    with pytest.raises(l2g.BatchNotSupported):
        l2g.run_ls_pf_batch(converted, net, [changed], False)
