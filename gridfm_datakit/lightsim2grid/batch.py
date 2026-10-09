"""Batched lightsim2grid power flows for the outage perturbations of one scenario.

The perturbations of a scenario all derive from the same operating point (the
OPF set points of one scenario) and only differ by branches and generators put out
of service. lightsim2grid can solve such a set in one call (``ScenarioSweepCPP``),
reusing its symbolic analysis and the admittance matrix of the base case, instead of
being updated and called once per perturbation (which re-does that set up each time).

The sweep returns the complex voltages. Everything else the datakit writes (branch
flows, generator outputs, the admittance matrix) is rebuilt here from them, in
numpy, for all the perturbations at once, following lightsim2grid's conventions:

* the active power mismatch of the slack bus is shared by the in-service generators
  of that bus with the weights lightsim2grid uses (read from a base case power flow),
* the reactive power of a bus is shared between its in-service generators in
  proportion to ``QMAX - QMIN``.

:func:`run_ls_pf_batch` raises :class:`BatchNotSupported` whenever it cannot guarantee
the result (anything else than outages, several reference buses, ...) so that the
caller can fall back to one power flow per perturbation.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from gridfm_datakit.network import Network, branch_vectors
from gridfm_datakit.utils.idx_brch import BR_STATUS, BR_X, F_BUS, SHIFT, T_BUS, TAP
from gridfm_datakit.utils.idx_bus import BS, BUS_I, BUS_TYPE, GS, PD, QD, REF
from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS, PG, QMAX, QMIN

from .convert import ConvertedNetwork, initial_voltage
from .preprocess import get_ybus


class BatchNotSupported(Exception):
    """The perturbations cannot be solved as a batch; solve them one at a time."""


# a bus voltage magnitude below this is a bus the sweep did not solve (diverged / islanded)
_TOL_MISMATCH_MW = 1e-6


def _non_status_columns(n_cols: int, status_col: int) -> np.ndarray:
    return np.array([c for c in range(n_cols) if c != status_col])


def _check_only_outages(base: Network, perturbations: Sequence[Network]) -> None:
    """Raise unless every perturbation is ``base`` with branches / generators switched off.

    Args:
        base: The network the LSGrid is in sync with.
        perturbations: The networks to solve.

    Raises:
        BatchNotSupported: If a perturbation differs from ``base`` in anything else than the
            status of branches and generators (and the bus types that follow from it),
            or switches an element on.
    """
    br_cols = _non_status_columns(base.branches.shape[1], BR_STATUS)
    gen_cols = _non_status_columns(base.gens.shape[1], GEN_STATUS)
    bus_cols = _non_status_columns(base.buses.shape[1], BUS_TYPE)
    base_br, base_gen, base_bus = (
        base.branches[:, br_cols],
        base.gens[:, gen_cols],
        base.buses[:, bus_cols],
    )
    base_br_on = base.branches[:, BR_STATUS] > 0
    base_gen_on = base.gens[:, GEN_STATUS] > 0
    for p in perturbations:
        if (
            p.branches.shape != base.branches.shape
            or p.gens.shape != base.gens.shape
            or p.buses.shape != base.buses.shape
            or not np.array_equal(p.branches[:, br_cols], base_br)
            or not np.array_equal(p.gens[:, gen_cols], base_gen)
            or not np.array_equal(p.buses[:, bus_cols], base_bus)
            or np.any((p.branches[:, BR_STATUS] > 0) & ~base_br_on)
            or np.any((p.gens[:, GEN_STATUS] > 0) & ~base_gen_on)
        ):
            raise BatchNotSupported("a perturbation is not only a set of outages")


def _slack_shares(
    ls_net: Any,
    net: Network,
    v_init: np.ndarray,
    dc: bool,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Share of the slack mismatch taken by each generator of the reference bus.

    Read from a base case power flow, so that the weights are lightsim2grid's own.

    Args:
        ls_net: The LSGrid, in sync with ``net``.
        net: The base network.
        v_init: Initial voltage.
        dc: Use the DC power flow.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.

    Returns:
        One entry per generator (0 for those not at the reference bus or out of
        service), summing to 1.

    Raises:
        BatchNotSupported: If the base case power flow fails or the slack is unusual.
    """
    v = (ls_net.dc_pf if dc else ls_net.ac_pf)(v_init, max_iter, tol)
    if len(v) == 0:
        raise BatchNotSupported("the base case power flow did not converge")
    gen_p = np.asarray(ls_net.get_gen_res()[0])
    on = net.gens[:, GEN_STATUS] > 0
    ref_buses = np.flatnonzero(net.buses[:, BUS_TYPE] == REF)
    if ref_buses.size != 1:
        raise BatchNotSupported("several (or no) reference buses")
    at_ref = on & (
        net.gens[:, GEN_BUS].astype(int) == int(net.buses[ref_buses[0], BUS_I])
    )
    if not at_ref.any():
        raise BatchNotSupported("no generator at the reference bus")
    delta = np.where(at_ref, gen_p - net.gens[:, PG], 0.0)
    if abs(delta.sum()) > _TOL_MISMATCH_MW:
        share = delta / delta.sum()
    else:  # no mismatch to read the weights from: lightsim2grid weights are proportional to p
        p = np.where(at_ref, np.abs(net.gens[:, PG]), 0.0)
        share = p / p.sum() if p.sum() > 0 else at_ref / at_ref.sum()
    return share


def _make_sweep(ls_net: Any, algorithms: Sequence[str]) -> Any:
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP

    sweep = ScenarioSweepCPP(ls_net)
    available = {a.name: a for a in sweep.available_default_algorithms()}
    for name in algorithms:
        if name in available:
            sweep.change_algorithm(available[name])
            break
    return sweep


def _solve_sweep(
    ls_net: Any,
    masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    v_init: np.ndarray,
    algorithms: Sequence[str],
    max_iter: int,
    tol: float,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    line_mask, trafo_mask, gen_mask = masks
    sweep = _make_sweep(ls_net, algorithms)
    try:
        if line_mask.shape[1]:
            sweep.set_contingency_lines(line_mask)
        if trafo_mask.shape[1]:
            sweep.set_contingency_trafos(trafo_mask)
        if gen_mask.any():
            sweep.set_contingency_gens(gen_mask)
        if not (line_mask.shape[1] or trafo_mask.shape[1] or gen_mask.any()):
            raise BatchNotSupported("no outage to solve")
        sweep.compute(v_init, max_iter, tol)
    except BatchNotSupported:
        raise
    except Exception as exc:  # e.g. a generator regulating a remote bus
        raise BatchNotSupported(
            f"lightsim2grid cannot solve this batch: {exc}"
        ) from exc
    converged = np.array(sweep.converged_mask(), dtype=bool)
    return sweep, np.asarray(sweep.get_voltages()), converged


def _solve_dc_sweeps(
    ls_net: Any,
    masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    v_init: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """DC sweep, with the rows that open a branch apart from the others.

    lightsim2grid 1.1.0 solves a generator-only row with the admittance matrix left by
    the branch outage of the row before it, so the two kinds of rows are never mixed
    in one sweep (a branch row is always solved on the matrix restored for it).

    Args:
        ls_net: The LSGrid.
        masks: Line, trafo and generator outage masks, one row per perturbation.
        v_init: Initial voltage.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.

    Returns:
        The complex voltages and the convergence mask, for all rows.
    """
    line_mask, trafo_mask, gen_mask = masks
    n_rows = line_mask.shape[0]
    opens_branch = line_mask.any(axis=1) | trafo_mask.any(axis=1)
    v = np.zeros((n_rows, ls_net.total_bus()), dtype=complex)
    converged = np.zeros(n_rows, dtype=bool)
    for group in (np.flatnonzero(opens_branch), np.flatnonzero(~opens_branch)):
        if group.size == 0:
            continue
        _, v_g, ok_g = _solve_sweep(
            ls_net,
            (line_mask[group], trafo_mask[group], gen_mask[group]),
            v_init,
            ("DC_KLU", "DC_SparseLU"),
            max_iter,
            tol,
        )
        v[group], converged[group] = v_g, ok_g
    return v, converged


def _ybus_removal(
    base_ybus: csr_matrix,
    net: Network,
    vecs: Tuple[np.ndarray, ...],
    removed: List[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Ybus of each outage, as the base Ybus minus the branches that are out.

    Args:
        base_ybus: Ybus of the base case (datakit bus indexing, sorted, no explicit zero).
        net: The base network.
        vecs: ``branch_vectors`` of the base network.
        removed: For each perturbation, the indices of the branches put out of service.

    Returns:
        For each perturbation, ``(rows, cols, values)`` of the non-zero entries in
        row-major order.
    """
    n_buses = base_ybus.shape[0]
    Ytt, Yff, Yft, Ytf = vecs
    nl = net.branches.shape[0]
    f = net.branches[:, F_BUS].real.astype(np.int64)
    t = net.branches[:, T_BUS].real.astype(np.int64)
    in_service = net.branches[:, BR_STATUS] > 0

    indptr, indices = base_ybus.indptr, base_ybus.indices
    rows = np.repeat(np.arange(n_buses), np.diff(indptr))
    keys = (
        rows.astype(np.int64) * n_buses + indices
    )  # sorted: row-major, sorted indices

    def position(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        k = r.astype(np.int64) * n_buses + c
        pos = np.searchsorted(keys, k)
        if np.any(pos >= keys.size) or np.any(
            keys[np.minimum(pos, keys.size - 1)] != k
        ):
            raise BatchNotSupported("a branch entry is missing from the base Ybus")
        return pos

    pos = np.zeros((4, nl), dtype=np.int64)
    pos[:, in_service] = np.stack(
        [
            position(f[in_service], f[in_service]),
            position(f[in_service], t[in_service]),
            position(t[in_service], f[in_service]),
            position(t[in_service], t[in_service]),
        ],
    )
    values = np.stack([Yff, Yft, Ytf, Ytt])  # (4, nl)

    # how many terms (branches, plus a non-zero shunt on the diagonal) build each entry
    n_terms = np.zeros(keys.size, dtype=np.int64)
    np.add.at(n_terms, pos[:, in_service].ravel(), 1)
    shunt = (net.buses[:, GS] != 0) | (net.buses[:, BS] != 0)
    diag = position(
        net.buses[shunt, BUS_I].astype(int), net.buses[shunt, BUS_I].astype(int)
    )
    n_terms[diag] += 1

    out = []
    for rem in removed:
        data = base_ybus.data.copy()
        if rem.size:
            rem = rem[in_service[rem]]
            np.subtract.at(data, pos[:, rem].ravel(), values[:, rem].ravel())
            left = n_terms.copy()
            np.subtract.at(left, pos[:, rem].ravel(), 1)
            keep = left > 0
            out.append((rows[keep], indices[keep], data[keep]))
        else:
            out.append((rows, indices, data))
    return out


def _gen_results(
    net: Network,
    gen_on: np.ndarray,
    s_gen_bus: np.ndarray,
    slack_share: np.ndarray,
    bus_row: np.ndarray,
    with_q: bool,
) -> np.ndarray:
    """Active (and reactive) power of the in-service generators, one row of ``gen_on`` each.

    Args:
        net: The base network.
        gen_on: ``(n_rows, n_gens)`` which generators are in service.
        s_gen_bus: ``(n_rows, n_buses)`` complex power produced at each bus (pu),
            columns in the bus order of ``net.buses``.
        slack_share: Share of the slack mismatch of each generator (sums to 1).
        bus_row: Row of ``net.buses`` of every bus index.
        with_q: Also share the reactive power.

    Returns:
        ``(n_rows, n_gens, 2)`` pu values (0 for generators out of service).
    """
    base_mva = float(net.baseMVA)
    n_rows = gen_on.shape[0]
    gen_bus_row = bus_row[net.gens[:, GEN_BUS].astype(int)]
    p_set = net.gens[:, PG] / base_mva
    ref_row = np.flatnonzero(net.buses[:, BUS_TYPE] == REF)[0]
    at_ref = gen_bus_row == ref_row

    p = np.where(gen_on, p_set, 0.0)
    # slack: the generators of the reference bus take the mismatch, weighted and renormalised
    weight = np.where(gen_on & at_ref, slack_share, 0.0)
    total_w = weight.sum(axis=1, keepdims=True)
    if np.any(total_w <= 0):
        raise BatchNotSupported("no generator left to take the slack")
    weight = weight / total_w
    mismatch = s_gen_bus[:, ref_row].real - p[:, at_ref].sum(axis=1)
    p = p + weight * mismatch[:, None]

    q = np.zeros_like(p)
    if with_q:
        rng = net.gens[:, QMAX] - net.gens[:, QMIN]
        rng_on = np.where(gen_on, rng, 0.0)
        bus_rng = np.zeros((n_rows, net.buses.shape[0]))
        np.add.at(bus_rng.T, gen_bus_row, rng_on.T)
        n_on_bus = np.zeros((n_rows, net.buses.shape[0]))
        np.add.at(n_on_bus.T, gen_bus_row, gen_on.T.astype(float))
        denom = bus_rng[:, gen_bus_row]
        equal = n_on_bus[:, gen_bus_row]
        frac = np.where(
            denom > 0,
            rng_on / np.where(denom > 0, denom, 1.0),
            gen_on / np.where(equal > 0, equal, 1.0),
        )
        q = s_gen_bus.imag[:, gen_bus_row] * frac
        q = np.where(gen_on, q, 0.0)
    return np.stack([p, q], axis=-1)


def run_ls_pf_batch(
    converted: ConvertedNetwork,
    base: Network,
    perturbations: Sequence[Network],
    include_dc: bool,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Tuple[List[Optional[Dict[str, Any]]], List[Optional[Dict[str, Any]]]]:
    """Solve the outages of one scenario with one lightsim2grid call per power flow type.

    Results have the layout of ``run_ls_pf(..., as_arrays=True)`` (with the Ybus of the AC
    result), so that ``pf_post_processing`` takes them unchanged.

    Args:
        converted: The LSGrid of ``base``, in sync with it (see ``update_lightsim2grid``).
        base: The network of the scenario before its topology perturbations.
        perturbations: The perturbed networks; each is ``base`` with some branches and
            generators out of service.
        include_dc: Also run the DC power flow.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.

    Returns:
        The AC and DC results, one entry per perturbation, None where the power flow did
        not converge (the DC list is all None if ``include_dc`` is False).

    Raises:
        BatchNotSupported: If the batch cannot be solved reliably.
    """
    ls_net, mapping = converted.ls_net, converted.mapping_l2g
    n_rows = len(perturbations)
    if n_rows == 0:
        return [], []
    _check_only_outages(base, perturbations)

    base_mva = float(base.baseMVA)
    n_buses, nl = base.buses.shape[0], base.branches.shape[0]
    bus_idx = base.buses[:, BUS_I].astype(int)
    bus_row = np.empty(bus_idx.max() + 1, dtype=np.int64)
    bus_row[bus_idx] = np.arange(n_buses)

    br_on = np.stack(
        [p.branches[:, BR_STATUS] > 0 for p in perturbations]
    )  # (rows, nl)
    gen_on = np.stack(
        [p.gens[:, GEN_STATUS] > 0 for p in perturbations]
    )  # (rows, n_gens)
    base_br_on = base.branches[:, BR_STATUS] > 0
    base_gen_on = base.gens[:, GEN_STATUS] > 0
    masks = (
        (~br_on & base_br_on)[:, mapping.line_rows],
        (~br_on & base_br_on)[:, mapping.trafo_rows],
        ~gen_on & base_gen_on,
    )

    v_init = initial_voltage(base)
    t0 = time.perf_counter()
    slack_ac = _slack_shares(ls_net, base, v_init, False, max_iter, tol)
    base_ybus = get_ybus(
        ls_net, base
    )  # datakit bus indexing; valid after the AC base case
    _, v_ac, ok_ac = _solve_sweep(
        ls_net, masks, v_init, ("NR_KLU", "NR_SparseLU"), max_iter, tol
    )
    t_ac = (time.perf_counter() - t0) / n_rows

    # branch quantities (pu), per perturbation
    Ytt, Yff, Yft, Ytf = branch_vectors(base.branches, nl)
    f_row = bus_row[base.branches[:, F_BUS].real.astype(int)]
    t_row = bus_row[base.branches[:, T_BUS].real.astype(int)]
    stat = br_on.astype(float)

    rows = np.flatnonzero(ok_ac)
    V = v_ac[rows]
    Vf, Vt = V[:, f_row], V[:, t_row]
    Sf = Vf * np.conj(Yff * Vf + Yft * Vt) * stat[rows]
    St = Vt * np.conj(Ytf * Vf + Ytt * Vt) * stat[rows]
    ysh = (base.buses[:, GS] + 1j * base.buses[:, BS]) / base_mva
    s_inj = V * np.conj(ysh * V)
    np.add.at(s_inj.T, f_row, Sf.T)
    np.add.at(s_inj.T, t_row, St.T)
    s_load = (base.buses[:, PD] + 1j * base.buses[:, QD]) / base_mva
    s_gen_bus = s_inj + s_load
    gens = _gen_results(base, gen_on[rows], s_gen_bus, slack_ac, bus_row, with_q=True)

    removed = [np.flatnonzero(~br_on[r] & base_br_on) for r in rows]
    vecs = (Ytt, Yff, Yft, Ytf)
    ybus = _ybus_removal(base_ybus, base, vecs, removed)

    res_ac: List[Optional[Dict[str, Any]]] = [None] * n_rows
    vm, va = np.abs(V), np.angle(V)
    for k, r in enumerate(rows):
        bus_vmva = np.empty((n_buses, 2))
        bus_vmva[bus_idx] = np.column_stack((vm[k], va[k]))
        flows = np.column_stack((Sf[k].real, Sf[k].imag, St[k].real, St[k].imag))
        res_ac[r] = {
            "solution": {
                "baseMVA": base_mva,
                "per_unit": True,
                "pf": True,
                "arrays": (flows[br_on[r]], gens[k][gen_on[r]], bus_vmva),
                "Ybus": ybus[k],
            },
            "solve_time": t_ac,
        }

    res_dc: List[Optional[Dict[str, Any]]] = [None] * n_rows
    if include_dc:
        t0 = time.perf_counter()
        slack_dc = _slack_shares(ls_net, base, v_init, True, max_iter, tol)
        v_dc, ok_dc = _solve_dc_sweeps(ls_net, masks, v_init, max_iter, tol)
        t_dc = (time.perf_counter() - t0) / n_rows
        rows_dc = np.flatnonzero(ok_dc)
        # lightsim2grid's DC model: flow = (theta_f - theta_t - shift) / (x * tap). Computed
        # from the angles, since the sweep's own flows leave out the phase shifts.
        theta = np.angle(v_dc[rows_dc])
        tap = np.where(base.branches[:, TAP] == 0, 1.0, base.branches[:, TAP])
        shift = np.deg2rad(base.branches[:, SHIFT])
        pf = (theta[:, f_row] - theta[:, t_row] - shift) / (
            base.branches[:, BR_X] * tap
        )
        pf = pf * stat[rows_dc]
        p_out = np.zeros((rows_dc.size, n_buses))
        np.add.at(p_out.T, f_row, pf.T)
        np.add.at(p_out.T, t_row, -pf.T)
        s_gen_dc = (p_out + s_load.real) + 0j
        gens_dc = _gen_results(
            base, gen_on[rows_dc], s_gen_dc, slack_dc, bus_row, with_q=False
        )
        angle = theta
        for k, r in enumerate(rows_dc):
            bus_vmva = np.empty((n_buses, 2))
            bus_vmva[bus_idx] = np.column_stack((np.abs(v_dc[rows_dc][k]), angle[k]))
            flows = np.column_stack((pf[k], np.zeros(nl), -pf[k], np.zeros(nl)))
            res_dc[r] = {
                "solution": {
                    "baseMVA": base_mva,
                    "per_unit": True,
                    "pf": True,
                    "arrays": (flows[br_on[r]], gens_dc[k][gen_on[r]], bus_vmva),
                },
                "solve_time": t_dc,
            }
    return res_ac, res_dc
