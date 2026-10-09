# lightsim2grid — batch power flow for outage perturbations: feature spec

**Audience:** lightsim2grid maintainers
**Written from:** gridfm-datakit, branch `ccr-fac6f705-ydji4p`, lightsim2grid 1.1.0 (PyPI wheel, `NR_KLU` / `DC_KLU` available)
**Status:** proposal. Everything under "Evidence" was measured; items marked *(unverified)* were not.

---

## 1. Goal

gridfm-datakit generates power-flow datasets. For every load scenario it solves an OPF once, then
solves **many topology variants of that one operating point** (a few branches and/or generators
put out of service) and writes the full solution of each variant.

`ScenarioSweepCPP` already solves such a set in one call. The datakit now uses it, but because the
sweep only returns voltages, the datakit rebuilds everything else in numpy, using conventions
it had to infer from lightsim2grid's behaviour. This spec lists what lightsim2grid should return
or fix so that the batch is solved **and fully reported** C++ side.

The two reasons, in order of weight:

1. **Correctness.** The slack sharing and the reactive-power split are lightsim2grid rules that the
   datakit now reimplements (§5, F2). A second implementation can silently drift.
2. **Speed.** The numpy rebuild is 11–13 % of the datakit's inner loop (§3).

## 2. Workload

| | |
|---|---|
| Grids | pglib-opf, 118 – 2 869 buses (4 582 branches, 510 generators); larger expected |
| Rows per sweep | ~20 (one scenario); default config: random outages, k = 1, branches and generators |
| Outage kinds | branches (lines and transformers), non-slack generators; k ≥ 2 possible (mixed rows) |
| Power flows | AC (`NR_KLU`) and DC (`DC_KLU`) for every row; a base-case solve per scenario |
| Between sweeps | the grid is updated in place (loads, generator set points) and the sweep rebuilt |
| Process model | one worker per process, one `LSGrid` per worker, scenarios sequential |
| Output per row | see §4 |

## 3. Evidence

### 3.1 Where the datakit's time goes now

Best of 5, the real inner loop (OPF stubbed), DC enabled, ms per perturbation:

| | 793 buses | 2 869 buses |
|---|---|---|
| **total** | **2.62** | **10.35** |
| C++ AC sweep | 0.78 (30 %) | 3.57 (35 %) |
| C++ DC sweeps | 0.16 (6 %) | 0.64 (6 %) |
| C++ base-case solves (slack weights, §5 F2) | 0.07 (3 %) | 0.28 (3 %) |
| **numpy rebuild of what the sweep does not return** | **0.34 (13 %)** | **1.17 (11 %)** |
| output formatting, topology generation, guards, rest | 1.27 (48 %) | 4.69 (45 %) |

Of the numpy rebuild, the generator slack/Q recompute is only ~0.06 ms per row; the rest is branch
flows, bus injections, the per-row Ybus and the DC reconstruction. Moving all of it C++ side removes
**11–13 % of the loop** and, more importantly, the duplicated conventions.

### 3.2 Why a batch at all (single calls after a topology change)

Per power flow, case 2 869, KLU, one line opened:

| | AC | DC |
|---|---|---|
| unchanged grid, already converged (floor) | 0.97 ms | 0.21 ms |
| after `deactivate_powerline`, one `ac_pf` / `dc_pf` call | 7.6 ms | 2.4 ms |
| `ScenarioSweepCPP`, same outages | 2.6 ms | — |

After a topology change the single-call path pays Ybus + symbolic-analysis set-up again on every call
(DC is one linear solve, so its extra ~2.2 ms is pure set-up). The sweep avoids it.

### 3.3 Threads *(unverified in the pipeline)*

Standalone, 2 869 buses, AC sweep, 20 rows: 1 thread 3.65 ms/row, 2 threads 2.11, 4 threads 1.48.
Not measured inside the datakit, whose workers are already one per process.

---

## 4. What the datakit needs per row

For each of the R rows (R ≈ 20) of a sweep, at the solved state:

**AC** (per unit or MW/MVAr, but stated):

* complex bus voltage (exists: `get_voltages`)
* converged flag (exists: `converged_mask`)
* for every branch **in input order**: `P, Q` at both ends — zero for a branch that is out
* for every generator **in input order**: `P, Q` — zero for a generator that is out
* the bus admittance matrix Ybus of the row's topology (non-zeros)

**DC**:

* bus voltage angles, converged flag
* for every branch: `P` at the origin end (the other end is `-P`)
* for every generator: `P`

Per scenario: the base-case solve, to read the slack participation (§5 F2).

---

## 5. Feature list

Priorities: **P0** wrong results today; **P1** removes datakit reimplementation; **P2** speed or
ergonomics.

### B1 (P0) — DC sweep: a generator-only row after a line-outage row is solved on the wrong matrix

* **Observed:** `DC_KLU` and `DC_SparseLU`. With rows `[open line 5, lose generator 4]`, row 1 has a
  0.395° angle error against a grid with generator 4 deactivated; with the rows in the other order it
  is exact (6.7e-13°). AC (`NR_KLU`) is exact in every combination I tried.
  A line-outage row followed by another line row is fine; the problem needs the second row to change
  only generators. Reproducer: appendix A.1.
* **Suspected cause** *(unverified)*: the Ybus edit of the line row is not undone when the next row
  edits no admittance.
* **Required:** every row is solved on its own topology, whatever the rows before it.
* **Current workaround:** DC rows that open a branch are solved in a separate sweep from the others.
  Costs a second `compute`.
* **Acceptance:** for random mixes of line / trafo / generator rows in any order, every DC row equals a
  fresh `dc_pf` of that topology (≤ 1e-12°).

### B2 (P0) — DC sweep flows ignore phase shifters

* **Observed:** `compute_power_flows()` / `get_power_flows()` after a DC sweep differ from `dc_pf` +
  `get_trafo_res1` on the phase-shifting transformers (case 2 869: 12 of 4 582 branches, up to 73 MW).
  The angles are right. Reproducer: appendix A.1.
* **lightsim2grid's DC model**, fitted to its own `dc_pf` output to 1e-12 pu:
  `P_from = (θ_f − θ_t − shift) / (x · tap)` (`tap = 1` when 0; `shift` in radians, hence the
  `−shift` sign), `P_to = −P_from`.
* **Required:** sweep DC flows follow the same model as `dc_pf`.
* **Current workaround:** flows recomputed from the angles with the formula above.

### F1 (P1) — AC branch flows, both ends, from the sweep

* `P, Q` at the origin and extremity of every branch for every converged row, in the **order of the
  grid model's branches** (lines and trafos interleaved as in the input, or a documented permutation),
  in a single `(R, n_branch, 4)` array. Zero for branches that are out of service in that row.
* **Why:** the datakit's input is a MATPOWER branch table with lines and transformers mixed.
  `get_flows` returns amps and `get_power_flows` only the origin-side active power.
* **Today:** `S_f = V_f · conj(Y_ff V_f + Y_ft V_t)` etc. in numpy (0.18 ms/row at 2 869).
* **Acceptance:** equals `get_line_res1/2` + `get_trafo_res1/2` of a one-at-a-time solve (≤ 1e-9).

### F2 (P1) — generator `P` and `Q` from the sweep, with lightsim2grid's own rules

* `(R, n_gen, 2)`; zero for generators out of service in the row.
* The rules the datakit had to infer, and must match, are:
  * **Slack P.** The in-service generators of the slack bus share the active-power mismatch; the share
    of each is read today from a base-case `ac_pf` (`(P − P_set) / ΣΔ`), because the weights are not
    exposed. When a contingency removes one of them, the others are re-normalised
    (`GeneratorContainer.hpp`: "re-weights the distributed slack").
  * **Reactive power.** The reactive power of a PV bus is split between its in-service generators in
    proportion to `QMAX − QMIN` (checked on pglib 793, on two buses with two generators each: ratios match to 1e-4; 2 869 has no bus
    with several generators). Equal split when the sum of ranges is 0 is a guess *(unverified)*.
* **Required:** per-row `P, Q` straight from the solved state, so that none of the above is
  reimplemented. If the rules are configurable, state which setting matters.
* **Current workaround:** reconstruction in numpy plus a self-check against a one-at-a-time solve.
* **Acceptance:** equals `get_gen_res` of a one-at-a-time solve (≤ 1e-9 MW), including rows that remove
  a generator of the slack bus.

### F3 (P1) — per-row Ybus

* Return Ybus of each row as a sparse matrix (CSR) in the bus order of the grid model, or as the
  non-zeros of the base Ybus plus the entries removed by the row.
* **Why:** the dataset stores the admittance matrix of every variant. The datakit builds it today as
  "base Ybus minus the open branches" with exact bookkeeping of which entries become structurally zero
  (0.14 ms/row).
* **Acceptance:** same pattern as `get_Ybus` after a one-at-a-time solve of that topology (no explicit
  zeros), values within 1e-15 relative.

### F4 (P1) — DC outputs through the same API

* Angles (exists), flows both ends (B2), generator `P` (F2), with the DC slack rules the AC path uses.
* Ideally one entry point returning an AC or DC "result object" with identical accessors, so the
  datakit does not need two code paths.

### F5 (P1) — expose the slack participation per generator

* A getter for each generator's **current, normalised** slack share (today `get_slack_weights` is per
  bus, and the per-generator split inside a bus is not visible).
* **Why:** the datakit runs one base-case power flow per scenario only to read these shares
  (0.28 ms/row at 2 869, the AC and DC base solves together).

### F6 (P2) — slack weights must follow the generator set points, or say so

* **Observed:** the distributed-slack weights are fixed when the grid is built (proportional to the
  set points at that time). `change_p_gen` does not update them; only `update_slack_weights` does.
  After halving one of three equal slack generators' set points, the three still take **0.333 / 0.333 /
  0.333** of the mismatch (proportional to the new set points would be 0.2 / 0.4 / 0.4); after
  `update_slack_weights`: 0.2 / 0.4 / 0.4. Reproducer: appendix A.2.
* **Why it matters:** the datakit changes generator set points in place for every scenario and reuses one
  `LSGrid` per worker, so a slack bus with several generators of different set points is shared with
  the weights of whichever scenario built the grid. The datakit side needs fixing too (it should call
  `update_slack_weights`); the request here is to **document** the behaviour, or to refresh the weights
  in `change_p_gen` for generators taking part in the slack.

### F7 (P2) — make KLU the default solver when it is compiled in

* An `LSGrid` from `init_from_matpower` solves with `NR_SparseLU` / `DC_SparseLU`; the sweep wrapper
  switches to KLU when available. On pglib 118 – 2 869 buses KLU is 3 – 4× faster (AC) and ~2× faster
  (DC), with voltages equal to ≤ 1.1e-12 pu and the same convergence on every perturbation.
* **Current workaround:** `change_algorithm(NR_KLU)` and `change_algorithm(DC_KLU)` after loading.

### F8 (P2) — cheaper single-call path after a topology change

* The single `ac_pf` / `dc_pf` call after `deactivate_powerline` costs ~3× (AC) and ~4× (DC) the sweep
  per row (§3.2 and the DC sweeps row of §3.1). Reusing the symbolic analysis and updating Ybus incrementally when a branch is
  (de)activated would speed up every caller that is not batched.
* Independent of the batch; listed because the datakit still has a one-at-a-time fallback.

### F9 (P2) — reuse a sweep object across scenarios *(unverified need)*

* A `ScenarioSweepCPP` is rebuilt for every scenario (0.07 – 0.13 ms/row). If the grid's loads,
  generator set points or branch parameters are changed in place, allow the same sweep object to be
  recomputed without reconstruction, keeping its symbolic analysis. Not measured as a bottleneck.

### F10 (P2) — threads in a multi-process caller

* Document `nb_thread` guidance when the caller already runs one process per core (oversubscription),
  and whether `compute` releases the GIL (allowing a thread pool instead of processes).

### F11 (P2) — per-row timing and iterations

* Expose per-row `solve_time` and iteration count. The datakit stores a `solve_time` per sample and
  currently writes the batch total divided by the number of rows.

---

## 6. Semantics to confirm or document

1. A contingency mask entry on an element **already out of service** in the grid: accepted and
   harmless? *(not tested; the datakit only masks elements in service, but a mask built from the
   target state would be simpler)*.
2. A row whose contingency **islands** the grid: `converged_mask` is False and the row's voltages are
   zero. (The datakit filters islanding variants upstream, so this is not exercised.)
3. `set_contingency_gens` raises for generators regulating a remote bus or held by a control group. The
   datakit catches this and falls back to one power flow per perturbation; confirm the exception type.
4. All rows start from the same `v_init`; `init_from_n_powerflow` is not used.

## 7. Acceptance, as an executable reference

`tests/lightsim2grid/test_lightsim2grid_batch.py` in gridfm-datakit compares the batched path with one
power flow per perturbation on `ieee14` for N-1, for random k ≤ 3 mixes of branch and generator
outages, and for the vectorised post-processing. Once F1–F5 exist, its numpy rebuild
(`gridfm_datakit/lightsim2grid/batch.py`: `_gen_results`, `_ybus_removal`, the flow formulas in
`solve_ls_pf_batch`) can be deleted and the same tests keep passing unchanged.

## 8. Expected gain, honestly

* F1 – F5 together remove the numpy rebuild: **11 – 13 % of the datakit inner loop**, plus the slack
  base-case solve (~3 %).
* The loop would still be ~half non-C++ (dense output formatting 17 %, topology generation 9 – 18 %,
  guards and bookkeeping), which these features do not touch.
* B1 and B2 are worth fixing regardless of speed: they return wrong DC results.

---

## Appendix A — reproducers

Files used: `pglib_opf_case118_ieee.m`, `pglib_opf_case793_goc.m`, `pglib_opf_case2869_pegase.m` from
<https://github.com/power-grid-lib/pglib-opf>.

### A.1 B1 and B2

```python
import sys
import numpy as np
from lightsim2grid import network
from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP

def dc_sweep(grid, line_mask, gen_mask, algo="DC_KLU"):
    sweep = ScenarioSweepCPP(grid)
    sweep.change_algorithm({a.name: a for a in sweep.available_default_algorithms()}[algo])
    sweep.set_contingency_lines(line_mask)
    if len(grid.get_trafos()):
        sweep.set_contingency_trafos(np.zeros((line_mask.shape[0], len(grid.get_trafos())), dtype=bool))
    sweep.set_contingency_gens(gen_mask)
    sweep.compute(np.ones(grid.total_bus(), dtype=complex), 50, 1e-8)
    return sweep

# B1: a generator-only row right after a line-outage row
grid = network.init_from_matpower(sys.argv[1])            # pglib_opf_case118_ieee.m
LINE, GEN = 5, 4                                           # generator 4 produces 252.5 MW
n_line, n_gen = len(grid.get_lines()), len(grid.get_generators())
line_mask = np.zeros((2, n_line), dtype=bool); line_mask[0, LINE] = True   # row 0: open a line
gen_mask = np.zeros((2, n_gen), dtype=bool);   gen_mask[1, GEN] = True     # row 1: lose a generator
V = np.asarray(dc_sweep(grid, line_mask, gen_mask).get_voltages())
ref = network.init_from_matpower(sys.argv[1]); ref.deactivate_gen(GEN)
V_ref = np.asarray(ref.dc_pf(np.ones(ref.total_bus(), dtype=complex), 50, 1e-8))
print(np.degrees(np.abs(np.angle(V[1]) - np.angle(V_ref))).max())          # 0.395 (expected ~1e-14)
swapped = np.asarray(dc_sweep(grid, line_mask[::-1].copy(), gen_mask[::-1].copy()).get_voltages())
print(np.degrees(np.abs(np.angle(swapped[0]) - np.angle(V_ref))).max())    # 6.7e-13: exact

# B2: DC flows of the sweep vs dc_pf, phase shifters
grid = network.init_from_matpower(sys.argv[2])            # pglib_opf_case2869_pegase.m
sweep = dc_sweep(grid, np.zeros((1, len(grid.get_lines())), dtype=bool),
                 np.zeros((1, len(grid.get_generators())), dtype=bool))
sweep.compute_power_flows(); p_sweep = np.asarray(sweep.get_power_flows())[0]   # lines then trafos, MW
grid.dc_pf(np.ones(grid.total_bus(), dtype=complex), 50, 1e-8)
p_ref = np.concatenate([np.asarray(grid.get_line_res1()[0]), np.asarray(grid.get_trafo_res1()[0])])
err = np.abs(p_sweep - p_ref)
print(int((err > 1e-6).sum()), "of", err.size, "branches differ, max", err.max(), "MW")   # 12 of 4582, 73 MW
```

### A.2 F6 — slack weights are fixed at build time

```python
import sys
import numpy as np
from lightsim2grid import network

g = network.init_from_matpower(sys.argv[1])    # pglib_opf_case793_goc.m: generators 53, 54, 55 on the slack bus
v0 = np.ones(g.total_bus(), dtype=complex)
ids, p_set = [53, 54, 55], np.array([300.0, 599.6645, 599.6645])
g.change_p_gen(53, 300.0)                      # was 599.6645 like the other two
g.ac_pf(v0, 100, 1e-8)
p = np.asarray(g.get_gen_res()[0])[ids]
print((p - p_set) / (p - p_set).sum())         # [0.333 0.333 0.333]; proportional to the new set points: [0.2 0.4 0.4]
g.update_slack_weights(np.isin(np.arange(len(g.get_generators())), ids))
g.ac_pf(v0, 100, 1e-8)
p = np.asarray(g.get_gen_res()[0])[ids]
print((p - p_set) / (p - p_set).sum())         # [0.2 0.4 0.4]
```

## Appendix B — how the numbers were obtained

* **Inner loop:** the real `process_scenario_pf_mode` with `solver = lightsim2grid`, `run_opf` replaced by
  the base-case power flow (no Julia in the build container), topology = random outages (k = 1, 20
  variants per scenario), loads varied ±10 % per scenario, DC on, `NR_KLU` / `DC_KLU`, one process,
  shared 4-core container (run-to-run noise ±25 %, hence best of 5).
* **Single call vs sweep:** `ac_pf` / `dc_pf` after `deactivate_powerline` on 40 lines, against
  `ScenarioSweepCPP` with the same 40 single-line contingencies.
* **Equivalence:** batched vs one-at-a-time results agree to ≤ 4.4e-11 (AC, DC, generator outputs,
  Ybus, convergence) on pglib 118, 793, 1 354, 2 383 and 2 869 buses, including k = 3 mixed rows.
