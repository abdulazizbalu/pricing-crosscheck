# Phase 1 measurement protocol

Protocol version 1.0. This document was written and merged before any
measurement results were generated. Every number used by the measurements
(grids, engine settings, tolerances, seeds) is in `specs/phase1.json`; this
document explains the choices. If the two disagree, that is a bug; the JSON
file is what the code reads.

Changes made after results were seen are listed in
[Deviations from protocol](#deviations-from-protocol) with the reason. The
original text above that section is not edited after merge, except for
typographical corrections.

## 1. Model and market conventions

- Black-Scholes-Merton dynamics under the pricing measure:
  dS = (r - q) S dt + sigma S dW, with constant rate r, continuous dividend
  yield q and volatility sigma.
- Spot S0 = 100.
- Valuation date 2025-01-15. Maturity = valuation date + D calendar days.
  Day count Actual/365 Fixed, no calendar, so T = D / 365 exactly in every
  implementation. Flat curves with continuous compounding.
- Greeks units: delta = dV/dS, gamma = d2V/dS2, vega = dV/dsigma per unit of
  volatility (not per vol point), theta = dV/dt in calendar time, per year
  (that is, theta = -dV/dT).

## 2. Instruments

| Instrument | Variants | Notes |
|---|---|---|
| European | call, put | strike K |
| American | call, put | exercise at any time in [valuation date, maturity] |
| Single barrier | down/up, in/out, call/put | rebate 0, continuous monitoring, spot strictly on the live side |
| Cash-or-nothing digital | call, put | pays 1 at maturity if S_T > K (call) or S_T < K (put) |

## 3. Parameter grids

Full grids (Cartesian products, `specs/phase1.json`, key `grids`):

| Grid | Axes | Points per option variant |
|---|---|---|
| vanilla (European, American, digital) | K in {50, 80, 95, 100, 105, 120, 200}; D in {1, 7, 91, 365, 3650}; sigma in {0.01, 0.1, 0.3, 1.0}; r in {-0.02, 0, 0.05}; q in {0, 0.03} | 840 |
| barrier | H down {70, 90, 99, 99.9} or up {100.1, 101, 110, 130}; K in {90, 100, 110}; D in {30, 365, 1825}; sigma in {0.1, 0.3}; (r, q) in {(0, 0), (0.05, 0.02), (-0.01, 0)} | 216 |

The stressed regions are deliberate: deep in and out of the money (K = 50,
200), one-day and ten-year maturities, sigma = 0.01 and 1.0, zero and
negative rates, spot 0.1% from the barrier.

Monte Carlo engines that are slow per point run on fixed subsets
(`american_mc_subset`, `barrier_mc_subset`).

The `fast` profile (CI on every pull request) uses smaller subsets defined in
`profiles.fast`. It runs the same code and settings; only the grids are
smaller.

## 4. Engines compared

QuantLib engines (Python wheel, version pinned in `requirements.lock` and
recorded in every results file):

| Instrument | QuantLib engines and settings |
|---|---|
| European | AnalyticEuropeanEngine; BinomialCRRVanillaEngine (1000 steps); BinomialLRVanillaEngine (1001 steps); FdBlackScholesVanillaEngine (tGrid 400, xGrid 400, dampingSteps 0 and 2); MCEuropeanEngine (pseudorandom, 1 time step, antithetic, 131072 samples) |
| American | BaroneAdesiWhaleyApproximationEngine; BjerksundStenslandApproximationEngine; QdFpAmericanEngine (high-precision scheme); BinomialCRRVanillaEngine (1000); FdBlackScholesVanillaEngine (400 x 400, dampingSteps 0 and 2); MCAmericanEngine (Longstaff-Schwartz, 50 time steps, antithetic, 32768 samples, 4096 calibration samples, monomial basis of order 3; subset grid) |
| Barrier | AnalyticBarrierEngine; BinomialCRRBarrierEngine (1000); FdBlackScholesBarrierEngine (400 x 400); MCBarrierEngine (one time step per calendar day, Brownian-bridge correction on, antithetic, 65536 samples; subset grid) |
| Digital | AnalyticEuropeanEngine; FdBlackScholesVanillaEngine (400 x 400, dampingSteps 0 and 2). The binomial and MC European engines are also called; if they refuse the payoff, this is recorded as "not available", not as a disagreement |

Our own implementations (pure NumPy/SciPy, written without reading QuantLib
source code):

| Name | Method |
|---|---|
| `cf` | Closed forms: Black-Scholes-Merton European, cash-or-nothing digital, single barriers with rebate 0 (continuous monitoring) |
| `own_crr` | Cox-Ross-Rubinstein binomial tree, European and American |
| `own_pde` | Crank-Nicolson in log-spot with Rannacher start-up (implicit Euler half steps), strike placed on a grid node, payoff cell-averaged; barrier placed on the boundary node; American exercise by a penalty iteration |
| `own_mc` | Seeded Monte Carlo: exact lognormal sampling at maturity; for barriers, exact sampling on a daily grid plus the Brownian-bridge crossing probability between dates, which removes the monitoring bias for continuous barriers under constant parameters |

ORE: `open-source-risk-engine` 1.8.17.0 is installable from PyPI on CPython
3.10-3.12 (manylinux, macOS arm64, Windows wheels). For these Phase 1
products its pricing engines are the QuantLib engines it bundles, so it would
not be an independent implementation. It is planned as an optional second
engine for products where ORE adds its own code; it is not used in Phase 1.

## 5. Reference values

| Comparison | Reference |
|---|---|
| European prices and Greeks | `cf` closed form |
| Digital prices and Greeks | `cf` closed form |
| Barrier prices | `cf` closed form (continuous monitoring) |
| American prices | `ref_pde`: `own_pde` with the penalty method on three grids, (space nodes, time steps) = (800, 400), (1600, 800), (3200, 1600). Reference value = finest grid. Uncertainty estimate u = abs(V3 - V2). |

Rules:

1. A reference is used only after it passed the validation tests of
   section 7. A failing reference is treated as a bug until shown otherwise.
2. No engine is compared against itself or against a value derived from its
   own output. QuantLib values are never used to build or tune a reference.
3. American reference uncertainty: a point is used only if u <= 0.1 x the
   numerical tolerance at that point. Points that fail this are excluded from
   the American comparison and reported separately with their count.
4. Our own numerical engines (`own_crr`, `own_pde`, `own_mc`) are also
   compared with the closed forms on the full grid; for American options they
   are compared with `ref_pde`, which shares code with `own_pde` (same solver,
   much finer grid). That comparison is a self-convergence check and is
   labelled as such; it is not counted as independent evidence.

## 6. Tolerances and what counts as a disagreement

Let d = engine value - reference value and tol the tolerance below. A
**disagreement** is |d| > tol.

| Class | Applies to | tol | Why |
|---|---|---|---|
| exact | closed form vs closed form (QuantLib analytic engines vs `cf`) | max(1e-9, 1e-9 x abs(ref)) | Same formula in double precision. Differences should come only from rounding and the normal CDF implementation (about 1e-15 relative); 1e-9 leaves room for cancellation of up to six orders of magnitude. |
| numerical | trees, PDE, analytic approximations | max(1e-3, 1e-4 x abs(ref)) | 1e-3 is 1e-5 of spot; 1e-4 relative is one basis point of the price. This is meant as a practitioner-level acceptance threshold, not as the expected error of each method at the fixed settings. Exceeding it is what the disagreement map reports and explains. |
| Monte Carlo | MC engines | numerical tol + 4 x SE | SE is the engine's own standard error estimate. Under a correct unbiased estimator a 4-sigma exceedance has probability about 6e-5 per point. |

Relative differences are reported as d / abs(ref) when abs(ref) > 1e-12 and
left empty otherwise.

Greeks: a Greek disagrees if |d| > max(abs_floor, 1e-3 x abs(ref)) with
abs_floor = 1e-4 (delta), 1e-5 (gamma), 1e-3 (vega), 1e-3 (theta).

Identity checks (section 9) use the class of the engine that produced the
values: exact for analytic engines, numerical (applied to the sum of the
magnitudes involved) for deterministic numerical engines, and numerical plus
4 x (sum of SEs) for MC.

## 7. Validation of the references (before use)

Closed forms (`cf`), tested against model identities and limits:

- put-call parity; digital call + digital put = exp(-rT);
- digital call = -dC/dK (numerical derivative of the closed-form call);
- sigma -> 0: European -> exp(-rT) max(F - K, 0) with F = S exp((r - q)T);
- T -> 0: price -> payoff;
- deep in the money call -> S exp(-qT) - K exp(-rT); deep out of the money -> 0;
- comparison with numerical integration of the payoff against the lognormal
  density (scipy.integrate.quad), independent of the closed form;
- Greeks: against high-order finite differences of the closed form, and the
  Black-Scholes PDE identity theta + 0.5 sigma^2 S^2 gamma + (r - q) S delta - r V = 0;
- barriers: in + out = vanilla; barrier far from spot: knock-out -> vanilla,
  knock-in -> 0; spot at the barrier: knock-out = 0, knock-in = vanilla;
  up-and-out call with H <= K and down-and-out put with H >= K are worthless;
  comparison with `own_mc` with bridge correction.

Numerical references must converge to the closed form at their theoretical
order on the convergence points (`convergence.points`, `barrier_points`):

| Method | Expected order | Acceptance |
|---|---|---|
| `own_crr` European | 1 in steps N (with odd-even oscillation) | fitted order in [0.8, 1.2] on even N |
| `own_pde` European, digital, knock-out | 2 in grid size (space and time refined together) | fitted order in [1.8, 2.2] |
| `own_mc` European, digital | standard error N^(-1/2), unbiased | fitted SE slope in [-0.55, -0.45]; abs(error) <= 4 SE at every sample size |
| `own_mc` barrier with bridge | unbiased for continuous monitoring | abs(error) <= 4 SE at every monitoring frequency |
| `ref_pde` American | converges; agrees with `own_crr` | u <= 0.1 tol on the convergence points; American call with q = 0, r >= 0 equals the European closed form within numerical tol; abs(ref_pde - own_crr(6400 and 6401 steps averaged)) <= numerical tol |

The fitted order is the least-squares slope of log abs(error) against
log(resolution) over the last `fit_last_levels` = 5 levels.

## 8. Greeks

Instruments: European and digital (closed-form Greeks exist). Methods:

1. Analytic formula (`cf`) - reference.
2. QuantLib engine Greeks where the engine provides them: analytic (delta,
   gamma, vega, theta), finite differences (delta, gamma, theta from the grid),
   binomial (delta, gamma, theta; vanilla only).
3. Bump and revalue on the closed form with two documented bump sets:
   - desk: spot +/- 1% (relative), vol +/- 0.01 (absolute), theta as the
     one-day change (V(T - 1/365) - V(T)) x 365, one-sided;
   - fine: spot +/- 1e-4 relative, vol +/- 1e-4 absolute, time +/- 1e-4 T,
     central differences.
4. Bump-size sweep: relative bumps eps in {1e-1, ..., 1e-8} for spot (h = eps
   S), vol (h = eps sigma) and time (h = eps T), central differences, on the
   `greeks.sweep_points`. Expected shape: truncation error ~ eps^2 for large
   eps, round-off growing like machine epsilon / eps (first derivatives) or
   / eps^2 (gamma) for small eps.

## 9. Identity checks

| Identity | Engines | Condition |
|---|---|---|
| Put-call parity C - P = S exp(-qT) - K exp(-rT) | every European engine | all points |
| In-out parity KI + KO = vanilla | every barrier engine; the vanilla comes from the same engine family with the same settings (analytic, CRR, FD); for MC from the closed form | all points |
| Digital = limit of call spread (C(K - h) - C(K + h)) / (2h) | closed form, QuantLib analytic, QuantLib FD | h in {1, 0.1, 0.01, 0.001}; report observed order in h |
| Digital call + digital put = exp(-rT) | every digital engine | all points |
| American >= European | every American engine vs European closed form | violation if A < E - tol |
| No early exercise of an American call | every American engine vs European closed form | q = 0 and r >= 0 |
| No early exercise of an American put | every American engine vs European closed form | r <= 0 and q >= 0 |

An identity that an engine satisfies by construction (for example if it
prices one leg as the difference of the others) carries no information. If
an identity holds to machine precision for a numerical engine, this is
reported and the engine is labelled as satisfying it by construction only
after checking its source code.

For the American call with r < 0 and q = 0 the no-early-exercise argument
does not apply (K exp(-rT) > K). The early-exercise premium of `ref_pde`
there is reported as an observation, not as a check.

## 10. Convergence measurements

On `convergence.points` and `convergence.barrier_points`:

- trees (QuantLib CRR, LR, CRR barrier; `own_crr`): steps in {25, 50, ..., 6400};
- PDE (QuantLib FD with dampingSteps 0 and 2; `own_pde`): space nodes = time
  steps in {25, 50, ..., 1600};
- MC (QuantLib MC, `own_mc`): samples 2^10 ... 2^20; for barriers also
  monitoring steps per year in {12, 52, 365, 1460}, with and without bridge
  correction.

Observed order: least-squares slope over the last five levels, plus local
orders log2(e_k / e_(k+1)).

## 11. Disagreement map and explanations

For every disagreement (engine, point) the engine is re-run with its
resolution multiplied by 2 and 4 (tree steps; FD space nodes and time steps
together; MC samples by 4 and 16 with the same seed rule). From d1, d2, d4
the measurement assigns one category, by these rules fixed in advance:

| Category | Rule |
|---|---|
| discretization, converging | abs(d) decreases under refinement with a local order >= 0.5, or abs(d4) <= tol |
| MC noise | MC only: abs(d) <= tol at 4 x or 16 x samples and the exceedance is within 5 SE |
| MC bias | MC only: abs(d) does not shrink with samples (bias exceeds 4 SE at 16 x samples) |
| approximation, no refinement parameter | BAW, Bjerksund-Stensland: no refinement exists; the explanation rests on the reference being converged and QdFp agreeing with it |
| not explained | none of the above: listed as an open item |

Each disagreement is also tagged with region labels (not mutually
exclusive), computed from the inputs:

- near expiry: D <= 7;
- near-zero vol: sigma <= 0.01;
- near barrier: abs(ln(H / S)) <= 0.01;
- deep in/out of the money: abs(ln(K / F)) / (sigma sqrt(T)) > 3;
- negative rate: r < 0.

Explanations in the documentation (barrier discretization, gamma near
expiry, MC noise, PDE grid alignment with the strike) are stated only with a
refinement or convergence experiment that supports them.

## 12. Random seeds

Base seed 20250115. Each (point, engine) pair gets seed =
1 + (first four bytes of SHA-256 of "base|point_id|engine" read as an unsigned
integer) mod 2147483646. Seeds do not depend on execution order or the number
of worker processes. QuantLib seed 0 (clock-based) is never used. `own_mc`
uses numpy's PCG64 generator.

## 13. Reproducibility

`python -m pricing_crosscheck run --profile full` regenerates `results/full`
(CSV plus generated markdown summary) and the charts in `docs/`. Every CSV
starts with comment lines holding the content hash (code, specs, lock file,
library versions) and the exact versions. CI regenerates the fast profile on
every pull request and compares it with the committed files; the full grid
runs on manual dispatch and weekly.

## Deviations from protocol

None yet.
