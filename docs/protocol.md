# Phase 1 measurement protocol

Protocol version 1.0. This document is merged before any measurement results
are generated. Every number used by the measurements (grids, engine settings,
tolerances, seeds) is in `specs/phase1.json`; this document explains the
choices. If the two disagree, that is a bug; the JSON file is what the code
reads.

Changes made after results were seen are listed in
[Deviations from protocol](#deviations-from-protocol) with the reason. The
text above that section is not edited after merge, except for typographical
corrections.

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
  (theta = -dV/dT).
- Notional scale used by tolerances: 100 (the spot) for European, American
  and barrier options; 1 (the cash amount) for digitals.

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
| barrier | H down {70, 90, 99, 99.9} or up {100.1, 101, 110, 130}; K in {90, 100, 110}; D in {30, 365, 1825}; sigma in {0.1, 0.3}; (r, q) in {(0, 0), (0.05, 0.02), (-0.01, 0)} | 216 per barrier type |

The stressed regions are deliberate: deep in and out of the money (K = 50,
200), one-day and ten-year maturities, sigma = 0.01 and 1.0, zero and
negative rates, spot 0.1% from the barrier.

Monte Carlo engines that are slow per point run on fixed subsets
(`american_mc_subset`, `barrier_mc_subset`).

The `fast` profile (CI on every pull request) uses smaller subsets defined in
`profiles.fast`. It runs the same code and settings; only the grids are
smaller. An unknown profile name is an error.

## 4. Engines compared

QuantLib engines (Python wheel, version pinned in `requirements.lock` and
recorded in every results file):

| Instrument | QuantLib engines and settings |
|---|---|
| European | AnalyticEuropeanEngine; BinomialCRRVanillaEngine (1000 steps); BinomialLRVanillaEngine (1001 steps; QuantLib increases even counts by one); FdBlackScholesVanillaEngine (tGrid 400, xGrid 400, dampingSteps 0 and 2, default Douglas scheme); MCEuropeanEngine (pseudorandom, 1 time step, antithetic, requiredSamples 131072) |
| American | BaroneAdesiWhaleyApproximationEngine; BjerksundStenslandApproximationEngine; QdFpAmericanEngine (high-precision scheme); BinomialCRRVanillaEngine (1000); FdBlackScholesVanillaEngine (400 x 400, dampingSteps 0 and 2); MCAmericanEngine (Longstaff-Schwartz, 50 exercise dates, so it prices a Bermudan; antithetic, requiredSamples 32768, 4096 calibration samples, monomial basis of order 3; subset grid) |
| Barrier | AnalyticBarrierEngine; BinomialCRRBarrierEngine (1000 steps; QuantLib may adjust the step count and applies a Derman-Kani correction); FdBlackScholesBarrierEngine (400 x 400); MCBarrierEngine (one time step per calendar day, `isBiased=False`, which selects the path pricer that uses the barrier-crossing probability between dates; `brownianBridge=False`, which only concerns path construction; antithetic, requiredSamples 65536; subset grid) |
| Digital | AnalyticEuropeanEngine; FdBlackScholesVanillaEngine (400 x 400, dampingSteps 0 and 2). The binomial and MC European engines are also called; they are expected to refuse the payoff |

"Samples" for QuantLib MC engines means `requiredSamples` as QuantLib defines
it; for `own_mc` it means simulated paths (antithetic pairs = samples / 2).
Comparisons use each engine's own standard error, so equal path counts are
not required.

Our own implementations (pure NumPy/SciPy, written without reading QuantLib
source code):

| Name | Method |
|---|---|
| `cf` | Closed forms: Black-Scholes-Merton European, cash-or-nothing digital, single barriers with rebate 0 (Reiner-Rubinstein, continuous monitoring) |
| `own_crr` | Cox-Ross-Rubinstein binomial tree, European and American |
| `own_pde` | Crank-Nicolson with Rannacher start-up (each of the first two steps replaced by two implicit Euler half-steps). Vanilla, digital and American: log-forward variable (no drift term), domain of +/- 8 standard deviations around the forward, strike on a grid node when it lies in the domain, payoff cell-averaged, value at the spot by cubic interpolation, far boundaries from the zero-volatility limit. Knock-outs: log-spot variable, barrier on the boundary node and strike on a node. American exercise by a penalty iteration |
| `own_mc` | Seeded Monte Carlo: exact lognormal sampling at maturity; for barriers exact sampling on a daily grid plus the Brownian-bridge crossing probability between dates, which removes the monitoring bias for continuous barriers under constant parameters |

ORE: `open-source-risk-engine` 1.8.17.0 is installable from PyPI on CPython
3.10-3.12 (manylinux, macOS arm64 and Windows wheels). For these Phase 1
products its pricing engines are the QuantLib engines it bundles, so it would
not be an independent implementation. It is planned as an optional second
engine for products where ORE adds its own code; it is not used in Phase 1.

### Engine failures

An engine call that raises an exception or returns a non-finite value is
recorded as **not available**, with the message, and counted per engine and
instrument. It is never a disagreement and never part of a difference
statistic. Payoff refusals expected in section 4 are recorded the same way.

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
   are compared with `ref_pde`. `own_pde` and `ref_pde` share code (same
   solver, finer grid): that comparison is a self-convergence check, labelled
   as such and not counted as independent evidence.

## 6. Tolerances and what counts as a disagreement

Let d = engine value - reference value and tol the tolerance below. A
**disagreement** is |d| > tol.

| Class | Applies to | tol | Why |
|---|---|---|---|
| exact | QuantLib analytic engines vs `cf` (prices and Greeks) | max(1e-9, 1e-9 x abs(ref)) | Same formula in double precision. Differences should come only from rounding and the normal CDF implementation (about 1e-15 relative); 1e-9 leaves room for cancellation of up to six orders of magnitude. |
| numerical | trees, PDE, analytic approximations | max(1e-5 x notional, 1e-4 x abs(ref)) | 1e-5 of notional is 1e-3 in price units for spot-based options and 1e-5 for a digital paying 1; 1e-4 relative is one basis point of the price. This is an acceptance threshold, not the expected error of each method at the fixed settings. Exceeding it is what the disagreement map reports and explains. |
| Monte Carlo | MC engines | numerical tol + 4 x SE | SE is the engine's own standard error estimate. Under a correct unbiased estimator a 4-sigma exceedance has probability about 6e-5 per point. |

Relative differences are reported as d / abs(ref) when abs(ref) > 1e-12 and
left empty otherwise.

Greeks: analytic-vs-analytic comparisons use the exact class. Numerical and
bumped Greeks disagree if |d| > max(floor x notional, 1e-3 x abs(ref)) with
floor = 1e-6 (delta), 1e-7 (gamma), 1e-5 (vega), 1e-5 (theta); for spot-based
options this is 1e-4, 1e-5, 1e-3, 1e-3.

Identity checks (section 9) use the class of the engine that produced the
values: exact for analytic engines, numerical (applied to the largest
magnitude involved) for deterministic numerical engines, and numerical plus
4 x (sum of the SEs involved) for MC.

## 7. Validation of the references (before use)

Closed forms (`cf`), tested against model identities and limits:

- put-call parity; digital call + digital put = exp(-rT);
- digital call = -dC/dK (numerical derivative of the closed-form call);
- sigma -> 0, calls and puts: European -> exp(-rT) max(phi (F - K), 0) with
  F = S exp((r - q)T);
- T -> 0: price -> payoff;
- deep in the money call -> S exp(-qT) - K exp(-rT); deep out of the money -> 0;
- comparison with numerical integration of the payoff against the lognormal
  density (scipy.integrate.quad), independent of the closed form;
- Greeks: against Richardson-extrapolated finite differences of the closed
  form, and the Black-Scholes PDE identity
  theta + 0.5 sigma^2 S^2 gamma + (r - q) S delta - r V = 0;
- barriers: in + out = vanilla; barrier far from spot: knock-out -> vanilla,
  knock-in -> 0; spot at the barrier: knock-out = 0, knock-in = vanilla;
  up-and-out call with H <= K and down-and-out put with H >= K are worthless;
  down-and-out call (H <= K) and up-and-out put (H >= K) against the method of
  images, C(S) - (S/H)^(1 - 2(r - q)/sigma^2) C(H^2/S); convergence of
  `own_pde` knock-outs and `own_mc` with bridge correction to the closed form.

Numerical references must converge to the closed form at their theoretical
order on the convergence points (`convergence.points`, `barrier_points`):

| Method | Expected order | Acceptance |
|---|---|---|
| `own_crr` European | 1 in steps N; the error constant oscillates with the strike's position relative to the nodes, so the local slope over a finite range can exceed 1 | fitted order >= 0.8 on even N, and N x abs(error) at the last level <= 2 x the largest N x abs(error) at earlier levels |
| `own_pde` European, digital, knock-out | 2 in grid size (space and time refined together) | fitted order in [1.8, 2.2] |
| `own_mc` European, digital | standard error N^(-1/2), unbiased | fitted SE slope in [-0.55, -0.45] over the last five sample sizes; abs(error) <= 4 SE at every sample size |
| `own_mc` barrier with bridge | unbiased for continuous monitoring | abs(error) <= 4 SE at every monitoring frequency |
| `ref_pde` American | converges; agrees with `own_crr` | u <= 0.1 tol on the convergence points; American call with q = 0, r >= 0 and American put with r <= 0, q >= 0 equal the European closed form within numerical tol; abs(ref_pde - own_crr) <= numerical tol, with own_crr the average of 6400 and 6401 steps |

Control experiments, expected to fail the criteria above and so showing that
the tests can fail: `own_pde` digital without strike alignment and payoff
smoothing (order below 1.5); `own_mc` barrier without bridge correction at
12 dates per year (bias above 4 SE).

Fitted order: least-squares slope of -log abs(error) against log(resolution)
over the last five levels whose abs(error) exceeds the rounding floor
1e-8 x max(1, abs(ref)); at least three such levels are required, otherwise
the order is reported as not measurable and the acceptance becomes
abs(error) <= numerical tol at every level from 100 nodes or steps upwards.
Local orders log2(e_k / e_(k+1)) are reported alongside.

## 8. Greeks

Instruments: European and digital (closed-form Greeks exist). Methods:

1. Analytic formula (`cf`) - reference.
2. QuantLib engine Greeks where the engine provides them: analytic (delta,
   gamma, vega, theta), finite differences (delta, gamma, theta from the
   grid), binomial (delta, gamma, theta; vanilla only). If a QuantLib Greek is
   observed to satisfy the Black-Scholes PDE identity with the engine's own
   value, delta and gamma to rounding level, it is labelled as derived from
   that identity rather than independently computed.
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
| Digital call + digital put = exp(-rT) | every digital engine | all points |
| American >= European | every American engine vs European closed form | violation if A < E - tol |
| No early exercise of an American call | every American engine vs European closed form | q = 0 and r >= 0 |
| No early exercise of an American put | every American engine vs European closed form | r <= 0 and q >= 0 |

Observations (reported, not pass/fail checks):

- Digital as the limit of a call spread (C(K - h) - C(K + h)) / (2h), with h
  in {1, 0.1, 0.01, 0.001}, for the closed form, QuantLib analytic and
  QuantLib FD. For FD the call-price error divided by 2h dominates at small h,
  so the FD row shows that effect rather than testing the identity.
- For the American call with r < 0 and q = 0 the no-early-exercise argument
  does not apply (K exp(-rT) > K). The early-exercise premium of `ref_pde`
  there is reported.

An identity that an engine satisfies by construction carries no information.
For the QuantLib FD barrier engine the in-out residual is expected to be zero
to machine precision (QuantLib's FD knock-in is consistent with vanilla minus
knock-out on the same grid); that row is reported as not informative and its
knock-in prices are judged only by the direct comparison with `cf`. Any other
identity that holds to machine precision for a numerical engine is reported
and labelled the same way.

## 10. Convergence measurements

On `convergence.points` and `convergence.barrier_points`:

- trees (QuantLib CRR, LR, CRR barrier; `own_crr`): steps in {25, 50, ..., 6400}
  (QuantLib LR uses the next odd count; the CRR barrier engine may adjust
  steps);
- PDE (QuantLib FD with dampingSteps 0 and 2; `own_pde`): space nodes = time
  steps in {25, 50, ..., 1600};
- MC (QuantLib MC, `own_mc`): samples 2^10 ... 2^20; for barriers also
  monitoring steps per year in {12, 52, 365, 1460}, with and without the
  crossing correction (QuantLib `isBiased` False and True; `own_mc` bridge on
  and off);
- each level uses its own seed (section 12).

Orders are fitted as in section 7.

## 11. Disagreement map and explanations

For each disagreement (engine e, point x) with difference d0 and tolerance
tol0, the category is the first rule that applies:

1. **Engine failure**: not a disagreement (section 4).
2. **Approximation engine** (BAW, Bjerksund-Stensland): category
   "approximation error". This is not counted as explained. The size of the
   error is reported (count, median, maximum, by region label), together with
   evidence that the reference is converged there (u) and the QdFp difference
   at the same point.
3. **No refinement parameter** (QdFp at the high-precision scheme, analytic
   engines): category "not explained", listed individually.
4. **Deterministic numerical engine** (trees, FD, `own_pde`, `own_crr`):
   re-run at 2x and 4x resolution (tree steps; FD space nodes and time steps
   together), giving d1 and d2.
   - |d2| <= tol0: "discretization, resolved at 4x";
   - otherwise if |d0| > |d1| > |d2| and log2(|d0| / |d2|) / 2 >= 0.5:
     "discretization, converging slowly";
   - otherwise: "not explained".
5. **Monte Carlo, European/digital/barrier**: re-run with 4x and 16x samples,
   new seeds, giving (d1, SE1), (d2, SE2).
   - |d2| <= tol_num + 4 SE2: "MC noise" (the first exceedance was not
     reproduced with more samples);
   - otherwise, for barrier MC only, re-run with 4x and 16x monitoring
     frequency at the original sample count; if |d| then decreases with a
     local order >= 0.5 or falls within tol_num + 4 SE: "monitoring
     discretization";
   - otherwise: "MC bias, not explained".
6. **Longstaff-Schwartz** (Bermudan with 50 dates, regression estimate): re-run
   with 2x and 4x time steps together with 4x and 16x samples.
   - |d2| <= tol_num + 4 SE2: "Bermudan/LSM discretization, converging";
   - otherwise: "LSM bias, not explained".

Here tol_num is the numerical tolerance at the point. Each disagreement is
also tagged with region labels (not mutually exclusive):

- near expiry: D <= 7;
- near-zero vol: sigma <= 0.01;
- near barrier: abs(H / S - 1) <= 0.01;
- deep in/out of the money: abs(ln(K / F)) / (sigma sqrt(T)) > 3;
- negative rate: r < 0.

Explanations in the documentation (barrier discretization, gamma near
expiry, MC noise, PDE grid alignment with the strike) are stated only with a
refinement or convergence experiment that supports them.

## 12. Random seeds

Base seed 20250115. Each run gets seed = 1 + (first four bytes of SHA-256 of
"base|instrument|point_id|engine|run" read as an unsigned integer) mod
2147483646, where `run` names the sample count, monitoring frequency and
refinement level. Different instruments, sample sizes and refinement levels
therefore use independent random numbers. Seeds do not depend on execution
order or the number of worker processes. QuantLib seed 0 (clock-based) is
never used. `own_mc` uses numpy's PCG64 generator.

## 13. Reproducibility

`python -m pricing_crosscheck run --profile full` regenerates `results/full`
(CSV plus generated markdown summary) and the charts in `docs/`. Every CSV
starts with comment lines holding the content hash (code, specs, lock file,
library versions) and the exact versions. CI regenerates the fast profile on
every pull request and compares it with the committed files; the full grid
runs on manual dispatch and weekly.

## Deviations from protocol

None yet.
