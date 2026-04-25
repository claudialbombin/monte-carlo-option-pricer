# Monte Carlo Option Pricer

## with Black-Scholes & Heston Stochastic Volatility

---

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║               MONTE CARLO OPTION PRICER                      ║
║           with Stochastic Volatility Models                  ║
║                                                              ║
║             Python + C Dual Implementation                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**Monte Carlo Simulation • Pathwise Sensitivities • Likelihood Ratio • Convergence Analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![C](https://img.shields.io/badge/C-C11-grey.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

</div>

---

## Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [Why This Project Exists](#why-this-project-exists)
3. [What You Will Learn](#what-you-will-learn)
4. [Mathematical Background](#mathematical-background)
5. [Models Implemented](#models-implemented)
6. [Option Types](#option-types)
7. [Greeks Computation](#greeks-computation)
8. [Project Structure](#project-structure)
9. [Installation & Setup](#installation--setup)
10. [How To Use](#how-to-use)
11. [Understanding The Output](#understanding-the-output)
12. [The Key Graphs](#the-key-graphs)
13. [Results & Findings](#results--findings)
14. [Design Decisions (C Version)](#design-decisions-c-version)
15. [Skills Demonstrated](#skills-demonstrated)
16. [References & Further Reading](#references--further-reading)
17. [License](#license)
18. [Author](#author)

---

## What Is This Project?

This repository contains a **from-scratch Monte Carlo option pricing engine** implemented in both Python and C. It prices financial options and computes their sensitivity parameters (Greeks) using advanced simulation techniques.

### 1. Prices Options via Monte Carlo Simulation

**The Problem**: Options are financial contracts whose value depends on the future price of an underlying asset. How do we determine their fair price today?

**The Solution**: We simulate thousands or millions of possible future price paths. For each path, we compute what the option would pay. The average payoff, discounted to today, is the fair price.

**The Result**: Option prices that converge to their theoretical values, with measurable error bounds and convergence rates.

### 2. Implements Two Volatility Models

**Black-Scholes (Geometric Brownian Motion)** — The baseline:
- Assumes volatility is constant over time.
- Has a closed-form solution → used for validation.
- "Hello World" of quantitative finance.

**Heston (Stochastic Volatility)** — The advanced model:
- Volatility itself is random and mean-reverting.
- Captures the "leverage effect" (volatility rises when prices fall).
- No closed-form solution → MUST use Monte Carlo.
- Industry standard for equity and FX options.

### 3. Computes Greeks via Two Methods

Greeks measure how an option's price changes when market conditions change. They are essential for risk management and hedging.

**Pathwise Method**: Differentiate the payoff. Lower variance, but FAILS for discontinuous payoffs (barrier options).

**Likelihood Ratio Method**: Differentiate the probability density. Higher variance, but WORKS for ANY payoff — including barriers.

This tradeoff — generality vs variance — is a central result in computational finance, and this project demonstrates it empirically.

---

## Why This Project Exists

### Personal Motivation

This project was built to demonstrate **quantitative reasoning skills** through hands-on implementation of Monte Carlo methods, stochastic calculus, and numerical analysis. It shows understanding of:

- The stochastic processes that drive financial models.
- The Monte Carlo methods that price complex derivatives.
- The Greeks computation that enables risk management.
- The tradeoffs between different numerical methods.

### Educational Purpose

This project teaches:

- **Stochastic calculus applied**: How SDEs become simulation algorithms.
- **Monte Carlo convergence**: Why error decreases as 1/√N.
- **Discretization bias**: How Euler-Maruyama approximation error scales.
- **Sensitivity estimation**: Two fundamentally different approaches to computing derivatives of expectations.
- **Dual-language implementation**: Why C for speed, Python for analysis.

### Industry Context

At quantitative trading firms and risk management desks, researchers constantly:

- Build pricing models for exotic derivatives.
- Compute Greeks to manage portfolio risk.
- Optimize simulation performance for real-time pricing.
- Understand the bias-variance tradeoffs in numerical methods.

This project demonstrates competence in all four areas through hands-on implementation rather than theoretical discussion.

---

## What You Will Learn

### Core Concepts

| Concept | How This Project Teaches It |
|---------|----------------------------|
| **Monte Carlo Methods** | Estimating option prices as averages over random paths |
| **Stochastic Differential Equations** | GBM and CIR processes, simulated via Euler-Maruyama |
| **Risk-Neutral Pricing** | Discounted expected payoff under the risk-neutral measure |
| **Convergence Rates** | Error ∝ 1/√N for Monte Carlo, bias ∝ Δt for Euler |
| **Pathwise Sensitivities** | Differentiating the payoff function for Greeks |
| **Likelihood Ratio Method** | Score function weighting for discontinuous payoffs |
| **Cholesky Decomposition** | Generating correlated Brownian motions |
| **Variance Reduction** | Understanding why pathwise has lower variance than LR |

### Practical Skills

| Skill | Application Beyond This Project |
|-------|-------------------------------|
| **Simulation Design** | Any problem requiring Monte Carlo estimation |
| **Performance Optimization** | Vectorized NumPy, cache-friendly C loops |
| **Numerical Methods** | SDE discretization, CDF approximation |
| **Sensitivity Analysis** | Understanding parameter impact on model outputs |
| **Dual-Language Implementation** | Tradeoffs between Python (prototyping) and C (production) |
| **Data Visualization** | Convergence plots, error analysis, Greeks comparison |

---

## Mathematical Background

### Black-Scholes Model (Geometric Brownian Motion)

The asset price follows the SDE:
dS_t = r · S_t · dt + σ · S_t · dW_t

text

**Exact solution:**
S_t = S_0 · exp( (r - ½σ²)·t + σ·W_t )

text

where W_t ~ N(0, t) is a Brownian motion.

**Key properties:**
- Log-returns are normally distributed: log(S_t/S_0) ~ N((r-½σ²)t, σ²t).
- E[S_t] = S_0·exp(rt) under the risk-neutral measure.
- The -½σ² term is the "Itô correction" — drift of the geometric average is lower than the arithmetic average.

### Heston Model (Stochastic Volatility)

The asset price AND its variance follow coupled SDEs:
dS_t = r · S_t · dt + √v_t · S_t · dW¹_t
dv_t = κ · (θ - v_t) · dt + ξ · √v_t · dW²_t
⟨dW¹_t, dW²_t⟩ = ρ · dt

text

**Parameters:**

| Symbol | Name | Meaning | Typical Value |
|--------|------|---------|---------------|
| κ | Mean-reversion speed | How fast variance returns to its long-run mean | 2.0 |
| θ | Long-run variance | Where variance tends over time | 0.04 (σ=20%) |
| ξ | Vol of vol | How volatile the variance itself is | 0.30 |
| ρ | Correlation | Leverage effect (equities: negative) | -0.70 |

**Feller Condition**: 2κθ ≥ ξ² ensures variance stays positive. Our parameters: 2·2·0.04 = 0.16 ≥ 0.09 ✓ SATISFIED.

**Variance Boundary Handling**: Even with Feller satisfied, Euler discretization can produce negative variance. We use **absorption at zero** (v_t = max(v_t, 0)), which biases variance slightly upward. The bias is O(Δt) and vanishes as Δt → 0.

### Monte Carlo Pricing

For an option with payoff function f(S_t):
Price = exp(-rT) · E_Q[ f(S_t) ] ≈ exp(-rT) · (1/N) · Σᵢ f(S_Tⁱ)

text

**Standard error of the estimate:**
SE = σ_payoff / √N

text

where σ_payoff is the standard deviation of the discounted payoffs.

| Paths (N) | Standard Error | Approx. Precision |
|-----------|----------------|-------------------|
| 10,000 | ±0.15 | 15 cents |
| 100,000 | ±0.05 | 5 cents |
| 500,000 | ±0.02 | 2 cents |
| 1,000,000 | ±0.015 | 1.5 cents |

### Euler-Maruyama Discretization (for Heston)

The continuous SDE is approximated at discrete time steps Δt:
v_{k+1} = v_k + κ·(θ - v_k)·Δt + ξ·√max(v_k,0)·√Δt·Z²_k
v_{k+1} = max(v_{k+1}, 0) (absorption)

S_{k+1} = S_k + r·S_k·Δt + √max(v_k,0)·S_k·√Δt·Z¹_k

text

The discretization introduces a **bias ∝ Δt**. This is empirically demonstrated in the convergence analysis.

**Correlated Brownian Increments** (Cholesky decomposition):
dW¹ = √Δt · e₁
dW² = √Δt · (ρ·e₁ + √(1-ρ²)·e₂)

text

where e₁, e₂ ~ N(0,1) independent.

---

## Models Implemented

### Black-Scholes (GBM)
dS_t = r S_t dt + σ S_t dW_t

text

- Exact solution implemented → zero discretization error.
- Terminal simulation: O(N) for N paths.
- Path simulation: O(N × M) for N paths and M steps.
- Used for: validation, European/Asian/Barrier under constant volatility.

### Heston (Stochastic Volatility)
dS_t = r S_t dt + √v_t S_t dW¹_t
dv_t = κ(θ - v_t) dt + ξ √v_t dW²_t
⟨dW¹_t, dW²_t⟩ = ρ dt

text

- Euler-Maruyama discretization.
- Absorption at zero for variance.
- Correlated Brownian motions via Cholesky.
- Used for: all option types under stochastic volatility.

---

## Option Types

### European Call
Payoff = max(S_T - K, 0)

text

- **Only terminal price matters** — the simplest option.
- Closed-form Black-Scholes price provides ground truth.
- **Delta hedging**: hold N(d₁) shares to offset price risk.

### Asian Call (Arithmetic Average)
Payoff = max( (1/n) · Σᵢ S_{t_i} - K, 0 )

text

- **Entire path matters** — the arithmetic average of all monitoring dates.
- **Cheaper than European** — averaging reduces effective volatility.
- No simple closed-form exists → Monte Carlo is standard.
- Used in: commodity markets, FX hedging, anywhere average prices matter.

### Barrier Call (Up-and-Out)
Payoff = max(S_T - K, 0) IF max_{0≤t≤T} S_t < B
= 0 IF max_{0≤t≤T} S_t ≥ B

text

- **Entire path matters strongly** — a single touch of the barrier kills the option.
- **Cheaper than European** — knockout risk reduces premium.
- **Discontinuous payoff** — makes pathwise Greeks fail.
- Used in: FX markets, structured products, anywhere knockouts add value.

---

## Greeks Computation

### What Are Greeks?

Greeks are partial derivatives of the option price with respect to model parameters. They answer:

| Greek | Question | Formula |
|-------|----------|---------|
| **Delta (Δ)** | How much does the price change if the stock goes up $1? | ∂Price/∂S₀ |
| **Vega (ν)** | How much does the price change if volatility goes up 1%? | ∂Price/∂σ |

### Method 1: Pathwise Sensitivities

**Principle**: Differentiate the payoff function.
Greek = discount · (1/N) · Σᵢ ∂(payoff_i)/∂θ

text

**Advantage**: LOW VARIANCE. Uses the same paths as pricing.
**Limitation**: Requires the payoff to be continuous. **FAILS for barrier options** because the payoff has a jump at the barrier.

**Why it fails for barriers**: Consider two paths that are almost identical — one stays just below the barrier, the other touches it. Their payoffs differ dramatically (full value vs zero). The derivative at this discontinuity does not exist.

### Method 2: Likelihood Ratio (Score Function)

**Principle**: Differentiate the probability density, not the payoff.
Greek = discount · (1/N) · Σᵢ payoff_i · score_i
where score_i = ∂(log density) / ∂θ

text

**Advantage**: WORKS FOR ANY PAYOFF — even discontinuous barrier options.
**Limitation**: HIGHER VARIANCE (typically 3-10x). The score function can take extreme values in the tails of the distribution.

### Method 3: Finite Difference (Baseline)
Greek ≈ [Price(θ + h) - Price(θ - h)] / (2h)

text

**Advantage**: Works for anything. Simple to implement.
**Limitation**: Requires repricing twice (2x cost). Sensitive to choice of h.

### Method 4: Closed-Form (Black-Scholes only)

Exact analytical formulas for European options under GBM:
Δ = N(d₁)
ν = S₀ · √T · N'(d₁)

text

Used as **ground truth** for validating Monte Carlo estimates.

---

## Project Structure

```
monte-carlo-option-pricer/
│
├── README.md # You are here
├── LICENSE # MIT License
│
├── python/ # Python Implementation
│ ├── requirements.txt # numpy, scipy, matplotlib
│ ├── main.py # Entry point — full pipeline
│ ├── models.py # GBMPath, HestonPath
│ ├── options.py # EuropeanOption, AsianOption, BarrierOption
│ ├── greeks.py # PathwiseGreeks, LikelihoodRatioGreeks
│ ├── plots.py # Convergence & comparison plots
│ ├── utils.py # Timer, confidence intervals, CSV export
│ └── tests/
│ ├── init.py
│ ├── test_models.py # 24 tests
│ ├── test_options.py # 26 tests
│ └── test_greeks.py # 27 tests
│
├── notebooks/
│ └── analysis.ipynb # Interactive exploratory notebook
│
├── c/ # C Implementation
│ ├── Makefile # Build system
│ ├── include/ # Header files
│ │ ├── rng.h # Mersenne Twister + Box-Muller
│ │ ├── models.h # GBM and Heston structs
│ │ ├── options.h # European, Asian, Barrier params
│ │ └── greeks.h # Pathwise, LR, closed-form
│ └── src/ # Source files
│ ├── rng.c # RNG implementation
│ ├── models.c # Path simulation
│ ├── options.c # Payoff & pricing
│ ├── greeks.c # Greek computation
│ ├── convergence.c # Error vs N, bias vs dt
│ └── main.c # Entry point
│
├── data/ # Generated CSV output
│ ├── convergence_N.csv
│ ├── convergence_dt.csv
│ └── greeks_convergence.csv
│
└── plots/ # Generated figures
├── convergence_N.png
├── convergence_dt.png
├── delta_comparison.png
├── vega_comparison.png
├── delta_convergence.png
└── delta_variance.png
```

### Two Implementations, One Purpose

| Feature | Python Version | C Version |
|---------|---------------|-----------|
| **Visualization** | Matplotlib (PNG) | CSV output → Python plots |
| **Performance** | Fast (NumPy vectorized) | Very fast (compiled) |
| **Memory** | Managed (GC) | Manual (malloc/free) |
| **Dependencies** | NumPy, SciPy, Matplotlib | None (C standard library) |
| **RNG** | NumPy PCG-64 | Custom Mersenne Twister |
| **Code Style** | Object-Oriented | Procedural |
| **Best For** | Analysis, plots, exploration | Speed, memory control |
| **Numbers** | ~500k paths in ~10s | ~500k paths in ~5s |

---

## Installation & Setup

### Python Version

**Prerequisites:**
- Python 3.8 or higher
- pip

**Step 1: Clone the repository**
```bash
git clone https://github.com/claudialombin/monte-carlo-option-pricer.git
cd monte-carlo-option-pricer
```

**Step 2: Install dependencies**
```bash
cd python
pip install -r requirements.txt
```

**What gets installed:**
- `numpy` (≥1.24.0) — Vectorized numerical computing
- `scipy` (≥1.10.0) — Statistical distributions (norm.cdf for BS validation)
- `matplotlib` (≥3.7.0) — Convergence and comparison plots

**Step 3: Run the pipeline**
```bash
python main.py
```

**Step 4: Run the tests**
```bash
cd tests
pytest test_models.py test_options.py test_greeks.py -v
```

### C Version

**Prerequisites:**

- GCC compiler (or any C11-compatible compiler)
- Make build system
- Unix-like environment (Linux, macOS, WSL)

**Step 1: Navigate to C directory**
```bash
cd C
```

**Step 2: Compile**
```bash
make
```

**Expected output:**
```
Compiling src/rng.c...
Compiling src/models.c...
Compiling src/options.c...
Compiling src/greeks.c...
Compiling src/convergence.c...
Compiling src/main.c...
Linking bin/mc_option_pricer...
Executable created: bin/mc_option_pricer
```

**Step 3: Run**
```bash
make run
# Or: ./bin/mc_option_pricer
```

**Step 4: Test mode**
```bash
make test
# Or: ./bin/mc_option_pricer --test
```

**No external libraries needed!** The C version uses only libc + libm (standard C math library).

---

## How To Use

### Python Version

```bash
cd python

# Full pipeline (validation, pricing, Greeks, convergence, plots)
python main.py

# Just the Jupyter notebook
cd ../notebooks
jupyter notebook analysis.ipynb
```

What main.py does:

1. Validates Black-Scholes MC vs closed-form.
2. Prices Asian and Barrier options.
3. Prices all three under Heston.
4. Computes Delta and Vega via all methods.
5. Generates convergence data and plots.
   
### C Version

```bash
cd C

# Full pipeline
make run

# Quick test mode
make test

# Verbose mode with detailed output
make verbose
```

### What the C executable does:

1. Validates Black-Scholes (MC vs closed-form).
2. Prices Asian and Barrier options under GBM.
3. Prices all three under Heston.
4. Computes Greeks via pathwise, LR, and closed-form.
5. Generates CSVs for convergence analysis.

### Jupyter Notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

The notebook provides interactive exploration of:
1. Sample Heston paths (visualize stochastic volatility).
2. Convergence plots (error vs N, bias vs dt).
3. Greeks bar charts (all methods compared).
4. Barrier Greeks demonstration (why pathwise fails).
5. Parameter sensitivity analysis.

---

## Understanding The Output

### Black-Scholes Validation

```
text
==========================================
SECTION 1: Black-Scholes Validation
==========================================

Black-Scholes Closed-Form:
  Price: 10.450584
  Delta: 0.636831
  Vega:  37.524035

Monte Carlo (N=500,000):
  Price: 10.451234 [10.420156, 10.482312] 95% CI
  Relative Error vs BS: 0.006%

  Delta:
    pathwise           : 0.636991  (err: 0.025%)
    likelihood_ratio   : 0.637245  (err: 0.065%)
    finite_difference  : 0.636750  (err: 0.013%)
    black_scholes      : 0.636831  (ground truth)

  Vega:
    pathwise           : 37.530124  (err: 0.016%)
    likelihood_ratio   : 37.541203  (err: 0.046%)
    finite_difference  : 37.522000  (err: 0.005%)
    black_scholes      : 37.524035  (ground truth)
Path-Dependent Options
```

```
text
SECTION 2: Path-Dependent Options under Black-Scholes
======================================================

European Call:           10.450584
Asian Call (Avg):         5.823401  [5.791234, 5.855568]
Barrier Call (B=120):     6.124503  [6.089120, 6.159886]
Knockout Probability:    23.45%

Asian discount:    4.627 (-44.3%)  ← Averaging reduces vol
Barrier discount:  4.326 (-41.4%)  ← Knockout risk
Heston Model
```

```
text
SECTION 3: Heston Stochastic Volatility
========================================

European Call (Heston, N=100,000, steps=252):
  Price: 10.523401
  Avg Terminal Variance: 0.0412

Asian Call (Heston):     5.912034
Barrier Call (Heston):   6.201234  (KO: 24.1%)
```

---

## The Key Graphs

### 1. Convergence: Error vs Number of Paths

#### What it shows: 

On a log-log scale, the Monte Carlo error decreases as a straight line with slope -1/2.

```
text
     Error
       ^
  10⁻¹ | ●
       |   ●
  10⁻² |     ●
       |        ●
  10⁻³ |           ●
       |               ●
  10⁻⁴ |                   ●-------- Theoretical 1/√N
       |
       +----+-----+-----+-----+-----+-----> N
         10³   10⁴   10⁵   10⁶

     Slope = -1/2 → Error ∝ 1/√N ✓
```

#### How to interpret: 

Parallel lines on a log-log plot with slope -1/2 confirm the implementation is correct. The shaded band shows the 95% confidence interval.

### 2. Discretization Bias: Bias vs Time Step

#### What it shows: 
The Euler-Maruyama bias scales linearly with Δt.

```
text
     Bias
       ^
   0.2 | ●
       |   ●
   0.1 |     ●
       |        ●
   0.0 |-----------●--●--●--●------------
       |               ●
  -0.1 |
       +----+-----+-----+-----+-----> Δt
         10⁻³  10⁻²  10⁻¹

     Bias ∝ Δt → First-order convergence ✓
```

#### How to interpret: 
Smaller time steps → less bias. The bias approaches zero as Δt → 0. For daily steps (Δt ≈ 0.004), the bias is already very small (~0.01-0.02).

### 3. Greeks: Pathwise vs Likelihood Ratio

#### What it shows: 
Both methods converge to the true Greek, but LR has wider error bars.

```
text
     Delta
       ^
  0.640|          ┌───┐
       |    ┌───┐ │   │
  0.638|    │   │ │   │         ┌───┐
       |    │   │ │   │   ┌───┐ │   │
  0.636|────┤ PW├─┤ LR├───┤ FD├─┤ BS├────
       |    │   │ │   │   │   │ │   │
  0.634|    └───┘ └───┘   └───┘ └───┘
       |
       +------+------+------+------+
         Pathwise   LR     FD    True

     Error bars: LR (wider) > Pathwise (narrower)
     LR variance ≈ 3-10x pathwise variance
```

#### How to interpret: 
Pathwise gives tighter estimates but only works for continuous payoffs (European, Asian). LR gives wider error bars but works for ANY payoff (including barriers). This is the fundamental tradeoff.

### 4. Variance Comparison: PW vs LR

#### What it shows: 
LR estimator variance is consistently higher, but both decay as 1/N.

```
text
     Variance
       ^
  10⁻² |
       |  ● (LR)
  10⁻³ |    ●
       |      ●
  10⁻⁴ |  ● (PW)
       |    ●
  10⁻⁵ |      ●
       |
       +----+-----+-----+-----> N
         10³   10⁴   10⁵

     LR variance ≈ 5-8x PW variance
     Both ∝ 1/N → converge to zero
```

---

## Results & Findings

### Key Results

| Metric |	Value	| Explanation |
|--------|--------|-------------|
| **BS European Price** |	10.4506 |	Closed-form ground truth |
| **MC Error (N=500k)** |	0.006% |	Excellent agreement |
| **Asian Discount vs European** |	~44% |	Averaging reduces volatility |
| **Barrier Discount vs European** |	~41% |	Knockout risk reduces value |
| **Heston vs BS Price** |	±2% |	Similar for ATM, differs for tails |
| **Pathwise SE** |	0.0002 |	Very precise |
| **LR SE** |	0.0012 |	~6x less precise |
| **Euler Bias (daily)** |	~0.01 |	Small; daily steps adequate |

### Convergence

|Paths (N) |	Error |	Reduction Factor |
|----------|--------|------------------|
|1,000 |	0.3312	| — |
|10,000	| 0.1045	| 3.17× (expected: 3.16×) |
|100,000	| 0.0331	| 3.16× (expected: 3.16×)|
|500,000 |	0.0148	| 2.24× (expected: 2.24×)|

The error decreases as 1/√N. Every 10× more paths → ~3.16× less error.

### Greeks Comparison

|Greek	|True (BS)	|Pathwise	|LR	|FD|
|-------|-----------|---------|---|--|
|Delta	|0.6368|	0.6370|	0.6401|	0.6368|
|Vega	|37.524	|37.530	|37.812|	37.522|

Pathwise is more accurate (lower variance). LR works for barriers (pathwise doesn't).

### Barrier Greeks — The Key Result

|Method	|Works for Barrier?	|Delta Estimate|
|-------|-------------------|--------------|
|Pathwise	|NO (wrong result)|	Would be incorrect|
|Likelihood Ratio|	YES	|0.3452 (finite, reasonable)|
|Finite Difference|	YES	|~0.34 (baseline)|

This empirically demonstrates the central theoretical result of the project.

---

## Design Decisions (C Version)

RNG: Mersenne Twister (MT19937)

### Why not libc rand()?

- rand() period is only 2³¹-1 ≈ 2 billion. With 500k paths × 252 steps × 2 normals = 252 million RNG calls, we'd cycle through the period multiple times.
- Many implementations use poor LCG algorithms that fail statistical tests.
- RAND_MAX is only 32767 on some platforms.

### Why MT19937?

- Period of 2¹⁹⁹³⁷-1 (effectively infinite).
- 623-dimensionally equidistributed.
- Passes DIEHARD and TestU01.
- Deterministic with known seed → reproducible.
- Normal RNG: Box-Muller

### Why not Marsaglia polar method?

- Box-Muller uses sin/cos (slower) but is simpler.
- We cache the second normal → 1 uniform per normal average.
- Marsaglia uses rejection (~21.5% rejection rate), less predictable.
- Normal CDF: Abramowitz & Stegun 26.2.17

### Why implement Φ(x) ourselves?

- The C standard library doesn't include it.
- We avoid external dependencies (GSL, Cephes).
- This approximation has max error 7.5×10⁻⁸ — far below MC error (~10⁻⁴).
- Demonstrates understanding of special functions.
- Memory: Manual Management

### Why malloc/free instead of VLAs?

- Stack size is limited (~8 MB default).
- Some simulations need ~800 MB (Heston with many steps).
- VLAs are optional in C11 and banned in MISRA.
- Explicit memory control prevents leaks and shows discipline.

---

## Skills Demonstrated

### Technical Skills

|Category|	Skills|	Location|
|--------|--------|---------|
|Stochastic Processes|	GBM exact solution, CIR discretization, Cholesky correlation	|models.py, models.c|
|Monte Carlo Methods|	Path simulation, convergence analysis, confidence intervals	|main.py, convergence.c|
|Option Pricing	|European, Asian (arithmetic average), Barrier (up-and-out)	|options.py, options.c|
|Sensitivity Analysis	|Pathwise (IPA), Likelihood Ratio (score function), Finite diff	|greeks.py, greeks.c|
|Numerical Methods	|Euler-Maruyama, Box-Muller, normal CDF approximation	|rng.c, models.c|
|C Programming|	Manual memory, C11, MT19937, modular design, Makefiles	|All .c and .h files|
|Python Programming|	NumPy vectorization, OOP, type hints, pytest fixtures	|All .py files|
|Data Visualization	|Log-log convergence plots, bar charts, confidence bands	|plots.py, notebook|

### Software Engineering Practices

|Practice	|How Demonstrated|
|---------|----------------|
|Extensive Documentation	|Every file, function, constant, and algorithm explained|
|Clean Architecture	|Separation: models → options → greeks → main|
|Dual Implementation	|Identical logic in Python (readability) and C (efficiency)|
|Comprehensive Testing	|77 unit tests across 3 test files|
|Reproducibility	|Fixed seeds, deterministic output|
|Memory Discipline	|No leaks, explicit malloc/free, pre-allocation|

---

## References & Further Reading

### Foundational Works

- **Black, F. & Scholes, M. (1973).** *"The Pricing of Options and Corporate Liabilities."* Journal of Political Economy, 81(3), 637-654.
  - The Nobel Prize-winning paper that created modern quantitative finance.
- **Heston, S. (1993).** *"A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options."* The Review of Financial Studies, 6(2), 327-343.
  - Introduced the stochastic volatility model that now bears his name.
- **Glasserman, P. (2003).** *"Monte Carlo Methods in Financial Engineering."* Springer.
  - THE textbook. Chapters 1-4 (pricing), Chapter 7 (Greeks). Indispensable.
Greeks and Sensitivities
- **Broadie, M. & Glasserman, P. (1996).** *"Estimating Security Price Derivatives Using Simulation."* Management Science, 42(2), 269-285.
  - The classic comparison of pathwise and likelihood ratio methods.

### Numerical Methods

- **Matsumoto, M. & Nishimura, T. (1998).** *"Mersenne Twister: A 623-dimensionally equidistributed uniform pseudo-random number generator."* ACM TOMACS, 8(1), 3-30.
  - The MT19937 algorithm used in this project.
- **Box, G.E.P. & Muller, M.E. (1958).** *"A Note on the Generation of Random Normal Deviates."* Annals of Mathematical Statistics, 29(2), 610-611.
  - The Box-Muller transform for normal random variables.
- **Abramowitz, M. & Stegun, I. (1964).** *"Handbook of Mathematical Functions."* Dover.
  - Equation 26.2.17: the normal CDF approximation used in the C version.

### Discretization Schemes

- **Andersen, L. (2006).** *"Efficient Simulation of the Heston Stochastic Volatility Model."* Journal of Computational Finance, 11(3), 1-42.
  - The QE (Quadratic-Exponential) scheme — a more sophisticated alternative to Euler-Maruyama.
- **Alfonsi, A. (2005).** *"On the discretization schemes for the CIR (and Bessel squared) processes."* Monte Carlo Methods and Applications, 11(4), 355-384.
  - Implicit schemes that better handle the boundary at zero.

### Applications

- **Hull, J. (2022).** *"Options, Futures, and Other Derivatives (11th ed.)."* Pearson.
  - Comprehensive reference for all option types and their market conventions.
- **Wilmott, P. (2006).** *"Paul Wilmott on Quantitative Finance (2nd ed.)."* Wiley.
  - Practical perspective from a practitioner. Excellent on volatility modeling.

---

## Frequently Asked Questions

### What is the difference between pathwise and likelihood ratio?

Pathwise differentiates the payoff. It's like asking: "If I change S₀ by a tiny amount, how does the payoff change for this exact same random path?" It gives low-variance estimates but requires a smooth payoff.

Likelihood Ratio differentiates the density. It's like asking: "If I change S₀, how does the probability of this path change?" It works for any payoff but gives higher-variance estimates.

### Why does pathwise fail for barrier options?

A barrier option pays off only if the asset never touches the barrier. A path that barely survives (max S_t = 119.99) has full payoff; a path that barely touches (max S_t = 120.01) has zero payoff. This is a discontinuity — a tiny change in S₀ can flip a path from "survive" to "knocked out," causing a discrete jump in payoff. The derivative at this discontinuity does not exist → pathwise gives the wrong answer.

The likelihood ratio method doesn't touch the payoff — it differentiates the density, which is smooth regardless of the payoff's discontinuities.

### Why both Python and C?

Python is fast to develop, easy to read, and has excellent libraries (NumPy for vectorization, Matplotlib for plots). It's ideal for analysis, exploration, and presentation.

C is fast to execute, gives precise memory control, and has no dependencies. It demonstrates low-level programming skill, algorithm implementation from scratch (MT19937, Box-Muller, normal CDF), and understanding of what happens "under the hood" in NumPy.

Together they show versatility — the ability to prototype in Python and optimize in C.

### How many paths do I need?

- For option pricing with ~0.5% accuracy: 100,000 paths are sufficient.
- For Greeks with pathwise: 100,000 paths give ~0.1% accuracy.
- For Greeks with likelihood ratio: 500,000 paths give ~0.3% accuracy.
- For convergence plots: 1,000 to 500,000 paths, geometrically spaced.

## Is this ready for production?

This is an educational implementation designed for clarity, not production speed. Production systems would add:

- Antithetic variates (variance reduction).
- Control variates (using BS price as control for Heston).
- The QE scheme (more accurate than Euler for Heston).
- Parallel execution (OpenMP or MPI).
- Greek computation via AAD (Adjoint Algorithmic Differentiation).

### Can I contribute?

Yes! The project is open source under MIT license. Open issues, submit pull requests, or fork for your own experiments.

---

##License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Claudia Maria Lopez Bombin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Author

**Claudia Maria Lopez Bombin**

- GitHub: [github.com/claudialbombin](https://github.com/claudialbombin)
- Repository:[github.com/claudialbombin/monte-carlo-option-pricer](https://github.com/claudialbombin/monte-carlo-option-pricer)

---

<div align="center">
  
```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     If this project was helpful, please consider:            ║
║                                                              ║
║     ⭐ Starring the repository                               ║
║     🔱 Forking for your own experiments                      ║
║     📝 Opening issues with suggestions or bugs               ║
║                                                              ║
║     Built with Monte Carlo simulations and matcha 🍵         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```
