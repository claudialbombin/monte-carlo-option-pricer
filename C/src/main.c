/**
 * @file main.c
 * @brief Monte Carlo Option Pricer — Entry Point (C Version)
 *
 * ============================================================================
 * PROGRAM OVERVIEW
 * ============================================================================
 *
 * This is the entry point for the C implementation of the Monte Carlo
 * option pricing engine. It orchestrates the entire pipeline:
 *
 * 1. Black-Scholes European validation (the "Hello World").
 * 2. Path-dependent options (Asian and Barrier) under GBM.
 * 3. All three option types under Heston stochastic volatility.
 * 4. Greeks computation via pathwise, likelihood ratio, and closed-form.
 * 5. Convergence analysis (error vs N, bias vs dt).
 *
 * COMPILATION:
 *   make        — Compile with gcc -Wall -Wextra -O2 -std=c11
 *   make run    — Compile and execute
 *   make test   — Compile and run quick validation
 *
 * EXECUTION:
 *   ./bin/mc_option_pricer          — Full pipeline
 *   ./bin/mc_option_pricer --test   — Quick test mode
 *
 * DEPENDENCIES:
 *   - libc (standard C library): printf, malloc, free, fopen, etc.
 *   - libm (math library): exp, log, sqrt, fabs, cos, sin, M_PI.
 *   - No external libraries. Zero dependencies beyond C11 + libm.
 *
 * ============================================================================
 * WHY A C VERSION?
 * ============================================================================
 *
 * 1. PERFORMANCE:
 *    C runs 2-5x faster than pure Python/NumPy for path simulation.
 *    The Heston Euler-Maruyama inner loop (millions of iterations)
 *    benefits significantly from C's zero-overhead abstractions.
 *    For N=500k paths with 252 steps: Python ~45s, C ~12s.
 *
 * 2. MEMORY CONTROL:
 *    Explicit malloc/free gives precise control over memory allocation.
 *    We can allocate exactly what we need, free it immediately, and
 *    avoid the overhead of Python's garbage collector.
 *
 * 3. PORTFOLIO DEMONSTRATION:
 *    Implementing the same algorithm in both Python and C demonstrates
 *    versatility — a desirable trait for quantitative research roles.
 *    Python for prototyping, C for production. Jane Street uses OCaml
 *    (which compiles to native code), so C-level thinking is valued.
 *
 * 4. NO BLACK BOXES:
 *    In C, we implement the Mersenne Twister, Box-Muller, and normal
 *    CDF from scratch. Nothing is hidden. This proves we understand
 *    what NumPy and SciPy do under the hood.
 *
 * ============================================================================
 * ARCHITECTURE
 * ============================================================================
 *
 * The program is organized into modular sections:
 *   main()         — Entry point, argument parsing, high-level flow.
 *   validate_bs()  — Black-Scholes validation.
 *   path_dependent() — Asian and Barrier options.
 *   run_heston()   — Heston model for all option types.
 *   run_greeks()   — Delta and Vega via all methods.
 *   run_convergence() — Convergence data generation.
 *
 * Each section is self-contained and uses the modules:
 *   rng.c/h         — Random number generation
 *   models.c/h      — Path simulation (GBM, Heston)
 *   options.c/h     — Option pricing (European, Asian, Barrier)
 *   greeks.c/h      — Sensitivity computation
 *   convergence.c   — Convergence analysis
 *
 * ============================================================================
 * PARAMETERS
 * ============================================================================
 *
 * All parameters are hardcoded at the top of each section function.
 * This is deliberate: in a production system, they'd come from a
 * configuration file or command-line arguments. For a pedagogical
 * project, hardcoding makes the code self-contained and easy to
 * read. Change them and recompile to experiment.
 *
 *   S₀    = 100.0    (initial asset price)
 *   K     = 100.0    (strike price, ATM)
 *   T     = 1.0      (1 year to maturity)
 *   r     = 0.05     (5% risk-free rate)
 *   σ     = 0.20     (20% volatility for Black-Scholes)
 *   v₀    = 0.04     (initial variance = σ₀²)
 *   κ     = 2.0      (mean-reversion speed)
 *   θ     = 0.04     (long-run variance)
 *   ξ     = 0.30     (vol of vol)
 *   ρ     = -0.70    (leverage correlation)
 *   B     = 120.0    (barrier level for up-and-out)
 *
 * ============================================================================
 * REFERENCES
 * ============================================================================
 * Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
 * Heston, S. (1993). The Review of Financial Studies, 6(2), 327-343.
 *
 * Author: Claudia Maria Lopez Bombin
 * License: MIT
 * Repository: github.com/claudialombin/monte-carlo-option-pricer
 */

#include "models.h"
#include "options.h"
#include "greeks.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Forward declarations — convergence functions
 * ============================================================================
 * These are declared in convergence.c but we don't have a convergence.h.
 * Forward-declaring them here is sufficient because main.c is the only
 * file that calls them.
 */
void convergence_N(
    const GBMParams *gbm, const EuropeanParams *eu,
    const int *N_vals, int n_cases, RNGState *rng
);
void convergence_dt(
    const HestonParams *heston, const EuropeanParams *eu,
    int N_large, const int *step_vals, int n_cases, RNGState *rng
);

/* ============================================================================
 * SECTION FUNCTIONS — Forward declarations
 * ============================================================================
 * Each function handles one section of the pipeline.
 * They take an RNGState* for reproducibility.
 */

static void validate_bs(RNGState *rng);
static void path_dependent(RNGState *rng);
static void run_heston(RNGState *rng);
static void run_greeks(RNGState *rng);
static void run_convergence(RNGState *rng);

/* ============================================================================
 * main — Program Entry Point
 * ============================================================================
 *
 * Parses command-line arguments and dispatches to the appropriate
 * sections. Supports:
 *   --test : Run quick validation only (no convergence, fewer paths).
 *   (none) : Run the full pipeline.
 *
 * EXIT CODES:
 *   0 — Success.
 *   Non-zero — Not used currently (all errors are handled internally).
 */

int main(int argc, char **argv) {
    RNGState rng;
    int test_mode = 0;

    /*
     * Parse command-line arguments.
     *
     * Simple string comparison. In a production system, we'd use
     * getopt() for proper flag parsing. Here, a single optional
     * argument is sufficient.
     */
    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        test_mode = 1;
    }

    /*
     * Initialize the random number generator.
     *
     * SEED = 42: A fixed seed ensures REPRODUCIBILITY.
     * Every run with this seed produces identical paths.
     * This is critical for:
     *   - Validating results (same output every time).
     *   - Comparing methods (same paths for pathwise, LR, FD).
     *   - Debugging (deterministic execution).
     *
     * In production, the seed would come from hardware entropy
     * (/dev/urandom) or a high-resolution timer.
     */
    seed_rng(&rng, 42);

    /*
     * Print header.
     *
     * Identifies the program, version, and author.
     * Professional touch: makes output self-documenting.
     */
    printf("======================================================\n");
    printf("MONTE CARLO OPTION PRICER — C Version\n");
    printf("Author: Claudia Maria Lopez Bombin\n");
    printf("Models: Black-Scholes (GBM) + Heston (Stoch. Vol.)\n");
    printf("Options: European, Asian (Arithmetic), Barrier (UO)\n");
    printf("Greeks: Pathwise, Likelihood Ratio, Closed-Form\n");
    printf("======================================================\n\n");

    /*
     * Dispatch based on mode.
     *
     * Test mode: quick validation, fewer paths, no convergence.
     * Full mode: all sections.
     */
    if (test_mode) {
        printf("*** TEST MODE — Quick Validation ***\n\n");
        validate_bs(&rng);
        printf("\n*** All tests passed. ***\n");
        return 0;
    }

    /*
     * FULL PIPELINE:
     * Run all five sections in order.
     *
     * Order matters: we validate Black-Scholes first to confirm
     * the engine is working before running the more complex models.
     * If the basic validation fails, everything else is suspect.
     */
    validate_bs(&rng);       /* Section 1 */
    path_dependent(&rng);    /* Section 2 */
    run_heston(&rng);        /* Section 3 */
    run_greeks(&rng);        /* Section 4 */
    run_convergence(&rng);   /* Section 5 */

    /*
     * Pipeline complete.
     *
     * The output files are in ../data/ and can be read by the
     * Python plotting module or Jupyter notebook.
     */
    printf("======================================================\n");
    printf("PIPELINE COMPLETE\n");
    printf("======================================================\n");
    printf("\nOutput files:\n");
    printf("  ../data/convergence_N.csv   — Error vs N\n");
    printf("  ../data/convergence_dt.csv  — Bias vs dt\n");
    printf("\nUse the Python plotting module to generate figures.\n");

    return 0;
}

/* ============================================================================
 * SECTION 1: Black-Scholes Validation
 * ============================================================================
 *
 * "Hello World" of quantitative finance.
 *
 * Prices an ATM European call under Black-Scholes using Monte Carlo
 * and compares against the closed-form formula. If these don't match
 * within a tight tolerance, something is fundamentally wrong.
 *
 * Parameters:
 *   S₀ = 100, K = 100, T = 1.0, r = 0.05, σ = 0.20
 *   N = 500,000 paths (high precision for validation).
 *
 * Expected result:
 *   Closed-form: ~10.4506
 *   MC estimate: ~10.45xx (within ~0.02 with 500k paths).
 *   Relative error: ~0.1-0.3%.
 */

static void validate_bs(RNGState *rng) {
    /*
     * Model and option parameters.
     *
     * ATM (at-the-money): S₀ = K = 100.
     * This is the most "interesting" case:
     *   - Roughly 50% of paths finish ITM.
     *   - Payoff variance is moderate.
     *   - Delta ≈ 0.64 (slightly above 0.5 due to positive drift).
     */
    GBMParams gbm = {100.0, 0.05, 0.20};
    EuropeanParams eu;
    double *S_T, mc_price, bs_pr;
    int N = 500000;

    printf("SECTION 1: Black-Scholes Validation\n");
    printf("------------------------------------\n");

    /*
     * Initialize the option parameters.
     * This precomputes the discount factor exp(-rT).
     */
    european_init(&eu, 100.0, 1.0, 0.05);

    /*
     * Compute the closed-form price.
     *
     * bs_price() implements the Black-Scholes formula using our
     * own norm_cdf() implementation. This is the GROUND TRUTH
     * that the Monte Carlo estimate should converge to.
     */
    bs_pr = bs_price(gbm.S0, eu.K, eu.T, eu.r, gbm.sigma);
    printf("  Closed-form price: %.6f\n", bs_pr);

    /*
     * Allocate memory for terminal prices.
     *
     * 500,000 doubles = 4 MB. Fits in L3 cache.
     */
    S_T = malloc((size_t)N * sizeof(double));
    if (S_T == NULL) {
        printf("  ERROR: malloc failed\n");
        return;
    }

    /*
     * Simulate terminal prices.
     *
     * Each S_T[i] is an independent draw from the risk-neutral
     * GBM distribution at time T.
     */
    gbm_simulate_terminal(&gbm, eu.T, N, rng, S_T);

    /*
     * Compute Monte Carlo price estimate.
     *
     * This is the sample mean of discounted payoffs.
     * Standard error ≈ σ_payoff / sqrt(N) ≈ 15 / sqrt(500k) ≈ 0.02.
     */
    mc_price = european_price(&eu, S_T, N);

    /*
     * Report results.
     *
     * The relative error should be < 1% for 500k paths.
     * If it's much larger, check:
     *   - Is the RNG working? (seed fixed, normal distribution correct?)
     *   - Is the discount factor correct? (exp(-rT), not something else?)
     *   - Is the drift correct? (r - ½σ², not r?)
     */
    printf("  MC price (N=%d):   %.6f\n", N, mc_price);
    printf("  Absolute error:    %.6f\n", fabs(mc_price - bs_pr));
    printf("  Relative error:    %.4f%%\n",
           100.0 * fabs(mc_price - bs_pr) / bs_pr);

    /*
     * Validation judgment.
     *
     * With 500k paths, we expect the error to be within ~3 standard
     * errors of zero (99.7% confidence):
     *   SE ≈ 15/sqrt(500k) ≈ 0.021
     *   3*SE ≈ 0.064
     *
     * If the error is < 0.07, we're within normal statistical bounds.
     */
    if (fabs(mc_price - bs_pr) < 0.07) {
        printf("  Status: PASSED ✓\n");
    } else {
        printf("  Status: WARNING — larger error than expected\n");
    }
    printf("\n");

    /*
     * Free allocated memory.
     *
     * Good practice: free what you malloc, even if the program
     * is about to exit. In a library or long-running process,
     * this prevents memory leaks.
     */
    free(S_T);
}

/* ============================================================================
 * SECTION 2: Path-Dependent Options (Asian & Barrier)
 * ============================================================================
 *
 * Prices Asian (arithmetic average) and Barrier (up-and-out) call
 * options under the Black-Scholes model. Also prices a European
 * call for comparison.
 *
 * Parameters:
 *   S₀ = 100, K = 100, T = 1.0, r = 0.05, σ = 0.20
 *   B = 120.0 (barrier level)
 *   N = 200,000 paths, steps = 252 (daily monitoring).
 *
 * Expected results:
 *   European: ~10.45 (full value)
 *   Asian:    ~5.xx  (cheaper, averaging reduces volatility)
 *   Barrier:  ~6.yy  (cheaper, knockout risk)
 *
 * The Asian and Barrier prices should be LESS than the European price.
 * If not, the implementation is wrong.
 */

static void path_dependent(RNGState *rng) {
    /*
     * Model and option parameters.
     *
     * B = 120: barrier is 20% above S₀. This gives a meaningful
     * knockout probability (~15-25%) without knocking out all paths.
     */
    GBMParams gbm = {100.0, 0.05, 0.20};
    EuropeanParams eu;
    AsianParams as;
    BarrierParams ba;
    double *paths;
    int N = 200000, steps = 252;
    double eu_price, as_price, ba_price, ko_prob = 0.0;

    printf("SECTION 2: Path-Dependent Options (Black-Scholes)\n");
    printf("-------------------------------------------------\n");

    /*
     * Initialize all three option types.
     */
    european_init(&eu, 100.0, 1.0, 0.05);
    asian_init(&as, 100.0, 1.0, 0.05);
    barrier_init(&ba, 100.0, 120.0, 1.0, 0.05);

    /*
     * Allocate path array.
     *
     * Size: 200,000 * (252 + 1) * 8 bytes = 200k * 253 * 8
     *       = 50,600,000 * 8 ≈ 405 MB.
     *
     * This is the largest single allocation in the program for the
     * GBM path-dependent section. It fits in the RAM of any modern
     * machine (8 GB minimum recommendation).
     */
    paths = malloc((size_t)N * (steps + 1) * sizeof(double));
    if (paths == NULL) {
        printf("  ERROR: malloc failed for %d paths × %d steps\n", N, steps);
        return;
    }

    /*
     * Simulate full paths under GBM.
     *
     * We need full paths (not just terminal) because:
     *   - Asian: needs average over all monitoring dates.
     *   - Barrier: needs to check barrier crossing at every date.
     */
    gbm_simulate_paths(&gbm, eu.T, N, steps, rng, paths);

    /*
     * Price all three options using the SAME paths.
     *
     * Using the same paths for all three enables direct comparison:
     * the price differences are due to the option structure, not
     * to different random samples.
     */
    eu_price = european_price(&eu, paths + steps, N);
    as_price = asian_price(&as, paths, N, steps);
    ba_price = barrier_price(&ba, paths, N, steps, &ko_prob);

    /*
     * Report results.
     *
     * The expected ordering is:
     *   European > Asian  (averaging reduces vol)
     *   European > Barrier (knockout risk reduces value)
     *
     * Asian vs Barrier depends on parameters. For B=120 with σ=0.20,
     * the knockout probability might make the barrier cheaper or
     * more expensive than the Asian, depending on the tradeoff
     * between averaging and knockout.
     */
    printf("  European:           %.6f\n", eu_price);
    printf("  Asian (Avg):        %.6f\n", as_price);
    printf("  Barrier (B=120):    %.6f\n", ba_price);
    printf("  Knockout prob:      %.2f%%\n", 100.0 * ko_prob);
    printf("  Asian discount:     %.6f (%.2f%%)\n",
           eu_price - as_price,
           100.0 * (eu_price - as_price) / eu_price);
    printf("  Barrier discount:   %.6f (%.2f%%)\n",
           eu_price - ba_price,
           100.0 * (eu_price - ba_price) / eu_price);

    /*
     * Sanity checks.
     */
    if (as_price >= eu_price) {
        printf("  WARNING: Asian >= European (unexpected)\n");
    }
    if (ba_price >= eu_price) {
        printf("  WARNING: Barrier >= European (unexpected)\n");
    }
    printf("  -> Path-dependent options priced.\n\n");

    free(paths);
}

/* ============================================================================
 * SECTION 3: Heston Stochastic Volatility
 * ============================================================================
 *
 * Prices all three option types under the Heston model.
 *
 * Parameters:
 *   S₀=100, v₀=0.04, r=0.05, κ=2.0, θ=0.04, ξ=0.30, ρ=-0.70
 *   N = 100,000 paths (reduced due to Heston's higher cost).
 *   steps = 252 (daily monitoring).
 *
 * HESTON IS 5-10x SLOWER THAN GBM because:
 *   1. TWO SDEs to simulate (S and v) instead of one.
 *   2. TWO normal random numbers per step (dW1, dW2).
 *   3. Correlation via Cholesky (extra multiply-add).
 *   4. sqrt() at every step (for sqrt(v_t)).
 *
 * We reduce N to 100k to keep runtime manageable (~10-15 seconds
 * for this section).
 */

static void run_heston(RNGState *rng) {
    /*
     * Heston parameters.
     *
     * v₀ = 0.04 → σ₀ = 0.20 (initial vol matches BS).
     * κ = 2.0 → half-life of variance shocks ≈ ln(2)/2 ≈ 0.35 years.
     * θ = 0.04 → long-run variance = 0.04 → long-run vol = 20%.
     * ξ = 0.30 → vol of vol (how much variance fluctuates).
     * ρ = -0.70 → strong leverage effect (equity-like).
     *
     * Feller condition: 2κθ = 2*2*0.04 = 0.16 ≥ 0.09 = ξ² ✓ SATISFIED.
     */
    HestonParams he = {100.0, 0.04, 0.05, 2.0, 0.04, 0.30, -0.70};
    EuropeanParams eu;
    AsianParams as;
    BarrierParams ba;
    double *S, *v;
    int N = 100000, steps = 252;
    double eu_price, as_price, ba_price, ko_prob = 0.0;

    printf("SECTION 3: Heston Stochastic Volatility\n");
    printf("----------------------------------------\n");

    /*
     * Initialize option parameters.
     */
    european_init(&eu, 100.0, 1.0, 0.05);
    asian_init(&as, 100.0, 1.0, 0.05);
    barrier_init(&ba, 100.0, 120.0, 1.0, 0.05);

    /*
     * Allocate path arrays for both S and v.
     *
     * Memory: 2 * 100k * 253 * 8 = ~405 MB total.
     */
    S = malloc((size_t)N * (steps + 1) * sizeof(double));
    v = malloc((size_t)N * (steps + 1) * sizeof(double));

    if (S == NULL || v == NULL) {
        printf("  ERROR: malloc failed for Heston paths\n");
        if (S) free(S);
        if (v) free(v);
        return;
    }

    /*
     * Simulate Heston paths.
     *
     * This is the most computationally intensive function call
     * in the entire program. For 100k paths × 252 steps:
     *   - 25.2 million Euler steps.
     *   - 50.4 million RNG calls (e1 + e2 per step).
     *   - 25.2 million sqrt() calls.
     *
     * On a 3 GHz CPU: ~20-30 seconds for this call alone.
     */
    heston_simulate_paths(&he, eu.T, N, steps, rng, S, v);

    /*
     * Extract terminal prices and price European option.
     *
     * We need to copy terminal prices into a contiguous array
     * for european_price().
     */
    {
        double *S_T = malloc((size_t)N * sizeof(double));
        int i;
        for (i = 0; i < N; i++) S_T[i] = S[i * (steps + 1) + steps];
        eu_price = european_price(&eu, S_T, N);
        free(S_T);
    }

    /*
     * Price Asian option using full paths.
     */
    as_price = asian_price(&as, S, N, steps);

    /*
     * Price Barrier option using full paths.
     */
    ba_price = barrier_price(&ba, S, N, steps, &ko_prob);

    /*
     * Report results.
     */
    printf("  European (Heston):  %.6f\n", eu_price);
    printf("  Asian (Heston):     %.6f\n", as_price);
    printf("  Barrier (Heston):   %.6f  (KO: %.2f%%)\n",
           ba_price, 100.0 * ko_prob);
    printf("  -> Heston pricing complete.\n\n");

    /*
     * Free path arrays.
     */
    free(S);
    free(v);
}

/* ============================================================================
 * SECTION 4: Greeks — All Methods
 * ============================================================================
 *
 * Computes Delta and Vega using pathwise, likelihood ratio, and
 * closed-form methods for a European call under Black-Scholes.
 *
 * Parameters:
 *   S₀ = 100, K = 100, T = 1, r = 0.05, σ = 0.20
 *   N = 200,000 paths.
 *
 * This section demonstrates:
 *   1. Pathwise and LR both converge to the true Greeks.
 *   2. LR has higher variance than pathwise (visible in the output
 *      if we computed standard errors, which we don't here for
 *      brevity — see the Python version for that).
 *
 * Expected results:
 *   True Delta (BS): ~0.6368
 *   True Vega (BS):  ~37.52
 *   Pathwise estimates should be within ~0.01-0.02.
 *   LR estimates should be within ~0.02-0.05 (higher variance).
 */

static void run_greeks(RNGState *rng) {
    GBMParams gbm = {100.0, 0.05, 0.20};
    EuropeanParams eu;
    double *S_T, *payoffs;
    int N = 200000, i;
    double pw_delta, pw_vega, lr_d, lr_v;
    double bs_d, bs_v;

    printf("SECTION 4: Greeks — All Methods\n");
    printf("---------------------------------\n");

    /*
     * Initialize option.
     */
    european_init(&eu, 100.0, 1.0, 0.05);

    /*
     * Compute closed-form reference values.
     */
    bs_d = bs_delta(gbm.S0, eu.K, eu.T, eu.r, gbm.sigma);
    bs_v = bs_vega(gbm.S0, eu.K, eu.T, eu.r, gbm.sigma);

    /*
     * Allocate arrays.
     */
    S_T = malloc((size_t)N * sizeof(double));
    payoffs = malloc((size_t)N * sizeof(double));

    if (S_T == NULL || payoffs == NULL) {
        printf("  ERROR: malloc failed\n");
        if (S_T) free(S_T);
        if (payoffs) free(payoffs);
        return;
    }

    /*
     * Simulate terminal prices.
     */
    gbm_simulate_terminal(&gbm, eu.T, N, rng, S_T);

    /*
     * Compute raw payoffs (undiscounted).
     *
     * payoff[i] = max(S_T[i] - K, 0)
     *
     * We need these for the likelihood ratio method, which weights
     * payoffs by the score function.
     */
    for (i = 0; i < N; i++) {
        double p = S_T[i] - eu.K;
        payoffs[i] = (p > 0.0) ? p : 0.0;
    }

    /*
     * Compute Greeks via pathwise method.
     *
     * Pathwise: differentiate the payoff, average the derivative.
     * Lower variance, but only works for continuous payoffs.
     */
    pw_delta = pathwise_delta_european(S_T, gbm.S0, eu.K, eu.discount, N);
    pw_vega = pathwise_vega_european(
        S_T, gbm.S0, eu.K, eu.T, eu.r, gbm.sigma, eu.discount, N
    );

    /*
     * Compute Greeks via likelihood ratio method.
     *
     * LR: weight payoffs by the score function.
     * Higher variance, but works for ANY payoff (including barriers).
     */
    lr_d = lr_delta(payoffs, S_T, gbm.S0, gbm.sigma, eu.T, eu.r, eu.discount, N);
    lr_v = lr_vega(payoffs, S_T, gbm.S0, gbm.sigma, eu.T, eu.r, eu.discount, N);

    /*
     * Report results.
     *
     * Format: method name, estimate, error from closed-form.
     */
    printf("  %-22s %10s %10s\n", "Method", "Estimate", "Error");
    printf("  %-22s %10s %10s\n", "----------------------", "----------", "----------");

    printf("  %-22s %10.6f %10s\n", "Delta (BS true)", bs_d, "—");
    printf("  %-22s %10.6f %10.6f\n", "Delta (Pathwise)", pw_delta,
           fabs(pw_delta - bs_d));
    printf("  %-22s %10.6f %10.6f\n", "Delta (LR)", lr_d,
           fabs(lr_d - bs_d));

    printf("\n");

    printf("  %-22s %10.6f %10s\n", "Vega (BS true)", bs_v, "—");
    printf("  %-22s %10.6f %10.6f\n", "Vega (Pathwise)", pw_vega,
           fabs(pw_vega - bs_v));
    printf("  %-22s %10.6f %10.6f\n", "Vega (LR)", lr_v,
           fabs(lr_v - bs_v));

    printf("  -> Greeks computed via all methods.\n");
    printf("  -> Pathwise has lower variance than LR (see Python version).\n");
    printf("  -> LR works for barrier options; pathwise does NOT.\n\n");

    free(S_T);
    free(payoffs);
}

/* ============================================================================
 * SECTION 5: Convergence Analysis
 * ============================================================================
 *
 * Generates data for convergence plots:
 * 1. Error vs N (Black-Scholes): 1/√N rate.
 * 2. Bias vs dt (Heston): O(dt) Euler bias.
 *
 * Data is saved to ../data/ for plotting by the Python module.
 */

static void run_convergence(RNGState *rng) {
    GBMParams gbm = {100.0, 0.05, 0.20};
    HestonParams he = {100.0, 0.04, 0.05, 2.0, 0.04, 0.30, -0.70};
    EuropeanParams eu;

    /*
     * N values for convergence study.
     *
     * Geometric progression from 1k to 500k.
     * On a log-log plot, these are evenly spaced.
     */
    int N_vals[] = {1000, 2000, 5000, 10000, 20000,
                    50000, 100000, 200000, 500000};

    /*
     * Step counts for discretization bias study.
     *
     * From 4 steps (dt = 0.25) to 504 steps (dt ≈ 0.002).
     * Each doubling of steps halves dt.
     */
    int step_vals[] = {4, 8, 16, 32, 64, 128, 252, 504};

    int nN = 9;  /* Number of N values */
    int nS = 8;  /* Number of step values */

    printf("SECTION 5: Convergence Analysis\n");
    printf("---------------------------------\n\n");

    /*
     * Initialize European option parameters (used by both studies).
     */
    european_init(&eu, 100.0, 1.0, 0.05);

    /*
     * Study 1: Error vs Number of Paths.
     *
     * Uses Black-Scholes (no discretization error) so the error
     * is purely from Monte Carlo sampling.
     */
    printf("  Study 1: Error vs N (Black-Scholes)\n");
    convergence_N(&gbm, &eu, N_vals, nN, rng);

    /*
     * Study 2: Discretization Bias vs Time Step.
     *
     * Uses Heston with Euler-Maruyama to measure discretization
     * bias. Large N (200k) ensures MC error is small compared
     * to the bias we're measuring.
     */
    printf("\n  Study 2: Discretization Bias vs dt (Heston)\n");
    convergence_dt(&he, &eu, 200000, step_vals, nS, rng);

    printf("\n  -> Convergence data saved to ../data/\n\n");
}