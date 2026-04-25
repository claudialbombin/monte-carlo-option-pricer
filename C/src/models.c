/**
 * @file models.c
 * @brief Stochastic process simulation — Black-Scholes & Heston paths
 *
 * ============================================================================
 * WHAT THIS FILE DOES
 * ============================================================================
 *
 * Generates asset price paths for Monte Carlo option pricing under two
 * models. All simulation is vectorized over paths: we simulate all paths
 * simultaneously rather than looping path-by-path in the outer loop.
 *
 * MEMORY LAYOUT:
 *   Paths are stored in row-major 1D arrays:
 *     paths[i * (n_steps + 1) + j]
 *   where i = path index (0 to n_paths-1), j = time step (0 to n_steps).
 *   Column j=0 is the initial price S_0, column j=n_steps is S_T.
 *
 * WHY ROW-MAJOR?
 *   A single path is contiguous in memory. This is cache-friendly when
 *   we compute path-dependent payoffs (Asian averaging, barrier monitoring)
 *   because those operations traverse an entire path sequentially.
 *   If we used column-major, barriers would jump between paths at each
 *   time step, thrashing the cache.
 *
 * MEMORY ALLOCATION:
 *   The caller pre-allocates all arrays. This module NEVER calls malloc.
 *   This design lets the caller reuse buffers across multiple simulations
 *   and control memory lifetime explicitly.
 *
 * ============================================================================
 * MODEL 1: Black-Scholes (Geometric Brownian Motion)
 * ============================================================================
 *
 * SDE:  dS_t = r * S_t * dt + sigma * S_t * dW_t
 *
 * EXACT SOLUTION:
 *   S_t = S_0 * exp( (r - 0.5*sigma^2) * t + sigma * W_t )
 *
 * This is one of the rare SDEs with a closed-form solution. We don't
 * need Euler-Maruyama discretization. We can jump directly to any time t
 * by generating W_t ~ N(0, t) and applying the formula above.
 *
 * For TERMINAL simulation, this is O(1) per path with ZERO discretization
 * error — the holy grail of Monte Carlo.
 *
 * For FULL PATH simulation (needed for Asian/barrier options), we build
 * the Brownian path W_{t_j} from independent increments:
 *   dW_j = sqrt(dt) * Z_j    with Z_j ~ N(0,1) i.i.d.
 *   W_{t_k} = sum_{j=1}^{k} dW_j
 * Then apply the exact solution at each monitoring date.
 *
 * TRICK: We first accumulate the drift + diffusion in log-space, do a
 * cumulative sum, and exponentiate at the end. This avoids calling
 * exp() at every intermediate step (which is slower) — we compute
 * log-prices, cumsum them, and do ONE exp per path per step.
 *
 * ============================================================================
 * MODEL 2: Heston (Stochastic Volatility)
 * ============================================================================
 *
 * SDE SYSTEM:
 *   dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t
 *   dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW2_t
 *   Corr(dW1_t, dW2_t) = rho * dt
 *
 * UNLIKE GBM, THIS HAS NO CLOSED-FORM SOLUTION FOR S_t.
 * We MUST use discretization. The simplest scheme is Euler-Maruyama.
 *
 * EULER-MARUYAMA SCHEME (one step of size dt):
 *
 *   v_{k+1} = v_k + kappa * (theta - v_k) * dt
 *                   + xi * sqrt(max(v_k, 0)) * dW2
 *   v_{k+1} = max(v_{k+1}, 0)       ← ABSORPTION AT ZERO
 *
 *   S_{k+1} = S_k + r * S_k * dt
 *                 + sqrt(max(v_k, 0)) * S_k * dW1
 *
 * CORRELATED BROWNIAN INCREMENTS:
 *   dW1 = sqrt(dt) * e1
 *   dW2 = sqrt(dt) * (rho * e1 + sqrt(1 - rho^2) * e2)
 *   where e1, e2 ~ N(0,1) independent.
 *
 * This is Cholesky decomposition of the covariance matrix:
 *   Cov(dW1, dW2) = [[1, rho], [rho, 1]] * dt
 *   L = [[1, 0], [rho, sqrt(1-rho^2)]]
 *   [dW1, dW2]^T = L * [e1*dt, e2*dt]^T
 *
 * VARIANCE BOUNDARY HANDLING — ABSORPTION AT ZERO:
 *
 * The CIR process (Cox-Ingersoll-Ross) for variance:
 *   dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dW2_t
 *
 * The process is theoretically non-negative if the Feller condition holds:
 *   2 * kappa * theta >= xi^2
 *
 * With our parameters (kappa=2, theta=0.04, xi=0.30):
 *   2 * 2 * 0.04 = 0.16
 *   xi^2 = 0.09
 *   0.16 >= 0.09 → Feller condition SATISFIED
 *
 * BUT: Euler discretization can still produce negative values because
 * it uses a discrete approximation. The probability of v_k becoming
 * negative in a step depends on dt. Smaller dt → lower probability.
 *
 * OUR FIX: Truncation (absorption):
 *   v_k = max(v_k, 0)  at each step
 *
 * This biases the variance slightly upward (we replace negative values
 * with zero instead of letting them revert). The bias is O(dt) and
 * VANISHES as dt → 0. This is the standard industry fix.
 *
 * ALTERNATIVES (not implemented, documented for completeness):
 * - Reflection: v = |v|. Less bias but changes the distribution shape.
 * - Alfonsi (2005) implicit scheme: more accurate, more complex.
 * - Quadratic-Exponential (QE) scheme: Andersen (2006). Popular in
 *   production systems. Handles large dt better.
 *
 * ============================================================================
 * PERFORMANCE NOTES
 * ============================================================================
 *
 * The Heston simulation is the bottleneck of the entire project:
 * - Each path × each step needs 2 normal random numbers.
 * - For N=500k paths and 252 steps = 126M paths × 2 normals = 252M RNG calls.
 * - At ~20M RNG calls/second, this takes ~12 seconds.
 *
 * Optimizations applied:
 * - Single loop over paths with inner loop over steps (cache-friendly).
 * - Local pointers Si = S + i*(m+1) to avoid repeated index arithmetic.
 * - Precomputed: dt, sqrt_dt, rho_c = sqrt(1-rho^2).
 * - max_d() as a static inline function (typically inlined by compiler).
 *
 * ============================================================================
 * REFERENCES
 * ============================================================================
 * Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.
 *   Chapter 3: Generating Sample Paths.
 *   Chapter 4: Variance Reduction Techniques.
 * Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic
 *   Volatility." Review of Financial Studies, 6(2), 327-343.
 * Andersen, L. (2006). "Efficient Simulation of the Heston Stochastic
 *   Volatility Model." Journal of Computational Finance, 11(3), 1-42.
 *
 * Author: Claudia Maria Lopez Bombin
 * License: MIT
 */

#include "models.h"
#include <math.h>

/* ============================================================================
 * HELPER: Cumulative sum of an array in-place
 * ============================================================================
 *
 * Given array arr[0..len-1], replaces arr[i] with sum_{k=0}^{i} arr[k].
 *
 * Used to convert independent Brownian INCREMENTS into a Brownian PATH.
 *   dW[0], dW[1], ..., dW[len-1]  →  W[0]=0, W[i] = sum_{k=0}^{i-1} dW[k]
 *
 * We call this with arr = log_price_increments (starting at index 1,
 * since index 0 = log(S_0) = 0) and len = n_steps. After cumsum:
 *   arr[k] = drift*k*dt + sigma*W_{t_k}
 * Then: S_{t_k} = S_0 * exp(arr[k]).
 *
 * This is O(len), single pass. Cache-friendly: sequential memory access.
 */

static void cumsum(double *arr, int len) {
    int i;
    for (i = 1; i < len; i++) {
        arr[i] += arr[i - 1];
    }
}

/* ============================================================================
 * gbm_simulate_terminal — Black-Scholes terminal prices
 * ============================================================================
 *
 * Simulates S_T = S_0 * exp( (r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z )
 * for n_paths independent paths, with Z ~ N(0,1).
 *
 * ALGORITHM:
 *   For each path i = 0..n-1:
 *     Z = rand_normal(rng)                    // Standard normal
 *     S_T[i] = S0 * exp(drift + sigma*sqrtT * Z)
 *
 * COMPLEXITY: O(n) time, O(1) per path. Zero discretization error.
 *
 * @param params  GBM parameters (S0, r, sigma).
 * @param T       Time to maturity (years).
 * @param n       Number of paths.
 * @param rng     Initialized RNG state.
 * @param S_T     Output array of length n (pre-allocated by caller).
 */

void gbm_simulate_terminal(
    const GBMParams *p, double T, int n,
    RNGState *rng, double *S_T
) {
    /*
     * Precompute the deterministic components ONCE.
     * drift_term = (r - 0.5*sigma^2) * T
     * sigma_sqrtT = sigma * sqrt(T)
     *
     * The (r - 0.5*sigma^2) drift is the RISK-NEUTRAL drift.
     * Under the risk-neutral measure Q, the asset grows at the risk-free
     * rate r. But because of Ito's lemma, the EXPECTED terminal price
     * E[S_T] = S_0 * exp(r*T), not S_0 * exp((r - sigma^2/2)*T).
     * The sigma^2/2 term compensates for the convexity of exp().
     *
     * Proof: E[exp(sigma*W_T)] = exp(sigma^2*T/2) because W_T ~ N(0,T).
     * So E[S_T] = S_0 * exp((r - sigma^2/2)*T) * exp(sigma^2*T/2)
     *            = S_0 * exp(r*T).  ✓
     */
    double drift, diffusion;
    int i;

    drift = (p->r - 0.5 * p->sigma * p->sigma) * T;
    diffusion = p->sigma * sqrt(T);

    /*
     * Loop over paths. Each iteration generates ONE standard normal
     * and computes ONE terminal price. This is embarrassingly parallel
     * — each path is independent. A future optimization could use
     * OpenMP: #pragma omp parallel for.
     */
    for (i = 0; i < n; i++) {
        /*
         * Exact GBM solution:
         * S_T = S_0 * exp( (r - 0.5*sigma^2)*T + sigma*W_T )
         * where W_T = sqrt(T) * Z,  Z ~ N(0,1).
         */
        S_T[i] = p->S0 * exp(drift + diffusion * rand_normal(rng));
    }
}

/* ============================================================================
 * gbm_simulate_paths — Black-Scholes full paths
 * ============================================================================
 *
 * Generates n_paths × (n_steps+1) price matrix with monitoring at
 * equally spaced dates: t_j = j * T/n_steps for j = 0..n_steps.
 *
 * ALGORITHM:
 *   For each path i:
 *     1. Set paths[i][0] = S0 (initial price).
 *     2. For j = 1..n_steps, generate the LOG-PRICE INCREMENT:
 *          inc[j] = (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_j
 *     3. Compute cumulative sum of inc[] (excluding inc[0]):
 *          cum[j] = sum_{k=1}^{j} inc[k]
 *     4. paths[i][j] = S0 * exp(cum[j]) for j = 1..n_steps.
 *
 * This is more efficient than calling exp() at every intermediate step
 * because we work in log-space (where the GBM is just Brownian motion
 * with drift) and only exponentiate at the end.
 *
 * @param params   GBM parameters.
 * @param T        Time to maturity.
 * @param n        Number of paths.
 * @param steps    Number of time steps (monitoring dates = steps + 1).
 * @param rng      Initialized RNG state.
 * @param paths    Output array of size n*(steps+1), row-major, pre-allocated.
 */

void gbm_simulate_paths(
    const GBMParams *p, double T,
    int n, int steps, RNGState *rng, double *paths
) {
    double dt, sqrt_dt, drift_dt;
    int i, j;

    /*
     * dt = T / steps: time increment.
     * sqrt_dt: precomputed for generating dW = sqrt(dt) * Z.
     * drift_dt = (r - 0.5*sigma^2) * dt: drift per step.
     *
     * Precomputing these outside the loops saves:
     *   n * steps divisions (for dt each iteration)
     *   n * steps sqrt calls
     *   n * steps multiplications
     */
    dt = T / steps;
    sqrt_dt = sqrt(dt);
    drift_dt = (p->r - 0.5 * p->sigma * p->sigma) * dt;

    /*
     * Loop over paths. For each path:
     * 1. Set S_0.
     * 2. Fill log-price increments for steps 1..steps.
     * 3. Cumulative sum of increments to get the log-price path.
     * 4. Exponentiate to get prices.
     */
    for (i = 0; i < n; i++) {
        /*
         * row = pointer to the start of path i in the flat array.
         * This avoids repeatedly computing paths + i*(steps+1).
         */
        double *row = paths + i * (steps + 1);

        /* Column 0 = initial price (all paths start at S0) */
        row[0] = p->S0;

        /*
         * Fill log-price increments.
         * inc_j = drift_dt + sigma * sqrt_dt * Z_j
         * where Z_j ~ N(0,1) i.i.d.
         *
         * We store these at row[1..steps]. After cumsum, row[k] will
         * contain the cumulative log-drift+diffusion up to step k,
         * which equals (r - 0.5*sigma^2)*t_k + sigma*W_{t_k}.
         */
        for (j = 1; j <= steps; j++) {
            row[j] = drift_dt + p->sigma * sqrt_dt * rand_normal(rng);
        }

        /*
         * Cumulative sum converts independent increments into a path.
         * Before: row[k] = increment from t_{k-1} to t_k.
         * After:  row[k] = sum_{j=1}^{k} increment_j
         *                = (r - 0.5*sigma^2)*t_k + sigma*W_{t_k}
         *
         * Note: we pass row+1 and steps (the length of increments).
         * cumsum() starts accumulating from index 1.
         */
        cumsum(row + 1, steps);

        /*
         * Exponentiate: S_{t_k} = S_0 * exp(cumulative log-process)
         * This is the EXACT GBM solution at each monitoring date.
         */
        for (j = 1; j <= steps; j++) {
            row[j] = p->S0 * exp(row[j]);
        }
    }
}

/* ============================================================================
 * HELPER: max_d — Maximum of two doubles
 * ============================================================================
 *
 * Simple inline-style helper. Used for variance truncation in Heston.
 * In C99/C11, fmax() from math.h exists but we define our own to:
 * 1. Avoid potential function call overhead (compiler usually inlines).
 * 2. Keep the code explicit about what we're doing.
 * 3. Make the absorption-at-zero intent obvious.
 */

static double max_d(double a, double b) {
    return (a > b) ? a : b;
}

/* ============================================================================
 * heston_simulate_paths — Heston stochastic volatility paths
 * ============================================================================
 *
 * Simulates correlated asset price AND variance paths.
 *
 * EULER-MARUYAMA RECURRENCE:
 *
 *   For k = 1, 2, ..., steps:
 *
 *     1. Generate independent normals: e1, e2 ~ N(0,1).
 *
 *     2. Correlate via Cholesky:
 *          dW1 = sqrt(dt) * e1
 *          dW2 = sqrt(dt) * (rho*e1 + sqrt(1-rho^2)*e2)
 *
 *     3. Update variance (CIR process):
 *          v_pos = max(v_{k-1}, 0)          // enforce non-negativity
 *          v_k = v_{k-1} + kappa*(theta - v_{k-1})*dt
 *                         + xi*sqrt(v_pos)*dW2
 *          v_k = max(v_k, 0)                // absorption at zero
 *
 *     4. Update asset price (GBM with stochastic vol):
 *          S_k = S_{k-1} + r*S_{k-1}*dt
 *                         + sqrt(v_pos)*S_{k-1}*dW1
 *
 * WHY UPDATE VARIANCE FIRST:
 *   The variance process drives the asset process (through sqrt(v_t)),
 *   but the asset process does NOT feed back into variance (except
 *   through the correlation rho, which we handle via the Cholesky).
 *   Updating v before S is natural.
 *
 * NUMERICAL STABILITY:
 *   - sqrt(v_pos) is safe because we enforce v_pos >= 0.
 *   - The Euler scheme for S uses additive updates. For small dt,
 *     S stays positive almost surely. But unlike GBM's exact solution,
 *     Euler CAN theoretically produce negative S with very large dt
 *     or extreme variance. Our parameters make this vanishingly rare.
 *
 *   - Future improvement: use log-Euler scheme for S:
 *       log(S_k) = log(S_{k-1}) + (r - 0.5*v_pos)*dt + sqrt(v_pos)*dW1
 *     This guarantees S > 0 always. We use the simpler additive scheme
 *     here for clarity.
 *
 * @param params  Heston parameters.
 * @param T       Time to maturity.
 * @param n       Number of paths.
 * @param steps   Number of time steps.
 * @param rng     Initialized RNG state.
 * @param S       Output asset paths (n × (steps+1)), row-major.
 * @param v       Output variance paths (n × (steps+1)), row-major.
 */

void heston_simulate_paths(
    const HestonParams *p, double T,
    int n, int steps, RNGState *rng, double *S, double *v
) {
    double dt, sqrt_dt, rho_c;
    int i, j;

    /*
     * Precompute constants that depend on the time discretization.
     *
     * dt = T / steps: length of one Euler step.
     *   - With daily steps (252 for 1 year): dt ≈ 0.004.
     *   - With weekly steps (52): dt ≈ 0.019.
     *   - Bias is O(dt), so daily steps give ~5x less bias than weekly.
     *
     * sqrt_dt: used in dW1, dW2 = sqrt(dt) * e.
     *
     * rho_c = sqrt(1 - rho^2):
     *   This is the (2,2) element of the Cholesky factor.
     *   It scales the independent component of the variance shock.
     *   If rho = -0.70, rho_c = sqrt(1 - 0.49) = sqrt(0.51) ≈ 0.714.
     *   The variance shock is:
     *     dW2 = 0.70*dW1_unscaled + 0.714*independent_noise
     *   = -0.70*dW1_unscaled if dW1 and the independent part align.
     */
    dt = T / steps;
    sqrt_dt = sqrt(dt);
    rho_c = sqrt(1.0 - p->rho * p->rho);

    /*
     * Main loop: iterate over PATHS first (outer), then TIME STEPS (inner).
     *
     * Why paths in the outer loop?
     * - Each path is independent → good for potential parallelization.
     * - Inner loop accesses contiguous memory (Si[0], Si[1], ..., Si[steps]),
     *   which is cache-friendly: one cache line holds 8 doubles (64 bytes).
     * - If we put time steps in the outer loop, we'd jump between paths
     *   at each step, causing cache misses every time we cross a cache line.
     *
     * Trade-off: With paths outer, we precompute the RNG values per path.
     * This means the RNG is called sequentially (not an issue for MT).
     */
    for (i = 0; i < n; i++) {
        /*
         * Si = pointer to path i in the asset price array.
         * vi = pointer to path i in the variance array.
         *
         * These local pointers avoid the compiler having to recompute
         * S + i*(steps+1) and v + i*(steps+1) at every array access
         * inside the inner loop.
         */
        double *Si = S + i * (steps + 1);
        double *vi = v + i * (steps + 1);

        /* Set initial conditions for this path */
        Si[0] = p->S0;
        vi[0] = p->v0;

        /*
         * Time-stepping loop: Euler-Maruyama for steps 0 to steps-1.
         *
         * At iteration j, we have (S_j, v_j) and compute (S_{j+1}, v_{j+1}).
         */
        for (j = 0; j < steps; j++) {
            double e1, e2;      /* Independent standard normals */
            double dW1, dW2;    /* Correlated Brownian increments */
            double v_pos;       /* Non-negative variance for sqrt */
            double v_new;       /* Updated variance (before absorption) */

            /*
             * STEP 1: Generate two independent standard normals.
             *
             * e1 drives the asset shock directly.
             * e2 provides the independent component of the variance shock.
             *
             * Together they produce correlated shocks via Cholesky.
             */
            e1 = rand_normal(rng);
            e2 = rand_normal(rng);

            /*
             * STEP 2: Correlate the Brownian increments via Cholesky.
             *
             * Covariance matrix of (dW1, dW2) scaled by dt:
             *   Σ = [[1,  rho  ],
             *        [rho, 1    ]] * dt
             *
             * Cholesky factor L such that L @ L^T = [[1, rho], [rho, 1]]:
             *   L = [[1,          0        ],
             *        [rho,  sqrt(1-rho^2) ]]
             *
             * Then: [dW1, dW2]^T = sqrt(dt) * L * [e1, e2]^T
             *
             *   dW1 = sqrt(dt) * e1
             *   dW2 = sqrt(dt) * (rho*e1 + sqrt(1-rho^2)*e2)
             *
             * With rho = -0.70:
             *   dW2 = sqrt(dt) * (-0.70*e1 + 0.714*e2)
             *
             * Interpretation: When the asset shock is negative (e1 < 0),
             * dW2 tends to be POSITIVE (since -0.70 * negative = positive).
             * This means negative asset returns correlate with INCREASED
             * variance — the "leverage effect" observed in equity markets.
             */
            dW1 = sqrt_dt * e1;
            dW2 = sqrt_dt * (p->rho * e1 + rho_c * e2);

            /*
             * STEP 3: Update variance (CIR process).
             *
             * v_pos = max(v_j, 0) ensures we take sqrt of a non-negative
             * number. This is applied BEFORE the update because the
             * diffusion term xi*sqrt(v_t)*dW2 requires sqrt(v_t).
             *
             * The mean-reversion term kappa*(theta - v_j) pulls variance
             * toward its long-run mean theta.
             *   - If v_j > theta: drift is negative (v tends down).
             *   - If v_j < theta: drift is positive (v tends up).
             *
             * Speed of reversion: kappa = 2 means the half-life of a
             * deviation is ln(2)/kappa ≈ 0.35 years ≈ 4 months.
             */
            v_pos = max_d(vi[j], 0.0);

            /*
             * Euler step for variance:
             * v_{j+1} = v_j + drift*dt + diffusion*dW2
             *
             * drift = kappa*(theta - v_j): mean reversion.
             * diffusion = xi*sqrt(v_pos): volatility of variance.
             */
            v_new = vi[j]
                  + p->kappa * (p->theta - vi[j]) * dt
                  + p->xi * sqrt(v_pos) * dW2;

            /*
             * ABSORPTION AT ZERO:
             * If the Euler step produced a negative variance, truncate
             * to zero. This is the simplest practical fix for the
             * discretization error that sometimes pushes v below zero.
             *
             * This creates a small positive bias (we replace negative
             * values with zero, raising the average variance).
             * The bias is O(dt) and vanishes as dt → 0.
             *
             * For our parameters with daily steps (dt ≈ 0.004), the
             * probability of v_j going negative is very small (Feller
             * condition is satisfied with margin: 0.16 ≥ 0.09).
             *
             * We store the truncated value for the next iteration.
             */
            vi[j + 1] = max_d(v_new, 0.0);

            /*
             * STEP 4: Update asset price.
             *
             * Euler step for S:
             * S_{j+1} = S_j + r*S_j*dt + sqrt(v_pos)*S_j*dW1
             *
             * This is the standard Euler discretization of GBM with
             * stochastic volatility. Note we use v_pos (variance at
             * the START of the step), which is the standard Euler
             * approach (explicit scheme).
             *
             * The drift term r*S_j*dt represents the risk-neutral growth.
             * The diffusion sqrt(v_pos)*S_j*dW1 is the randomness.
             *
             * For small dt, S remains positive almost surely. The
             * probability of S going negative in a single Euler step
             * is approximately:
             *   P(S_{j+1} < 0) ≈ Φ(-(1 + r*dt) / (sqrt(v_pos*dt)))
             * For typical parameters, this is astronomically small
             * (many standard deviations in the tail).
             */
            Si[j + 1] = Si[j]
                      + p->r * Si[j] * dt
                      + sqrt(v_pos) * Si[j] * dW1;
        }
    }
}