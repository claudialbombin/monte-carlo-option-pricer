/**
 * @file options.c
 * @brief Option payoff and pricing functions
 *
 * ============================================================================
 * WHAT THIS FILE DOES
 * ============================================================================
 *
 * Computes discounted expected payoffs for three option types.
 * Each function operates on pre-simulated paths and returns the
 * Monte Carlo estimate of the option price.
 *
 * VECTORIZATION OVER PATHS:
 * All pricing functions loop over paths ONCE, computing the payoff
 * for each path and accumulating a running sum. This is O(n_paths)
 * for European, O(n_paths * n_steps) for Asian/barrier.
 *
 * DISCOUNTING:
 * All prices are discounted to present value using exp(-rT).
 * The discount factor is precomputed when the option parameters
 * are initialized and stored in params->discount.
 *
 *   Price = exp(-rT) * (1/N) * sum_{i=1}^{N} payoff_i
 *
 * This is the risk-neutral pricing formula: under the risk-neutral
 * measure Q, the expected discounted payoff equals today's price.
 *
 * ============================================================================
 * OPTION TYPES
 * ============================================================================
 *
 * 1. EUROPEAN CALL:
 *    Payoff = max(S_T - K, 0)
 *    Only the terminal price matters. The simplest option.
 *    Closed-form Black-Scholes price available for validation.
 *
 * 2. ASIAN CALL (Arithmetic Average Rate):
 *    Payoff = max(A - K, 0)
 *    where A = (1/n_steps) * sum_{j=1}^{n_steps} S_{t_j}
 *    The entire path matters. Averaging reduces volatility,
 *    making Asian options cheaper than European options with
 *    the same strike and maturity.
 *
 *    NOTE: We exclude S_0 from the average. Some Asian contracts
 *    include the initial fixing; we follow the convention of
 *    averaging over future prices only. Including S_0 would
 *    reduce variance further (the known initial price acts as
 *    a variance reduction mechanism).
 *
 *    WHY ARITHMETIC (NOT GEOMETRIC)?
 *    Real traded Asian options use arithmetic averaging. The
 *    geometric average DOES have a closed-form solution (because
 *    the product of lognormal random variables is lognormal),
 *    but it's NOT what trades in the market. We implement the
 *    arithmetic version because that's what a real pricing
 *    engine needs.
 *
 * 3. BARRIER CALL (Up-and-Out):
 *    Payoff = max(S_T - K, 0) IF max_{0<=t<=T} S_t < B
 *           = 0                 IF max_{0<=t<=T} S_t >= B
 *
 *    The option is "knocked out" (becomes worthless) if the
 *    underlying ever trades AT OR ABOVE the barrier level B
 *    during the life of the option.
 *
 *    WHY UP-AND-OUT?
 *    - Cheaper than vanilla calls (knockout risk → lower premium).
 *    - Useful for investors with a capped bullish view.
 *    - Common in FX markets (central bank intervention creates
 *      natural barriers).
 *
 *    DISCRETE MONITORING BIAS:
 *    We monitor at discrete time steps t_j. In reality, barriers
 *    are monitored continuously. A discrete approximation MISSES
 *    some barrier crossings that occur between observation dates.
 *    This means our price is SLIGHTLY HIGHER than the true
 *    continuously-monitored price.
 *
 *    The bias is O(1/sqrt(n_steps)) for a single barrier level.
 *    More precisely, the probability of missing a crossing in
 *    interval [t, t+dt] is approximately the probability that
 *    the maximum occurs between monitoring dates, which decays
 *    as monitoring frequency increases.
 *
 *    We output the knockout probability alongside the price so
 *    the user can assess the barrier's impact.
 *
 * ============================================================================
 * DESIGN: WHY THREE SEPARATE STRUCTS?
 * ============================================================================
 *
 * We could have used a single "OptionParams" struct with a type enum.
 * We chose separate structs because:
 * 1. Type safety: EuropeanParams* can't be accidentally passed to
 *    barrier_price() — the compiler catches the error.
 * 2. Memory: Asian and Barrier don't carry unnecessary fields.
 * 3. Clarity: Each option's interface is explicit about what it needs.
 *
 * The cost is some code duplication (three init functions, three price
 * functions). Given the pedagogical nature of this project, clarity
 * outweighs conciseness.
 *
 * ============================================================================
 * REFERENCES
 * ============================================================================
 * Hull, J. (2022). Options, Futures, and Other Derivatives (11th ed.).
 *   Chapter 15: The Black-Scholes-Merton Model.
 *   Chapter 26: Exotic Options.
 * Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
 *   Chapter 1: Foundations.
 *   Chapter 4: Variance Reduction.
 *
 * Author: Claudia Maria Lopez Bombin
 * License: MIT
 */

#include "options.h"
#include <math.h>
#include <stddef.h>

/* ============================================================================
 * european_init — Initialize European option parameters
 * ============================================================================
 *
 * Sets the strike, maturity, risk-free rate, and precomputes the
 * discount factor exp(-rT).
 *
 * WHY PRECOMPUTE THE DISCOUNT FACTOR?
 * - exp() is an expensive transcendental function (~50-100 CPU cycles).
 * - The discount factor is used exactly once per pricing call, so the
 *   savings are minor here. But it establishes a pattern: precompute
 *   everything that depends only on the option parameters.
 * - More importantly, it ensures the discount factor is computed
 *   consistently across all functions that use these parameters.
 *
 * @param params Pointer to EuropeanParams to initialize.
 * @param K      Strike price. Must be > 0.
 * @param T      Time to maturity in years. Must be > 0.
 * @param r      Risk-free rate, continuously compounded.
 */

void european_init(EuropeanParams *p, double K, double T, double r) {
    /*
     * Store raw parameters.
     * No validation here — the caller is responsible for providing
     * valid parameters. In a production system, we'd add checks
     * (K > 0, T > 0) and return error codes.
     */
    p->K = K;
    p->T = T;
    p->r = r;

    /*
     * Precompute discount factor.
     *
     * exp(-rT) is the present value of $1 received at time T.
     * Under continuous compounding with rate r, $1 today grows
     * to exp(rT) at time T. Conversely, $1 at time T is worth
     * exp(-rT) today.
     *
     * Example: r=0.05, T=1 → discount = exp(-0.05) ≈ 0.9512.
     * A payoff of $10 in 1 year is worth $9.51 today.
     */
    p->discount = exp(-r * T);
}

/* ============================================================================
 * european_price — Price a European call option
 * ============================================================================
 *
 * Computes: price = discount * (1/n) * sum_{i=1}^{n} max(S_T[i] - K, 0)
 *
 * ALGORITHM:
 *   sum = 0
 *   For each path i:
 *     payoff = S_T[i] - K
 *     if payoff > 0: sum += payoff
 *   price = discount * sum / n
 *
 * This is an unbiased estimator of the true option price.
 * The standard error of the estimate is:
 *   SE = discount * std(payoff) / sqrt(n)
 *
 * For ATM options, roughly 50% of paths are in-the-money,
 * so the payoff variance is moderate.
 *
 * @param params  European option parameters (with precomputed discount).
 * @param S_T     Array of terminal prices, length n_paths.
 * @param n       Number of paths.
 * @return        Discounted expected payoff (Monte Carlo estimate).
 */

double european_price(const EuropeanParams *p, const double *S_T, int n) {
    double sum = 0.0;
    int i;

    /*
     * Accumulate raw (undiscounted) payoffs.
     *
     * We loop over all paths and sum the positive parts of (S_T - K).
     * Using if (payoff > 0) instead of fmax() avoids an unnecessary
     * function call for paths that end out of the money (roughly half
     * the paths for ATM options).
     *
     * For DEEP in-the-money options (S_T >> K on most paths), the
     * branch is predictable → CPU branch predictor handles it well.
     * For ATM options, the branch is random → some branch misprediction
     * penalty, but it's dwarfed by the RNG cost in the simulation.
     */
    for (i = 0; i < n; i++) {
        double payoff = S_T[i] - p->K;
        if (payoff > 0.0) {
            sum += payoff;
        }
    }

    /*
     * Discount and average.
     *
     * sum / n = sample mean of raw payoffs (expected payoff at time T).
     * * discount = present value (price today).
     *
     * This division by n could cause precision loss for very large n
     * (n > 2^53 ≈ 9e15), but we never use that many paths.
     */
    return p->discount * sum / n;
}

/* ============================================================================
 * asian_init — Initialize Asian option parameters
 * ============================================================================
 *
 * Identical structure to european_init(). The Asian option has the
 * same parameters as a European option (the averaging is handled
 * by the pricing function, not by the parameters).
 *
 * @param params Pointer to AsianParams to initialize.
 * @param K      Strike price.
 * @param T      Time to maturity.
 * @param r      Risk-free rate.
 */

void asian_init(AsianParams *p, double K, double T, double r) {
    p->K = K;
    p->T = T;
    p->r = r;
    p->discount = exp(-r * T);
}

/* ============================================================================
 * asian_price — Price an arithmetic average rate Asian call
 * ============================================================================
 *
 * Computes: price = discount * (1/n) * sum_{i} max(avg_i - K, 0)
 * where avg_i = (1/n_steps) * sum_{j=1}^{n_steps} S_{t_j}^{(i)}
 *
 * ALGORITHM:
 *   For each path i:
 *     1. Sum the prices at all monitoring dates j=1..n_steps.
 *        We exclude j=0 (S_0) because the initial price is known
 *        and is not part of the averaging period.
 *     2. Divide by n_steps to get the arithmetic average.
 *     3. Payoff = max(average - K, 0).
 *     4. Accumulate into running sum.
 *   price = discount * sum / n_paths
 *
 * COMPLEXITY: O(n_paths * n_steps).
 * For n_paths=500k and n_steps=252, this is 126 million additions
 * and comparisons. On a modern CPU (~3 GHz, ~8 ops/cycle), this
 * takes about 5 milliseconds — negligible compared to path simulation.
 *
 * WHY ARITHMETIC AVERAGING MAKES THE OPTION CHEAPER:
 *   The variance of an average is lower than the variance of the
 *   underlying process:
 *     Var(average) ≈ Var(S_T) / n_steps  (for independent observations)
 *   For correlated observations (as in a price path):
 *     Var(average) = Var(S_T) * (1/3 + O(1/n_steps))
 *   The reduction factor is roughly 1/3 for continuous averaging.
 *   Lower variance → lower probability of extreme positive outcomes
 *   → lower call option price.
 *
 * @param params  Asian option parameters.
 * @param paths   Full paths array, row-major, size n*(steps+1).
 * @param n       Number of paths.
 * @param steps   Number of time steps (monitoring dates = steps).
 * @return        Discounted expected payoff.
 */

double asian_price(
    const AsianParams *p, const double *paths,
    int n, int steps
) {
    double sum = 0.0;
    int i, j;

    /*
     * Loop over paths.
     *
     * For each path, we sum the prices at all monitoring dates
     * (columns 1 through steps, excluding column 0 = S_0).
     *
     * We could precompute row pointers and use pointer arithmetic,
     * but the index calculation paths + i*(steps+1) + j is simple
     * enough that the compiler optimizes it to LEA instructions
     * (Load Effective Address) on x86 — no multiplication in the
     * inner loop.
     */
    for (i = 0; i < n; i++) {
        const double *row = paths + i * (steps + 1);
        double avg = 0.0;
        double payoff;

        /*
         * Sum prices at all monitoring dates.
         * j runs from 1 to steps (inclusive), skipping column 0.
         *
         * WHY SKIP S_0?
         * The option's averaging period starts at t=0 and the first
         * fixing is at t_1 = dt. S_0 is known at inception and is
         * not part of the floating average. Some OTC Asian options
         * DO include the first fixing at inception; for those, we'd
         * start j=0. Our convention matches the most common market
         * standard.
         *
         * Accumulating in double precision: for steps=252, we're
         * summing 252 doubles. The worst-case relative error from
         * floating-point addition is ~252 * epsilon ≈ 5.6e-14,
         * completely negligible.
         */
        for (j = 1; j <= steps; j++) {
            avg += row[j];
        }

        /*
         * Divide by number of monitoring dates to get arithmetic mean.
         *
         * avg/steps is the sample mean of the path. For large steps,
         * this approximates the continuous-time integral:
         *   (1/T) * integral_0^T S_t dt
         * by the rectangle rule with step size dt = T/steps.
         */
        avg /= steps;

        /*
         * Payoff = max(average - K, 0).
         *
         * The Asian call pays the positive difference between the
         * average price and the strike. If the average is below K,
         * the option expires worthless (but the holder does NOT
         * lose more than the premium paid).
         */
        payoff = avg - p->K;
        if (payoff > 0.0) {
            sum += payoff;
        }
    }

    /*
     * Discount and average over paths.
     *
     * As with the European option, this is:
     *   Price = e^{-rT} * E_Q[payoff]
     * approximated by the sample mean.
     */
    return p->discount * sum / n;
}

/* ============================================================================
 * barrier_init — Initialize barrier option parameters
 * ============================================================================
 *
 * Sets strike, barrier level, maturity, risk-free rate, and
 * precomputes the discount factor.
 *
 * The barrier level B must be > S_0 for an up-and-out call.
 * If B <= S_0, the option is immediately knocked out at t=0
 * (price = 0). We don't enforce this here — the caller should
 * ensure B > S_0 for a meaningful option.
 *
 * @param params Pointer to BarrierParams to initialize.
 * @param K      Strike price.
 * @param B      Barrier level (must be > S_0 for up-and-out).
 * @param T      Time to maturity.
 * @param r      Risk-free rate.
 */

void barrier_init(
    BarrierParams *p, double K, double B, double T, double r
) {
    p->K = K;
    p->B = B;
    p->T = T;
    p->r = r;
    p->discount = exp(-r * T);
}

/* ============================================================================
 * barrier_price — Price an up-and-out barrier call option
 * ============================================================================
 *
 * Computes the discounted expected payoff accounting for knockout.
 *
 * ALGORITHM:
 *   For each path i:
 *     1. Check if ANY monitoring date S_t >= B.
 *        If yes → knocked out → payoff = 0, increment ko_count.
 *        If no  → survived → payoff = max(S_T - K, 0).
 *     2. Accumulate into running sum.
 *   price = discount * sum / n
 *   ko_prob = ko_count / n
 *
 * THE KNOCKOUT DISCONTINUITY:
 *   Consider two almost-identical paths:
 *   - Path A: max(S_t) = 119.99, S_T = 130 → payoff = 30.
 *   - Path B: max(S_t) = 120.01, S_T = 130 → payoff = 0 (knocked out).
 *
 *   A tiny change in the path (0.02 in the maximum) causes a huge
 *   change in payoff (30 → 0). This is the DISCONTINUITY that makes
 *   the pathwise Greeks method fail. The payoff function has a jump
 *   at the barrier, so its derivative does not exist there.
 *
 *   The likelihood ratio method doesn't care — it differentiates
 *   the probability density, not the payoff. This is the central
 *   conceptual result of the Greeks module.
 *
 * EARLY EXIT OPTIMIZATION:
 *   Once we find S_t >= B, we stop checking that path (break out
 *   of the inner loop). This saves time for knocked-out paths,
 *   especially when the barrier is close to S_0 (many early knockouts).
 *
 * @param params      Barrier option parameters.
 * @param paths       Full paths, row-major, n*(steps+1).
 * @param n           Number of paths.
 * @param steps       Number of time steps.
 * @param ko_prob_out Output: estimated knockout probability (can be NULL).
 * @return            Discounted expected payoff.
 */

double barrier_price(
    const BarrierParams *p, const double *paths,
    int n, int steps, double *ko_prob_out
) {
    double sum = 0.0;
    int ko_count = 0;
    int i, j;

    /*
     * Loop over paths.
     *
     * For each path, we need to check the barrier condition at
     * EVERY monitoring date (including t=0? Yes, j starts at 0.
     * If S_0 >= B, the option is immediately knocked out — though
     * for an up-and-out call, B > S_0 by definition, so j=0
     * never triggers knockout).
     */
    for (i = 0; i < n; i++) {
        const double *row = paths + i * (steps + 1);
        int knocked = 0;

        /*
         * BARRIER MONITORING LOOP:
         * Check S_t at each monitoring date j = 0..steps.
         *
         * If S_{t_j} >= B at ANY j, the option is knocked out.
         * We set knocked = 1 and break immediately (early exit).
         *
         * WHY >= AND NOT >?
         * The standard convention is that touching the barrier
         * EXACTLY triggers knockout. "Up-and-out" means the option
         * ceases to exist if the barrier is REACHED OR EXCEEDED.
         * Using >= ensures correctness for exact touches.
         *
         * DISCRETE MONITORING VS CONTINUOUS:
         * With discrete monitoring at N_steps dates, we miss
         * crossings where:
         *   S_{t_j} < B  AND  S_{t_{j+1}} < B
         *   BUT   max_{t in [t_j, t_{j+1}]} S_t >= B
         *
         * The asset could cross above B and come back down between
         * monitoring dates. This is the discrete monitoring bias —
         * it causes us to OVERPRICE the option (we miss some
         * knockouts, so the expected payoff is too high).
         *
         * The bias scales as O(1/sqrt(n_steps)) and can be reduced
         * by increasing monitoring frequency (more steps).
         *
         * In practice, "continuous monitoring" is approximated by
         * using a barrier correction:
         *   B_effective = B * exp(beta * sigma * sqrt(dt))
         * where beta ≈ 0.5826... (the Riemann zeta function constant).
         * We don't apply this correction — we document the bias
         * as a known limitation.
         */
        for (j = 0; j <= steps; j++) {
            if (row[j] >= p->B) {
                knocked = 1;
                break;  /* Early exit: no need to check further dates */
            }
        }

        if (knocked) {
            /*
             * Path was knocked out: payoff = 0.
             * The option holder receives nothing, regardless of S_T.
             * This is the risk the holder takes in exchange for a
             * lower premium compared to a vanilla call.
             */
            ko_count++;
        } else {
            /*
             * Path survived: compute European payoff at maturity.
             *
             * payoff = max(S_T - K, 0)
             * S_T is at column 'steps' (the last monitoring date).
             *
             * Only surviving paths contribute to the expected payoff.
             * This makes barrier options PATH-DEPENDENT in a stronger
             * sense than Asian options: the entire path determines
             * survival, not just the average.
             */
            double payoff = row[steps] - p->K;
            if (payoff > 0.0) {
                sum += payoff;
            }
        }
    }

    /*
     * Output the knockout probability if the caller wants it.
     *
     * ko_prob = ko_count / n
     *
     * This is the fraction of paths that hit the barrier.
     * For an up-and-out call with B significantly above S_0,
     * this might be 10-40%. It depends on:
     *   - Distance S_0 to B: closer → more knockouts.
     *   - Volatility: higher → more knockouts.
     *   - Drift: upward drift (r > 0) → more knockouts for up barrier.
     *   - Time: longer maturity → more chances to hit.
     *
     * The standard error of ko_prob is sqrt(ko_prob*(1-ko_prob)/n),
     * which is ≤ 0.5/sqrt(n). For n=100k, SE ≤ 0.0016.
     */
    if (ko_prob_out != NULL) {
        *ko_prob_out = (double)ko_count / n;
    }

    /*
     * Return discounted expected payoff.
     *
     * Note: sum only includes SURVIVING paths.
     * We divide by n (ALL paths), not by (n - ko_count).
     * This correctly averages in the zero payoffs from knocked-out
     * paths, giving an unbiased estimate of E[payoff].
     */
    return p->discount * sum / n;
}