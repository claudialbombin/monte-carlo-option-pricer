/**
 * @file options.h
 * @brief Option payoff structures and pricing functions
 *
 * PURPOSE:
 * Defines the three option types and their payoff/pricing logic:
 * - European call: max(S_T - K, 0)
 * - Asian call (arithmetic average): max(avg(S_t) - K, 0)
 * - Barrier call (up-and-out): max(S_T - K, 0) if max(S_t) < B, else 0
 *
 * DESIGN:
 * - Each option type has a params struct and a pricing function.
 * - Payoff computation is vectorized over paths.
 * - Discount factor exp(-rT) is precomputed.
 *
 * Author: Claudia Maria Lopez Bombin
 * License: MIT
 */

#ifndef OPTIONS_H
#define OPTIONS_H

/* =========================================================================
 * European Option
 * ========================================================================= */

/**
 * @brief Parameters for a European call option.
 */
typedef struct {
    double K;           /**< Strike price */
    double T;           /**< Time to maturity (years) */
    double r;           /**< Risk-free rate */
    double discount;    /**< Precomputed exp(-rT) */
} EuropeanParams;

/**
 * @brief Initialize European option parameters.
 *
 * Precomputes the discount factor exp(-rT).
 *
 * @param params Pointer to struct to initialize.
 * @param K      Strike price.
 * @param T      Time to maturity.
 * @param r      Risk-free rate.
 */
void european_init(EuropeanParams *params, double K, double T, double r);

/**
 * @brief Price a European call from terminal prices.
 *
 * price = exp(-rT) * mean(max(S_T - K, 0))
 *
 * @param params  European option parameters.
 * @param S_T     Terminal prices array (length n_paths).
 * @param n_paths Number of paths.
 * @return Monte Carlo price estimate.
 */
double european_price(
    const EuropeanParams *params,
    const double *S_T,
    int n_paths
);

/* =========================================================================
 * Asian Option (Arithmetic Average)
 * ========================================================================= */

/**
 * @brief Parameters for an arithmetic Asian call option.
 */
typedef struct {
    double K;           /**< Strike price */
    double T;           /**< Time to maturity (years) */
    double r;           /**< Risk-free rate */
    double discount;    /**< Precomputed exp(-rT) */
} AsianParams;

/**
 * @brief Initialize Asian option parameters.
 *
 * @param params Pointer to struct to initialize.
 * @param K      Strike price.
 * @param T      Time to maturity.
 * @param r      Risk-free rate.
 */
void asian_init(AsianParams *params, double K, double T, double r);

/**
 * @brief Price an Asian call from full paths.
 *
 * Averages over columns 1..n_steps (excludes S_0).
 *
 * @param params   Asian option parameters.
 * @param paths    Full paths array (n_paths * (n_steps+1)), row-major.
 * @param n_paths  Number of paths.
 * @param n_steps  Number of time steps.
 * @return Monte Carlo price estimate.
 */
double asian_price(
    const AsianParams *params,
    const double *paths,
    int n_paths,
    int n_steps
);

/* =========================================================================
 * Barrier Option (Up-and-Out Call)
 * ========================================================================= */

/**
 * @brief Parameters for an up-and-out barrier call option.
 */
typedef struct {
    double K;           /**< Strike price */
    double B;           /**< Barrier level (B > S0 for up-and-out) */
    double T;           /**< Time to maturity (years) */
    double r;           /**< Risk-free rate */
    double discount;    /**< Precomputed exp(-rT) */
} BarrierParams;

/**
 * @brief Initialize barrier option parameters.
 *
 * @param params Pointer to struct to initialize.
 * @param K      Strike price.
 * @param B      Barrier level.
 * @param T      Time to maturity.
 * @param r      Risk-free rate.
 */
void barrier_init(
    BarrierParams *params,
    double K, double B, double T, double r
);

/**
 * @brief Price an up-and-out barrier call from full paths.
 *
 * @param params      Barrier option parameters.
 * @param paths       Full paths array (n_paths * (n_steps+1)), row-major.
 * @param n_paths     Number of paths.
 * @param n_steps     Number of time steps.
 * @param ko_prob_out Output: estimated knockout probability.
 * @return Monte Carlo price estimate.
 */
double barrier_price(
    const BarrierParams *params,
    const double *paths,
    int n_paths,
    int n_steps,
    double *ko_prob_out
);

#endif /* OPTIONS_H */