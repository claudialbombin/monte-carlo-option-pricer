/**
 * @file models.h
 * @brief Stochastic process simulation engines
 *
 * PURPOSE:
 * Defines the data structures and function declarations for simulating
 * asset price paths under two models:
 *
 * 1. Black-Scholes (Geometric Brownian Motion)
 *    dS_t = r * S_t * dt + sigma * S_t * dW_t
 *    Exact solution used — no discretization error.
 *
 * 2. Heston (Stochastic Volatility)
 *    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t
 *    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW2_t
 *    Corr(dW1_t, dW2_t) = rho * dt
 *    Euler-Maruyama discretization with absorption at zero.
 *
 * DESIGN:
 * - All path data stored in contiguous 1D arrays (row-major order)
 *   allocated on the heap.
 * - Functions operate on pre-allocated arrays to avoid repeated
 *   malloc/free calls during simulation.
 * - Vectorized inner loops over paths for cache efficiency.
 *
 * Author: Claudia Maria Lopez Bombin
 * License: MIT
 */

#ifndef MODELS_H
#define MODELS_H

#include "rng.h"

/* =========================================================================
 * Black-Scholes Model Parameters
 * ========================================================================= */

/**
 * @brief Parameters for Geometric Brownian Motion.
 */
typedef struct {
    double S0;     /**< Initial asset price */
    double r;      /**< Risk-free rate (continuously compounded) */
    double sigma;  /**< Constant volatility */
} GBMParams;

/* =========================================================================
 * Heston Model Parameters
 * ========================================================================= */

/**
 * @brief Parameters for the Heston stochastic volatility model.
 */
typedef struct {
    double S0;     /**< Initial asset price */
    double v0;     /**< Initial variance */
    double r;      /**< Risk-free rate */
    double kappa;  /**< Mean-reversion speed of variance */
    double theta;  /**< Long-run mean of variance */
    double xi;     /**< Volatility of variance (vol of vol) */
    double rho;    /**< Correlation between asset and variance shocks */
} HestonParams;

/* =========================================================================
 * Function Declarations — GBM
 * ========================================================================= */

/**
 * @brief Simulate terminal asset prices under GBM.
 *
 * Uses the exact solution: S_T = S0 * exp((r - sigma^2/2)*T + sigma*W_T)
 * where W_T ~ N(0, T). No discretization error.
 *
 * @param params    Model parameters.
 * @param T         Time to maturity (years).
 * @param n_paths   Number of paths to simulate.
 * @param rng       Pointer to initialized RNG state.
 * @param S_T       Output array of length n_paths (pre-allocated).
 */
void gbm_simulate_terminal(
    const GBMParams *params,
    double T,
    int n_paths,
    RNGState *rng,
    double *S_T
);

/**
 * @brief Simulate full paths under GBM (for path-dependent options).
 *
 * Generates paths at n_steps equally spaced monitoring dates.
 * Output is stored in row-major order: paths[i * (n_steps+1) + j]
 * where i = path index, j = time step.
 *
 * @param params    Model parameters.
 * @param T         Time to maturity (years).
 * @param n_paths   Number of paths.
 * @param n_steps   Number of time steps per path.
 * @param rng       Pointer to initialized RNG state.
 * @param paths     Output array (n_paths * (n_steps+1)), pre-allocated.
 */
void gbm_simulate_paths(
    const GBMParams *params,
    double T,
    int n_paths,
    int n_steps,
    RNGState *rng,
    double *paths
);

/* =========================================================================
 * Function Declarations — Heston
 * ========================================================================= */

/**
 * @brief Simulate asset and variance paths under Heston model.
 *
 * Uses Euler-Maruyama discretization with absorption at zero
 * for the variance process (v_t = max(v_t, 0)).
 *
 * Output arrays are row-major: array[i * (n_steps+1) + j]
 * where i = path index, j = time step.
 *
 * @param params    Heston model parameters.
 * @param T         Time to maturity (years).
 * @param n_paths   Number of paths.
 * @param n_steps   Number of time steps.
 * @param rng       Pointer to initialized RNG state.
 * @param S         Output asset price paths (n_paths * (n_steps+1)).
 * @param v         Output variance paths (n_paths * (n_steps+1)).
 */
void heston_simulate_paths(
    const HestonParams *params,
    double T,
    int n_paths,
    int n_steps,
    RNGState *rng,
    double *S,
    double *v
);

#endif /* MODELS_H */