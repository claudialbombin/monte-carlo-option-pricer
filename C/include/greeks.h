/**
 * @file greeks.h
 * @brief Option sensitivity computation — Pathwise & Likelihood Ratio
 *
 * PURPOSE:
 * Computes Delta (dPrice/dS0) and Vega (dPrice/d(sigma)) using two methods:
 *
 * 1. PATHWISE (Infinitesimal Perturbation Analysis):
 *    Differentiates the payoff function. Lower variance but requires
 *    continuous payoffs. FAILS for barrier options.
 *
 * 2. LIKELIHOOD RATIO (Score Function Method):
 *    Differentiates the transition density. Higher variance but works
 *    for ANY payoff — including discontinuous barrier options.
 *
 * Also provides closed-form Black-Scholes Greeks for European options
 * as ground truth for validation.
 *
 * Author: Claudia Maria Lopez Bombin
 * License: MIT
 */

#ifndef GREEKS_H
#define GREEKS_H

/* =========================================================================
 * Closed-Form Black-Scholes Greeks
 * ========================================================================= */

/**
 * @brief Compute Black-Scholes European call price (closed form).
 *
 * C = S0 * N(d1) - K * exp(-rT) * N(d2)
 *
 * @param S0    Initial asset price.
 * @param K     Strike price.
 * @param T     Time to maturity.
 * @param r     Risk-free rate.
 * @param sigma Volatility.
 * @return Black-Scholes call price.
 */
double bs_price(double S0, double K, double T, double r, double sigma);

/**
 * @brief Compute Black-Scholes Delta (closed form).
 *
 * Delta = N(d1)
 *
 * @param S0    Initial asset price.
 * @param K     Strike price.
 * @param T     Time to maturity.
 * @param r     Risk-free rate.
 * @param sigma Volatility.
 * @return Black-Scholes Delta.
 */
double bs_delta(double S0, double K, double T, double r, double sigma);

/**
 * @brief Compute Black-Scholes Vega (closed form).
 *
 * Vega = S0 * sqrt(T) * N'(d1)
 *
 * @param S0    Initial asset price.
 * @param K     Strike price.
 * @param T     Time to maturity.
 * @param r     Risk-free rate.
 * @param sigma Volatility.
 * @return Black-Scholes Vega.
 */
double bs_vega(double S0, double K, double T, double r, double sigma);

/* =========================================================================
 * Pathwise Greeks
 * ========================================================================= */

/**
 * @brief Compute pathwise Delta for a European call.
 *
 * For each path: delta_i = indicator(S_T > K) * (S_T / S0)
 * Greek = discount * mean(delta_i)
 *
 * @param S_T      Terminal prices (length n_paths).
 * @param S0       Initial asset price.
 * @param K        Strike price.
 * @param discount Discount factor exp(-rT).
 * @param n_paths  Number of paths.
 * @return Pathwise Delta estimate.
 */
double pathwise_delta_european(
    const double *S_T,
    double S0, double K,
    double discount,
    int n_paths
);

/**
 * @brief Compute pathwise Vega for a European call.
 *
 * @param S_T      Terminal prices (length n_paths).
 * @param S0       Initial asset price.
 * @param K        Strike price.
 * @param T        Time to maturity.
 * @param r        Risk-free rate.
 * @param sigma    Volatility.
 * @param discount Discount factor exp(-rT).
 * @param n_paths  Number of paths.
 * @return Pathwise Vega estimate.
 */
double pathwise_vega_european(
    const double *S_T,
    double S0, double K,
    double T, double r, double sigma,
    double discount,
    int n_paths
);

/* =========================================================================
 * Likelihood Ratio Greeks
 * ========================================================================= */

/**
 * @brief Compute likelihood ratio Delta.
 *
 * Greek = discount * mean(payoff_i * score_i)
 *
 * @param payoffs  Raw payoffs array (length n_paths).
 * @param S_T      Terminal prices (length n_paths).
 * @param S0       Initial asset price.
 * @param sigma    Volatility.
 * @param T        Time to maturity.
 * @param r        Risk-free rate.
 * @param discount Discount factor exp(-rT).
 * @param n_paths  Number of paths.
 * @return Likelihood ratio Delta estimate.
 */
double lr_delta(
    const double *payoffs,
    const double *S_T,
    double S0, double sigma,
    double T, double r,
    double discount,
    int n_paths
);

/**
 * @brief Compute likelihood ratio Vega.
 *
 * @param payoffs  Raw payoffs array (length n_paths).
 * @param S_T      Terminal prices (length n_paths).
 * @param S0       Initial asset price.
 * @param sigma    Volatility.
 * @param T        Time to maturity.
 * @param r        Risk-free rate.
 * @param discount Discount factor exp(-rT).
 * @param n_paths  Number of paths.
 * @return Likelihood ratio Vega estimate.
 */
double lr_vega(
    const double *payoffs,
    const double *S_T,
    double S0, double sigma,
    double T, double r,
    double discount,
    int n_paths
);

#endif /* GREEKS_H */