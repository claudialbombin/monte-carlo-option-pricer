/**
 * @file rng.h
 * @brief Random number generation — Mersenne Twister & Box-Muller
 *
 * PURPOSE:
 * Provides high-quality pseudo-random number generation for Monte Carlo
 * simulations. Implements the Mersenne Twister (MT19937) algorithm for
 * uniform random integers and the Box-Muller transform for generating
 * standard normal (Gaussian) random variables.
 *
 * WHY MT19937?
 * - Period of 2^19937-1 (effectively infinite for our purposes).
 * - Passes the DIEHARD statistical tests for randomness.
 * - Fast: generates 624 random numbers at a time via state vector.
 * - Deterministic with a known seed → reproducible simulations.
 *
 * WHY BOX-MULLER?
 * - Transforms two uniform [0,1) random numbers into two independent
 *   standard normal N(0,1) random numbers.
 * - We cache the second normal value to avoid wasting entropy.
 * - Alternative (not used): Marsaglia polar method (avoids sin/cos
 *   but uses rejection sampling).
 *
 * DESIGN:
 * - All functions are stateless aside from the module-level state.
 * - seed_rng() initializes the MT state from a 32-bit seed.
 * - rand_uniform() returns a double in [0, 1).
 * - rand_normal() returns a standard N(0,1) sample.
 *
 * Author: Claudia Maria Lopez Bombin
 * License: MIT
 */

#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* =========================================================================
 * Constants
 * ========================================================================= */

#define MT_N 624
#define MT_M 397

/* =========================================================================
 * RNG State Structure
 * ========================================================================= */

/**
 * @brief Mersenne Twister state with Box-Muller cache.
 *
 * Stores the MT state vector and index, plus a cached value for
 * the Box-Muller transform (generates normals in pairs, stores
 * the second for next call).
 */
typedef struct {
    uint32_t mt[MT_N];   /**< MT state vector (624 32-bit words) */
    int      mti;        /**< Current index into mt[] */
    double   bm_cached;  /**< Cached Box-Muller normal value */
    int      bm_has_cached; /**< 1 if cache is valid, 0 otherwise */
} RNGState;

/* =========================================================================
 * Function Declarations
 * ========================================================================= */

/**
 * @brief Initialize the RNG with a given seed.
 *
 * Uses the standard MT19937 initialization algorithm.
 * Must be called before any other RNG function.
 *
 * @param state Pointer to RNGState to initialize.
 * @param seed  32-bit integer seed.
 */
void seed_rng(RNGState *state, uint32_t seed);

/**
 * @brief Generate a uniform random double in [0, 1).
 *
 * Extracts a 32-bit random integer from the MT state and
 * scales it to [0, 1) by multiplying by 2^-32.
 *
 * @param state Pointer to initialized RNGState.
 * @return Double in [0, 1).
 */
double rand_uniform(RNGState *state);

/**
 * @brief Generate a standard normal N(0,1) random variable.
 *
 * Uses the Box-Muller transform on two uniform random numbers.
 * The second normal is cached and returned on the next call,
 * so every other call generates two uniforms.
 *
 * @param state Pointer to initialized RNGState.
 * @return Standard normal random variable.
 */
double rand_normal(RNGState *state);

#endif /* RNG_H */