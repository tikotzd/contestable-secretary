import numpy as np
import math
from scipy.integrate import quad
from scipy.special import comb
from scipy.stats import norm
from scipy.optimize import brentq


def calculate_continuous_mf_cdf_threshold(num_captains):
    """
    Finds the Cumulative Distribution Function (CDF) threshold value F(u_T)
    for the Max Form (MF) Contest Success Function (CSF) under continuous distributions.
    """
    if num_captains < 2:
        raise ValueError("Number of captains must be an integer >= 2")

    def mf_residual_function(cdf_val):
        # Term 1: Probability that all n-1 opponents draw a strictly lower utility
        acceptance_term = cdf_val ** (num_captains - 1)

        # Term 2: Expected probability of winning from rejection
        num_a = 1 - cdf_val ** (2 * num_captains - 1)
        den_a = 1 - cdf_val ** num_captains
        part_a = num_a / den_a

        num_b = 1 - cdf_val ** num_captains
        den_b = num_captains * (1 - cdf_val)
        part_b = num_b / den_b

        rejection_term = (1 / (num_captains - 1)) * (part_a - part_b)

        return acceptance_term - rejection_term

    # Search in [0, 1 - epsilon] to avoid division by zero at cdf_val=1
    return brentq(mf_residual_function, 0, 1 - 1e-9)


def calculate_continuous_uniform_pf_threshold(num_captains, lower_bound, upper_bound):
    """
    Main entry point for solving the integral equation for continuous UNIFORM distributions
    specifically with respect to the Power Form (PF) Contest Success Function.
    Normalizes the interval if necessary and selects the appropriate integration solver.
    """
    # Normalization Logic to [a/b, 1]
    if upper_bound > 1.0:
        scale_factor = upper_bound
        normalized_lower = lower_bound / scale_factor
        normalized_upper = 1.0

        normalized_threshold = calculate_continuous_uniform_pf_threshold(
            num_captains, normalized_lower, normalized_upper
        )

        if normalized_threshold is None:
            return None
        return normalized_threshold * scale_factor

    # Solver Selection
    if num_captains <= 20:
        return _solve_pf_uniform_exact_small_competition(num_captains, lower_bound, upper_bound)
    else:
        return _solve_pf_uniform_approx_large_competition(num_captains, lower_bound, upper_bound)


def _solve_pf_uniform_exact_small_competition(num_captains, lower_bound, upper_bound):
    """Internal Exact Solver (Irwin-Hall) for Power Form (PF) continuous UNIFORM distributions with small competition sizes."""
    if num_captains < 2:
        raise ValueError("Number of captains must be >= 2")
    dimensions = num_captains - 1

    def irwin_hall_pdf(x):
        if x < 0 or x > dimensions: return 0.0
        val = 0.0
        k_limit = int(math.floor(x))
        for k in range(k_limit + 1):
            term = comb(dimensions, k) * ((x - k) ** (dimensions - 1))
            val += term if k % 2 == 0 else -term
        return val / math.factorial(dimensions - 1)

    def safe_log_term(S, T):
        if S < 1e-13: return 0.0
        if T + S <= 0: return -1e9
        return S * (np.log(upper_bound + S) - np.log(T + S))

    integration_points = list(range(1, dimensions))

    def calculate_LHS(T):
        def integrand(y):
            S_real = (num_captains - 1) * lower_bound + (upper_bound - lower_bound) * y
            denom = T + S_real
            if abs(denom) < 1e-13: return 0.0
            return (T / denom) * irwin_hall_pdf(y)

        val, _ = quad(integrand, 0, dimensions, points=integration_points)
        return val

    def calculate_I_Same(T):
        def integrand(y):
            S_real = (num_captains - 1) * lower_bound + (upper_bound - lower_bound) * y
            val_inner = (upper_bound - T) - safe_log_term(S_real, T)
            return val_inner * irwin_hall_pdf(y)

        val, _ = quad(integrand, 0, dimensions, points=integration_points)
        return val * ((upper_bound - lower_bound) ** dimensions)

    def equation_residual(T):
        lhs = calculate_LHS(T)
        i_same = calculate_I_Same(T)

        vol_total = (upper_bound - lower_bound) ** dimensions
        i_diff = (vol_total * (upper_bound - T) - i_same) / (num_captains - 1)

        if abs(upper_bound - T) < 1e-9:
            coef1 = (num_captains - 1) * ((upper_bound - lower_bound) ** (num_captains - 2))
        else:
            coef1 = ((upper_bound - lower_bound) ** dimensions - (T - lower_bound) ** dimensions) / (upper_bound - T)

        coef2 = ((T - lower_bound) ** dimensions) / (upper_bound - lower_bound)

        denom_main = (upper_bound - lower_bound) ** num_captains - (T - lower_bound) ** num_captains
        if abs(denom_main) < 1e-13:
            main_mult = 1e12
        else:
            main_mult = ((upper_bound - lower_bound) ** 2) / denom_main

        rhs = main_mult * (coef1 * i_diff + coef2 * i_same)
        return lhs - rhs

    return _robust_root_search(equation_residual, lower_bound, upper_bound)


def _solve_pf_uniform_approx_large_competition(num_captains, lower_bound, upper_bound):
    """Internal Approximation Solver (Normal Distribution) for Power Form (PF) continuous UNIFORM distributions with large competition sizes."""
    dimensions = num_captains - 1
    mu = dimensions * 0.5
    sigma = math.sqrt(dimensions / 12.0)

    def approx_pdf(y):
        return norm.pdf(y, loc=mu, scale=sigma)

    def safe_log_term(S, T):
        if S < 1e-13: return 0.0
        if T + S <= 0: return -1e9
        return S * (np.log(upper_bound + S) - np.log(T + S))

    def calculate_LHS(T):
        def integrand(y):
            S_real = (num_captains - 1) * lower_bound + (upper_bound - lower_bound) * y
            denom = T + S_real
            if abs(denom) < 1e-13: return 0.0
            return (T / denom) * approx_pdf(y)

        val, _ = quad(integrand, 0, dimensions)
        return val

    def calculate_I_Same(T):
        def integrand(y):
            S_real = (num_captains - 1) * lower_bound + (upper_bound - lower_bound) * y
            val_inner = (upper_bound - T) - safe_log_term(S_real, T)
            return val_inner * approx_pdf(y)

        val, _ = quad(integrand, 0, dimensions)
        return val * ((upper_bound - lower_bound) ** dimensions)

    def equation_residual(T):
        lhs = calculate_LHS(T)
        i_same = calculate_I_Same(T)

        vol_total = (upper_bound - lower_bound) ** dimensions
        i_diff = (vol_total * (upper_bound - T) - i_same) / (num_captains - 1)

        if abs(upper_bound - T) < 1e-9:
            coef1 = (num_captains - 1) * ((upper_bound - lower_bound) ** (num_captains - 2))
        else:
            coef1 = ((upper_bound - lower_bound) ** dimensions - (T - lower_bound) ** dimensions) / (upper_bound - T)

        coef2 = ((T - lower_bound) ** dimensions) / (upper_bound - lower_bound)

        denom_main = (upper_bound - lower_bound) ** num_captains - (T - lower_bound) ** num_captains
        if abs(denom_main) < 1e-13:
            main_mult = 1e12
        else:
            main_mult = ((upper_bound - lower_bound) ** 2) / denom_main

        rhs = main_mult * (coef1 * i_diff + coef2 * i_same)
        return lhs - rhs

    return _robust_root_search(equation_residual, lower_bound, upper_bound)


def _robust_root_search(residual_func, lower_bound, upper_bound):
    """Shared helper: Scans the interval to find a sign change, then executes Brent's method."""
    steps = 40
    margin = (upper_bound - lower_bound) * 0.005
    xs = np.linspace(lower_bound + margin, upper_bound - margin, steps)

    signs = []
    for x in xs:
        val = residual_func(x)
        signs.append(np.sign(val))

    bracket = None
    for i in range(len(signs) - 1):
        if signs[i] * signs[i + 1] < 0:
            bracket = (xs[i], xs[i + 1])
            break

    if bracket is None:
        return None

    try:
        return brentq(residual_func, bracket[0], bracket[1])
    except ValueError:
        return None
