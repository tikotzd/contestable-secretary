import numpy as np
from typing import Callable


class FiniteHorizonContinuousDP:
    """
    Dynamic Programming implementation for the Finite-Horizon Contestable Secretary Problem
    under ANY continuous utility distribution and the Max Form (MF) Contest Success Function.

    This class corresponds to Algorithm 3, computing the exact pre-acceptance thresholds
    for all captains across a finite temporal horizon using the distribution's inverse CDF.
    """

    def __init__(self, num_captains: int, num_turns: int, inverse_cdf: Callable[[float], float]):
        """
        Initializes the dynamic programming environment for a general continuous distribution.

        :param num_captains: Total number of captains (n)
        :param num_turns: Total number of rounds per captain (t)
        :param inverse_cdf: A callable representing the inverse Cumulative Distribution Function F^{-1}(p).
                            It must accept a probability p in [0, 1] and return a utility value u.
        """
        if num_captains < 2 or num_turns < 1:
            raise ValueError("Game requires at least 2 captains and 1 round.")

        self.n = num_captains
        self.t = num_turns
        self.inverse_cdf = inverse_cdf

    def run_algorithm(self) -> np.ndarray:
        """
        Executes the primary backward induction DP framework to solve the SPNE.

        The algorithm solves the probabilities in the CDF space F(u) in [0, 1],
        and maps the optimal probability threshold back to the utility domain
        using the provided inverse_cdf function.

        :return: A 2D numpy array `thresholds` of shape (t + 1, n + 1) where
                 thresholds[tau][i] represents the pre-acceptance utility threshold
                 for captain i with tau turns remaining.
        """
        # win_probs tracks the expected win probability of each captain from the current game state.
        # Initialized to the terminal Forced Assignment phase, where all have an equal 1/n chance to win.
        win_probs = [1.0 / self.n for _ in range(self.n)]

        # Matrix to store the utility threshold values: [tau][captain_index_i]
        thresholds = np.full((self.t + 1, self.n + 1), np.nan)

        # Iterate backwards chronologically: from round tau=1 (last active round) up to tau=t.
        # Within each round, iterate backwards through the absolute captain index: n down to 1.
        for tau in range(1, self.t + 1):
            for i in range(self.n, 0, -1):
                idx = i - 1  # 0-based array index for captain c_i

                # sigma_tau_i: The maximum cumulative number of draws remaining for ALL OPPONENTS
                # if c_i chooses to accept an agent at this specific turn.
                sigma_tau_i = tau * (i - 1) + (tau + 1) * (self.n - i)

                # The Indifference Principle (Zero-crossing of advantage):
                # E[Win | Reject] = win_probs[idx]
                # E[Win | Accept T] = F(T)^(sigma_tau_i)
                # Optimal CDF Threshold F(u_T) is where they intersect: F(u_T)^(sigma_tau_i) = win_probs[idx]
                cdf_threshold = win_probs[idx] ** (1.0 / sigma_tau_i)

                # Remap the CDF threshold back to the actual utility domain using the Inverse CDF
                u_threshold = self.inverse_cdf(cdf_threshold)
                thresholds[tau, i] = u_threshold

                # State Update: Calculate the new expected win probabilities for the state
                # strictly BEFORE c_i evaluates their draw.
                new_win_probs = [0.0 for _ in range(self.n)]

                for l in range(1, self.n + 1):
                    l_idx = l - 1

                    if l == i:
                        # For the acting captain c_i:
                        # Integral of F(u)^(sigma_tau_i) from u_T to b, plus (Prob Reject * win_probs[idx])
                        new_win_probs[l_idx] = (1.0 - cdf_threshold ** (sigma_tau_i + 1)) / (
                                    sigma_tau_i + 1) + cdf_threshold * win_probs[l_idx]
                    else:
                        # For waiting opponents c_l:
                        # tau_l is the specific number of draws c_l has left to beat c_i's accepted utility.
                        tau_l = tau if l < i else tau + 1

                        # Exact closed form of the integral for opponent c_l winning if c_i accepts.
                        prob_l_wins_if_i_accepts = (tau_l / sigma_tau_i) * (
                                    1.0 - cdf_threshold - (1.0 - cdf_threshold ** (sigma_tau_i + 1)) / (
                                        sigma_tau_i + 1))

                        new_win_probs[l_idx] = prob_l_wins_if_i_accepts + cdf_threshold * win_probs[l_idx]

                # Lock in the updated game state for the preceding backward induction step
                win_probs = new_win_probs

        return thresholds
