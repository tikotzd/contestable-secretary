import numpy as np
import math


class FiniteHorizonDiscreteDP:
    """
    Dynamic Programming implementation for the Finite-Horizon Contestable Secretary Problem
    under discrete utility distributions and the Max Form (MF) Contest Success Function.
    """

    def __init__(self, num_captains, num_utilities, num_turns, probabilities):
        """
        Initializes the dynamic programming environment.

        :param num_captains: Total number of captains (n)
        :param num_utilities: Total number of discrete utility values (m)
        :param num_turns: Total number of rounds per captain (t)
        :param probabilities: 1D array/list of probabilities of length m+1 (index 0 is a dummy pad)
        """
        self.n = num_captains
        self.m = num_utilities
        self.t = num_turns

        # Initialize Probability Vector (1-indexed)
        self.p = np.array(probabilities, dtype=np.float64)

        # Precompute cumulative probabilities: S[a, b] = sum(p_k) from k=a to b
        self.S = np.zeros((self.m + 1, self.m + 1), dtype=np.float64)
        for a in range(1, self.m + 1):
            for b in range(a, self.m + 1):
                self.S[a, b] = np.sum(self.p[a:b + 1])

    def _initialize_matrices(self):
        """
        Allocates the multidimensional tensors required to cache expected winning
        probabilities and optimal thresholds across all reachable states.
        Dimensions typically track: [tau][alpha][K][ell][j]
        (and [d] for positional waiting distance).
        """
        # Expected Probability Tensors
        self.v_acting = np.full((self.t + 1, self.n + 1, self.n + 1, self.n + 1, self.m + 1), np.nan)
        self.v_waiting = np.full((self.t + 1, self.n + 1, self.n + 1, self.n + 1, self.n + 1, self.m + 1), np.nan)
        self.v_tied = np.full((self.t + 1, self.n + 1, self.n + 1, self.n + 1, self.m + 1), np.nan)

        # Threshold Output Matrices
        self.PreThresholds = np.full((self.t + 1, self.n + 1), np.nan)
        self.PostThresholds = np.full((self.t + 1, self.n + 1, self.n + 1, self.n + 1), np.nan)

    def _safe_pow(self, base, exp):
        """Safely evaluates powers, explicitly defining 0^0 = 1 for probability boundary conditions."""
        if base == 0 and exp == 0:
            return 1.0
        return base ** exp

    def _get_S(self, a, b):
        """Fetches the cumulative probability S_{a,b}. Returns 0.0 for invalid intervals."""
        if a > b or b == 0:
            return 0.0
        return self.S[a, b]

    def _get_next_turn_state(self, tau, alpha, ell, action):
        """
        Determines the exact state parameters (tau, alpha) for the subsequent draw
        based on the current action (Accept or Reject).
        """
        active_queue_size = self.n - ell

        # Round transition trigger: current captain is the last in the active queue
        if alpha == active_queue_size:
            next_tau = tau - 1
            next_alpha_A = 1
            next_alpha_R = 1
        else:
            next_tau = tau
            if action == "ACCEPT":
                next_alpha_A = alpha
                next_alpha_R = None
            elif action == "REJECT":
                next_alpha_A = None
                next_alpha_R = alpha + 1

        return next_tau, next_alpha_A, next_alpha_R

    def _compute_random_assignment(self):
        """
        Calculates and populates the expected probabilities of winning for all game states
        immediately preceding the random assignment phase (tau = 0).
        """
        terminal_tau = 0

        # Base Case: No captain has accepted any agent. All n captains draw simultaneously.
        self.v_waiting[terminal_tau, :, 0, 0, :, 0] = 1.0 / self.n
        self.v_acting[terminal_tau, :, 0, 0, 0] = 1.0 / self.n

        # Edge Case: ell = n. The game effectively terminates early as all captains are assigned.
        for K in range(1, self.n + 1):
            for j in range(1, self.m + 1):
                self.v_tied[terminal_tau, :, K, self.n, j] = 1.0 / K
                self.v_tied[1:self.t + 1, 1, K, self.n, j] = 1.0 / K
                self.v_acting[:, :, K, self.n, :] = 0.0
                self.v_waiting[:, :, K, self.n, :, :] = 0.0

        # Standard Random Assignment (ell < n)
        for K in range(0, self.n):
            for ell in range(self.n - 1, K - 1, -1):
                for j in range(1, self.m + 1):
                    # Unassigned Captains (Waiting Queue)
                    if self.n - ell > 0:
                        term1 = (1.0 - self._safe_pow(self._get_S(1, j), self.n - ell)) / (self.n - ell)

                        term2 = 0.0
                        for k in range(0, self.n - 1 - ell + 1):
                            comb_val = math.comb(self.n - 1 - ell, k)
                            prob_val = self._safe_pow(self.p[j], k + 1)
                            cum_val = self._safe_pow(self._get_S(1, j - 1), self.n - 1 - ell - k)
                            term2 += (1.0 / (k + K + 1)) * comb_val * prob_val * cum_val

                        self.v_waiting[terminal_tau, :, K, ell, :, j] = term1 + term2
                        self.v_acting[terminal_tau, :, K, ell, j] = term1 + term2

                    # Tied Captains (Already Assigned Max Utility)
                    v_tied_val = 0.0
                    for k in range(0, self.n - ell + 1):
                        if k + K > 0:
                            comb_val = math.comb(self.n - ell, k)
                            prob_val = self._safe_pow(self.p[j], k)
                            cum_val = self._safe_pow(self._get_S(1, j - 1), self.n - ell - k)
                            v_tied_val += (1.0 / (k + K)) * comb_val * prob_val * cum_val

                    self.v_tied[terminal_tau, :, K, ell, j] = v_tied_val

    def _get_v_waiting(self, tau, alpha, K, ell, d, j):
        """
        Unified interface for retrieving future expected probability values.
        If distance is 0, the token passes immediately back to the current captain,
        evaluating directly to the 'acting' tensor.
        """
        if d == 0:
            return self.v_acting[tau, alpha, K, ell, j]
        return self.v_waiting[tau, alpha, K, ell, d, j]

    def _compute_post_threshold(self, tau, alpha, K, ell):
        """Calculates the minimum utility required to accept a tie during the post-acceptance phase."""
        next_tau, next_alpha_A, _ = self._get_next_turn_state(tau, alpha, ell, "ACCEPT")
        _, _, next_alpha_R = self._get_next_turn_state(tau, alpha, ell, "REJECT")

        # Leverage monotonicity properties to optimize search bounds
        if ell + 1 <= self.n and not np.isnan(self.PostThresholds[tau, alpha, K, ell + 1]):
            curr_T_post = int(self.PostThresholds[tau, alpha, K, ell + 1])
        elif K - 1 >= 1 and not np.isnan(self.PostThresholds[tau, alpha, K - 1, ell]):
            curr_T_post = int(self.PostThresholds[tau, alpha, K - 1, ell])
        else:
            curr_T_post = 2

        for j in range(curr_T_post, self.m + 1):
            epowA = self.v_tied[next_tau, next_alpha_A, K + 1, ell + 1, j]
            epowR = self._get_v_waiting(next_tau, next_alpha_R, K, ell, self.n - ell - 1, j)

            if epowA >= epowR:
                self.PostThresholds[tau, alpha, K, ell] = j
                break

        # Fallback to m if no utility warrants acceptance
        if np.isnan(self.PostThresholds[tau, alpha, K, ell]):
            self.PostThresholds[tau, alpha, K, ell] = self.m

    def _compute_expected_probabilities_post(self, tau, alpha, K, ell):
        """Updates the probabilities of winning for all captains during a specific post-acceptance state."""
        next_tau, next_alpha_A, _ = self._get_next_turn_state(tau, alpha, ell, "ACCEPT")
        _, _, next_alpha_R = self._get_next_turn_state(tau, alpha, ell, "REJECT")

        post_T = self.PostThresholds[tau, alpha, K, ell]

        # Dynamic suffix sum accumulators mapping expectations to O(n) per index instead of O(n*m)
        running_sum_tied = 0.0
        running_sum_acting = 0.0
        running_sum_waiting = np.zeros(self.n)

        # Iterate backward to build the suffix sums on the fly
        for j in range(self.m, 1, -1):
            if not np.isnan(post_T) and j >= post_T:
                # Active captain ACCEPTS
                epowR_acting = self._get_v_waiting(next_tau, next_alpha_R, K, ell, self.n - ell - 1, j)

                self.v_acting[tau, alpha, K, ell, j] = (
                        self._get_S(1, j - 1) * epowR_acting +
                        self.p[j] * self.v_tied[next_tau, next_alpha_A, K + 1, ell + 1, j] +
                        running_sum_tied
                )

                self.v_waiting[tau, alpha, K, ell, 1, j] = (
                        self._get_S(1, j - 1) * self.v_acting[next_tau, next_alpha_R, K, ell, j] +
                        self.p[j] * self.v_acting[next_tau, next_alpha_A, K + 1, ell + 1, j] +
                        running_sum_acting
                )

                for d in range(2, self.n - ell):
                    self.v_waiting[tau, alpha, K, ell, d, j] = (
                            self._get_S(1, j - 1) * self._get_v_waiting(next_tau, next_alpha_R, K, ell, d - 1, j) +
                            self.p[j] * self._get_v_waiting(next_tau, next_alpha_A, K + 1, ell + 1, d - 1, j) +
                            running_sum_waiting[d - 1]
                    )

                self.v_tied[tau, alpha, K, ell, j] = (
                        self._get_S(1, j - 1) * self.v_tied[next_tau, next_alpha_R, K, ell, j] +
                        self.p[j] * self.v_tied[next_tau, next_alpha_A, K + 1, ell + 1, j]
                )

            else:
                # Active captain REJECTS
                epowR_acting = self._get_v_waiting(next_tau, next_alpha_R, K, ell, self.n - ell - 1, j)

                self.v_acting[tau, alpha, K, ell, j] = (self._get_S(1, j) * epowR_acting + running_sum_tied)
                self.v_waiting[tau, alpha, K, ell, 1, j] = (
                            self._get_S(1, j) * self.v_acting[next_tau, next_alpha_R, K, ell, j] + running_sum_acting)

                for d in range(2, self.n - ell):
                    self.v_waiting[tau, alpha, K, ell, d, j] = (
                            self._get_S(1, j) * self._get_v_waiting(next_tau, next_alpha_R, K, ell, d - 1, j) +
                            running_sum_waiting[d - 1]
                    )

                self.v_tied[tau, alpha, K, ell, j] = self._get_S(1, j) * self.v_tied[next_tau, next_alpha_R, K, ell, j]

            # Update accumulators for the subsequent iteration (j - 1)
            running_sum_tied += self.p[j] * self.v_tied[next_tau, next_alpha_A, 1, ell + 1, j]
            running_sum_acting += self.p[j] * self.v_acting[next_tau, next_alpha_A, 1, ell + 1, j]

            for d in range(2, self.n - ell):
                running_sum_waiting[d - 1] += self.p[j] * self.v_waiting[next_tau, next_alpha_A, 1, ell + 1, d - 1, j]

    def _compute_pre_threshold(self, tau, alpha):
        """Determines the minimum utility required to trigger the absolute first acceptance in the game."""
        next_tau, next_alpha_A, _ = self._get_next_turn_state(tau, alpha, 0, "ACCEPT")
        _, _, next_alpha_R = self._get_next_turn_state(tau, alpha, 0, "REJECT")

        # Temporal horizon bounds optimization
        if alpha + 1 <= self.n and not np.isnan(self.PreThresholds[tau, alpha + 1]):
            prev_T_pre = int(self.PreThresholds[tau, alpha + 1])
        else:
            prev_T_pre = 2

        for j in range(prev_T_pre, self.m + 1):
            epowA = self.v_tied[next_tau, next_alpha_A, 1, 1, j]
            epowR = self.v_waiting[next_tau, next_alpha_R, 0, 0, self.n - 1, 0]

            if epowA >= epowR:
                self.PreThresholds[tau, alpha] = j
                break

        if np.isnan(self.PreThresholds[tau, alpha]):
            self.PreThresholds[tau, alpha] = self.m

    def _compute_expected_probabilities_pre(self, tau, alpha):
        """Updates winning probabilities evaluated prior to the absolute first acceptance."""
        T_pre = int(self.PreThresholds[tau, alpha])
        next_tau, next_alpha_A, _ = self._get_next_turn_state(tau, alpha, 0, "ACCEPT")
        _, _, next_alpha_R = self._get_next_turn_state(tau, alpha, 0, "REJECT")

        sum_tied = sum(self.p[r] * self.v_tied[next_tau, next_alpha_A, 1, 1, r] for r in range(T_pre, self.m + 1))
        self.v_acting[tau, alpha, 0, 0, 0] = (self._get_S(1, T_pre - 1) * self.v_waiting[
            next_tau, next_alpha_R, 0, 0, self.n - 1, 0]) + sum_tied

        sum_acting = sum(self.p[r] * self.v_acting[next_tau, next_alpha_A, 1, 1, r] for r in range(T_pre, self.m + 1))
        self.v_waiting[tau, alpha, 0, 0, 1, 0] = (self._get_S(1, T_pre - 1) * self.v_acting[
            next_tau, next_alpha_R, 0, 0, 0]) + sum_acting

        for d in range(2, self.n):
            sum_waiting = sum(
                self.p[r] * self.v_waiting[next_tau, next_alpha_A, 1, 1, d - 1, r] for r in range(T_pre, self.m + 1))
            self.v_waiting[tau, alpha, 0, 0, d, 0] = (self._get_S(1, T_pre - 1) * self.v_waiting[
                next_tau, next_alpha_R, 0, 0, d - 1, 0]) + sum_waiting

    def run_algorithm(self):
        """
        Executes the primary backward induction DP framework to solve the SPNE.
        Returns the populated Pre-Acceptance and Post-Acceptance threshold structures.
        """
        self._initialize_matrices()
        self._compute_random_assignment()

        for tau in range(1, self.t + 1):
            for alpha in range(self.n, 0, -1):

                # Topological Sort: State resolution must strictly process states with larger ties
                # prior to processing states with smaller ties to maintain acyclic mathematical dependencies.
                if alpha < self.n:
                    for ell in range(self.n - alpha, 0, -1):
                        for K in range(1, ell + 1):
                            self._compute_post_threshold(tau, alpha, K, ell)
                            self._compute_expected_probabilities_post(tau, alpha, K, ell)

                # Safely compute the empty board (ell=0) once all localized post-acceptances are resolved.
                self._compute_pre_threshold(tau, alpha)
                self._compute_expected_probabilities_pre(tau, alpha)

        return self.PreThresholds, self.PostThresholds
