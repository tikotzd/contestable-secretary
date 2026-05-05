import numpy as np
from itertools import product


def calculate_G_MF(target_index, utilities, probabilities, cumulative_probs, num_captains):
    """
    Calculates the net advantage threshold function G(T) = E[P(Accept)] - E[P(Reject)]
    for the Max Form (MF) CSF at a given utility index.
    """
    m_size = len(utilities)
    S_j = cumulative_probs[target_index]
    S_prev = cumulative_probs[target_index - 1] if target_index > 0 else 0.0
    p_j = probabilities[target_index]

    # Edge case: Rejecting the absolute maximum utility leads to an infinite horizon
    if target_index == m_size - 1:
        expected_acceptance = (1 - (1 - p_j) ** num_captains) / (num_captains * p_j)
        expected_rejection = 0.0
        return expected_acceptance - expected_rejection

    # Expected Acceptance Payoff
    expected_acceptance = (S_j ** num_captains - S_prev ** num_captains) / (num_captains * p_j)

    # Expected Rejection Payoff
    if abs(1 - S_j) < 1e-9:
        expected_rejection = 0.0
    else:
        term_a = (1 - S_j ** (2 * num_captains - 1)) / (1 - S_j ** num_captains)
        term_b = (1 - S_j ** num_captains) / (num_captains * (1 - S_j))
        expected_rejection = (1 / (num_captains - 1)) * (term_a - term_b)

    return expected_acceptance - expected_rejection


def calculate_G_PF(target_index, utilities, probabilities, cumulative_probs, precomputed_H, num_captains):
    """
    Calculates the net advantage threshold function G(T) = E[P(Accept)] - E[P(Reject)]
    for the Power Form (PF) CSF at a given utility index.
    """
    m_size = len(utilities)
    S_j = cumulative_probs[target_index]
    expected_acceptance = precomputed_H[target_index]

    # Edge case: Rejecting the absolute maximum utility leads to an infinite horizon
    if target_index == m_size - 1:
        return expected_acceptance

    # Expected Rejection Payoff
    sum_prob_H = sum(probabilities[r] * precomputed_H[r] for r in range(target_index + 1, m_size))

    if abs(1 - S_j) < 1e-9:
        expected_rejection = 0.0
    else:
        term_A = (1 - S_j ** (num_captains - 1)) / (1 - S_j ** num_captains)
        term_B_part1 = (num_captains * S_j ** (num_captains - 1)) / (1.0 - S_j ** num_captains)
        term_B_part2 = 1.0 / (1.0 - S_j)
        term_B = term_B_part1 - term_B_part2

        expected_rejection = (1.0 / (num_captains - 1)) * (term_A + term_B * sum_prob_H)

    return expected_acceptance - expected_rejection


def find_discrete_threshold_binary_search(utilities, probabilities, num_captains, csf_type="MF"):
    """
    Central prototype implementing Algorithm 1.
    Uses a binary search to locate the optimal threshold index where G(T) >= 0.
    Returns the corresponding utility value.
    """
    m_size = len(utilities)
    cumulative_probs = np.cumsum(probabilities)

    # Precomputations for PF to avoid redundant combinatorial evaluation during binary search
    if csf_type == "PF":
        precomputed_H = np.zeros(m_size)
        rival_indices_iter = list(product(range(m_size), repeat=num_captains - 1))
        for j in range(m_size):
            u_j = utilities[j]
            prob_win_sum = 0.0
            for rivals in rival_indices_iter:
                rival_prob = np.prod([probabilities[r_idx] for r_idx in rivals])
                rival_util_sum = sum([utilities[r_idx] for r_idx in rivals])

                denom = u_j + rival_util_sum
                win_share = 1.0 / num_captains if denom == 0 else u_j / denom
                prob_win_sum += rival_prob * win_share
            precomputed_H[j] = prob_win_sum

    # Closure mapping an index to its respective G(T) value
    def evaluate_G(index):
        if csf_type == "MF":
            return calculate_G_MF(index, utilities, probabilities, cumulative_probs, num_captains)
        elif csf_type == "PF":
            return calculate_G_PF(index, utilities, probabilities, cumulative_probs, precomputed_H, num_captains)
        else:
            raise ValueError("Unsupported CSF type. Choose 'MF' or 'PF'.")

    # Fast return for trivial or immediate boundary cases
    if m_size == 1 or evaluate_G(0) >= 0:
        return utilities[0]

    # Standard Binary Search execution mapping directly to Algorithm 1
    start_idx = 1
    end_idx = m_size - 1
    current_T = (start_idx + end_idx) // 2

    while evaluate_G(current_T) < 0 or evaluate_G(current_T - 1) >= 0:
        if evaluate_G(current_T - 1) >= 0:
            end_idx = current_T - 1
        else:
            start_idx = current_T + 1

        # Failsafe boundary constraint
        if start_idx > end_idx:
            current_T = start_idx if start_idx < m_size else end_idx
            break

        current_T = (start_idx + end_idx) // 2

    return utilities[current_T]
