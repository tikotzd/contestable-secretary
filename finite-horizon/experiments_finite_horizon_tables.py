import numpy as np
from discrete_solver import FiniteHorizonDiscreteDP
from continuous_solver import FiniteHorizonContinuousDP


def reproduce_table_3():
    """
    Reproduces Table 3: Pre-acceptance threshold utilities for each captain
    with 7 remaining turns. Evaluated for n=3 captains and t=7 turns,
    under uniform distribution over integers {0, 1, ..., 10}.
    """
    n_captains = 3
    num_turns = 7
    # Utilities are {0, 1, ..., 10}, so m=11 discrete values
    num_utilities = 11
    # Probability vector padded with 0.0 at index 0
    probabilities = [0.0] + [1.0 / 11.0] * 11

    dp = FiniteHorizonDiscreteDP(n_captains, num_utilities, num_turns, probabilities)
    pre_thresh, _ = dp.run_algorithm()

    print("--- Table 3: Pre-acceptance thresholds (Discrete Uniform {0..10}) ---")
    print(f"{'Turns (tau)':<12} | {'alpha=1 (c_1)':<14} | {'alpha=2 (c_2)':<14} | {'alpha=3 (c_3)':<14}")
    print("-" * 63)

    for tau in range(num_turns, 0, -1):
        # Subtract 1 to map 1-indexed DP threshold back to the actual {0..10} utility
        t1 = int(pre_thresh[tau, 1]) - 1
        t2 = int(pre_thresh[tau, 2]) - 1
        t3 = int(pre_thresh[tau, 3]) - 1

        print(f"tau={tau:<8} | {t1:<14} | {t2:<14} | {t3:<14}")
    print("\n")


def reproduce_table_4():
    """
    Reproduces Table 4: Post-acceptance thresholds evaluated for n=3 captains
    and t=7 turns, under uniform distribution over integers {0, 1, ..., 10}.
    Columns display specific states mapping active pool index alpha,
    total acceptors l, and max holders K.
    """
    n_captains = 3
    num_turns = 7
    num_utilities = 11
    probabilities = [0.0] + [1.0 / 11.0] * 11

    dp = FiniteHorizonDiscreteDP(n_captains, num_utilities, num_turns, probabilities)
    _, post_thresh = dp.run_algorithm()

    print("--- Table 4: Post-acceptance thresholds (Discrete Uniform {0..10}) ---")
    print(
        f"{'Turns (tau)':<12} | {'l=1, alpha=1, K=1':<17} | {'l=1, alpha=2, K=1':<17} | {'l=2, alpha=1, K=2':<17} | {'l=2, alpha=1, K=1':<17}")
    print("-" * 89)

    for tau in range(num_turns, 0, -1):
        # Extract the specific combinations shown in Table 4 and subtract 1 to map to {0..10}
        col1 = int(post_thresh[tau, 1, 1, 1]) - 1  # alpha=1, K=1, l=1
        col2 = int(post_thresh[tau, 2, 1, 1]) - 1  # alpha=2, K=1, l=1
        col3 = int(post_thresh[tau, 1, 2, 2]) - 1  # alpha=1, K=2, l=2
        col4 = int(post_thresh[tau, 1, 1, 2]) - 1  # alpha=1, K=1, l=2

        print(f"tau={tau:<8} | {col1:<17} | {col2:<17} | {col3:<17} | {col4:<17}")
    print("\n")


def reproduce_table_5():
    """
    Reproduces Table 5: Pre-acceptance threshold utilities u_{tau,i} for each captain c_i
    with tau remaining turns. Evaluated for n=3 captains and t=7 turns,
    under a continuous uniform distribution U[0, 10].
    """
    n_captains = 3
    num_turns = 7

    # The inverse CDF for a Uniform[0, 10] distribution is simply 10 * p
    def inverse_cdf_uniform_0_to_10(p: float) -> float:
        return 10.0 * p

    dp_cont = FiniteHorizonContinuousDP(n_captains, num_turns, inverse_cdf_uniform_0_to_10)
    thresholds = dp_cont.run_algorithm()

    print("--- Table 5: Pre-acceptance thresholds (Continuous Uniform U[0,10]) ---")
    print(f"{'Turns (tau)':<12} | {'Captain c_1':<14} | {'Captain c_2':<14} | {'Captain c_3':<14}")
    print("-" * 63)

    for tau in range(num_turns, 0, -1):
        c1 = thresholds[tau, 1]
        c2 = thresholds[tau, 2]
        c3 = thresholds[tau, 3]

        print(f"tau={tau:<8} | {c1:<14.4f} | {c2:<14.4f} | {c3:<14.4f}")
    print("\n")


# ==========================================
#  Execution Block
# ==========================================
if __name__ == "__main__":
    reproduce_table_3()
    reproduce_table_4()
    reproduce_table_5()