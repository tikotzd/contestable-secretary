import numpy as np
import matplotlib.pyplot as plt
from continuous_solver import FiniteHorizonContinuousDP
from discrete_solver import FiniteHorizonDiscreteDP


def reproduce_figure_7():
    """
    Reproduces Figure 7: Pre-acceptance thresholds for the final three turns
    (the last three captains to act at tau=1) as a function of n,
    given a uniform continuous distribution U[0, 10].
    """
    n_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 22, 25, 30, 37, 47, 60, 75, 90, 120, 170, 250, 400, 650, 1000]

    last_turn_thresholds = []  # Captain i = n
    second_last_thresholds = []  # Captain i = n - 1
    third_last_thresholds = []  # Captain i = n - 2 (only for n >= 3)

    # Inverse CDF for Uniform[0, 10]
    def inverse_cdf_uniform_0_to_10(p: float) -> float:
        return 10.0 * p

    print("Calculating thresholds for Figure 7... (This is optimized by evaluating only t=1)")

    for n in n_values:
        # Since we only care about the final turn (tau=1) for all captains,
        # initializing the game with num_turns=1 is mathematically sufficient
        # and significantly speeds up the computation for large n.
        dp = FiniteHorizonContinuousDP(num_captains=n, num_turns=1, inverse_cdf=inverse_cdf_uniform_0_to_10)
        thresholds = dp.run_algorithm()

        # tau=1 is the final round for all captains.
        # The captains play in order, so the absolute last turn is captain n.
        last_turn_thresholds.append(thresholds[1, n])
        second_last_thresholds.append(thresholds[1, n - 1])

        if n >= 3:
            third_last_thresholds.append(thresholds[1, n - 2])
        else:
            # Mathematical edge case: for n=2, there is no third-to-last turn.
            # np.nan ensures matplotlib simply skips this point without breaking the line.
            third_last_thresholds.append(np.nan)

        print(f"Calculated for n={n}")

    # Set up the plot styling to match the thesis
    plt.figure(figsize=(9, 6))

    # Plotting the three lines
    plt.plot(n_values, third_last_thresholds, color='#e41a1c', marker='o', markersize=5,
             linestyle='-', label='Third-to-Last Turn')
    plt.plot(n_values, second_last_thresholds, color='#000080', marker='s', markersize=5,
             linestyle='-', label='Second-to-Last Turn')
    plt.plot(n_values, last_turn_thresholds, color='#4daf4a', marker='^', markersize=5,
             linestyle='-', label='Last Turn')

    # Axis formatting
    plt.xscale('log')
    plt.xlabel('n (Logarithmic Scale)', fontsize=12)
    plt.ylabel('Pre-Acceptance Threshold', fontsize=12)

    # Grid and Legend
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc='lower right', fontsize=11, framealpha=1, edgecolor='black')

    plt.title('Pre-acceptance thresholds for the final three turns (U[0,10])', fontsize=13)
    plt.tight_layout()
    plt.show()


def reproduce_figure_8():
    """
    Reproduces Figure 8: Post-acceptance thresholds of the last active captain's
    last turn (tau=1, alpha=n-l) for specific (K, l) configurations as a function of n.
    Evaluated under a discrete uniform distribution over integers {0, 1, ..., 100}.
    """
    n_values = list(range(2, 21))  # n from 2 to 20

    # Utilities are {0, 1, ..., 100}, meaning m=101 discrete values
    num_utilities = 101
    # Probability vector padded with 0.0 at index 0 (1-indexed for the DP)
    probabilities = [0.0] + [1.0 / 101.0] * 101

    # Lists to store the plotted thresholds
    k2_l2_thresholds = []  # (K=2, l=2)
    k1_l1_thresholds = []  # (K=1, l=1)
    k1_l2_thresholds = []  # (K=1, l=2)

    print("Calculating thresholds for Figure 8... (Optimized by evaluating only t=1)")

    for n in n_values:
        # Initialize the DP with t=1 since we only need the final turn's state
        dp = FiniteHorizonDiscreteDP(num_captains=n, num_utilities=num_utilities,
                                     num_turns=1, probabilities=probabilities)
        _, post_thresh = dp.run_algorithm()

        # 1. Configuration (K=1, l=1)
        # alpha = n - l = n - 1
        val_k1_l1 = int(post_thresh[1, n - 1, 1, 1]) - 1  # Subtract 1 to map to {0..100}
        k1_l1_thresholds.append(val_k1_l1)

        # 2 & 3. Configurations with l=2
        # Requires at least 3 captains total for an active captain to exist after 2 acceptances
        if n >= 3:
            # alpha = n - l = n - 2
            val_k2_l2 = int(post_thresh[1, n - 2, 2, 2]) - 1
            val_k1_l2 = int(post_thresh[1, n - 2, 1, 2]) - 1

            k2_l2_thresholds.append(val_k2_l2)
            k1_l2_thresholds.append(val_k1_l2)
        else:
            k2_l2_thresholds.append(np.nan)
            k1_l2_thresholds.append(np.nan)

        print(f"Calculated for n={n}")

    # Set up the plot styling to match the thesis
    plt.figure(figsize=(9, 6))

    # Plotting the three lines with distinct markers and dashed lines
    plt.plot(n_values, k2_l2_thresholds, color='#e41a1c', marker='o', markersize=6,
             linestyle='--', label=r'$(K=2, \ell=2)$')
    plt.plot(n_values, k1_l1_thresholds, color='#000080', marker='s', markersize=6,
             linestyle='--', label=r'$(K=1, \ell=1)$')
    plt.plot(n_values, k1_l2_thresholds, color='#4daf4a', marker='^', markersize=6,
             linestyle='--', label=r'$(K=1, \ell=2)$')

    # Axis formatting
    plt.xlabel('Competition Size (n)', fontsize=12)
    plt.ylabel('Post-Acceptance Threshold', fontsize=12)
    plt.xticks(range(2, 21, 2))  # Tick every 2 units to match the thesis

    # Grid and Legend
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc='lower right', fontsize=12, framealpha=1, edgecolor='black')

    plt.title('Post-acceptance thresholds of the last active captain (Discrete U{0..100})', fontsize=13)
    plt.tight_layout()
    plt.show()


def reproduce_figure_9():
    """
    Reproduces Figure 9: Pre-acceptance thresholds evaluated for n=3, t=6
    under four distinct discrete probability distributions over {0, 1, ..., 100}.
    Demonstrates the impact of stochastic dominance (or lack thereof) on optimal thresholds.
    """
    from discrete_solver import FiniteHorizonDiscreteDP

    n_captains = 3
    num_turns = 6
    num_utilities = 101  # Utilities {0, 1, ..., 100}

    # 1. Benchmark: Uniform [1/101] * 101
    dist_benchmark = [1.0 / 101.0] * 101

    # 2. Mixed Shift (F_new crosses F_old)
    dist_mixed = ([1.0 / 101.0] * 61 +
                  [1.5 / 101.0] * 10 +
                  [0.5 / 101.0] * 20 +
                  [1.5 / 101.0] * 10)

    # 3. Shifted to lower utilities: F_new >= F_old
    dist_above = ([1.0 / 101.0] * 61 +
                  [1.5 / 101.0] * 20 +
                  [0.5 / 101.0] * 20)

    # 4. Shifted to higher utilities: F_new <= F_old
    dist_below = ([1.0 / 101.0] * 61 +
                  [0.5 / 101.0] * 20 +
                  [1.5 / 101.0] * 20)

    # Configuration for plotting: (Name, Distribution, Color, LineStyle, Marker, Label)
    plot_configs = [
        ("Benchmark", dist_benchmark, 'black', '-', 'o', 'Benchmark'),
        ("Above", dist_above, '#000080', '--', 's', r'$F_{new} \geq F_{old}$'),
        ("Below", dist_below, '#e41a1c', '--', '^', r'$F_{new} \leq F_{old}$'),
        ("Mixed", dist_mixed, '#4daf4a', '--', 'd', 'Mixed Shift')
    ]

    # Generate the exact x-axis state order found in the thesis:
    # (1,3), (1,2), (1,1), (2,3), (2,2), (2,1) ... (6,3), (6,2), (6,1)
    states = [(tau, alpha) for tau in range(1, num_turns + 1) for alpha in range(n_captains, 0, -1)]
    x_labels = [f"({tau},{alpha})" for tau, alpha in states]
    x_positions = range(len(states))

    plt.figure(figsize=(12, 6))

    print("Calculating thresholds for Figure 9...")

    for name, dist, color, ls, marker, label in plot_configs:
        # Pad the distribution with 0.0 at index 0 for the 1-indexed DP engine
        probabilities = [0.0] + dist

        dp = FiniteHorizonDiscreteDP(num_captains=n_captains, num_utilities=num_utilities,
                                     num_turns=num_turns, probabilities=probabilities)
        pre_thresh, _ = dp.run_algorithm()

        # Extract the thresholds in the defined state order
        y_values = []
        for tau, alpha in states:
            # Map the 1-indexed DP result back to {0..100}
            u_threshold = int(pre_thresh[tau, alpha]) - 1
            y_values.append(u_threshold)

        # The benchmark line is thicker and solid
        line_width = 2.5 if name == "Benchmark" else 1.5

        plt.plot(x_positions, y_values, color=color, linestyle=ls, marker=marker,
                 markersize=6, linewidth=line_width, label=label)

        print(f"Calculated distribution: {name}")

    # Axis formatting
    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel(r'State ($\tau, \alpha$)', fontsize=12)
    plt.ylabel('Pre-Acceptance Threshold', fontsize=12)

    # Grid and Legend
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc='lower right', fontsize=12, framealpha=1, edgecolor='black')

    plt.title('Pre-acceptance thresholds for n=3, t=6 under probability shifts', fontsize=13)
    plt.tight_layout()
    plt.show()


# ==========================================
#  Execution Block
# ==========================================
if __name__ == "__main__":
    # reproduce_figure_7()
    # reproduce_figure_8()
    reproduce_figure_9()
