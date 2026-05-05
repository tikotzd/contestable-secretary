import matplotlib.pyplot as plt
from continuous_solvers import calculate_continuous_mf_cdf_threshold, calculate_continuous_uniform_pf_threshold


def plot_figure_3():
    """
    Reproduces Figure 3 from the thesis: Threshold values as a function of n
    for MF and PF CSFs given uniform continuous distribution U[0,1].
    """
    n_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 70, 100, 200, 400, 1000]

    mf_thresholds = []
    pf_thresholds = []

    print("Calculating thresholds for Figure 3... (This may take a moment for large n)")
    for n in n_values:
        # MF threshold for U[0,1] directly maps to the CDF value F(u) = u
        mf_val = calculate_continuous_mf_cdf_threshold(n)
        mf_thresholds.append(mf_val)

        # PF threshold for continuous U[0,1]
        pf_val = calculate_continuous_uniform_pf_threshold(n, 0.0, 1.0)
        pf_thresholds.append(pf_val)

        print(f"Calculated for n={n}: MF ≈ {mf_val:.4f}, PF ≈ {pf_val:.4f}")

    # Set up the plot
    plt.figure(figsize=(8, 6))

    # Plot the calculated data points with styling matching the thesis
    plt.plot(n_values, mf_thresholds, color='#e41a1c', marker='o', markersize=5,
             linestyle='-', label='MF threshold values')
    plt.plot(n_values, pf_thresholds, color='#000080', marker='s', markersize=5,
             linestyle='-', label='PF threshold values')

    # Add the horizontal asymptotic limit lines
    plt.axhline(y=1.0, color='#e41a1c', linestyle='--', alpha=0.8)
    plt.text(20, 1.03, r'Limit $\rightarrow 1$', color='#e41a1c', ha='center', fontsize=10)

    plt.axhline(y=0.5, color='#000080', linestyle='--', alpha=0.8)
    plt.text(20, 0.53, r'Limit $\rightarrow 0.5$', color='#000080', ha='center', fontsize=10)

    # Formatting the plot axes and grid
    plt.xscale('log')
    plt.xlabel('n (Logarithmic Scale)', fontsize=11)
    plt.ylabel('$u_T$', fontsize=11)
    plt.ylim(0, 1.1)

    # Grid and Legend
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc='lower right', fontsize=11, framealpha=1, edgecolor='black')

    # Display
    plt.title('Thresholds as a function of n for U[0,1] w.r.t. both CSFs', fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_figure_3()
