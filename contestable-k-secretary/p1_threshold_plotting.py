import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def reproduce_figure_11():
    """
    Reproduces Figure 11 from the thesis:
    Probability threshold of p_1 as a function of alpha = u_2 / u_1 for both MF and PF CSFs.
    Evaluated for n=2, m=2, k=2, given that c_1 holds one u_2 agent, c_2 is empty,
    and a u_1 agent has just arrived to c_1.
    """

    # ---------------------------------------------------------
    # 1. Mathematical Functions (Equations 15 and 16)
    # ---------------------------------------------------------

    def G_MF(p1: float) -> float:
        """
        Threshold function G_{k-MF}(1) for the Max Form CSF (Equation 15).
        Returns E[P(c_1, A, u_1)] - E[P(c_1, R, u_1)].
        """
        expected_acceptance = p1

        num = (1 + p1 + p1 ** 2) * (1 + 2 * p1 - p1 ** 2)
        den = 2 * (1 + p1) ** 2
        expected_rejection = num / den

        return expected_acceptance - expected_rejection

    def G_PF(p1: float, alpha: float) -> float:
        """
        Threshold function G_{k-PF}(1) for the Power Form CSF.
        Returns E[P(c_1, A, u_1)] - E[P(c_1, R, u_1)] parameterized by alpha.
        """
        # Expected Acceptance Payoff
        term_a1 = ((1 + alpha) / (3 + alpha)) * (p1 ** 2)
        term_a2 = p1 * (1 - p1)
        term_a3 = ((1 + alpha) / (1 + 3 * alpha)) * ((1 - p1) ** 2)
        expected_acceptance = term_a1 + term_a2 + term_a3

        # Expected Rejection Payoff
        term_r1 = (alpha / (1 + alpha)) * ((p1 ** 3) / (1 + p1))
        term_r2 = (2 * alpha / (1 + 3 * alpha)) * ((p1 + 2 * (p1 ** 2) - 2 * (p1 ** 4)) / ((1 + p1) ** 2))
        term_r3 = ((1 + alpha) / (1 + 3 * alpha)) * ((p1 ** 2) / ((1 + p1) ** 2))
        term_r4 = (1 - 2 * (p1 ** 2) + p1 ** 3) / (2 * (1 + p1))
        expected_rejection = term_r1 + term_r2 + term_r3 + term_r4

        return expected_acceptance - expected_rejection

    # ---------------------------------------------------------
    # 2. Calculation over the Domain of alpha
    # ---------------------------------------------------------

    print("Calculating thresholds for Figure 11...")

    # Generate alpha values logarithmically from 1.01 to 1000
    alpha_values = np.logspace(np.log10(1.01), 3, 25)

    # Calculate the static MF threshold (root of G_MF(p1) = 0)
    # The thesis notes this polynomial resolves to p_1 >= 0.7283
    mf_threshold_val = brentq(G_MF, 0.5, 0.9999)
    mf_thresholds = [mf_threshold_val] * len(alpha_values)

    # Calculate the dynamic PF threshold for each alpha (root of G_PF(p1, alpha) = 0)
    pf_thresholds = []
    for a in alpha_values:
        # We bracket the search between 0.5 and 0.9999 since the theoretical limit is 0.5107
        pf_root = brentq(lambda p: G_PF(p, a), 0.5, 0.9999)
        pf_thresholds.append(pf_root)

    print(f"MF Constant Threshold: {mf_threshold_val:.4f}")

    # ---------------------------------------------------------
    # 3. Plot Generation
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    # Plot MF Threshold (Solid red line)
    plt.plot(alpha_values, mf_thresholds, color='#e41a1c', linestyle='-',
             label=f'MF threshold ({mf_threshold_val:.4f})')

    # Plot PF Thresholds (Solid blue line with square markers)
    plt.plot(alpha_values, pf_thresholds, color='#000080', marker='s', markersize=5,
             linestyle='-', label='PF threshold values')

    # Add horizontal asymptotic limit line for PF
    plt.axhline(y=0.5107, color='#000080', linestyle='--', alpha=0.8)
    plt.text(10, 0.48, r'Limit $\rightarrow 0.5107$', color='#000080', ha='center', fontsize=11)

    # Axis formatting
    plt.xscale('log')
    plt.xlabel(r'$\alpha$ (Logarithmic Scale)', fontsize=12)
    plt.ylabel(r'$T_{p_1}$', fontsize=12)
    plt.ylim(0.45, 1.05)

    # Grid and Legend
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc='upper right', fontsize=12, framealpha=1, edgecolor='black')

    plt.title(r'Probability threshold of $p_1$ as a function of $\alpha = u_2 / u_1$ for both CSFs', fontsize=13)
    plt.tight_layout()
    plt.show()


# ==========================================
#  Execution Block
# ==========================================
if __name__ == "__main__":
    reproduce_figure_11()
