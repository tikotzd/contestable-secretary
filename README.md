# The Contestable Secretary Problem

This repository contains the official Python implementation and supplementary code for the Master's thesis: **The Contestable Secretary Problem**, authored by Daniel Tikotzky (Bar-Ilan University, 2026), under the supervision of Prof. Noa Agmon.

## Overview

This repository provides the computational frameworks and solvers used to derive the Subgame-Perfect Nash Equilibrium (SPNE) strategy profiles for a competitive multi-agent optimal stopping problem. In this model, $n$ **captains** sequentially draft agents from an online stream to represent them in a subsequent contest. Winning probabilities in the contest are determined by Contest Success Functions (CSFs)—specifically the **Max Form (MF)** and **Power Form (PF)**.

The codebase is divided into three primary environments analyzed in the thesis:
1. **The Infinite-Horizon Model:** Features an immediate forced assignment blocker upon the first acceptance.
2. **The Finite-Horizon Model:** Features a strict, predefined turn limit for each captain, solved via backward induction dynamic programming.
3. **The Contestable k-Secretary Model:** An advanced extension where captains must assemble teams of $k$ agents.

## Repository Structure

The repository is organized into three distinct directories corresponding to the models evaluated in the thesis. Each directory maintains a strict separation between the core mathematical solvers and the experimental scripts used to reproduce the thesis figures and tables.

### 1. `infinite-horizon/`
Contains the stationary threshold solvers for continuous and discrete utility distributions.
* **`continuous_solvers.py`**: Solves the integral equations to find optimal thresholds for continuous distributions (unified for MF and PF CSFs).
* **`discrete_solvers.py`**: Calculates the net advantage threshold function $G(T)$ and employs a binary search algorithm to locate the optimal acceptance threshold for discrete distributions.
* **`experiments_figure_3.py`**: Script to reproduce Figure 3 (Threshold values as a function of $n$ for MF and PF CSFs given a uniform continuous distribution $U[0,1]$).
* **`experiments_tables.py`**: Script to systematically verify and reproduce the threshold sensitivity analyses found in Tables 6–14.

### 2. `finite-horizon/`
Contains the dynamic programming (DP) algorithms required to solve the state-dependent, non-stationary thresholds.
* **`continuous_solver.py`**: Contains the `FiniteHorizonContinuousDP` class. Implements backward induction to calculate exact pre-acceptance thresholds across the temporal horizon for continuous distributions using the Inverse CDF.
* **`discrete_solver.py`**: Contains the `FiniteHorizonDiscreteDP` class. Evaluates both pre-acceptance and post-acceptance phases across all valid parameter states $(\tau, \alpha, K, l)$.
* **`experiments_finite_horizon_figures.py`**: Script to reproduce Figures 7, 8, and 9, demonstrating terminal-turn thresholds and the impacts of stochastic dominance.
* **`experiments_finite_horizon_tables.py`**: Script to reproduce Tables 3, 4, and 5, tracking pre- and post-acceptance thresholds dynamically over $t=7$ turns.

### 3. `contestable-k-secretary/`
Explores the mathematical and combinatorial complexities of multi-agent team drafting.
* **`p1_threshold_plotting.py`**: Script to reproduce Figure 11, calculating and plotting the threshold probability $T_{p_1}$ as a function of the utility ratio $\alpha = u_2 / u_1$ for both CSFs when $k=2$.

## Dependencies and Installation

This codebase is built using standard scientific Python libraries. To run the solvers and generate the plots, ensure you have the following installed:

* `Python 3.8+`
* `numpy`
* `scipy`
* `matplotlib`

You can install the required packages using pip:
```bash
pip install numpy scipy matplotlib
```

## Usage and Reproduction

To reproduce any specific figure or table from the thesis, simply run the corresponding experiment file. The scripts are fully self-contained and will automatically call the necessary solvers.

For example, to generate Figure 3 from the Infinite-Horizon chapter:
```bash
cd infinite-horizon
python experiments_figure_3.py
```

To view the dynamically computed finite-horizon matrices (Tables 3, 4, and 5):
```bash
cd finite-horizon
python experiments_finite_horizon_tables.py
```
