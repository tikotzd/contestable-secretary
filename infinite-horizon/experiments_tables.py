from discrete_solvers import find_discrete_threshold_binary_search


def prove_table_6():
    """Table 6: A strong enough mass shift from u <= u_T to u >= u_T increases u_T (MF)"""
    print("\n--- Proving Table 6 (MF) ---")
    n_captains = 2
    # MF relies purely on ordinal rank. We use 1, 2, 3, 4 to represent u_1, u_2, u_3, u_4
    utilities = [1, 2, 3, 4]

    p_initial = [0.25, 0.25, 0.25, 0.25]
    p_shift_1 = [0.15, 0.15, 0.25, 0.45]
    p_shift_2 = [0.05, 0.05, 0.25, 0.65]

    res_init = find_discrete_threshold_binary_search(utilities, p_initial, n_captains, "MF")
    res_s1 = find_discrete_threshold_binary_search(utilities, p_shift_1, n_captains, "MF")
    res_s2 = find_discrete_threshold_binary_search(utilities, p_shift_2, n_captains, "MF")

    print(f"Initial Values : p={p_initial} -> Threshold: u_{res_init}")
    print(f"Shift 1 (Weak) : p={p_shift_1} -> Threshold: u_{res_s1}")
    print(f"Shift 2 (Strong): p={p_shift_2} -> Threshold: u_{res_s2}")


def prove_table_7():
    """Table 7: A strong enough mass shift from u >= u_T to u <= u_T decreases u_T (MF)"""
    print("\n--- Proving Table 7 (MF) ---")
    n_captains = 2
    utilities = [1, 2, 3, 4]

    p_initial = [0.25, 0.25, 0.25, 0.25]
    p_shift_1 = [0.275, 0.275, 0.25, 0.2]
    p_shift_2 = [0.3, 0.3, 0.25, 0.15]

    res_init = find_discrete_threshold_binary_search(utilities, p_initial, n_captains, "MF")
    res_s1 = find_discrete_threshold_binary_search(utilities, p_shift_1, n_captains, "MF")
    res_s2 = find_discrete_threshold_binary_search(utilities, p_shift_2, n_captains, "MF")

    print(f"Initial Values : p={p_initial} -> Threshold: u_{res_init}")
    print(f"Shift 1 (Weak) : p={p_shift_1} -> Threshold: u_{res_s1}")
    print(f"Shift 2 (Strong): p={p_shift_2} -> Threshold: u_{res_s2}")


def prove_table_8():
    """Table 8: A strong enough rightwards-shifting of probability mass increases u_T (PF)"""
    print("\n--- Proving Table 8 (PF) ---")
    n_captains = 2
    utilities = [17, 18, 19, 20]

    p_initial = [0.35, 0.25, 0.25, 0.15]
    p_shift_1 = [0.32, 0.25, 0.25, 0.18]
    p_shift_2 = [0.25, 0.25, 0.25, 0.25]

    res_init = find_discrete_threshold_binary_search(utilities, p_initial, n_captains, "PF")
    res_s1 = find_discrete_threshold_binary_search(utilities, p_shift_1, n_captains, "PF")
    res_s2 = find_discrete_threshold_binary_search(utilities, p_shift_2, n_captains, "PF")

    print(f"Initial Values : p={p_initial} -> Threshold: {res_init}")
    print(f"Shift 1 (Weak) : p={p_shift_1} -> Threshold: {res_s1}")
    print(f"Shift 2 (Strong): p={p_shift_2} -> Threshold: {res_s2}")


def prove_table_9():
    """Table 9: A strong enough leftwards-shifting of probability mass decreases u_T (PF)"""
    print("\n--- Proving Table 9 (PF) ---")
    n_captains = 2
    utilities = [15, 18, 19, 20, 27]

    p_initial = [0.2, 0.2, 0.2, 0.2, 0.2]
    p_shift_1 = [0.2, 0.2, 0.2, 0.3, 0.1]
    p_shift_2 = [0.2, 0.2, 0.2, 0.35, 0.05]

    res_init = find_discrete_threshold_binary_search(utilities, p_initial, n_captains, "PF")
    res_s1 = find_discrete_threshold_binary_search(utilities, p_shift_1, n_captains, "PF")
    res_s2 = find_discrete_threshold_binary_search(utilities, p_shift_2, n_captains, "PF")

    print(f"Initial Values : p={p_initial} -> Threshold: {res_init}")
    print(f"Shift 1 (Weak) : p={p_shift_1} -> Threshold: {res_s1}")
    print(f"Shift 2 (Strong): p={p_shift_2} -> Threshold: {res_s2}")


def prove_table_10():
    """Table 10: Incrementing sub-threshold utilities can cause T to shift towards any direction (PF)"""
    print("\n--- Proving Table 10 (PF) ---")
    n_captains = 2
    probabilities = [0.25, 0.25, 0.25, 0.25]

    u_initial = [1, 2, 11, 24]
    u_inc_1 = [1, 4, 11, 24]
    u_inc_2 = [1, 10, 11, 24]
    u_inc_3 = [9, 10, 11, 24]

    res_init = find_discrete_threshold_binary_search(u_initial, probabilities, n_captains, "PF")
    res_i1 = find_discrete_threshold_binary_search(u_inc_1, probabilities, n_captains, "PF")
    res_i2 = find_discrete_threshold_binary_search(u_inc_2, probabilities, n_captains, "PF")
    res_i3 = find_discrete_threshold_binary_search(u_inc_3, probabilities, n_captains, "PF")

    print(f"Initial Values          : u={u_initial} -> T={u_initial.index(res_init) + 1} (value {res_init})")
    print(f"Increment 1 (T remains) : u={u_inc_1} -> T={u_inc_1.index(res_i1) + 1} (value {res_i1})")
    print(f"Increment 2 (T decreases): u={u_inc_2} -> T={u_inc_2.index(res_i2) + 1} (value {res_i2})")
    print(f"Increment 3 (T increases): u={u_inc_3} -> T={u_inc_3.index(res_i3) + 1} (value {res_i3})")


def prove_table_11():
    """Table 11: A strong enough incrementing of super-threshold utilities increments T (PF)"""
    print("\n--- Proving Table 11 (PF) ---")
    n_captains = 2
    probabilities = [0.25, 0.25, 0.25, 0.25]

    u_initial = [9, 10, 11, 12]
    u_inc_1 = [9, 10, 11, 18]
    u_inc_2 = [9, 10, 11, 24]

    res_init = find_discrete_threshold_binary_search(u_initial, probabilities, n_captains, "PF")
    res_i1 = find_discrete_threshold_binary_search(u_inc_1, probabilities, n_captains, "PF")
    res_i2 = find_discrete_threshold_binary_search(u_inc_2, probabilities, n_captains, "PF")

    print(f"Initial Values          : u={u_initial} -> T={u_initial.index(res_init) + 1} (value {res_init})")
    print(f"Increment 1 (T remains) : u={u_inc_1} -> T={u_inc_1.index(res_i1) + 1} (value {res_i1})")
    print(f"Increment 2 (T increases): u={u_inc_2} -> T={u_inc_2.index(res_i2) + 1} (value {res_i2})")


def prove_table_12():
    """Table 12: A strong enough decrementing of sub-threshold utilities decrements T (PF)"""
    print("\n--- Proving Table 12 (PF) ---")
    n_captains = 2
    probabilities = [0.25, 0.25, 0.25, 0.25]

    u_initial = [9, 10, 12, 20]
    u_dec_1 = [7, 10, 12, 20]
    u_dec_2 = [6, 10, 12, 20]

    res_init = find_discrete_threshold_binary_search(u_initial, probabilities, n_captains, "PF")
    res_d1 = find_discrete_threshold_binary_search(u_dec_1, probabilities, n_captains, "PF")
    res_d2 = find_discrete_threshold_binary_search(u_dec_2, probabilities, n_captains, "PF")

    print(f"Initial Values          : u={u_initial} -> T={u_initial.index(res_init) + 1} (value {res_init})")
    print(f"Decrement 1 (T remains) : u={u_dec_1} -> T={u_dec_1.index(res_d1) + 1} (value {res_d1})")
    print(f"Decrement 2 (T decreases): u={u_dec_2} -> T={u_dec_2.index(res_d2) + 1} (value {res_d2})")


def prove_table_13():
    """Table 13: Decrementing u_T retains/increases T (PF)"""
    print("\n--- Proving Table 13 (PF) ---")
    n_captains = 2
    probabilities = [0.25, 0.25, 0.25, 0.25]

    u_initial = [1, 2, 14, 15]
    u_dec_1 = [1, 2, 10, 15]
    u_dec_2 = [1, 2, 2.1, 15]

    res_init = find_discrete_threshold_binary_search(u_initial, probabilities, n_captains, "PF")
    res_d1 = find_discrete_threshold_binary_search(u_dec_1, probabilities, n_captains, "PF")
    res_d2 = find_discrete_threshold_binary_search(u_dec_2, probabilities, n_captains, "PF")

    print(f"Initial Values          : u={u_initial} -> T={u_initial.index(res_init) + 1} (value {res_init})")
    print(f"Decrement 1 (T remains) : u={u_dec_1} -> T={u_dec_1.index(res_d1) + 1} (value {res_d1})")
    print(f"Decrement 2 (T increases): u={u_dec_2} -> T={u_dec_2.index(res_d2) + 1} (value {res_d2})")


def prove_table_14():
    """Table 14: Decrementing u_T decrements T (PF)"""
    print("\n--- Proving Table 14 (PF) ---")
    n_captains = 2
    # Probabilities as defined in the thesis table: 1/2, 1/10, 3/10, 1/10
    probabilities = [0.5, 0.1, 0.3, 0.1]

    u_initial = [1, 1.5, 11, 12]
    u_dec_1 = [1, 1.5, 2, 12]

    res_init = find_discrete_threshold_binary_search(u_initial, probabilities, n_captains, "PF")
    res_d1 = find_discrete_threshold_binary_search(u_dec_1, probabilities, n_captains, "PF")

    print(f"Initial Values          : u={u_initial} -> T={u_initial.index(res_init) + 1} (value {res_init})")
    print(f"Decrement 1 (T decreases): u={u_dec_1} -> T={u_dec_1.index(res_d1) + 1} (value {res_d1})")


if __name__ == "__main__":
    prove_table_6()
    prove_table_7()
    prove_table_8()
    prove_table_9()
    prove_table_10()
    prove_table_11()
    prove_table_12()
    prove_table_13()
    prove_table_14()
