def make_equal(min_value_to_equalize, vector):
    # Equalize all values greater than a specified value (assume all values sum up to 1)
    make_equal_mask = vector >= min_value_to_equalize
    nr_equal_candidates = sum(make_equal_mask)

    # Compute the left-over-share and then split it evenly
    left_over_share = 1 - sum(vector[~make_equal_mask])
    vector[make_equal_mask] = left_over_share / nr_equal_candidates

    return vector
