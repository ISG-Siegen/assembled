from difflib import SequenceMatcher


# from Levenshtein import ratio as levenshtein_ratio  # use this as it is less radical than sequence matcher


def filter_duplicates_manually(mt, min_sim_pre_filter: bool = True, min_sim: float = 0.85):
    """A function that can help you to manually filter duplicates of a metatask.

    Does not really work if the description is not a string; is not really sophisticated; ...

    Parameters
    ----------
    mt: Metatask
        The metatask for which to filter stuff.
    min_sim_pre_filter : bool, default=True
        Whether you want to pre-filter the possible duplicates based on their similarity using the edit distance.
    min_sim : float, default=0.85
        Minimal percentage of similarity to be considered for manual filtering.
    """
    # -- Delayed Import because it is an optional function (perhaps not used at all)

    # -- Get only predictors with high similarity
    similar_predictors = set()
    sim_pred_to_sim_val = {}
    to_remove_predictors = set()
    for outer_pred_name, outer_pred_dsp in mt.predictor_descriptions.items():
        # If pred already in the similar list, no need to check if other preds are similar to it.
        #   (If any would be, they would be found in their own loop)
        if outer_pred_name in similar_predictors:
            continue

        # Loop over all predictors and find sim
        for inner_pred_name, inner_pred_dsp in mt.predictor_descriptions.items():
            # Skip if the same predictor
            if outer_pred_name == inner_pred_name:
                continue
            else:
                # Find difference between the strings
                len_to_check = len(outer_pred_dsp) if len(outer_pred_dsp) < len(inner_pred_dsp) else len(
                    inner_pred_dsp)
                for i in range(len_to_check):
                    if outer_pred_dsp[i] != inner_pred_dsp[i]:
                        difference1 = outer_pred_dsp[i:]
                        difference2 = inner_pred_dsp[i:]
                        break
                else:
                    difference1 = outer_pred_dsp
                    difference2 = inner_pred_dsp

                # Add both to similar list if higher than min similarity, afterwards skip
                sim_val = SequenceMatcher(None, outer_pred_dsp, inner_pred_dsp, autojunk=False).ratio()
                # sim_val = levenshtein_ratio(difference1, difference2)
                if sim_val >= min_sim or (not min_sim_pre_filter):
                    similar_predictors.add(outer_pred_name)
                    sim_pred_to_sim_val[outer_pred_name] = sim_val
                    similar_predictors.add(inner_pred_name)
                    sim_pred_to_sim_val[inner_pred_name] = sim_val

                    # Validate by hand
                    print("\n\n## Start Similarity Case ##")
                    print("[1]", outer_pred_name, "|", outer_pred_dsp)
                    print("[2]", inner_pred_name, "|", inner_pred_dsp)
                    print("Difference: "
                          "\n     [1] {} ".format(difference1),
                          "\n     [2] {} ".format(difference2))
                    print("Similarity between the difference (edit distance):", sim_val)
                    to_keep = input("Keep 1,2 or both? ('1', '2', 'both' / '\\n') \n")

                    if to_keep == "1":
                        to_remove_predictors.add(inner_pred_name)
                    elif to_keep == "2":
                        to_remove_predictors.add(outer_pred_name)
                    elif to_keep in ["", "both"]:
                        pass
                    else:
                        raise ValueError("Unable to understand your answer!")
                    print("\n## Finished Similarity Case ##")
                    break

    # To list for index usage
    to_remove_predictors = list(to_remove_predictors)

    if to_remove_predictors:
        mt.remove_predictors(to_remove_predictors)
        print("Removed: {}".format(to_remove_predictors))
    else:
        print("No predictors removed.")
