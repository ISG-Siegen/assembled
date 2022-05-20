from assembled.metatask import MetaTask
from results.data_utils import get_valid_benchmark_ids
from ensemble_techniques.collect_ensemble_techniques import get_benchmark_techniques

# -- Main
if __name__ == "__main__":

    # --- Input para
    path_to_benchmark_data = "../results/openml_benchmark/benchmark_metatasks"
    valid_task_ids = get_valid_benchmark_ids(path_to_benchmark_data)
    test_split_frac = 0.5

    # This seed is used to create a random state for each individual technique. This random state is reproducible across
    #   runs even with chaining number of techniques or datasets/tasks.
    #   To guarantee this, we must re-initialize the techniques for every task.
    rng_seed = 3151278530

    # The following is not a random state object nor part of the above rng to avoid the problem that adding
    #   a new technique would change the random state and thus make it less comparable across runs.
    test_split_rng = 581640921

    # --- Iterate over tasks to gather results
    nr_tasks = len(valid_task_ids)
    for task_nr, task_id in enumerate(valid_task_ids, start=1):
        mt = MetaTask()
        mt.read_metatask_from_files(path_to_benchmark_data, task_id)
        print("#### Process Task {} for dataset {} ({}/{}) ####".format(mt.openml_task_id, mt.dataset_name,
                                                                        task_nr, nr_tasks))
        out_path = "../results/benchmark_output/results_for_metatask_{}.csv".format(task_id)

        # -- Get techniques (initialize new for each task due to randomness and clean start)
        techniques_to_benchmark = get_benchmark_techniques(rng_seed)

        # -- Run Techniques on Metatask
        nr_techniques = len(techniques_to_benchmark)
        counter_techniques = 1
        for technique_name, technique_run_args in techniques_to_benchmark.items():
            print("### Benchmark Ensemble Technique: {} ({}/{})###".format(technique_name, counter_techniques,
                                                                           nr_techniques))
            counter_techniques += 1
            mt.run_ensemble_on_all_folds(technique_name=technique_name, **technique_run_args,
                                         meta_train_test_split_fraction=0.5, output_file_path=out_path,
                                         meta_train_test_split_random_state=test_split_rng)

    # ----------------- TODOs
    # x) collect metadata about evaluation: time taken; parameters; passthrough;
