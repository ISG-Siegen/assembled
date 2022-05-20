from assembled.benchmaker import BenchMaker
from sklearn.metrics import accuracy_score


class TestBenchMaker:

    def test_init(self):
        bmer = BenchMaker("example_metatasks", "example_benchmark_metatasks",
                          manual_filter_duplicates=False, min_number_predictors=5,
                          remove_bad_predictors=True, remove_worse_than_random_predictors=True,
                          remove_constant_predictors=True,
                          metric_info=(accuracy_score, "acc", True))

        bmer.build_benchmark()
