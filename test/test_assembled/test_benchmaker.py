from assembled.benchmaker import BenchMaker
from sklearn.metrics import accuracy_score
import pathlib


class TestBenchMaker:

    def test_init(self):
        base_path = pathlib.Path(__file__).parent.resolve()

        bmer = BenchMaker(base_path / "example_metatasks", base_path / "example_benchmark_metatasks",
                          manual_filter_duplicates=False, min_number_predictors=5,
                          remove_bad_predictors=True, remove_worse_than_random_predictors=True,
                          remove_constant_predictors=True,
                          metric_info=(accuracy_score, "acc", True))

        bmer.build_benchmark()
