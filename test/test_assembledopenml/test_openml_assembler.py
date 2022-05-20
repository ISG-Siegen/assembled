from assembledopenml.openml_assembler import OpenMLAssembler
import pytest


class TestOpenMLAssembler:
    """This can take a long time because of API Calls (init also must do an API call)"""

    def test_assembler_init(self):
        with pytest.raises(ValueError):
            OpenMLAssembler("möp")

    def test_assembler_run(self):
        omla = OpenMLAssembler(nr_base_models=5)

        mt = omla.run(3)

        assert mt.dataset is not None
        assert mt.dataset_name is not None
        assert mt.target_name is not None
        assert mt.class_labels is not None
        assert mt.cat_feature_names is not None
        assert mt.task_type is not None
        assert mt.is_classification is not None
        assert mt.is_regression is not None
        assert mt.openml_task_id is not None
        assert mt.folds is not None

        assert len(mt.predictors) == 5
