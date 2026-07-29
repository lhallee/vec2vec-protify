import numpy as np
import pytest
import torch

from types import SimpleNamespace
from transformers import EvalPrediction

try:
    from src.protify.probes import trainers as trainers_module
    from src.protify.probes.trainers import TrainerArguments, TrainerMixin
    from src.protify.metrics import (
        compute_multi_label_classification_metrics,
        evaluate_at_threshold,
        fit_thresholds,
    )
except ImportError:
    try:
        from protify.probes import trainers as trainers_module
        from protify.probes.trainers import TrainerArguments, TrainerMixin
        from protify.metrics import (
            compute_multi_label_classification_metrics,
            evaluate_at_threshold,
            fit_thresholds,
        )
    except ImportError:
        from ..probes import trainers as trainers_module
        from ..probes.trainers import TrainerArguments, TrainerMixin
        from ..metrics import (
            compute_multi_label_classification_metrics,
            evaluate_at_threshold,
            fit_thresholds,
        )


def test_validation_fitted_threshold_is_immutable_on_test():
    validation_probabilities = np.array([[0.45], [0.40], [0.20], [0.10]])
    validation_labels = np.array([[1], [1], [0], [0]])
    fitted = fit_thresholds(validation_probabilities, validation_labels)

    assert fitted == pytest.approx(0.40)

    test_probabilities = np.array([[0.35], [0.45], [0.15]])
    first = evaluate_at_threshold(
        test_probabilities, np.array([[0], [1], [0]]), fitted
    )
    second = evaluate_at_threshold(
        test_probabilities, np.array([[1], [0], [1]]), fitted
    )

    assert first["threshold"] == pytest.approx(fitted)
    assert second["threshold"] == pytest.approx(fitted)
    assert first["f1"] != second["f1"]


def test_trainer_metric_defaults_to_fixed_half_threshold():
    logits = np.array([[-0.6], [-0.2], [0.2], [0.6]])
    first = compute_multi_label_classification_metrics(
        EvalPrediction(predictions=logits, label_ids=np.array([[0], [0], [1], [1]]))
    )
    second = compute_multi_label_classification_metrics(
        EvalPrediction(predictions=logits, label_ids=np.array([[1], [1], [0], [0]]))
    )

    assert first["threshold"] == pytest.approx(0.5)
    assert second["threshold"] == pytest.approx(0.5)


def test_trainer_metric_accepts_validation_fitted_threshold():
    validation_probabilities = np.array([[0.45], [0.40], [0.20], [0.10]])
    validation_labels = np.array([[1], [1], [0], [0]])
    fitted = fit_thresholds(validation_probabilities, validation_labels)
    test_probabilities = np.array([[0.35], [0.45], [0.15]])
    logits = np.log(test_probabilities / (1.0 - test_probabilities))

    result = compute_multi_label_classification_metrics(
        EvalPrediction(
            predictions=logits,
            label_ids=np.array([[0], [1], [0]]),
        ),
        threshold=fitted,
    )

    assert result["threshold"] == pytest.approx(fitted)
    assert result["f1"] == pytest.approx(1.0)


def test_legacy_eval_optimization_is_explicit_and_warns():
    probabilities = np.array([[0.45], [0.40], [0.20], [0.10]])
    logits = np.log(probabilities / (1.0 - probabilities))
    prediction = EvalPrediction(
        predictions=logits,
        label_ids=np.array([[1], [1], [0], [0]]),
    )

    with pytest.warns(FutureWarning, match="leaks held-out outcomes"):
        result = compute_multi_label_classification_metrics(
            prediction, legacy_optimize_on_eval=True
        )

    assert result["threshold"] != pytest.approx(0.5)


def test_per_label_thresholds_and_degenerate_labels_are_safe():
    probabilities = np.array(
        [
            [0.10, 0.40, 0.90],
            [0.20, 0.35, 0.80],
            [0.30, 0.20, 0.70],
            [0.40, 0.10, 0.60],
        ]
    )
    labels = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )

    fitted = fit_thresholds(probabilities, labels, mode="per_label")
    result = evaluate_at_threshold(probabilities, labels, fitted)

    assert fitted.shape == (3,)
    assert fitted[0] == pytest.approx(0.5)
    assert fitted[1] == pytest.approx(0.35)
    assert fitted[2] == pytest.approx(0.5)
    assert result["threshold_mode"] == "per_label"
    np.testing.assert_allclose(result["threshold"], fitted)
    assert np.isfinite(result["f1"])


def test_threshold_shape_is_checked():
    probabilities = np.full((3, 2), 0.5)
    labels = np.zeros((3, 2), dtype=int)

    with pytest.raises(ValueError, match="scalar or have shape"):
        evaluate_at_threshold(probabilities, labels, np.array([0.2, 0.3, 0.4]))


def _multilabel_trainer_mixin(num_runs=1):
    trainer_args = TrainerArguments(
        model_save_dir="threshold-test",
        task_type="multilabel",
        num_runs=num_runs,
        balanced_regression_metrics=False,
        make_plots=False,
        torch_compile=False,
    )
    mixin = TrainerMixin(trainer_args=trainer_args)
    mixin.probe_args = SimpleNamespace(tokenwise=False, num_labels=1)
    return mixin


def test_parallel_workflow_reuses_each_validation_threshold_on_test():
    mixin = _multilabel_trainer_mixin(num_runs=2)
    validation_probabilities = np.array(
        [
            [[0.45], [0.80]],
            [[0.40], [0.79]],
            [[0.20], [0.69]],
            [[0.10], [0.68]],
        ]
    )
    validation_logits = np.log(
        validation_probabilities / (1.0 - validation_probabilities)
    ).astype(np.float32)
    labels = np.array([[1], [1], [0], [0]], dtype=np.float32)
    thresholds = mixin._parallel_probe_fit_multilabel_thresholds(
        validation_logits, labels
    )

    test_probabilities = np.array(
        [
            [[0.35], [0.72]],
            [[0.45], [0.65]],
            [[0.15], [0.52]],
            [[0.25], [0.48]],
        ]
    )
    test_logits = np.log(test_probabilities / (1.0 - test_probabilities)).astype(
        np.float32
    )
    first = mixin._parallel_probe_metrics_by_run(
        test_logits,
        labels,
        "dataset",
        "test",
        multilabel_thresholds=thresholds,
        run_seeds=[42, 43],
    )
    second = mixin._parallel_probe_metrics_by_run(
        test_logits,
        1.0 - labels,
        "dataset",
        "test",
        multilabel_thresholds=thresholds,
        run_seeds=[42, 43],
    )

    assert thresholds[0] < 0.5
    assert thresholds[1] > 0.5
    assert [metrics["test_threshold"] for metrics in first] == pytest.approx(
        thresholds
    )
    assert [metrics["test_threshold"] for metrics in second] == pytest.approx(
        thresholds
    )
    assert all(
        metrics["test_threshold_fit_split"] == "validation" for metrics in first
    )
    assert all(
        metrics["test_threshold_applied_without_refit"] is True
        for metrics in first
    )
    assert [metrics["test_threshold_fit_seed"] for metrics in first] == [42, 43]


def test_sequential_train_workflow_fits_validation_and_applies_to_test(
    monkeypatch,
):
    validation_dataset = [object()] * 4
    test_dataset = [object()] * 3
    validation_probabilities = np.array([[0.45], [0.40], [0.20], [0.10]])
    validation_logits = np.log(
        validation_probabilities / (1.0 - validation_probabilities)
    ).astype(np.float32)
    validation_labels = np.array([[1], [1], [0], [0]], dtype=np.float32)
    test_probabilities = np.array([[0.35], [0.45], [0.15]])
    test_logits = np.log(test_probabilities / (1.0 - test_probabilities)).astype(
        np.float32
    )
    test_labels = np.array([[0], [1], [0]], dtype=np.float32)

    class FakeAccelerator:
        def free_memory(self):
            return None

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.model = kwargs["model"]
            self.accelerator = FakeAccelerator()
            self.compute_metrics = kwargs["compute_metrics"]

        def evaluate(self, dataset):
            del dataset
            return {"eval_loss": 0.25}

        def train(self):
            return SimpleNamespace(metrics={"train_runtime": 0.1})

        def predict(self, dataset):
            if dataset is validation_dataset:
                return (
                    validation_logits,
                    validation_labels,
                    {"test_loss": 0.2},
                )
            assert dataset is test_dataset
            return (
                test_logits,
                test_labels,
                {"test_loss": 0.3},
            )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

    mixin = _multilabel_trainer_mixin()
    expected_threshold = mixin._fit_multilabel_threshold_from_logits(
        validation_logits,
        validation_labels,
    )
    monkeypatch.setattr(trainers_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    model, valid_metrics, test_metrics, _predictions, _labels = mixin._train(
        model=FakeModel(),
        train_dataset=[object()] * 4,
        valid_dataset=validation_dataset,
        test_dataset=test_dataset,
        data_collator=None,
        tokenizer=None,
        log_id="run",
        model_name="model",
        data_name="dataset",
        probe=True,
        skip_plot=True,
    )

    assert valid_metrics["eval_threshold"] == pytest.approx(expected_threshold)
    assert test_metrics["test_threshold"] == pytest.approx(expected_threshold)
    assert test_metrics["test_f1"] == pytest.approx(1.0)
    assert test_metrics["test_threshold_fit_split"] == "validation"
    assert test_metrics["test_threshold_applied_without_refit"] is True
    assert model.config.multilabel_threshold == pytest.approx(expected_threshold)
    assert (
        model.config.multilabel_threshold_provenance["fit_split"] == "validation"
    )


def test_multilabel_training_rejects_missing_validation_split():
    mixin = _multilabel_trainer_mixin()

    with pytest.raises(ValueError, match="requires a validation split"):
        mixin._train(
            model=None,
            train_dataset=[],
            valid_dataset=None,
            test_dataset=[],
            data_collator=None,
            tokenizer=None,
            log_id="run",
            model_name="model",
            data_name="dataset",
        )
