import pytest

from tinytune.config import TrainConfig


def test_defaults_are_valid():
    TrainConfig().validate()


def test_label_count_must_match():
    with pytest.raises(ValueError):
        TrainConfig(num_labels=3).validate()


def test_push_to_hub_requires_repo_id():
    with pytest.raises(ValueError):
        TrainConfig(push_to_hub=True).validate()
    TrainConfig(push_to_hub=True, hub_repo_id="user/repo").validate()


@pytest.mark.parametrize("field,value", [("epochs", 0), ("batch_size", 0), ("train_samples", -1)])
def test_rejects_non_positive_values(field, value):
    with pytest.raises(ValueError):
        TrainConfig(**{field: value}).validate()


def test_to_dict_roundtrip():
    cfg = TrainConfig(epochs=3)
    assert TrainConfig(**cfg.to_dict()) == cfg
