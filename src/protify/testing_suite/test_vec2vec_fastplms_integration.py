from __future__ import annotations

import torch
import torch.nn as nn

from src.protify.base_models import get_base_models
from src.protify.base_models import vec2vec


def _tiny_config(**overrides) -> vec2vec.Vec2VecConfig:
    values = {
        "encoder_names": ["ESM2-8", "ESM2-35"],
        "encoder_paths": ["Synthyra/ESM2-8M", "Synthyra/ESM2-35M"],
        "encoder_dims": [8, 10],
        "d_adapter": 8,
        "d_hidden": 8,
        "d_transform": 8,
        "adapter_depth": 1,
        "transform_depth": 1,
        "disc_dim": 8,
        "disc_depth": 1,
        "norm_style": "layer",
    }
    values.update(overrides)
    return vec2vec.Vec2VecConfig(**values)


class TensorSourceEncoder(nn.Module):
    """FastPLMs-style source encoder with no tokenizer attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs = None

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.last_kwargs = kwargs
        hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4)
        if kwargs.get("output_attentions"):
            return hidden, torch.ones(1)
        return hidden


def test_build_uses_protify_source_factory_and_forwards_full_batch(
    monkeypatch,
):
    config = _tiny_config()
    translator = vec2vec.Vec2VecModel(config)
    source_model = TensorSourceEncoder()
    source_tokenizer = object()
    observed = {}

    def fake_get_base_model(name, masked_lm=False, dtype=None, model_path=None):
        observed.update(name=name, dtype=dtype, model_path=model_path)
        return source_model, source_tokenizer

    monkeypatch.setattr(
        vec2vec,
        "_load_vec2vec_config_for_inference",
        lambda path: (config, False),
    )
    monkeypatch.setattr(get_base_models, "get_base_model", fake_get_base_model)
    monkeypatch.setattr(
        vec2vec.Vec2VecModel,
        "from_pretrained",
        classmethod(lambda cls, path, config: translator),
    )

    model, tokenizer = vec2vec.build_vec2vec_model(
        "unused",
        dtype=torch.bfloat16,
        model_path="checkpoint",
    )
    output = model(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        output_attentions=True,
        sequence_ids=torch.tensor([[0, 0, 0], [1, 1, 1]]),
    )

    assert tokenizer is source_tokenizer
    assert observed == {
        "name": "ESM2-8",
        "dtype": torch.bfloat16,
        "model_path": "Synthyra/ESM2-8M",
    }
    assert source_model.last_kwargs["output_attentions"] is True
    assert "sequence_ids" in source_model.last_kwargs
    assert output.shape == (2, 10)
    assert output.dtype == next(translator.parameters()).dtype


def test_source_encoder_output_must_be_residue_embeddings():
    class PooledSource(nn.Module):
        def forward(self, input_ids, **kwargs):
            return torch.ones(input_ids.shape[0], 4)

    config = _tiny_config()
    model = vec2vec.Vec2VecForEmbedding(
        config,
        PooledSource(),
        vec2vec.Vec2VecModel(config),
        "ESM2-8",
        "ESM2-35",
    )

    try:
        model(input_ids=torch.ones(2, 3, dtype=torch.long))
    except ValueError as error:
        assert "residue embeddings" in str(error)
    else:
        raise AssertionError("Expected pooled source output to be rejected")
