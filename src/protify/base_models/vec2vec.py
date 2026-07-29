"""
HuggingFace-compatible vec2vec implementation for embedding translation.
Based on: "Harnessing the Universal Geometry of Embeddings" (arXiv:2505.12540)

Kept in sync with ProteinRepresentationEnhancement/models/vec2vec.py so that
any checkpoint pushed by the training repo loads cleanly here for downstream
supervised probing.
"""

import base64
import hashlib
import json
import math
import struct
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union
from einops import rearrange
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from pooler import Pooler

from .base_tokenizer import BaseSequenceTokenizer
from .supported_models import all_presets_with_paths


# =============================================================================
# Utilities (mirrored from models/utils.py + models/attention.py upstream)
# =============================================================================

def _source_scaler_from_config(
    config: "Vec2VecConfig",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode and verify the train-fitted source scaler from checkpoint config."""

    manifest = getattr(config, "preprocessing_manifest", None)
    if not isinstance(manifest, Mapping):
        raise ValueError(
            "Standardized Vec2Vec inference requires preprocessing_manifest"
        )
    if (
        manifest.get("preprocessing_profile")
        != config.preprocessing_profile
        or manifest.get("input_standardize") is not True
    ):
        raise ValueError(
            "Checkpoint preprocessing manifest disagrees with Vec2Vec config"
        )
    state = manifest.get("scaler_state_a")
    if not isinstance(state, Mapping):
        raise ValueError(
            "Standardized Vec2Vec inference requires inline source scaler state"
        )
    if (
        state.get("schema_version") != "embedding_preprocessing_v1"
        or state.get("dtype") != "float64"
        or state.get("fit_split") != "train"
    ):
        raise ValueError("Unsupported Vec2Vec source scaler state")
    feature_count = state.get("feature_count")
    if (
        isinstance(feature_count, bool)
        or not isinstance(feature_count, int)
        or feature_count <= 0
    ):
        raise ValueError("Vec2Vec source scaler width must be positive")

    canonical = json.dumps(
        dict(state),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    observed_digest = hashlib.sha256(canonical).hexdigest()
    if observed_digest != manifest.get("scaler_stats_sha256_a"):
        raise ValueError("Vec2Vec source scaler checksum mismatch")

    def decode(field: str) -> torch.Tensor:
        value = state.get(field)
        if not isinstance(value, str):
            raise ValueError(f"Vec2Vec source scaler {field} is malformed")
        try:
            packed = base64.b64decode(value.encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError) as error:
            raise ValueError(
                f"Vec2Vec source scaler {field} is malformed"
            ) from error
        if len(packed) != feature_count * 8:
            raise ValueError(
                f"Vec2Vec source scaler {field} has the wrong width"
            )
        return torch.tensor(
            struct.unpack(f"<{feature_count}d", packed),
            dtype=torch.float64,
        )

    mean = decode("mean_base64_le")
    scale = decode("scale_base64_le")
    if (
        not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(scale).all())
        or bool((scale <= 0).any())
    ):
        raise ValueError("Vec2Vec source scaler contains invalid values")
    return mean, scale

def _linear_layer(input_size: int, output_size: int, bias: bool = False) -> nn.Linear:
    layer = nn.Linear(input_size, output_size, bias=bias)
    nn.init.xavier_normal_(layer.weight)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def _parameter_layer(size):
    param = nn.Parameter(torch.randn(size))
    nn.init.xavier_normal_(param)
    return param


Linear = partial(_linear_layer, bias=False)


def correction_fn_256(expansion_ratio: float, hidden_size: int) -> int:
    return int(((expansion_ratio * hidden_size) + 255) // 256 * 256)


class AttentionPooler(nn.Module):
    """
    Cross-attention pool (b, L, hidden_size) -> (b, n_tokens, hidden_size).
    Used by Vec2VecLearnedPooling to pool matrix embeddings before translation.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, n_tokens: int = 1):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_heads = intermediate_size // 64
        self.d_head = 64
        self.input = Linear(hidden_size, intermediate_size)
        self.Q = _parameter_layer((1, n_tokens, intermediate_size))
        self.Wq = Linear(intermediate_size, intermediate_size)
        self.Wv = Linear(intermediate_size, intermediate_size)
        self.Wk = Linear(intermediate_size, intermediate_size)
        self.Wo = Linear(intermediate_size, hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, _ = x.size()
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(b, 1, self.n_tokens, L).bool()
        x = self.input(x)
        q = self.Wq(self.Q).expand(b, -1, -1)
        v = self.Wv(x)
        k = self.Wk(x)
        q, k, v = map(self.reshaper, (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)
        attn = rearrange(attn, "b h s d -> b s (h d)")
        return self.Wo(attn)


# =============================================================================
# Configuration
# =============================================================================

class Vec2VecConfig(PretrainedConfig):
    """Configuration for Vec2Vec model."""

    model_type = "vec2vec"

    def __init__(
        self,
        encoder_names: List[str] = None,
        encoder_paths: List[str] = None,
        encoder_dims: List[int] = None,
        d_adapter: int = 2048,
        d_hidden: int = 2048,
        d_transform: int = 2048,
        adapter_depth: int = 3,
        transform_depth: int = 4,
        disc_dim: int = 2048,
        disc_depth: int = 5,
        weight_init: str = "kaiming",
        norm_style: str = "batch",
        normalize_embeddings: bool = False,
        preprocessing_profile: Optional[str] = None,
        input_standardize: Optional[bool] = None,
        input_l2_normalize: Optional[bool] = None,
        output_l2_normalize: Optional[bool] = None,
        expansion_ratio: float = 2.0,
        learned_pooling: bool = False,
        # Loss coefficients (only read during training; kept here so configs load cleanly)
        loss_coefficient_rec: float = 1.0,
        loss_coefficient_vsp: float = 1.0,
        loss_coefficient_cc_trans: float = 10.0,
        loss_coefficient_cc_vsp: float = 10.0,
        loss_coefficient_cc_rec: float = 0.0,
        loss_coefficient_reverse_rec: float = 0.0,
        loss_coefficient_gen: float = 1.0,
        loss_coefficient_latent_gen: float = 1.0,
        loss_coefficient_similarity_gen: float = 0.0,
        loss_coefficient_disc: float = 1.0,
        loss_coefficient_r1_penalty: float = 0.0,
        loss_coefficient_learned_pooling_vsp: float = 1.0,
        loss_coefficient_contrastive: float = 0.0,
        contrastive_temperature: float = 0.04,
        loss_coefficient_round_trip_contrastive: float = 0.0,
        loss_coefficient_b_mlm: float = 0.0,
        noise_level: float = 0.0,
        max_grad_norm: float = 1000.0,
        rec_sim_type: str = "cosine",
        trans_sim_type: str = "cosine",
        vsp_sim_type: str = "cosine",
        sim_type: Optional[str] = None,
        sigmoid: bool = False,
        gan_style: str = "least_squares",
        architecture_version: str = "pps_vec2vec_v2",
        discriminator_layout: str = "output2_latent2",
        similarity_batch_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_names = (
            list(encoder_names)
            if encoder_names is not None
            else ["model_a", "model_b"]
        )
        self.encoder_paths = (
            list(encoder_paths)
            if encoder_paths is not None
            else list(self.encoder_names)
        )
        self.encoder_dims = (
            list(encoder_dims)
            if encoder_dims is not None
            else [768, 768]
        )
        if not (
            len(self.encoder_names)
            == len(self.encoder_paths)
            == len(self.encoder_dims)
        ):
            raise ValueError(
                "encoder_names, encoder_paths, and encoder_dims must have equal lengths"
            )
        if len(self.encoder_names) != 2:
            raise ValueError(
                "pps_vec2vec_v2 requires exactly two encoders; received "
                f"{len(self.encoder_names)}"
            )
        if len(set(self.encoder_names)) != len(self.encoder_names):
            raise ValueError("encoder_names must be unique")
        if architecture_version != "pps_vec2vec_v2":
            raise ValueError(
                "Only architecture_version='pps_vec2vec_v2' is trainable. "
                "Use Vec2VecModel.load_legacy_translator_state_dict() for legacy "
                "inference."
            )
        if discriminator_layout != "output2_latent2":
            raise ValueError(
                "Only discriminator_layout='output2_latent2' is supported"
            )
        if loss_coefficient_similarity_gen > 0 and (
            similarity_batch_size is None or similarity_batch_size < 2
        ):
            raise ValueError(
                "A positive loss_coefficient_similarity_gen requires a fixed "
                "similarity_batch_size >= 2"
            )
        if (
            loss_coefficient_contrastive > 0
            or loss_coefficient_round_trip_contrastive > 0
        ):
            try:
                resolved_temperature = float(contrastive_temperature)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Enabled InfoNCE requires a finite "
                    "contrastive_temperature > 0"
                ) from exc
            if (
                not math.isfinite(resolved_temperature)
                or resolved_temperature <= 0
            ):
                raise ValueError(
                    "Enabled InfoNCE requires a finite "
                    "contrastive_temperature > 0"
                )
            contrastive_temperature = resolved_temperature
        profile_flags = {
            "native": (False, False),
            "l2": (False, True),
            "standard": (True, False),
            "standard_l2": (True, True),
        }
        if preprocessing_profile is None:
            # Preserve the single-switch interpretation of old checkpoints.
            preprocessing_profile = (
                "l2" if bool(normalize_embeddings) else "native"
            )
        preprocessing_profile = str(preprocessing_profile).lower()
        if preprocessing_profile not in profile_flags:
            raise ValueError(
                "preprocessing_profile must be one of "
                f"{sorted(profile_flags)}"
            )
        profile_standardize, profile_l2 = profile_flags[
            preprocessing_profile
        ]
        if input_standardize is None:
            input_standardize = profile_standardize
        if input_l2_normalize is None:
            input_l2_normalize = profile_l2
        if (
            bool(input_standardize),
            bool(input_l2_normalize),
        ) != (profile_standardize, profile_l2):
            raise ValueError(
                "preprocessing_profile disagrees with the resolved input "
                "standardization/L2 flags"
            )
        if output_l2_normalize is None:
            output_l2_normalize = bool(input_l2_normalize)
        self.d_adapter = d_adapter
        self.d_hidden = d_hidden
        self.d_transform = d_transform
        self.adapter_depth = adapter_depth
        self.transform_depth = transform_depth
        self.disc_dim = disc_dim
        self.disc_depth = disc_depth
        self.weight_init = weight_init
        self.norm_style = norm_style
        self.preprocessing_profile = preprocessing_profile
        self.input_standardize = bool(input_standardize)
        self.input_l2_normalize = bool(input_l2_normalize)
        self.output_l2_normalize = bool(output_l2_normalize)
        self.normalize_embeddings = self.output_l2_normalize
        self.expansion_ratio = expansion_ratio
        self.learned_pooling = learned_pooling
        self.loss_coefficient_rec = loss_coefficient_rec
        self.loss_coefficient_vsp = loss_coefficient_vsp
        self.loss_coefficient_cc_trans = loss_coefficient_cc_trans
        self.loss_coefficient_cc_vsp = loss_coefficient_cc_vsp
        self.loss_coefficient_cc_rec = loss_coefficient_cc_rec
        self.loss_coefficient_reverse_rec = loss_coefficient_reverse_rec
        self.loss_coefficient_gen = loss_coefficient_gen
        self.loss_coefficient_latent_gen = loss_coefficient_latent_gen
        self.loss_coefficient_similarity_gen = loss_coefficient_similarity_gen
        self.loss_coefficient_disc = loss_coefficient_disc
        self.loss_coefficient_r1_penalty = loss_coefficient_r1_penalty
        self.loss_coefficient_learned_pooling_vsp = loss_coefficient_learned_pooling_vsp
        self.loss_coefficient_contrastive = loss_coefficient_contrastive
        self.contrastive_temperature = contrastive_temperature
        self.loss_coefficient_round_trip_contrastive = (
            loss_coefficient_round_trip_contrastive
        )
        self.loss_coefficient_b_mlm = loss_coefficient_b_mlm
        self.noise_level = noise_level
        self.max_grad_norm = max_grad_norm
        if sim_type is not None:
            rec_sim_type = sim_type
            trans_sim_type = sim_type
            vsp_sim_type = sim_type if sim_type in ("cosine", "dot") else "cosine"
        self.rec_sim_type = rec_sim_type
        self.trans_sim_type = trans_sim_type
        self.vsp_sim_type = vsp_sim_type
        # ``None`` means there is no legacy global override. Preserve it
        # through config serialization so distinct per-term settings are not
        # collapsed to ``rec_sim_type`` on reload.
        self.sim_type = sim_type
        self.sigmoid = sigmoid
        self.gan_style = gan_style
        self.architecture_version = architecture_version
        self.discriminator_layout = discriminator_layout
        self.similarity_batch_size = similarity_batch_size

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Reject silent promotion of an unversioned legacy Hub config."""

        missing = {
            "architecture_version",
            "discriminator_layout",
        }.difference(config_dict)
        if missing:
            raise ValueError(
                "Unversioned Vec2Vec configs are legacy and cannot be loaded "
                "through the generic from_pretrained path. Construct an "
                "explicit pps_vec2vec_v2 config and call "
                "Vec2VecModel.from_legacy_pretrained_for_inference(), or use "
                "load_legacy_translator_state_dict() for a raw state dict."
            )
        return super().from_dict(config_dict, **kwargs)

    def get_encoder_dims_dict(self) -> Dict[str, int]:
        return dict(zip(self.encoder_names, self.encoder_dims))

    def get_encoder_paths_dict(self) -> Dict[str, str]:
        return dict(zip(self.encoder_names, self.encoder_paths))

    def get_path_for_name(self, name: str) -> str:
        paths_dict = self.get_encoder_paths_dict()
        return paths_dict.get(name, name)


# =============================================================================
# Model Outputs
# =============================================================================

@dataclass
class Vec2VecOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reconstructions: Optional[Dict[str, torch.Tensor]] = None
    translations: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    latents: Optional[Dict[str, torch.Tensor]] = None
    metrics: Optional[Dict[str, float]] = None


# =============================================================================
# Model Components
# =============================================================================

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _initialize_weights(self, weight_init: str):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
                elif weight_init == "xavier":
                    nn.init.xavier_normal_(module.weight)
                elif weight_init == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.normal_(module.bias, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def _add_residual(self, input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if input_x.shape[1] < x.shape[1]:
            padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device)
            input_x = torch.cat([input_x, padding], dim=1)
        elif input_x.shape[1] > x.shape[1]:
            input_x = input_x[:, :x.shape[1]]
        return x + input_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MLPWithResidual(BaseModule):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        norm_style: str = "batch",
        weight_init: str = "kaiming",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        norm_layer = nn.BatchNorm1d if norm_style == "batch" else nn.LayerNorm

        for layer_idx in range(depth):
            if layer_idx == 0:
                h_dim = out_dim if depth == 1 else hidden_dim
                self.layers.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.SiLU()))
            elif layer_idx < depth - 1:
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    norm_layer(hidden_dim),
                    nn.Dropout(p=0.1),
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(p=0.1),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, out_dim),
                ))
        self._initialize_weights(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            input_x = x
            x = layer(x)
            x = self._add_residual(input_x, x)
        return x


class Discriminator(BaseModule):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 2048,
        depth: int = 5,
        weight_init: str = "kaiming",
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        if depth >= 2:
            layers = [nn.Linear(latent_dim, hidden_dim), nn.Dropout(0.0)]
            for _ in range(depth - 2):
                layers.extend([
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.0),
                ])
            layers.extend([nn.SiLU(), nn.Linear(hidden_dim, 1)])
            self.layers.append(nn.Sequential(*layers))
        else:
            self.layers.append(nn.Linear(latent_dim, 1))

        self._initialize_weights(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Main Model
# =============================================================================

class Vec2VecModel(PreTrainedModel):
    """
    Vec2Vec model for embedding translation between different spaces.

    Architecture:
        Input -> In Adapter -> Transform -> Out Adapter -> Output
    """

    config_class = Vec2VecConfig
    _LEGACY_PROVENANCE_STATE_KEY = "_legacy_inference_only_marker"
    _DISCRIMINATOR_STATE_PREFIXES = (
        "output_discriminators.",
        "latent_discriminators.",
        "similarity_discriminator.",
        "discriminators.",
    )

    def __init__(self, config: Vec2VecConfig):
        super().__init__(config)
        self.config = config
        self.register_buffer(
            "_legacy_inference_only_marker",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self._legacy_inference_only = bool(
            getattr(config, "legacy_inference_only", False)
        )
        encoder_dims = config.get_encoder_dims_dict()

        self.transform = MLPWithResidual(
            depth=config.transform_depth,
            in_dim=config.d_adapter,
            hidden_dim=config.d_transform,
            out_dim=config.d_adapter,
            norm_style=config.norm_style,
            weight_init=config.weight_init,
        )

        self.in_adapters = nn.ModuleDict()
        self.out_adapters = nn.ModuleDict()

        for name, dim in encoder_dims.items():
            self.in_adapters[name] = MLPWithResidual(
                config.adapter_depth, dim, config.d_hidden, config.d_adapter,
                config.norm_style, config.weight_init,
            )
            self.out_adapters[name] = MLPWithResidual(
                config.adapter_depth, config.d_adapter, config.d_hidden, dim,
                config.norm_style, config.weight_init,
            )

        self.output_discriminators = nn.ModuleDict()
        self.latent_discriminators = nn.ModuleDict()
        for name, dim in encoder_dims.items():
            self.output_discriminators[name] = Discriminator(
                dim, config.disc_dim, config.disc_depth, config.weight_init
            )
            self.latent_discriminators[name] = Discriminator(
                config.d_adapter,
                config.disc_dim,
                config.disc_depth,
                config.weight_init,
            )

        self.similarity_discriminator: Optional[Discriminator] = None
        if config.loss_coefficient_similarity_gen > 0:
            self.similarity_discriminator = Discriminator(
                2 * config.similarity_batch_size,
                config.disc_dim,
                config.disc_depth,
                config.weight_init,
            )

        self.post_init()
        self.register_load_state_dict_post_hook(
            self._restore_legacy_inference_provenance
        )
        if self._legacy_inference_only:
            self._freeze_legacy_for_inference()

    def train(self, mode: bool = True):
        if mode and (
            self._legacy_inference_only
            or bool(getattr(self.config, "legacy_inference_only", False))
        ):
            raise RuntimeError(
                "Legacy three-discriminator checkpoints are inference-only and "
                "cannot be resumed for training"
            )
        return super().train(mode)

    def _freeze_legacy_for_inference(self) -> None:
        self._legacy_inference_only = True
        self._legacy_inference_only_marker.fill_(True)
        self.config.legacy_inference_only = True
        self.config.checkpoint_provenance = (
            "legacy_translator_inference_only"
        )
        nn.Module.train(self, False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _restore_legacy_inference_provenance(
        self,
        module: nn.Module,
        incompatible_keys,
    ) -> None:
        del module, incompatible_keys
        if (
            self._legacy_inference_only
            or bool(self._legacy_inference_only_marker)
            or bool(getattr(self.config, "legacy_inference_only", False))
        ):
            self._freeze_legacy_for_inference()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Restore the legacy inference-only freeze after Hugging Face loading."""

        loaded = super().from_pretrained(*args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        if (
            bool(getattr(model, "_legacy_inference_only", False))
            or bool(getattr(model, "_legacy_inference_only_marker", False))
            or bool(getattr(model.config, "legacy_inference_only", False))
        ):
            model._freeze_legacy_for_inference()
        return loaded

    @classmethod
    def from_legacy_pretrained_for_inference(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config: Vec2VecConfig,
        **kwargs,
    ):
        """Explicitly migrate a legacy Hub/local translator for inference."""

        if not isinstance(config, Vec2VecConfig):
            raise TypeError(
                "config must be an explicit pps_vec2vec_v2 Vec2VecConfig"
            )
        warnings.warn(
            "Loading a legacy Vec2Vec repository for frozen translator-only "
            "inference. Continuing training is prohibited.",
            UserWarning,
            stacklevel=2,
        )
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **kwargs,
        )
        model._freeze_legacy_for_inference()
        return model

    def named_discriminator_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        for name, discriminator in self.output_discriminators.items():
            yield f"output:{name}", discriminator
        for name, discriminator in self.latent_discriminators.items():
            yield f"latent:{name}", discriminator
        if self.similarity_discriminator is not None:
            yield "similarity", self.similarity_discriminator

    @property
    def discriminators(self) -> Mapping[str, nn.Module]:
        """Return a non-registering compatibility view of all discriminators."""
        return dict(self.named_discriminator_modules())

    def named_discriminator_parameters(
        self,
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for module_name, discriminator in self.named_discriminator_modules():
            for parameter_name, parameter in discriminator.named_parameters():
                yield f"{module_name}.{parameter_name}", parameter

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_discriminator_parameters():
            yield parameter

    def named_generator_parameters(
        self,
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        discriminator_ids = {id(parameter) for parameter in self.discriminator_parameters()}
        for name, parameter in self.named_parameters():
            if id(parameter) not in discriminator_ids:
                yield name, parameter

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_generator_parameters():
            yield parameter

    def load_legacy_translator_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ):
        warnings.warn(
            "Loading a legacy three-discriminator checkpoint for translator-only "
            "inference. Newly initialized v4 discriminators are ignored.",
            UserWarning,
            stacklevel=2,
        )
        target_state = {
            name: value
            for name, value in self.state_dict().items()
            if (
                not name.startswith(self._DISCRIMINATOR_STATE_PREFIXES)
                and name != self._LEGACY_PROVENANCE_STATE_KEY
            )
        }
        filtered_state = {
            name: value for name, value in state_dict.items() if name in target_state
        }
        missing = sorted(set(target_state) - set(filtered_state))
        unexpected = sorted(
            name
            for name in state_dict
            if name not in target_state
            and not name.startswith(self._DISCRIMINATOR_STATE_PREFIXES)
            and name != self._LEGACY_PROVENANCE_STATE_KEY
        )
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Legacy translator state is incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )
        incompatible = self.load_state_dict(filtered_state, strict=False)
        self._freeze_legacy_for_inference()
        return incompatible

    def add_encoder(self, name: str, dim: int, overwrite: bool = False):
        if name in self.in_adapters and not overwrite:
            print(f"Encoder {name} already exists, skipping...")
            return
        if name not in self.in_adapters:
            raise ValueError(
                "pps_vec2vec_v2 has a fixed two-encoder topology. Construct a "
                "new two-sided model instead of adding a third encoder."
            )

        self.in_adapters[name] = MLPWithResidual(
            self.config.adapter_depth, dim, self.config.d_hidden, self.config.d_adapter,
            self.config.norm_style, self.config.weight_init,
        )
        self.out_adapters[name] = MLPWithResidual(
            self.config.adapter_depth, self.config.d_adapter, self.config.d_hidden, dim,
            self.config.norm_style, self.config.weight_init,
        )
        self.output_discriminators[name] = Discriminator(
            dim, self.config.disc_dim, self.config.disc_depth, self.config.weight_init
        )
        self.latent_discriminators[name] = Discriminator(
            self.config.d_adapter,
            self.config.disc_dim,
            self.config.disc_depth,
            self.config.weight_init,
        )

        try:
            encoder_index = self.config.encoder_names.index(name)
        except ValueError as exc:
            raise RuntimeError(
                f"Encoder {name!r} exists in the model but not in its config"
            ) from exc
        self.config.encoder_dims[encoder_index] = dim

    def _get_latent(
        self,
        emb: torch.Tensor,
        encoder_name: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.in_adapters[encoder_name](emb)
        return self.transform(z)

    def _decode(
        self,
        latent: torch.Tensor,
        encoder_name: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.out_adapters[encoder_name](latent)
        if self.config.output_l2_normalize:
            out = F.normalize(out, p=2, dim=1)
        return out

    def translate(
        self,
        embeddings: torch.Tensor,
        src: str,
        tgt: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self._get_latent(embeddings, src, attention_mask)
        return self._decode(latent, tgt, attention_mask)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        noise_level: float = None,
        return_latents: bool = False,
    ) -> Vec2VecOutput:
        noise_level = noise_level if noise_level is not None else self.config.noise_level

        reconstructions: Dict[str, torch.Tensor] = {}
        translations: Dict[str, Dict[str, torch.Tensor]] = {}
        latents: Dict[str, torch.Tensor] = {}

        for src_name, emb in inputs.items():
            if self.training and noise_level > 0.0:
                emb = emb + torch.randn_like(emb) * noise_level
                if self.config.input_l2_normalize:
                    emb = F.normalize(emb, p=2, dim=1)

            latent = self._get_latent(emb, src_name)
            if return_latents:
                latents[src_name] = latent

            for tgt_name in inputs.keys():
                decoded = self._decode(latent, tgt_name)
                if tgt_name == src_name:
                    reconstructions[src_name] = decoded
                else:
                    if tgt_name not in translations:
                        translations[tgt_name] = {}
                    translations[tgt_name][src_name] = decoded

        return Vec2VecOutput(
            reconstructions=reconstructions,
            translations=translations,
            latents=latents if return_latents else None,
        )


class Vec2VecLearnedPooling(Vec2VecModel):
    """
    Vec2Vec variant that pools matrix embeddings (b, L, d) internally with a
    learned AttentionPooler before the standard translator.
    """

    config_class = Vec2VecConfig

    def __init__(self, config: Vec2VecConfig):
        super().__init__(config)
        encoder_dims = config.get_encoder_dims_dict()

        self.input_l2_normalize = config.input_l2_normalize
        self.poolers = nn.ModuleDict()
        for name, dim in encoder_dims.items():
            intermediate_size = correction_fn_256(config.expansion_ratio, dim)
            self.poolers[name] = AttentionPooler(
                hidden_size=dim,
                intermediate_size=intermediate_size,
                n_tokens=1,
            )

        self.post_init()

    def add_encoder(self, name: str, dim: int, overwrite: bool = False):
        super().add_encoder(name, dim, overwrite)
        if name not in self.poolers or overwrite:
            intermediate_size = correction_fn_256(self.config.expansion_ratio, dim)
            self.poolers[name] = AttentionPooler(
                hidden_size=dim,
                intermediate_size=intermediate_size,
                n_tokens=1,
            )

    def pool(
        self,
        embeddings: torch.Tensor,
        encoder_name: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.poolers[encoder_name](embeddings, attention_mask).squeeze(1)
        if self.input_l2_normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled

    def _get_latent(
        self,
        emb: torch.Tensor,
        encoder_name: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if emb.dim() == 3:
            emb = self.pool(emb, encoder_name, attention_mask)
        z = self.in_adapters[encoder_name](emb)
        return self.transform(z)

    def translate(
        self,
        embeddings: torch.Tensor,
        src: str,
        tgt: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self._get_latent(embeddings, src, attention_mask)
        return self._decode(latent, tgt)


# =============================================================================
# Protify integration
# =============================================================================

presets = {
    # Legacy, unversioned three-discriminator Hub artifacts. The loader below
    # accepts these only through the explicit frozen inference migration path.
    # Reviewed pps_vec2vec_v2 results must use an explicit versioned checkpoint
    # until replacement Hub destinations are published.
    'vec2vec-ESM2-8-ESM2-35': 'lhallee/ESM2-8-ESM2-35-sequence-sequence',
    'vec2vec-ESM2-8-ESM2-150': 'lhallee/ESM2-8-ESM2-150-sequence-sequence',
    'vec2vec-ESM2-8-ESM2-650': 'lhallee/ESM2-8-ESM2-650-sequence-sequence',
    'vec2vec-ESM2-8-ESM2-3B': 'lhallee/ESM2-8-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-35-ESM2-150': 'lhallee/ESM2-35-ESM2-150-sequence-sequence',
    'vec2vec-ESM2-35-ESM2-650': 'lhallee/ESM2-35-ESM2-650-sequence-sequence',
    'vec2vec-ESM2-35-ESM2-3B': 'lhallee/ESM2-35-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-150-ESM2-650': 'lhallee/ESM2-150-ESM2-650-sequence-sequence',
    'vec2vec-ESM2-150-ESM2-3B': 'lhallee/ESM2-150-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-650-ESM2-3B': 'lhallee/ESM2-650-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-650-ModernBERT-base-contrastive': 'lhallee/ESM2-650-ModernBERT-base-sequence-sequence-contrastive',
    'vec2vec-ESM2-650-ModernBERT-large-contrastive': 'lhallee/ESM2-650-ModernBERT-large-sequence-sequence-contrastive',
}


class Vec2VecTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: Any):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        return self.tokenizer(sequences, **kwargs)


class Vec2VecForEmbedding(nn.Module):
    """
    Wraps a frozen base PLM + a Vec2Vec translator so Protify sees a single
    "encoder" whose output is the *translated* embedding.

    Direction convention (matches ProteinRepresentationEnhancement training):
        - model_name_a := encoder_names[0]  (source side at training time)
        - model_name_b := encoder_names[1]  (target side)
        - base_model loads model_name_a and its hidden states are pooled
          (mean+var, unless learned_pooling is set) then translated A -> B.

    For the canonical small-to-big ablation, pairs are trained with the SMALLER
    model as encoder_names[0], so Protify embeds with the cheap model and
    returns an approximation of the larger model's pooled embedding.
    """

    # forward() returns a fully pooled + translated (B, D) vector; callers
    # (e.g. Protify's embedder) must not apply their own pooler on top.
    already_pooled: bool = True

    def __init__(
        self,
        config: Vec2VecConfig,
        base_model: nn.Module,
        vec2vec_model: Vec2VecModel,
        model_name_a: str,
        model_name_b: str,
    ):
        super().__init__()
        self.base_model = base_model
        self.vec2vec_model = vec2vec_model
        self.config = config
        self.learned_pooling = bool(getattr(config, 'learned_pooling', False))
        self.pooler = None if self.learned_pooling else Pooler(['mean', 'var'])
        self.model_name_a = model_name_a
        self.model_name_b = model_name_b
        self.input_l2_normalize = config.input_l2_normalize
        if config.input_standardize:
            if self.learned_pooling:
                raise ValueError(
                    "Train-fitted standardization is incompatible with "
                    "learned-pooling Vec2Vec checkpoints"
                )
            scaler_mean, scaler_scale = _source_scaler_from_config(config)
        else:
            scaler_mean = torch.empty(0, dtype=torch.float64)
            scaler_scale = torch.empty(0, dtype=torch.float64)
        self.register_buffer(
            "_source_scaler_mean",
            scaler_mean,
            persistent=False,
        )
        self.register_buffer(
            "_source_scaler_scale",
            scaler_scale,
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        # input_ids: (b, l); attention_mask: (b, l) or None
        base_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        if isinstance(base_output, torch.Tensor):
            base_state = base_output
        elif isinstance(base_output, tuple):
            base_state = base_output[0]
        elif hasattr(base_output, "last_hidden_state"):
            base_state = base_output.last_hidden_state
        else:
            raise TypeError(
                "Vec2Vec source encoder must return a tensor, a tensor-first "
                "tuple, or an object with last_hidden_state"
            )
        if not isinstance(base_state, torch.Tensor) or base_state.ndim != 3:
            raise ValueError(
                "Vec2Vec source encoder must produce residue embeddings with "
                "shape (batch, sequence, hidden)"
            )
        # base_state: (b, l, d_a)
        # Translator weights are loaded fp32; under autocast, base_state may be
        # bf16 which collides with the Linear weight dtype. Cast base output to
        # the translator's parameter dtype so autocast does not hand a bf16
        # input to an fp32 Linear mid-way through the wrapper.
        translator_dtype = next(self.vec2vec_model.parameters()).dtype
        if self.learned_pooling:
            # Translator has its own AttentionPooler; pass raw hidden states.
            translated = self.vec2vec_model.translate(
                base_state.to(translator_dtype),
                src=self.model_name_a,
                tgt=self.model_name_b,
                attention_mask=attention_mask,
            )  # (b, d_b)
        else:
            base_vec = self.pooler(  # (b, 2 * d_a)
                base_state,
                attention_mask=attention_mask,
            )
            if self.config.input_standardize:
                if base_vec.shape[1] != self._source_scaler_mean.numel():
                    raise ValueError(
                        "Pooled source width does not match the checkpoint "
                        "preprocessing scaler"
                    )
                mean = self._source_scaler_mean.to(  # (2 * d_a,)
                    device=base_vec.device,
                    dtype=base_vec.dtype,
                )
                scale = self._source_scaler_scale.to(  # (2 * d_a,)
                    device=base_vec.device,
                    dtype=base_vec.dtype,
                )
                base_vec = (base_vec - mean) / scale  # (b, 2 * d_a)
            if self.input_l2_normalize:
                epsilon = max(1e-12, torch.finfo(base_vec.dtype).tiny)
                base_vec = F.normalize(  # (b, 2 * d_a)
                    base_vec,
                    p=2,
                    dim=1,
                    eps=epsilon,
                )
            translated = self.vec2vec_model.translate(
                base_vec.to(translator_dtype),
                src=self.model_name_a,
                tgt=self.model_name_b,
            )  # (b, d_b)
        return translated  # (b, d_b)


def _source_encoder_spec(config: Vec2VecConfig) -> Tuple[str, Optional[str]]:
    """Resolve the source-side Protify dispatch name and checkpoint path."""

    source_name = config.encoder_names[0]
    source_path = None
    encoder_paths = getattr(config, "encoder_paths", None)
    if encoder_paths and len(encoder_paths) >= 1:
        candidate = encoder_paths[0]
        if candidate and candidate != "model_a" and not (
            candidate == source_name and source_name in all_presets_with_paths
        ):
            source_path = candidate
    if source_path is None:
        source_path = all_presets_with_paths.get(source_name)
    return source_name, source_path


def get_vec2vec_tokenizer(preset: str, model_path: str = None):
    path = model_path or all_presets_with_paths[preset]
    config, _ = _load_vec2vec_config_for_inference(path)
    source_name, source_path = _source_encoder_spec(config)
    if "vec2vec" in source_name.lower():
        raise ValueError("A Vec2Vec checkpoint cannot use Vec2Vec as its source encoder")

    # Import lazily to avoid a module-level cycle with get_base_models.
    from .get_base_models import get_tokenizer

    tokenizer = get_tokenizer(source_name, model_path=source_path)
    if tokenizer is None:
        raise ValueError("Vec2Vec source encoder requires a tokenizer")
    return tokenizer


def _load_vec2vec_config_for_inference(model_path: str):
    """Load current config metadata or explicitly migrate a legacy preset."""

    config_dict, _ = Vec2VecConfig.get_config_dict(model_path)
    legacy = not {
        "architecture_version",
        "discriminator_layout",
    }.issubset(config_dict)
    if legacy:
        warnings.warn(
            "Loading an unversioned three-discriminator Vec2Vec preset for "
            "frozen translator-only inference. Continued training is "
            "prohibited.",
            UserWarning,
            stacklevel=2,
        )
        config_dict = dict(config_dict)
        config_dict.update(
            {
                "architecture_version": "pps_vec2vec_v2",
                "discriminator_layout": "output2_latent2",
                "legacy_inference_only": True,
                "checkpoint_provenance": (
                    "legacy_translator_inference_only"
                ),
            }
        )
    return Vec2VecConfig.from_dict(config_dict), legacy


def build_vec2vec_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype = None,
    model_path: str = None,
    **kwargs,
):
    if masked_lm:
        raise ValueError("Masked LM is not supported for Vec2VecForEmbedding")

    model_path = model_path or presets[preset]
    config, legacy = _load_vec2vec_config_for_inference(model_path)

    # Preserve training-time direction: encoder_names[0] is the source side.
    encoder_names = config.encoder_names
    assert len(encoder_names) >= 2, f"Vec2Vec checkpoint needs >=2 encoders, got {encoder_names}"
    model_name_a = encoder_names[0]
    model_name_b = encoder_names[1]

    if "vec2vec" in model_name_a.lower():
        raise ValueError("A Vec2Vec checkpoint cannot use Vec2Vec as its source encoder")
    source_name, source_path = _source_encoder_spec(config)

    # Use Protify's family adapters so Vec2Vec follows the same FastPLMs 1.0
    # loading, dtype, tokenizer, and model-specific batching contracts as every
    # other source encoder.
    from .get_base_models import get_base_model

    base_model, base_tokenizer = get_base_model(
        source_name,
        dtype=dtype,
        model_path=source_path,
    )

    translator_cls = Vec2VecLearnedPooling if getattr(config, 'learned_pooling', False) else Vec2VecModel
    if legacy:
        vec2vec_model = (
            translator_cls.from_legacy_pretrained_for_inference(
                model_path,
                config=config,
            )
        )
    else:
        vec2vec_model = translator_cls.from_pretrained(
            model_path,
            config=config,
        )

    model = Vec2VecForEmbedding(config, base_model, vec2vec_model, model_name_a, model_name_b)
    return model, base_tokenizer


def get_vec2vec_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    raise ValueError("Vec2VecForTraining is not supported yet")


if __name__ == '__main__':
    # py -m src.protify.base_models.vec2vec
    model, tokenizer = build_vec2vec_model('vec2vec-ESM2-8-ESM2-35')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
