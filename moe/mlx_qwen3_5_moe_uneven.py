"""mlx-lm model type "qwen3_5_moe_uneven": Qwen3.5-MoE whose layers each keep a different number of routed experts
(xmodel README 6h). text_config["uneven_experts"] lists the count per layer; everything else is the stock model.

build_uneven_mlx.py copies this file into mlx_lm/models/ as qwen3_5_moe_uneven.py, which is where mlx-lm looks a
model type up, so `mlx_lm server`, adapters and `mlx_lm.load` work on such a model unchanged.
"""

import copy

from .qwen3_5_moe import Model as _Base
from .qwen3_5_moe import ModelArgs  # noqa: F401
from .qwen3_next import Qwen3NextSparseMoeBlock


class Model(_Base):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        text = self.language_model.args
        layers = self.language_model.model.layers
        counts = args.text_config["uneven_experts"]
        assert len(counts) == len(layers), (len(counts), len(layers))
        for layer, n in zip(layers, counts):
            a = copy.copy(text)
            a.num_experts = n
            layer.mlp = Qwen3NextSparseMoeBlock(a)
