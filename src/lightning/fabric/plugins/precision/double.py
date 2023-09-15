# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, ContextManager, Literal

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.nn import Module

from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager, _Float64ContextManager


class DoublePrecision(Precision):
    """Plugin for training with double (``torch.float64``) precision."""

    precision: Literal["64-true"] = "64-true"

    def convert_module(self, module: Module) -> Module:
        return module.double()

    def init_context(self) -> ContextManager:
        return _choose_context_manager(torch.double)

    def forward_context(self) -> ContextManager:
        return _choose_context_manager(torch.double)

    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.double)

    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())


def _choose_context_manager(dtype: torch.dtype) -> ContextManager:
    if torch._dynamo.is_compiling():
        # workaround for https://github.com/pytorch/pytorch/issues/109309#issuecomment-1720381886
        # TODO: raise an error if the default dtype is not float32. However, we would need to know that
        # `fullgraph=False` or else this would get raised even if graph breaks are okay. AFAIK there's no API to
        # know this.
        return _Float64ContextManager()
    return _DtypeContextManager(dtype)
