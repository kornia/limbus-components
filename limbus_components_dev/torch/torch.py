"""Some components from torch."""
from typing import Any, List
import asyncio

import torch
from limbus.core import Component, InputParams, OutputParams, ComponentState


class Stack(Component):
    """Stack a list of tensors."""
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:  # noqa: D102
        inputs.declare("dim", int, value=0)
        inputs.declare("tensors", List[torch.Tensor])

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:  # noqa: D102
        outputs.declare("tensor", torch.Tensor)

    async def forward(self) -> ComponentState:  # noqa: D102
        dim, inp = await asyncio.gather(self._inputs.dim.receive(), self._inputs.tensors.receive())
        out = torch.stack(inp, dim=dim)
        await self._outputs.tensor.send(out)
        return ComponentState.OK


class Cat(Component):
    """Cat a list of tensors."""
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:  # noqa: D102
        inputs.declare("dim", int, value=0)
        inputs.declare("tensors", List[torch.Tensor])

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:  # noqa: D102
        outputs.declare("tensor", torch.Tensor)

    async def forward(self) -> ComponentState:  # noqa: D102
        dim, inp = await asyncio.gather(self._inputs.dim.receive(), self._inputs.tensors.receive())
        out = torch.cat(inp, dim=dim)
        await self._outputs.tensor.send(out)
        return ComponentState.OK


class Unbind(Component):
    """Unbind a tensor."""
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:  # noqa: D102
        inputs.declare("dim", int, value=0)
        inputs.declare("tensor", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:  # noqa: D102
        outputs.declare("tensors", List[torch.Tensor])

    async def forward(self) -> ComponentState:  # noqa: D102
        dim, inp = await asyncio.gather(self._inputs.dim.receive(), self._inputs.tensor.receive())
        out = list(torch.unbind(inp, dim=dim))
        await self._outputs.tensors.send(out)
        return ComponentState.OK
