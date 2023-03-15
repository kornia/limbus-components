"""Component for torch.stack."""
import typing
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Stack(Component):
    r"""Stack.

    Args:
        name (str): name of the component.

    Input params:
        tensors (typing.List[torch.Tensor]): Check original documentation.
        dim (int, optional): Check original documentation. Default: 0.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    stack(tensors, dim=0, *, out=None) -> Tensor

    Concatenates a sequence of tensors along a new dimension.

    All tensors need to be of the same size.

    Arguments:
        tensors (sequence of Tensors): sequence of tensors to concatenate
        dim (int): dimension to insert. Has to be between 0 and the number
            of dimensions of concatenated tensors (inclusive)

    Keyword Args:
        out (Tensor, optional): the output tensor.

    """
    class InputsTyping(InputParams):  # noqa: D106
        tensors: InputParam
        dim: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = torch.stack

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("tensors", typing.List[torch.Tensor])
        inputs.declare("dim", int, 0)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        tensors, dim = await asyncio.gather(
            self._inputs.tensors.receive(),
            self._inputs.dim.receive()
        )
        out = self._callable(tensors=tensors, dim=dim)
        await self._outputs.out.send(out)
        return ComponentState.OK
