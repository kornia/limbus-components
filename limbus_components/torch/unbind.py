"""Component for torch.unbind."""
import typing
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Unbind(Component):
    r"""Unbind.

    Args:
        name (str): name of the component.

    Input params:
        input (torch.Tensor): Check original documentation.
        dim (int, optional): Check original documentation. Default: 0.

    Output params:
        out (typing.Tuple[torch.Tensor, ...]): Check original documentation.

    Original documentation
    ----------------------

    unbind(input, dim=0) -> seq

    Removes a tensor dimension.

    Returns a tuple of all slices along a given dimension, already without it.

    Arguments:
        input (Tensor): the tensor to unbind
        dim (int): dimension to remove

    Example::

        >>> torch.unbind(torch.tensor([[1, 2, 3],
        >>>                            [4, 5, 6],
        >>>                            [7, 8, 9]]))
        (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))

    """
    class InputsTyping(InputParams):  # noqa: D106
        input: InputParam
        dim: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = torch.unbind

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("input", torch.Tensor)
        inputs.declare("dim", int, 0)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", typing.Tuple[torch.Tensor, ...])

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        input, dim = await asyncio.gather(
            self._inputs.input.receive(),
            self._inputs.dim.receive()
        )
        out = self._callable(input=input, dim=dim)
        await self._outputs.out.send(out)
        return ComponentState.OK
