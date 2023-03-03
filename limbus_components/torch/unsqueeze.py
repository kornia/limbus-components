"""Component for torch.unsqueeze."""
import typing
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Unsqueeze(Component):
    r"""Unsqueeze.

    Args:
        name (str): name of the component.

    Input params:
        input (torch.Tensor): Check original documentation.
        dim (int): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    unsqueeze(input, dim) -> Tensor

    Returns a new tensor with a dimension of size one inserted at the
    specified position.

    The returned tensor shares the same underlying data with this tensor.

    A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``
    can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
    applied at :attr:`dim` = ``dim + input.dim() + 1``.

    Args:
        input (Tensor): the input tensor.
        dim (int): the index at which to insert the singleton dimension

    Example::

        >>> x = torch.tensor([1, 2, 3, 4])
        >>> torch.unsqueeze(x, 0)
        tensor([[ 1,  2,  3,  4]])
        >>> torch.unsqueeze(x, 1)
        tensor([[ 1],
                [ 2],
                [ 3],
                [ 4]])

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
        self._callable = torch.unsqueeze

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("input", torch.Tensor)
        inputs.declare("dim", int)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        input, dim = await asyncio.gather(
            self._inputs.input.receive(),
            self._inputs.dim.receive()
        )
        out = self._callable(input=input, dim=dim)
        await self._outputs.out.send(out)
        return ComponentState.OK
