"""Component for torch.squeeze."""
import typing
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Squeeze(Component):
    r"""Squeeze.

    Args:
        name (str): name of the component.

    Input params:
        input (torch.Tensor): Check original documentation.
        dim (typing.Optional[int]): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    squeeze(input, dim=None) -> Tensor

    Returns a tensor with all specified dimensions of :attr:`input` of size `1` removed.

    For example, if `input` is of shape:
    :math:`(A \times 1 \times B \times C \times 1 \times D)` then the `input.squeeze()`
    will be of shape: :math:`(A \times B \times C \times D)`.

    When :attr:`dim` is given, a squeeze operation is done only in the given
    dimension(s). If `input` is of shape: :math:`(A \times 1 \times B)`,
    ``squeeze(input, 0)`` leaves the tensor unchanged, but ``squeeze(input, 1)``
    will squeeze the tensor to the shape :math:`(A \times B)`.

    .. note:: The returned tensor shares the storage with the input tensor,
              so changing the contents of one will change the contents of the other.

    .. warning:: If the tensor has a batch dimension of size 1, then `squeeze(input)`
              will also remove the batch dimension, which can lead to unexpected
              errors. Consider specifying only the dims you wish to be squeezed.

    Args:
        input (Tensor): the input tensor.
        dim (int or tuple of ints, optional): if given, the input will be squeezed
               only in the specified dimensions.

            .. versionchanged:: 2.0
               :attr:`dim` now accepts tuples of dimensions.

    Example::

        >>> x = torch.zeros(2, 1, 2, 1, 2)
        >>> x.size()
        torch.Size([2, 1, 2, 1, 2])
        >>> y = torch.squeeze(x)
        >>> y.size()
        torch.Size([2, 2, 2])
        >>> y = torch.squeeze(x, 0)
        >>> y.size()
        torch.Size([2, 1, 2, 1, 2])
        >>> y = torch.squeeze(x, 1)
        >>> y.size()
        torch.Size([2, 2, 1, 2])
        >>> y = torch.squeeze(x, (1, 2, 3))
        torch.Size([2, 2, 2])

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
        self._callable = torch.squeeze

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("input", torch.Tensor)
        inputs.declare("dim", typing.Optional[int])

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
