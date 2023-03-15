"""Component for torch.select."""
import typing
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Select(Component):
    r"""Select.

    Args:
        name (str): name of the component.

    Input params:
        input (torch.Tensor): Check original documentation.
        dim (int, optional): Check original documentation. Default: 0.
        index (int, optional): Check original documentation. Default: 0.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    select(input, dim, index) -> Tensor

    Slices the :attr:`input` tensor along the selected dimension at the given index.
    This function returns a view of the original tensor with the given dimension removed.

    .. note:: If :attr:`input` is a sparse tensor and returning a view of
              the tensor is not possible, a RuntimeError exception is
              raised. In this is the case, consider using
              :func:`torch.select_copy` function.

    Args:
        input (Tensor): the input tensor.
        dim (int): the dimension to slice
        index (int): the index to select with

    .. note::

        :meth:`select` is equivalent to slicing. For example,
        ``tensor.select(0, index)`` is equivalent to ``tensor[index]`` and
        ``tensor.select(2, index)`` is equivalent to ``tensor[:,:,index]``.

    """
    class InputsTyping(InputParams):  # noqa: D106
        input: InputParam
        dim: InputParam
        index: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = torch.select

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("input", torch.Tensor)
        inputs.declare("dim", int, 0)
        inputs.declare("index", int, 0)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        input, dim, index = await asyncio.gather(
            self._inputs.input.receive(),
            self._inputs.dim.receive(),
            self._inputs.index.receive()
        )
        out = self._callable(input=input, dim=dim, index=index)
        await self._outputs.out.send(out)
        return ComponentState.OK
