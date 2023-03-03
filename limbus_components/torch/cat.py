"""Component for torch.cat."""
import typing
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Cat(Component):
    r"""Cat.

    Args:
        name (str): name of the component.

    Input params:
        tensors (typing.List[torch.Tensor]): Check original documentation.
        dim (int, optional): Check original documentation. Default: 0.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    cat(tensors, dim=0, *, out=None) -> Tensor

    Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating
    dimension) or be empty.

    :func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
    and :func:`torch.chunk`.

    :func:`torch.cat` can be best understood via examples.

    Args:
        tensors (sequence of Tensors): any python sequence of tensors of the same type.
            Non-empty tensors provided must have the same shape, except in the
            cat dimension.
        dim (int, optional): the dimension over which the tensors are concatenated

    Keyword Args:
        out (Tensor, optional): the output tensor.

    Example::

        >>> x = torch.randn(2, 3)
        >>> x
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> torch.cat((x, x, x), 0)
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> torch.cat((x, x, x), 1)
        tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
                 -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
                 -0.5790,  0.1497]])

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
        self._callable = torch.cat

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
