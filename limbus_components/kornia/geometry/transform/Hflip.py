"""Component for kornia.geometry.transform.Hflip."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Hflip(Component):
    r"""Hflip.

    Args:
        name (str): name of the component.

    Input params:
        input (torch.Tensor): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Horizontally flip a tensor image or a batch of tensor images.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The horizontally flipped image tensor.

    Examples:
        >>> hflip = Hflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> hflip(input)
        tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 0.]]]])

    """
    class InputsTyping(InputParams):  # noqa: D106
        input: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._obj = kornia.geometry.transform.Hflip()
        self._callable = self._obj.forward

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("input", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        input = await self._inputs.input.receive()
        out = self._callable(input=input)
        await self._outputs.out.send(out)
        return ComponentState.OK
