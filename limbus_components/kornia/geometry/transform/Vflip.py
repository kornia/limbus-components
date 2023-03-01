"""Component for kornia.geometry.transform.Vflip."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, ComponentState


class Vflip(Component):
    r"""Vflip.

    Args:
        name (str): name of the component.

    Input params:
        input (torch.Tensor): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Vertically flip a tensor image or a batch of tensor images.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The vertically flipped image tensor.

    Examples:
        >>> vflip = Vflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> vflip(input)
        tensor([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])

    """
    def __init__(self, name: str, ) -> None:
        super().__init__(name)
        self._obj = kornia.geometry.transform.Vflip()
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
