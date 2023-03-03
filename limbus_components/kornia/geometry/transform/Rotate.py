"""Component for kornia.geometry.transform.Rotate."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Rotate(Component):
    r"""Rotate.

    Args:
        name (str): name of the component.
        angle (torch.Tensor): Check original documentation.
        center (typing.Optional[torch.Tensor], optional): Check original documentation. Default: None.
        mode (str, optional): Check original documentation. Default: "bilinear".
        padding_mode (str, optional): Check original documentation. Default: "zeros".
        align_corners (bool, optional): Check original documentation. Default: True.

    Input params:
        input (torch.Tensor): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Rotate the tensor anti-clockwise about the centre.

    Args:
        angle: The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center: The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The rotated tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> angle = torch.tensor([90.])
        >>> out = Rotate(angle)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])

    """
    class InputsTyping(InputParams):  # noqa: D106
        input: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self,
                 name: str,
                 angle: torch.Tensor,
                 center: typing.Optional[torch.Tensor] = None,
                 mode: str = "bilinear",
                 padding_mode: str = "zeros",
                 align_corners: bool = True) -> None:
        super().__init__(name)
        self._obj = kornia.geometry.transform.Rotate(angle, center, mode, padding_mode, align_corners)
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
