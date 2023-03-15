"""Component for kornia.geometry.transform.Scale."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Scale(Component):
    r"""Scale.

    Args:
        name (str): name of the component.
        scale_factor (torch.Tensor): Check original documentation.
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

    Scale the tensor by a factor.

    Args:
        scale_factor: The scale factor apply. The tensor
          must have a shape of (B) or (B, 2), where B is batch size.
          If (B), isotropic scaling will perform.
          If (B, 2), x-y-direction specific scaling will perform.
        center: The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The scaled tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> scale_factor = torch.tensor([[2., 2.]])
        >>> out = Scale(scale_factor)(img)
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
                 scale_factor: torch.Tensor,
                 center: typing.Optional[torch.Tensor] = None,
                 mode: str = "bilinear",
                 padding_mode: str = "zeros",
                 align_corners: bool = True) -> None:
        super().__init__(name)
        self._obj = kornia.geometry.transform.Scale(scale_factor, center, mode, padding_mode, align_corners)
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
