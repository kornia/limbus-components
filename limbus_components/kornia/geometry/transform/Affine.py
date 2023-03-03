"""Component for kornia.geometry.transform.Affine."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Affine(Component):
    r"""Affine.

    Args:
        name (str): name of the component.
        angle (typing.Optional[torch.Tensor], optional): Check original documentation. Default: None.
        translation (typing.Optional[torch.Tensor], optional): Check original documentation. Default: None.
        scale_factor (typing.Optional[torch.Tensor], optional): Check original documentation. Default: None.
        shear (typing.Optional[torch.Tensor], optional): Check original documentation. Default: None.
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

    Apply multiple elementary affine transforms simultaneously.

    Args:
        angle: Angle in degrees for counter-clockwise rotation around the center. The tensor
            must have a shape of (B), where B is the batch size.
        translation: Amount of pixels for translation in x- and y-direction. The tensor must
            have a shape of (B, 2), where B is the batch size and the last dimension contains dx and dy.
        scale_factor: Factor for scaling. The tensor must have a shape of (B), where B is the
            batch size.
        shear: Angles in degrees for shearing in x- and y-direction around the center. The
            tensor must have a shape of (B, 2), where B is the batch size and the last dimension contains sx and sy.
        center: Transformation center in pixels. The tensor must have a shape of (B, 2), where
            B is the batch size and the last dimension contains cx and cy. Defaults to the center of image to be
            transformed.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Raises:
        RuntimeError: If not one of ``angle``, ``translation``, ``scale_factor``, or ``shear`` is set.

    Returns:
        The transformed tensor with same shape as input.

    Example:
        >>> img = torch.rand(1, 2, 3, 5)
        >>> angle = 90. * torch.rand(1)
        >>> out = Affine(angle)(img)
        >>> print(out.shape)
        torch.Size([1, 2, 3, 5])

    """
    class InputsTyping(InputParams):  # noqa: D106
        input: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self,
                 name: str,
                 angle: typing.Optional[torch.Tensor] = None,
                 translation: typing.Optional[torch.Tensor] = None,
                 scale_factor: typing.Optional[torch.Tensor] = None,
                 shear: typing.Optional[torch.Tensor] = None,
                 center: typing.Optional[torch.Tensor] = None,
                 mode: str = "bilinear",
                 padding_mode: str = "zeros",
                 align_corners: bool = True) -> None:
        super().__init__(name)
        self._obj = kornia.geometry.transform.Affine(angle,
                                                     translation,
                                                     scale_factor,
                                                     shear,
                                                     center,
                                                     mode,
                                                     padding_mode,
                                                     align_corners)
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
