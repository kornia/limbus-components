"""Component for kornia.geometry.transform.Rescale."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Rescale(Component):
    r"""Rescale.

    Args:
        name (str): name of the component.
        factor (typing.Union[float, typing.Tuple[float, float]]): Check original documentation.
        interpolation (str, optional): Check original documentation. Default: "bilinear".
        align_corners (bool, optional): Check original documentation. Default: True.
        antialias (bool, optional): Check original documentation. Default: False.

    Input params:
        input (torch.Tensor): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Rescale the input torch.Tensor with the given factor.

    Args:
        factor: Desired scaling factor in each direction. If scalar, the value is used
            for both the x- and y-direction.
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'trilinear'`` | ``'area'``.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The rescaled tensor with the shape according to the given factor.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = Rescale((2, 3))(img)
        >>> print(out.shape)
        torch.Size([1, 3, 8, 12])

    """
    class InputsTyping(InputParams):  # noqa: D106
        input: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self,
                 name: str,
                 factor: typing.Union[float, typing.Tuple[float, float]],
                 interpolation: str = "bilinear",
                 align_corners: bool = True,
                 antialias: bool = False) -> None:
        super().__init__(name)
        self._obj = kornia.geometry.transform.Rescale(factor, interpolation, align_corners, antialias)
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
