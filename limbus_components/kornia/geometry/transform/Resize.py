"""Component for kornia.geometry.transform.Resize."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class Resize(Component):
    r"""Resize.

    Args:
        name (str): name of the component.
        size (typing.Union[int, typing.Tuple[int, int]]): Check original documentation.
        interpolation (str, optional): Check original documentation. Default: "bilinear".
        align_corners (typing.Optional[bool], optional): Check original documentation. Default: None.
        side (str, optional): Check original documentation. Default: "short".
        antialias (bool, optional): Check original documentation. Default: False.

    Input params:
        input (torch.Tensor): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Resize the input torch.Tensor to the given size.

    Args:
        size: Desired output size. If size is a sequence like (h, w),
            output size will be matched to this. If size is an int, smaller edge of the image will
            be matched to this number. i.e, if height > width, then image will be rescaled
            to (size * height / width, size)
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            'bicubic' | 'trilinear' | 'area'.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The resized tensor with the shape of the given size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = Resize((6, 8))(img)
        >>> print(out.shape)
        torch.Size([1, 3, 6, 8])

    .. raw:: html

        <gradio-app space="kornia/kornia-resize-antialias"></gradio-app>

    """
    class InputsTyping(InputParams):  # noqa: D106
        input: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self,
                 name: str,
                 size: typing.Union[int, typing.Tuple[int, int]],
                 interpolation: str = "bilinear",
                 align_corners: typing.Optional[bool] = None,
                 side: str = "short",
                 antialias: bool = False) -> None:
        super().__init__(name)
        self._obj = kornia.geometry.transform.Resize(size, interpolation, align_corners, side, antialias)
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
