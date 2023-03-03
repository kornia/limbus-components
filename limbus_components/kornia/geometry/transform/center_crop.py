"""Component for kornia.geometry.transform.center_crop."""
import typing
import kornia
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class CenterCrop(Component):
    r"""CenterCrop.

    Args:
        name (str): name of the component.

    Input params:
        input_tensor (torch.Tensor): Check original documentation.
        size (typing.Tuple[int, int]): Check original documentation.
        mode (str, optional): Check original documentation. Default: "bilinear".
        padding_mode (str, optional): Check original documentation. Default: "zeros".
        align_corners (bool, optional): Check original documentation. Default: True.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Crop the 2D images (4D tensor) from the center.

    Args:
        input_tensor: the 2D image tensor with shape (B, C, H, W).
        size: a tuple with the expected height and width
          of the output patch.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.

    Returns:
        the output tensor with patches.

    Examples:
        >>> input = torch.tensor([[[
        ...     [1., 2., 3., 4.],
        ...     [5., 6., 7., 8.],
        ...     [9., 10., 11., 12.],
        ...     [13., 14., 15., 16.],
        ...  ]]])
        >>> center_crop(input, (2, 4), mode='nearest', align_corners=True)
        tensor([[[[ 5.,  6.,  7.,  8.],
                  [ 9., 10., 11., 12.]]]])

    """
    class InputsTyping(InputParams):  # noqa: D106
        input_tensor: InputParam
        size: InputParam
        mode: InputParam
        padding_mode: InputParam
        align_corners: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = kornia.geometry.transform.center_crop

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("input_tensor", torch.Tensor)
        inputs.declare("size", typing.Tuple[int, int])
        inputs.declare("mode", str, "bilinear")
        inputs.declare("padding_mode", str, "zeros")
        inputs.declare("align_corners", bool, True)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        input_tensor, size, mode, padding_mode, align_corners = await asyncio.gather(
            self._inputs.input_tensor.receive(),
            self._inputs.size.receive(),
            self._inputs.mode.receive(),
            self._inputs.padding_mode.receive(),
            self._inputs.align_corners.receive()
        )
        out = self._callable(input_tensor=input_tensor,
                             size=size,
                             mode=mode,
                             padding_mode=padding_mode,
                             align_corners=align_corners)
        await self._outputs.out.send(out)
        return ComponentState.OK
