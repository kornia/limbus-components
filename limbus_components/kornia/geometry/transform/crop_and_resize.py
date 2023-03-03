"""Component for kornia.geometry.transform.crop_and_resize."""
import typing
import kornia
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class CropAndResize(Component):
    r"""CropAndResize.

    Args:
        name (str): name of the component.

    Input params:
        input_tensor (torch.Tensor): Check original documentation.
        boxes (torch.Tensor): Check original documentation.
        size (typing.Tuple[int, int]): Check original documentation.
        mode (str, optional): Check original documentation. Default: "bilinear".
        padding_mode (str, optional): Check original documentation. Default: "zeros".
        align_corners (bool, optional): Check original documentation. Default: True.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Extract crops from 2D images (4D tensor) and resize given a bounding box.

    Args:
        input_tensor: the 2D image tensor with shape (B, C, H, W).
        boxes : a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx4x2, where each box is defined in the following (clockwise)
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y order.
            The coordinates would compose a rectangle with a shape of (N1, N2).
        size: a tuple with the height and width that will be
            used to resize the extracted patches.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | 'reflection'.
        align_corners: mode for grid_generation.

    Returns:
        Tensor: tensor containing the patches with shape BxCxN1xN2.

    Example:
        >>> input = torch.tensor([[[
        ...     [1., 2., 3., 4.],
        ...     [5., 6., 7., 8.],
        ...     [9., 10., 11., 12.],
        ...     [13., 14., 15., 16.],
        ... ]]])
        >>> boxes = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]])  # 1x4x2
        >>> crop_and_resize(input, boxes, (2, 2), mode='nearest', align_corners=True)
        tensor([[[[ 6.,  7.],
                  [10., 11.]]]])

    """
    class InputsTyping(InputParams):  # noqa: D106
        input_tensor: InputParam
        boxes: InputParam
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
        self._callable = kornia.geometry.transform.crop_and_resize

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("input_tensor", torch.Tensor)
        inputs.declare("boxes", torch.Tensor)
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
        input_tensor, boxes, size, mode, padding_mode, align_corners = await asyncio.gather(
            self._inputs.input_tensor.receive(),
            self._inputs.boxes.receive(),
            self._inputs.size.receive(),
            self._inputs.mode.receive(),
            self._inputs.padding_mode.receive(),
            self._inputs.align_corners.receive()
        )
        out = self._callable(input_tensor=input_tensor,
                             boxes=boxes,
                             size=size,
                             mode=mode,
                             padding_mode=padding_mode,
                             align_corners=align_corners)
        await self._outputs.out.send(out)
        return ComponentState.OK
