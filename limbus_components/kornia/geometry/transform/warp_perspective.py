"""Component for kornia.geometry.transform.warp_perspective."""
import typing
import kornia
import torch
from torch import tensor
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class WarpPerspective(Component):
    r"""WarpPerspective.

    Args:
        name (str): name of the component.

    Input params:
        src (torch.Tensor): Check original documentation.
        M (torch.Tensor): Check original documentation.
        dsize (typing.Tuple[int, int]): Check original documentation.
        mode (str, optional): Check original documentation. Default: "bilinear".
        padding_mode (str, optional): Check original documentation. Default: "zeros".
        align_corners (bool, optional): Check original documentation. Default: True.
        fill_value (torch.Tensor, optional): Check original documentation. Default: tensor([0., 0., 0.]).

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Apply a perspective transformation to an image.

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/warp_perspective_10_1.png

    The function warp_perspective transforms the source image using
    the specified matrix:

    .. math::
        \text{dst} (x, y) = \text{src} \left(
        \frac{M^{-1}_{11} x + M^{-1}_{12} y + M^{-1}_{13}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}} ,
        \frac{M^{-1}_{21} x + M^{-1}_{22} y + M^{-1}_{23}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}}
        \right )

    Args:
        src: input image with shape :math:`(B, C, H, W)`.
        M: transformation matrix with shape :math:`(B, 3, 3)`.
        dsize: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'`` | ``'fill'``.
        align_corners: interpolation flag.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.

    Returns:
        the warped input image :math:`(B, C, H, W)`.

    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> H = torch.eye(3)[None]
       >>> out = warp_perspective(img, H, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])

    .. note::
        This function is often used in conjunction with :func:`get_perspective_transform`.

    .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/
        latest/warp_perspective.html>`_.

    """
    class InputsTyping(InputParams):  # noqa: D106
        src: InputParam
        M: InputParam
        dsize: InputParam
        mode: InputParam
        padding_mode: InputParam
        align_corners: InputParam
        fill_value: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = kornia.geometry.transform.warp_perspective

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("src", torch.Tensor)
        inputs.declare("M", torch.Tensor)
        inputs.declare("dsize", typing.Tuple[int, int])
        inputs.declare("mode", str, "bilinear")
        inputs.declare("padding_mode", str, "zeros")
        inputs.declare("align_corners", bool, True)
        inputs.declare("fill_value", torch.Tensor, tensor([0., 0., 0.]))

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        src, M, dsize, mode, padding_mode, align_corners, fill_value = await asyncio.gather(  # noqa: N806
            self._inputs.src.receive(),
            self._inputs.M.receive(),
            self._inputs.dsize.receive(),
            self._inputs.mode.receive(),
            self._inputs.padding_mode.receive(),
            self._inputs.align_corners.receive(),
            self._inputs.fill_value.receive()
        )
        out = self._callable(src=src,
                             M=M,
                             dsize=dsize,
                             mode=mode,
                             padding_mode=padding_mode,
                             align_corners=align_corners,
                             fill_value=fill_value)
        await self._outputs.out.send(out)
        return ComponentState.OK
