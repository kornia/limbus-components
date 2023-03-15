"""Component for kornia.geometry.transform.HomographyWarper."""
import typing
import kornia
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class HomographyWarper(Component):
    r"""HomographyWarper.

    Args:
        name (str): name of the component.
        height (int): Check original documentation.
        width (int): Check original documentation.
        mode (str, optional): Check original documentation. Default: "bilinear".
        padding_mode (str, optional): Check original documentation. Default: "zeros".
        normalized_coordinates (bool, optional): Check original documentation. Default: True.
        align_corners (bool, optional): Check original documentation. Default: False.

    Input params:
        patch_src (torch.Tensor): Check original documentation.
        src_homo_dst (typing.Optional[torch.Tensor], optional): Check original documentation. Default: None.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Warp tensors by homographies.

    .. math::

        X_{dst} = H_{src}^{\{dst\}} * X_{src}

    Args:
        height: The height of the destination tensor.
        width: The width of the destination tensor.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        normalized_coordinates: whether to use a grid with normalized coordinates.
        align_corners: interpolation flag.

    """
    class InputsTyping(InputParams):  # noqa: D106
        patch_src: InputParam
        src_homo_dst: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self,
                 name: str,
                 height: int,
                 width: int,
                 mode: str = "bilinear",
                 padding_mode: str = "zeros",
                 normalized_coordinates: bool = True,
                 align_corners: bool = False) -> None:
        super().__init__(name)
        self._obj = kornia.geometry.transform.HomographyWarper(height,
                                                               width,
                                                               mode,
                                                               padding_mode,
                                                               normalized_coordinates,
                                                               align_corners)
        self._callable = self._obj.forward

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("patch_src", torch.Tensor)
        inputs.declare("src_homo_dst", typing.Optional[torch.Tensor], None)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        patch_src, src_homo_dst = await asyncio.gather(
            self._inputs.patch_src.receive(),
            self._inputs.src_homo_dst.receive()
        )
        out = self._callable(patch_src=patch_src, src_homo_dst=src_homo_dst)
        await self._outputs.out.send(out)
        return ComponentState.OK
