"""Component for kornia.color.hls_to_rgb."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class HlsToRgb(Component):
    r"""HlsToRgb.

    Args:
        name (str): name of the component.

    Input params:
        image (torch.Tensor): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Convert a HLS image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: HLS image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hls_to_rgb(input)  # 2x3x4x5

    """
    class InputsTyping(InputParams):  # noqa: D106
        image: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = kornia.color.hls_to_rgb

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("image", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        image = await self._inputs.image.receive()
        out = self._callable(image=image)
        await self._outputs.out.send(out)
        return ComponentState.OK
