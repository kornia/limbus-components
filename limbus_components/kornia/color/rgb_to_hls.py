"""Component for kornia.color.rgb_to_hls."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class RgbToHls(Component):
    r"""RgbToHls.

    Args:
        name (str): name of the component.

    Input params:
        image (torch.Tensor, optional): Check original documentation. Default: torch.tensor(0).

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Convert a RGB image to HLS.

    .. image:: _static/img/rgb_to_hls.png

    The image data is assumed to be in the range of (0, 1).

    NOTE: this method cannot be compiled with JIT in pytohrch < 1.7.0

    Args:
        image: RGB image to be converted to HLS with shape :math:`(*, 3, H, W)`.
        eps: epsilon value to avoid div by zero.

    Returns:
        HLS version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hls(input)  # 2x3x4x5

    """
    class InputsTyping(InputParams):  # noqa: D106
        image: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = kornia.color.rgb_to_hls

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("image", torch.Tensor, torch.tensor(0))

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
