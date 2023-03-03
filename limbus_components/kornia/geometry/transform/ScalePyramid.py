"""Component for kornia.geometry.transform.ScalePyramid."""
import typing
import kornia
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class ScalePyramid(Component):
    r"""ScalePyramid.

    Args:
        name (str): name of the component.
        n_levels (int, optional): Check original documentation. Default: 3.
        init_sigma (float, optional): Check original documentation. Default: 1.6.
        min_size (int, optional): Check original documentation. Default: 15.
        double_image (bool, optional): Check original documentation. Default: False.

    Input params:
        x (torch.Tensor): Check original documentation.

    Output params:
        out0 (typing.List[torch.Tensor]): Check original documentation.
        out1 (typing.List[torch.Tensor]): Check original documentation.
        out2 (typing.List[torch.Tensor]): Check original documentation.

    Original documentation
    ----------------------

    Create an scale pyramid of image, usually used for local feature detection.

    Images are consequently smoothed with Gaussian blur and downscaled.

    Args:
        n_levels: number of the levels in octave.
        init_sigma: initial blur level.
        min_size: the minimum size of the octave in pixels.
        double_image: add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this.

    Returns:
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output 1st: :math:`[(B, C, NL, H, W), (B, C, NL, H/2, W/2), ...]`
        - Output 2nd: :math:`[(B, NL), (B, NL), (B, NL), ...]`
        - Output 3rd: :math:`[(B, NL), (B, NL), (B, NL), ...]`

    Examples:
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = ScalePyramid(3, 15)(input)

    """
    class InputsTyping(InputParams):  # noqa: D106
        x: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out0: OutputParam
        out1: OutputParam
        out2: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self,
                 name: str,
                 n_levels: int = 3,
                 init_sigma: float = 1.6,
                 min_size: int = 15,
                 double_image: bool = False) -> None:
        super().__init__(name)
        self._obj = kornia.geometry.transform.ScalePyramid(n_levels, init_sigma, min_size, double_image)
        self._callable = self._obj.forward

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("x", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out0", typing.List[torch.Tensor])
        outputs.declare("out1", typing.List[torch.Tensor])
        outputs.declare("out2", typing.List[torch.Tensor])

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        x = await self._inputs.x.receive()
        out0, out1, out2 = self._callable(x=x)
        await asyncio.gather(
            self._outputs.out0.send(out0),
            self._outputs.out1.send(out1),
            self._outputs.out2.send(out2)
        )
        return ComponentState.OK
