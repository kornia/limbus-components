"""Component for kornia.contrib.histogram_matching."""
import typing
import kornia
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class HistogramMatching(Component):
    r"""HistogramMatching.

    Args:
        name (str): name of the component.

    Input params:
        source (torch.Tensor): Check original documentation.
        template (torch.Tensor): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Adjust the pixel values of an image to match its histogram towards a target image.

    `Histogram matching <https://en.wikipedia.org/wiki/Histogram_matching>`_ is the transformation
    of an image so that its histogram matches a specified histogram. In this implementation, the
    histogram is computed over the flattened image array. Code referred to
    `here <https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x>`_.

    Args:
        source: Image to transform.
        template: Template image. It can have different dimensions to source.

    Returns:
        The transformed output image as the same shape as the source image.

    Note:
        This function does not matches histograms element-wisely if input a batched tensor.

    """
    class InputsTyping(InputParams):  # noqa: D106
        source: InputParam
        template: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = kornia.contrib.histogram_matching

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("source", torch.Tensor)
        inputs.declare("template", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        source, template = await asyncio.gather(
            self._inputs.source.receive(),
            self._inputs.template.receive()
        )
        out = self._callable(source=source, template=template)
        await self._outputs.out.send(out)
        return ComponentState.OK
