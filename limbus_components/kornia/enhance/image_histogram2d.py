"""Component for kornia.enhance.image_histogram2d."""
import typing
import kornia
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class ImageHistogram2D(Component):
    r"""ImageHistogram2D.

    Args:
        name (str): name of the component.

    Input params:
        image (torch.Tensor): Check original documentation.
        min (float, optional): Check original documentation. Default: 0.0.
        max (float, optional): Check original documentation. Default: 255.0.
        n_bins (int, optional): Check original documentation. Default: 256.
        bandwidth (typing.Optional[float], optional): Check original documentation. Default: None.
        centers (typing.Optional[torch.Tensor], optional): Check original documentation. Default: None.
        return_pdf (bool, optional): Check original documentation. Default: False.
        kernel (str, optional): Check original documentation. Default: "triangular".
        eps (float, optional): Check original documentation. Default: 1e-10.

    Output params:
        out (torch.Tensor): Check original documentation.
        out2 (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Estimate the histogram of the input image(s).

    The calculation uses triangular kernel density estimation.

    Args:
        image: Input tensor to compute the histogram with shape
          :math:`(H, W)`, :math:`(C, H, W)` or :math:`(B, C, H, W)`.
        min: Lower end of the interval (inclusive).
        max: Upper end of the interval (inclusive). Ignored when
          :attr:`centers` is specified.
        n_bins: The number of histogram bins. Ignored when
          :attr:`centers` is specified.
        bandwidth: Smoothing factor. If not specified or equal to -1,
          :math:`(bandwidth = (max - min) / n_bins)`.
        centers: Centers of the bins with shape :math:`(n_bins,)`.
          If not specified or empty, it is calculated as centers of
          equal width bins of [min, max] range.
        return_pdf: If True, also return probability densities for
          each bin.
        kernel: kernel to perform kernel density estimation
          ``(`triangular`, `gaussian`, `uniform`, `epanechnikov`)``.

    Returns:
        Computed histogram of shape :math:`(bins)`, :math:`(C, bins)`,
          :math:`(B, C, bins)`.
        Computed probability densities of shape :math:`(bins)`, :math:`(C, bins)`,
          :math:`(B, C, bins)`, if return_pdf is ``True``. Tensor of zeros with shape
          of the histogram otherwise.

    """
    class InputsTyping(InputParams):  # noqa: D106
        image: InputParam
        min: InputParam
        max: InputParam
        n_bins: InputParam
        bandwidth: InputParam
        centers: InputParam
        return_pdf: InputParam
        kernel: InputParam
        eps: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam
        out2: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = kornia.enhance.image_histogram2d

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("image", torch.Tensor)
        inputs.declare("min", float, 0.0)
        inputs.declare("max", float, 255.0)
        inputs.declare("n_bins", int, 256)
        inputs.declare("bandwidth", typing.Optional[float], None)
        inputs.declare("centers", typing.Optional[torch.Tensor], None)
        inputs.declare("return_pdf", bool, False)
        inputs.declare("kernel", str, "triangular")
        inputs.declare("eps", float, 1e-10)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)
        outputs.declare("out2", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        image, min, max, n_bins, bandwidth, centers, return_pdf, kernel, eps = await asyncio.gather(
            self._inputs.image.receive(),
            self._inputs.min.receive(),
            self._inputs.max.receive(),
            self._inputs.n_bins.receive(),
            self._inputs.bandwidth.receive(),
            self._inputs.centers.receive(),
            self._inputs.return_pdf.receive(),
            self._inputs.kernel.receive(),
            self._inputs.eps.receive()
        )
        out, out2 = self._callable(image=image,
                                   min=min,
                                   max=max,
                                   n_bins=n_bins,
                                   bandwidth=bandwidth,
                                   centers=centers,
                                   return_pdf=return_pdf,
                                   kernel=kernel,
                                   eps=eps)
        await asyncio.gather(
            self._outputs.out.send(out),
            self._outputs.out2.send(out2)
        )
        return ComponentState.OK
