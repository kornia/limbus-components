"""Component for kornia.enhance.equalize_clahe."""
import typing
import kornia
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class EqualizeClahe(Component):
    r"""EqualizeClahe.

    Args:
        name (str): name of the component.

    Input params:
        input (torch.Tensor): Check original documentation.
        clip_limit (float, optional): Check original documentation. Default: 40.0.
        grid_size (typing.Tuple[int, int], optional): Check original documentation. Default: (8, 8).
        slow_and_differentiable (bool, optional): Check original documentation. Default: False.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Apply clahe equalization on the input tensor.

    .. image:: _static/img/equalize_clahe.png

    NOTE: Lut computation uses the same approach as in OpenCV, in next versions this can change.

    Args:
        input: images tensor to equalize with values in the range [0, 1] and shape :math:`(*, C, H, W)`.
        clip_limit: threshold value for contrast limiting. If 0 clipping is disabled.
        grid_size: number of tiles to be cropped in each direction (GH, GW).
        slow_and_differentiable: flag to select implementation

    Returns:
        Equalized image or images with shape as the input.

    Examples:
        >>> img = torch.rand(1, 10, 20)
        >>> res = equalize_clahe(img)
        >>> res.shape
        torch.Size([1, 10, 20])

        >>> img = torch.rand(2, 3, 10, 20)
        >>> res = equalize_clahe(img)
        >>> res.shape
        torch.Size([2, 3, 10, 20])

    """
    class InputsTyping(InputParams):  # noqa: D106
        input: InputParam
        clip_limit: InputParam
        grid_size: InputParam
        slow_and_differentiable: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = kornia.enhance.equalize_clahe

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("input", torch.Tensor)
        inputs.declare("clip_limit", float, 40.0)
        inputs.declare("grid_size", typing.Tuple[int, int], (8, 8))
        inputs.declare("slow_and_differentiable", bool, False)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        input, clip_limit, grid_size, slow_and_differentiable = await asyncio.gather(
            self._inputs.input.receive(),
            self._inputs.clip_limit.receive(),
            self._inputs.grid_size.receive(),
            self._inputs.slow_and_differentiable.receive()
        )
        out = self._callable(input=input,
                             clip_limit=clip_limit,
                             grid_size=grid_size,
                             slow_and_differentiable=slow_and_differentiable)
        await self._outputs.out.send(out)
        return ComponentState.OK
