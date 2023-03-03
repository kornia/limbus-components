"""Component for kornia.enhance.normalize_min_max."""
import typing
import kornia
import torch
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class NormalizeMinMax(Component):
    r"""NormalizeMinMax.

    Args:
        name (str): name of the component.

    Input params:
        x (torch.Tensor): Check original documentation.
        min_val (float, optional): Check original documentation. Default: 0.0.
        max_val (float, optional): Check original documentation. Default: 1.0.
        eps (float, optional): Check original documentation. Default: 1e-06.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Normalise an image/video tensor by MinMax and re-scales the value between a range.

    The data is normalised using the following formulation:

    .. math::
        y_i = (b - a) * \frac{x_i - \text{min}(x)}{\text{max}(x) - \text{min}(x)} + a

    where :math:`a` is :math:`\text{min_val}` and :math:`b` is :math:`\text{max_val}`.

    Args:
        x: The image tensor to be normalised with shape :math:`(B, C, *)`.
        min_val: The minimum value for the new range.
        max_val: The maximum value for the new range.
        eps: Float number to avoid zero division.

    Returns:
        The normalised image tensor with same shape as input :math:`(B, C, *)`.

    Example:
        >>> x = torch.rand(1, 5, 3, 3)
        >>> x_norm = normalize_min_max(x, min_val=-1., max_val=1.)
        >>> x_norm.min()
        tensor(-1.)
        >>> x_norm.max()
        tensor(1.0000)

    """
    class InputsTyping(InputParams):  # noqa: D106
        x: InputParam
        min_val: InputParam
        max_val: InputParam
        eps: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._callable = kornia.enhance.normalize_min_max

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("x", torch.Tensor)
        inputs.declare("min_val", float, 0.0)
        inputs.declare("max_val", float, 1.0)
        inputs.declare("eps", float, 1e-06)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        x, min_val, max_val, eps = await asyncio.gather(
            self._inputs.x.receive(),
            self._inputs.min_val.receive(),
            self._inputs.max_val.receive(),
            self._inputs.eps.receive()
        )
        out = self._callable(x=x, min_val=min_val, max_val=max_val, eps=eps)
        await self._outputs.out.send(out)
        return ComponentState.OK
