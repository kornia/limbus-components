"""Component for kornia.augmentation.RandomCrop."""
import typing
import kornia
import torch
import inspect
import asyncio

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState

NoneType = type(None)


class RandomCrop(Component):
    r"""RandomCrop.

    Args:
        name (str): name of the component.
        size (typing.Tuple[int, int]): Check original documentation.
        padding (typing.Union[int, typing.Tuple[int, int], typing.Tuple[int, int, int, int], NoneType], optional):
            Check original documentation. Default: None.
        pad_if_needed (typing.Optional[bool], optional): Check original documentation. Default: False.
        fill (int, optional): Check original documentation. Default: 0.
        padding_mode (str, optional): Check original documentation. Default: "constant".
        resample (typing.Union[str, int, kornia.constants.Resample], optional):
            Check original documentation. Default: "BILINEAR".
        same_on_batch (bool, optional): Check original documentation. Default: False.
        align_corners (bool, optional): Check original documentation. Default: True.
        p (float, optional): Check original documentation. Default: 1.0.
        keepdim (bool, optional): Check original documentation. Default: False.
        cropping_mode (str, optional): Check original documentation. Default: "slice".

    Input params:
        input (torch.Tensor): Check original documentation.
        params (typing.Optional[typing.Dict[str, torch.Tensor]], optional):
            Check original documentation. Default: None.
        kwargs (inspect._empty): Check original documentation.

    Output params:
        out (torch.Tensor): Check original documentation.

    Original documentation
    ----------------------

    Crop random patches of a tensor image on a given size.

    .. image:: _static/img/RandomCrop.png

    Args:
        size: Desired output size (out_h, out_w) of the crop.
            Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
        padding: Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed: It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
        padding_mode: Type of padding. Should be: constant, reflect, replicate.
        resample: the interpolation mode.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation for the whole batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
        cropping_mode: The used algorithm to crop. ``slice`` will use advanced slicing to extract the tensor based
                       on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                       to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                       differentiability.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> inputs = torch.arange(1*1*3*3.).view(1, 1, 3, 3)
        >>> aug = RandomCrop((2, 2), p=1., cropping_mode="resample")
        >>> out = aug(inputs)
        >>> out
        tensor([[[[3., 4.],
                  [6., 7.]]]])
        >>> aug.inverse(out, padding_mode="replicate")
        tensor([[[[3., 4., 4.],
                  [3., 4., 4.],
                  [6., 7., 7.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomCrop((2, 2), p=1., cropping_mode="resample")
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """
    class InputsTyping(InputParams):  # noqa: D106
        input: InputParam
        params: InputParam
        kwargs: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self,
                 name: str,
                 size: typing.Tuple[int, int],
                 padding: typing.Union[int, typing.Tuple[int, int], typing.Tuple[int, int, int, int], NoneType] = None,
                 pad_if_needed: typing.Optional[bool] = False,
                 fill: int = 0,
                 padding_mode: str = "constant",
                 resample: typing.Union[str, int, kornia.constants.Resample] = "BILINEAR",
                 same_on_batch: bool = False,
                 align_corners: bool = True,
                 p: float = 1.0,
                 keepdim: bool = False,
                 cropping_mode: str = "slice") -> None:
        super().__init__(name)
        self._obj = kornia.augmentation.RandomCrop(size,
                                                   padding,
                                                   pad_if_needed,
                                                   fill,
                                                   padding_mode,
                                                   resample,
                                                   same_on_batch,
                                                   align_corners,
                                                   p,
                                                   keepdim,
                                                   cropping_mode)
        self._callable = self._obj.forward

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("input", torch.Tensor)
        inputs.declare("params", typing.Optional[typing.Dict[str, torch.Tensor]], None)
        inputs.declare("kwargs", inspect._empty)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        input, params, kwargs = await asyncio.gather(
            self._inputs.input.receive(),
            self._inputs.params.receive(),
            self._inputs.kwargs.receive()
        )
        out = self._callable(input=input, params=params, kwargs=kwargs)
        await self._outputs.out.send(out)
        return ComponentState.OK
