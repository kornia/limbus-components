"""Component for kornia.contrib.FaceDetector."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class FaceDetector(Component):
    r"""FaceDetector.

    Args:
        name (str): name of the component.
        top_k (int, optional): Check original documentation. Default: 5000.
        confidence_threshold (float, optional): Check original documentation. Default: 0.3.
        nms_threshold (float, optional): Check original documentation. Default: 0.3.
        keep_top_k (int, optional): Check original documentation. Default: 750.

    Input params:
        image (torch.Tensor): Check original documentation.

    Output params:
        out (typing.List[torch.Tensor]): Check original documentation.

    Original documentation
    ----------------------

    Detect faces in a given image using a CNN.

    By default, it uses the method described in :cite:`facedetect-yu`.

    Args:
        top_k: the maximum number of detections to return before the nms.
        confidence_threshold: the threshold used to discard detections.
        nms_threshold: the threshold used by the nms for iou.
        keep_top_k: the maximum number of detections to return after the nms.

    Return:
        A list of B tensors with shape :math:`(N,15)` to be used with :py:class:`kornia.contrib.FaceDetectorResult`.

    Example:
        >>> img = torch.rand(1, 3, 320, 320)
        >>> detect = FaceDetector()
        >>> res = detect(img)

    """
    class InputsTyping(InputParams):  # noqa: D106
        image: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self,
                 name: str,
                 top_k: int = 5000,
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.3,
                 keep_top_k: int = 750) -> None:
        super().__init__(name)
        self._obj = kornia.contrib.FaceDetector(top_k, confidence_threshold, nms_threshold, keep_top_k)
        self._callable = self._obj.forward

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("image", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", typing.List[torch.Tensor])

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        image = await self._inputs.image.receive()
        out = self._callable(image=image)
        await self._outputs.out.send(out)
        return ComponentState.OK
