"""Some predefined components."""
import time
from typing import Any, List, Optional, Tuple
from pathlib import Path
import logging
import asyncio

import cv2
import numpy as np
import PIL
import torch
import kornia
from limbus.core import Component, ComponentState, Params, InputParams, OutputParams, NoValue
from limbus.widgets import WidgetState, WidgetComponent, BaseWidgetComponent
from limbus import widgets


log = logging.getLogger(__name__)


class ImageReader(WidgetComponent):
    """Component that read images.

    Args:
        name (str): component name.
        path (Path): path to an image or image folder.
        batch_size (int): number of images to read in a batch. Default: 1.

    Outpus:
        image (torch.Tensor): a batch of images (NxCxHxW).

    Viz params:
        title (str): title of the window. Default: "".

    """
    # the viz state by default is disabled but can be enabled by the user with the widget_state property.
    WIDGET_STATE: WidgetState = WidgetState.DISABLED

    def __init__(self, name: str, path: Path, batch_size: int = 1):
        super().__init__(name)
        self._value: List[Path] = []
        self._batch_size = batch_size
        self._idx = 0
        if Path(path).is_dir():
            self._value = sorted(list(Path(path).glob("*")))
        else:
            self._value.append(Path(path))

    @staticmethod
    def register_outputs(outputs: Params) -> None:  # noqa: D102
        outputs.declare("image", torch.Tensor)

    @staticmethod
    def register_properties(properties: Params) -> None:  # noqa: D102
        properties.declare("title", str, "")

    async def forward(self) -> ComponentState:  # noqa: D102
        images: List[torch.Tensor] = []
        batch_size = 0
        while batch_size < self._batch_size:
            if self._idx >= len(self._value):
                return ComponentState.STOPPED
            try:
                self._idx += 1
                images.append(
                    kornia.image_to_tensor(np.asarray(PIL.Image.open(str(self._value[self._idx]))))
                )
                batch_size += 1
            except:
                # avoid crashing the whole pipeline when there is a corrupted image or non-image file
                pass
        batch = torch.stack(images)
        # images must be in the range [0, 1]
        batch = batch.div(255.).clamp(0, 1)
        widgets.get().show_images(self, self._properties.get_param("title"), batch)
        await self._outputs.image.send(batch)
        return ComponentState.OK


class Webcam(WidgetComponent):
    """Component that read images from a webcam.

    Args:
        name (str): component name.
        batch_size (int): number of images to read in a batch. Default: 1.

    Outpus:
        image (torch.Tensor): a batch of images (NxCxHxW).

    Viz params:
        title (str): title of the window. Default: "".

    """
    # the viz state by default is disabled but can be enabled by the user with the widget_state property.
    WIDGET_STATE: WidgetState = WidgetState.DISABLED

    def __init__(self, name: str, batch_size: int = 1):
        super().__init__(name)
        self._batch_size = batch_size
        self._cap = cv2.VideoCapture(0)
        self._width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps: float = self._cap.get(cv2.CAP_PROP_FPS)

    def finish_pipeline(self):  # noqa: D102
        self._cap.release()

    @staticmethod
    def register_outputs(outputs: Params) -> None:  # noqa: D102
        outputs.declare("image", torch.Tensor)

    @staticmethod
    def register_properties(properties: Params) -> None:  # noqa: D102
        properties.declare("title", str, "")
        # NOTE: When a widget does not have a title we assing the name of the component, so if we have >1
        # widgets without default title we will have several widgets with the same title and will be overriden.
        # TODO: allow to have several widgets with the same title.
        properties.declare("text_title", str, "txt")

    async def forward(self) -> ComponentState:  # noqa: D102
        images: List[torch.Tensor] = []
        batch_size = 0
        while batch_size < self._batch_size:
            frame: np.ndarray
            ret: bool
            ret, frame = self._cap.read()
            if not ret:
                return ComponentState.STOPPED
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(
                kornia.image_to_tensor(frame)
            )
            batch_size += 1
            if batch_size > 1:
                time.sleep(1 / self._fps)

        batch = torch.stack(images)
        # images must be in the range [0, 1]
        batch = batch.div(255.).clamp(0, 1)
        widgets.get().show_images(self, self._properties.get_param("title"), batch)
        widgets.get().show_text(self, self._properties.get_param("text_title"),
                                f"{self._fps} fps, {self._width}x{self._height}")
        await self._outputs.image.send(batch)
        return ComponentState.OK


class DrawBoxes(Component):
    """Component that draw boxes on images.

    Args:
        name (str): component name.

    Inputs:
        images (torch.Tensor): a batch of images (BxCxHxW).
        boxes (List[torch.Tensor]): a batch of boxes Bx(Nx4x2).

    Outputs:
        image (torch.Tensor): a batch of images (BxCxHxW).

    """
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def register_inputs(inputs: Params) -> None:  # noqa: D102
        inputs.declare("images", torch.Tensor)
        inputs.declare("boxes", List[torch.Tensor])

    @staticmethod
    def register_outputs(outputs: Params) -> None:  # noqa: D102
        outputs.declare("out", torch.Tensor)

    def _draw_boxes(self, images: torch.Tensor, boxes: List[torch.Tensor]) -> torch.Tensor:
        imgs: np.ndarray = kornia.tensor_to_image(images, keepdim=True)
        imgs = (imgs * 255).astype(np.uint8)
        res_imgs: List[np.ndarray] = []
        for idx, image in enumerate(imgs):
            for box in boxes[idx]:
                image = cv2.polylines(image.copy(), [box.int().numpy()], True, (0, 255, 0), 4)
            res_imgs.append(image)
        return kornia.image_to_tensor(np.stack(res_imgs))

    async def forward(self) -> ComponentState:  # noqa: D102
        images, boxes = await asyncio.gather(self._inputs.images.receive(),
                                             self._inputs.boxes.receive())
        await self.outputs.out.send(self._draw_boxes(images, boxes))
        return ComponentState.OK


class CropBoxes(Component):
    """Component that crop the input image using the bounding boxes.

    Args:
        name (str): component name.

    Inputs:
        images (torch.Tensor): a batch of images (BxCxHxW).
        boxes (List[torch.Tensor]): a batch of boxes Bx(Nx4x2).

    Outputs:
        faces (torch.Tensor): a batch of faces (FxCxHxW).

    """
    def __init__(self, name: str, size: Tuple[int, int]):
        super().__init__(name)
        self._size: Tuple[int, int] = size

    @staticmethod
    def register_inputs(inputs: Params) -> None:  # noqa: D102
        inputs.declare("images", torch.Tensor)
        inputs.declare("boxes", List[torch.Tensor])

    @staticmethod
    def register_outputs(outputs: Params) -> None:  # noqa: D102
        outputs.declare("crops", torch.Tensor)

    async def forward(self) -> ComponentState:  # noqa: D102
        images: torch.Tensor
        boxes_x_image: List[torch.Tensor]
        images, boxes_x_image = await asyncio.gather(self._inputs.images.receive(),
                                                     self._inputs.boxes.receive())
        crops_x_image: List[torch.Tensor] = []
        for idx in range(len(boxes_x_image)):
            boxes = boxes_x_image[idx]
            image = images[idx]
            if boxes.shape[0] == 0:
                crops_x_image.append(torch.empty((0, images.shape[0], *self._size)).to(image))
            else:
                crops_x_image.append(kornia.geometry.transform.crop_and_resize(
                    image.expand(boxes.shape[0], *image.shape), boxes, self._size))
        await self._outputs.crops.send(torch.cat(crops_x_image))
        return ComponentState.OK


class ImageShow(BaseWidgetComponent):
    """Component to show images.

    Args:
        name (str): component name.

    Inputs:
        image (torch.Tensor): a batch of images (NxCxHxW).

    Viz params:
        title (str): title of the window. Default: "".
        nrow (int, optional): number of images to show in a row. Default: None.

    """

    @staticmethod
    def register_properties(properties: Params) -> None:  # noqa: D102
        # this line is like super() but for static methods.
        BaseWidgetComponent.register_properties(properties)  # adds the title param
        properties.declare("nrow", Optional[int], value=None)

    @staticmethod
    def register_inputs(inputs: Params) -> None:  # noqa: D102
        inputs.declare("image", torch.Tensor)

    async def _show(self, title: str) -> None:  # noqa: D102
        images = await self._inputs.image.receive()
        if isinstance(images, NoValue):
            images = torch.empty((0, 0, 0, 0))
        if images.numel() == 0:
            # TODO: temporal solution, replace by something better
            images = torch.zeros((1, 1, max(images.shape[2], 1), max(images.shape[3], 1))).to(images)
        widgets.get().show_images(self, title, images, nrow=self._properties.get_param("nrow"))


class Constant(Component):
    """Component that holds a constant.

    Args:
        name (str): component name.
        value (Any): constant value.

    Outputs:
        out (Any): constant value. Same value as the arg value.

    """
    def __init__(self, name: str, value: Any):
        super().__init__(name)
        self._value = value

    @staticmethod
    def register_outputs(outputs: Params) -> None:  # noqa: D102
        outputs.declare("out", Any, arg="value")

    async def forward(self) -> ComponentState:  # noqa: D102
        # TODO: next line could be autogenerated since in register_inputs() we are already linking both.
        await self._outputs.out.send(self._value)
        return ComponentState.OK


class Printer(BaseWidgetComponent):
    """Component to print the input in the console or in a text window if viz available.

    Args:
        name (str): component name.

    Inputs:
        inp (Any): input to print.

    Viz params:
        title (str): title of the window. Default: "".
        append (bool, optional): If True, the text is appended to the previous text. Default: False.

    """
    @staticmethod
    def register_properties(properties: Params) -> None:  # noqa: D102
        # this line is like super() but for static methods.
        BaseWidgetComponent.register_properties(properties)  # adds the title param
        properties.declare("append", bool, value=False)

    @staticmethod
    def register_inputs(inputs: Params) -> None:  # noqa: D102
        inputs.declare("inp", Any)

    async def _show(self, title: str) -> None:  # noqa: D102
        widgets.get().show_text(self, title,
                                str(await self._inputs.inp.receive()),
                                append=self._properties.get_param("append"))


class Accumulator(Component):
    """Component to accumulate sequential data.

    Args:
        name (str): component name.

    Inputs:
        in (Any): sequential input data source.

    Outputs:
        out (List[Any]): List with accumulated data.

    """
    def __init__(self, name: str, elements: int = 1):
        super().__init__(name)
        self._elements: int = elements

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:  # noqa: D102
        inputs.declare("inp", Any)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:  # noqa: D102
        outputs.declare("out", List[Any])

    async def forward(self) -> ComponentState:  # noqa: D102
        res: List[int] = []
        while len(res) < self._elements:
            res.append(await self._inputs.inp.receive())

        await self._outputs.out.send(res)
        return ComponentState.OK


# Example of a simple component created from the API
class Adder(Component):
    """Component to add two inputs and output the result.

    Args:
        name (str): component name.

    Inputs:
        a (torch.Tensor): first input.
        b (torch.Tensor): second input.

    Outputs:
        out (torch.Tensor): sum of the inputs.

    """
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def register_inputs(inputs: Params) -> None:  # noqa: D102
        inputs.declare("a", torch.Tensor)
        inputs.declare("b", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: Params) -> None:  # noqa: D102
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:  # noqa: D102
        a, b = await asyncio.gather(self._inputs.a.receive(), self._inputs.b.receive())
        await self._outputs.out.send(a + b)
        return ComponentState.OK


# temporal classes while we solve pending issues. TODO: allow components as parameters
class ImageStitcher(Component):
    """Component to stitch images together.

    Args:
        name (str): component name.
        estimator (str): method to compute homography, either "vanilla" or "ransac".
            Default: "ransac".
        blending_method (str): method to blend two images together.
            Only "naive" is currently supported. Default: "naive".

    Inputs:
        imgs (torch.Tensor): images to be stitched.

    Outputs:
        out (torch.Tensor): stitched image.

    """
    def __init__(self, name: str, estimator: str = 'ransac', blending_method: str = 'naive'):
        super().__init__(name)
        gftt_hardnet_matcher = kornia.feature.LocalFeatureMatcher(kornia.feature.GFTTAffNetHardNet(500),
                                                                  kornia.feature.DescriptorMatcher('snn', 0.8))
        self._is = kornia.contrib.ImageStitcher(gftt_hardnet_matcher, estimator, blending_method)

    @staticmethod
    def register_inputs(inputs: Params) -> None:  # noqa: D102
        inputs.declare("imgs", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: Params) -> None:  # noqa: D102
        outputs.declare("out", torch.Tensor)

    async def forward(self) -> ComponentState:  # noqa: D102
        await self._outputs.out.send(self._is(*((await self._inputs.imgs.receive()).unsqueeze(1))))
        return ComponentState.OK


class ImageRegistrator(Component):
    """Component to register images.

    Args:
        name (str): component name.

    Inputs:
        img_src (torch.Tensor): source image.
        img_dst (torch.Tensor): destination image.

    Outputs:
        homo (torch.Tensor): homography.

    """
    def __init__(self, name: str):
        super().__init__(name)
        self._ir = kornia.geometry.ImageRegistrator()

    @staticmethod
    def register_inputs(inputs: Params) -> None:  # noqa: D102
        inputs.declare("img_src", torch.Tensor)
        inputs.declare("img_dst", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: Params) -> None:  # noqa: D102
        outputs.declare("homo", torch.Tensor)

    async def forward(self) -> ComponentState:  # noqa: D102
        img_src, img_dst = await asyncio.gather(self._inputs.img_src.receive(),
                                                self._inputs.img_dst.receive())
        await self._outputs.homo.send(self._ir.register(img_src, img_dst))
        return ComponentState.OK
