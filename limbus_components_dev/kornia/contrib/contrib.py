"""Some predefined components for kornia."""
from typing import List, Optional
import logging
import asyncio

import cv2
import numpy as np
import torch
import kornia
from limbus.core import Component, InputParams, OutputParams, Params, ComponentState, InputParam, OutputParam
from limbus.widgets import WidgetState, BaseWidgetComponent, WidgetComponent
from limbus import widgets


log = logging.getLogger(__name__)


class ShowFaceLandmarks(BaseWidgetComponent):
    """Component that draw face landmarks on an image.

    Args:
        name (str): component name.

    Inputs:
        image (torch.Tensor): a batch of images (NxCxHxW).
        landmarks (torch.Tensor): a batch of landmarks (Nx15).

    Viz params:
        title (str): title of the window. Default: "".

    """
    class InputsTyping(OutputParams):  # noqa: D106
        image: InputParam
        landmarks: InputParam

    inputs: InputsTyping  # type: ignore

    # the viz state by default is disabled but can be enabled by the user with the widget_state property.
    WIDGET_STATE: WidgetState = WidgetState.DISABLED

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:  # noqa: D102
        inputs.declare("image", torch.Tensor)
        inputs.declare("landmarks", torch.Tensor)

    @staticmethod
    def register_properties(properties: Params) -> None:  # noqa: D102
        # this line is like super() but for static methods.
        BaseWidgetComponent.register_properties(properties)  # adds the title param
        properties.declare("nrow", Optional[int], value=None)
        properties.declare("draw_keypoints", bool, False)
        properties.declare("threshold", float, 0.8)

    def _draw_keypoint(self, img: np.ndarray,
                       det: kornia.contrib.FaceDetectorResult,
                       kpt_type: kornia.contrib.FaceKeypoint) -> np.ndarray:
        kpt = det.get_keypoint(kpt_type).int().tolist()
        return cv2.circle(img, kpt, 2, (255, 0, 0), 2)

    def _draw_landmarks(self, image: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        dets: List[kornia.contrib.FaceDetectorResult] = [kornia.contrib.FaceDetectorResult(o) for o in landmarks]
        frame_vis: np.ndarray = kornia.tensor_to_image(image).copy()
        frame_vis = (frame_vis * 255).astype(np.uint8)
        for b in dets:
            if b.score < self._properties.get_param("threshold"):
                continue

            # draw face bounding box
            line_thickness = 4
            line_length = 20

            x1, y1 = b.top_left.int().tolist()
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1 + line_length, y1), (0, 255, 0), thickness=line_thickness)
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1, y1 + line_length), (0, 255, 0), thickness=line_thickness)

            x1, y1 = b.top_right.int().tolist()
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1 - line_length, y1), (0, 255, 0), thickness=line_thickness)
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1, y1 + line_length), (0, 255, 0), thickness=line_thickness)

            x1, y1 = b.bottom_right.int().tolist()
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1 - line_length, y1), (0, 255, 0), thickness=line_thickness)
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1, y1 - line_length), (0, 255, 0), thickness=line_thickness)

            x1, y1 = b.bottom_left.int().tolist()
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1 + line_length, y1), (0, 255, 0), thickness=line_thickness)
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1, y1 - line_length), (0, 255, 0), thickness=line_thickness)

            if self._properties.get_param("draw_keypoints"):
                # draw facial keypoints
                frame_vis = self._draw_keypoint(frame_vis, b, kornia.contrib.FaceKeypoint.EYE_LEFT)
                frame_vis = self._draw_keypoint(frame_vis, b, kornia.contrib.FaceKeypoint.EYE_RIGHT)
                frame_vis = self._draw_keypoint(frame_vis, b, kornia.contrib.FaceKeypoint.NOSE)
                frame_vis = self._draw_keypoint(frame_vis, b, kornia.contrib.FaceKeypoint.MOUTH_LEFT)
                frame_vis = self._draw_keypoint(frame_vis, b, kornia.contrib.FaceKeypoint.MOUTH_RIGHT)

        return kornia.image_to_tensor(frame_vis)

    async def _show(self, title: str) -> None:  # noqa: D102
        images, landmarks = await asyncio.gather(self.inputs.image.receive(),
                                                 self.inputs.landmarks.receive())
        images = self._draw_landmarks(images, landmarks)
        widgets.get().show_images(self, title, images, nrow=self._properties.get_param("nrow"))


class FaceDetectorToBoxes(Component):
    """Component that get the bounding boxes from the FaceDetector.

    Args:
        name (str): component name.

    Inputs:
        landmarks (torch.Tensor): a batch of landmarks (Nx15).

    Outputs:
        faces (torch.Tensor): a batch of faces (Bx4x2).

    """
    class InputsTyping(OutputParams):  # noqa: D106
        landmarks: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        boxes: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:  # noqa: D102
        inputs.declare("landmarks", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:  # noqa: D102
        outputs.declare("boxes", torch.Tensor)

    @staticmethod
    def register_properties(properties: Params) -> None:  # noqa: D102
        # this line is like super() but for static methods.
        WidgetComponent.register_properties(properties)  # adds the title param
        properties.declare("threshold", float, 0.8)

    async def forward(self) -> ComponentState:  # noqa: D102
        landmarks = await self.inputs.landmarks.receive()
        dets: List[kornia.contrib.FaceDetectorResult] = [kornia.contrib.FaceDetectorResult(o) for o in landmarks]
        bboxes: List[torch.Tensor] = []
        for b in dets:
            if b.score < self._properties.get_param("threshold"):
                continue
            # order: top-left, top-right, bottom-right and bottom-left
            bboxes.append(torch.stack((b.top_left, b.top_right, b.bottom_right, b.bottom_left)))
        if len(bboxes) > 0:
            await self.outputs.boxes.send(torch.stack(bboxes))
        else:
            await self.outputs.boxes.send(torch.empty((0, 4, 2)).to(landmarks))
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
    class InputsTyping(OutputParams):  # noqa: D106
        imgs: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

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
        await self.outputs.out.send(self._is(*((await self._inputs.imgs.receive()).unsqueeze(1))))
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
    class InputsTyping(OutputParams):  # noqa: D106
        img_src: InputParam
        img_dst: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        homo: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

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
        img_src, img_dst = await asyncio.gather(self.inputs.img_src.receive(),
                                                self.inputs.img_dst.receive())
        await self.outputs.homo.send(self._ir.register(img_src, img_dst))
        return ComponentState.OK
