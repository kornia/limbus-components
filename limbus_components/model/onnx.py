from typing import List
import torch
import asyncio
try:
    import onnx
    import onnxruntime
finally:
    pass

from limbus.core import Component, ComponentState


class ONNXComponent(Component):
    """ONNX model component.

    Loads any ONNX model.

    Args:
        name: the name of the component.
        onnx_file: the onnx file path.

    Example:
        >>> model = ONNXComponent("mobilenet", "mobilenetv2-12-int8.onnx")
    """

    def __init__(self, name: str, onnx_file: str):
        super().__init__(name)
        self.onnx_file = onnx_file
        self.model = onnx.load(onnx_file)
        self.input_names = self.read_input_names()
        self.output_names = self.read_output_names()
        self.register_io()
        self.session = None

    def _session_singleton(self,):
        if self.session is not None:
            return self.session
        self.session = onnxruntime.InferenceSession(self.onnx_file)
        return self.session

    def read_input_names(self,) -> List[str]:
        return list([a.name for a in self.model.graph.input])

    def read_output_names(self,) -> List[str]:
        return list([a.name for a in self.model.graph.output])

    def register_io(self,):
        """Register inputs/outputs."""
        for name in self.input_names:
            self.inputs.declare(name, torch.Tensor)
        if len(self.output_names) == 1:
            self.outputs.declare("out", torch.Tensor)
        else:
            self.outputs.declare("out", [torch.Tensor] * len(self.outputs))

    async def forward(self) -> ComponentState:
        input_values = await asyncio.gather(
            *[getattr(self._inputs, param).receive() for param in self.input_names]
        )
        session = self._session_singleton()
        # Not tested on GPU. It shall support tensor types for onnxruntime-gpu
        out = session.run(None, {k: v.numpy() for k, v in zip(self.input_names, input_values)})
        if len(out) == 1:
            out = torch.from_numpy(out[0])
        else:
            out = [torch.from_numpy(o) for o in out]
        await self._outputs.out.send(out)
        return ComponentState.OK
