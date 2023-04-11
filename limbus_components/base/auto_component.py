from typing import Callable
import inspect
import asyncio

from limbus.core import Component, ComponentState


def extract_function_signature(function: Callable):
    sig = inspect.signature(function)
    parameters = sig.parameters
    return_annotation = sig.return_annotation
    return parameters, return_annotation


def auto_generate_doc_from_function(function: Callable):
    new_doc = f"""
    Original documentation
    ----------------------
    {function.__doc__}
    """
    return new_doc


class AutoComponent(Component):
    async def forward(self) -> ComponentState:
        input_values = await asyncio.gather(
            *[getattr(self._inputs, param).receive() for param in self.parameters.keys()]
        )
        out = self._callable(
            **{k:v for k, v in zip(self.parameters.keys(), input_values)}
        )
        await self._outputs.out.send(out)
        return ComponentState.OK


class ComponentFactory(Component):
    """Register Limbus component automatically."""

    @staticmethod
    def from_function(function: Callable) -> Component:
        parameters, return_annotation = extract_function_signature(function)

        def f(name: str):
            comp = AutoComponent(name)
            comp.__doc__ = auto_generate_doc_from_function(function)
            for value in parameters.values():
                if value.default == inspect._empty:
                    comp.inputs.declare(value.name, value.annotation)
                else:
                    comp.inputs.declare(value.name, value.annotation, value.default)
            comp.outputs.declare("out", return_annotation)

            input_cls_annotations = {value.name: value.annotation for value in parameters.values()}
            output_cls_annotations = {"out": return_annotation}
            comp._inputs.__annotations__ = input_cls_annotations
            comp._outputs.__annotations__ = output_cls_annotations

            setattr(comp, "_callable", function)
            setattr(comp, "parameters", parameters)
            setattr(comp, "return_annotation", return_annotation)
            return comp
        return f
