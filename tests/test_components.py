import pytest
import asyncio

import torch
from limbus.core import NoValue, ComponentState
from limbus.widgets import WidgetState
from limbus import widgets

from limbus_components.base import Constant, Adder, ImageShow, Printer  # type: ignore


class TestConstant:
    @pytest.mark.parametrize("value", ([1, 2., torch.tensor(3.)]))
    def test_smoke(self, value):
        c = Constant("k", value)
        assert c.state == ComponentState.INITIALIZED
        assert c.name == "k"
        assert isinstance(c.outputs.out.value, NoValue)
        asyncio.run(c())
        assert c.state == ComponentState.OK
        assert c.outputs.out.value == value


class TestAdder:
    def test_smoke(self):
        add = Adder("add")
        assert add.state == ComponentState.INITIALIZED
        assert add.name == "add"
        print(add.outputs.get_param("out"))
        assert isinstance(add.outputs.get_param("out"), NoValue)

        add.inputs.a.value = torch.tensor(2.)
        add.inputs.b.value = torch.tensor(3.)
        asyncio.run(add())
        assert add.state == ComponentState.OK
        assert add.outputs.out.value == torch.tensor(5.)


class TestImageShow:
    def test_viz_enabled(self):
        show = ImageShow("show")
        assert show.state == ComponentState.INITIALIZED
        assert show.name == "show"
        assert show.properties.get_param("title") == ""
        assert show.properties.get_param("nrow") is None

        show.inputs.image.value = torch.zeros((1, 3, 20, 20))
        asyncio.run(show())
        # by default the viz is done in Console. Images cannot be shown in Console and returns a log.warning.
        # log.warning("Console visualization does not show images.")
        # anyway the ComponentState returned is Ok in this case.
        assert show.state == ComponentState.OK

    def test_viz_disabled(self):
        # disable the Console visualization
        widgets.get()._enabled = False

        try:
            show = ImageShow("show")
            assert show.state == ComponentState.INITIALIZED
            show.inputs.image.value = torch.zeros((1, 3, 20, 20))
            asyncio.run(show())
            assert show.state == ComponentState.DISABLED
        except Exception as e:
            raise e
        finally:
            widgets.get()._enabled = True

    def test_viz_name(self):
        show = ImageShow("show")
        show.properties.set_param("title", "my title")
        assert show.name == "show"
        assert show.properties.get_param("title") == "my title"
        asyncio.run(show())
        assert show.state == ComponentState.OK

    def test_set_properties(self):
        assert ImageShow.WIDGET_STATE == WidgetState.ENABLED
        show = ImageShow("show")
        show.widget_state = WidgetState.ENABLED
        assert show.name == "show"
        assert show.set_properties(title="my_title", nrow=2)
        assert show.properties.get_param("title") == "my_title"
        assert show.properties.get_param("nrow") == 2
        assert not show.set_properties(title="my_title", xyz=2)


class TestPrinter:
    def test_viz_enabled(self):
        printer = Printer("printer")
        assert printer.name == "printer"
        assert printer.properties.get_param("title") == ""
        assert printer.properties.get_param("append") is False

        printer.inputs.inp.value = torch.zeros((1, 3, 20, 20))
        asyncio.run(printer())
        assert printer.state == ComponentState.OK

    def test_viz_disabled(self):
        # disable the Console visualization
        widgets.get()._enabled = False

        try:
            printer = Printer("printer")
            printer.inputs.inp.value = torch.zeros((1, 3, 20, 20))
            asyncio.run(printer())
            assert printer.state == ComponentState.DISABLED
        except Exception as e:
            raise e
        finally:
            widgets.get()._enabled = True

    def test_viz_name(self):
        printer = Printer("printer")
        printer.properties.set_param("title", "my title")
        assert printer.name == "printer"
        assert printer.properties.get_param("title") == "my title"
        asyncio.run(printer())
        assert printer.state == ComponentState.OK

    def test_set_properties(self):
        assert Printer.WIDGET_STATE == WidgetState.ENABLED
        printer = Printer("printer")
        printer.widget_state = WidgetState.ENABLED
        assert printer.name == "printer"
        assert printer.set_properties(title="my_title", append=True)
        assert printer.properties.get_param("title") == "my_title"
        assert printer.properties.get_param("append") is True
        assert not printer.set_properties(title="my_title", xyz=2)
