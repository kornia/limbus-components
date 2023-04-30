import pytest
import asyncio

import torch
from limbus.core import NoValue, ComponentState
from limbus.widgets import WidgetState
from limbus import widgets

from limbus_components_dev.base import Constant, Adder, ImageShow, Printer


class TestConstant:
    @pytest.mark.parametrize("value", ([1, 2., torch.tensor(3.)]))
    def test_smoke(self, value):
        c = Constant("k", value)
        assert ComponentState.INITIALIZED in c.state
        assert c.name == "k"
        assert isinstance(c.outputs.out.value, NoValue)
        asyncio.run(c())
        assert ComponentState.OK in c.state
        assert c.outputs.out.value == value


class TestAdder:
    def test_smoke(self):
        add = Adder("add")
        assert ComponentState.INITIALIZED in add.state
        assert add.name == "add"
        print(add.outputs.out.value)
        assert isinstance(add.outputs.out.value, NoValue)

        add.inputs.a.value = torch.tensor(2.)
        add.inputs.b.value = torch.tensor(3.)
        asyncio.run(add())
        assert ComponentState.OK in add.state
        assert add.outputs.out.value == torch.tensor(5.)


class TestImageShow:
    def test_viz_enabled(self):
        show = ImageShow("show")
        assert ComponentState.INITIALIZED in show.state
        assert show.name == "show"
        assert show.properties.title.value == ""
        assert show.properties.nrow.value is None

        show.inputs.image.value = torch.zeros((1, 3, 20, 20))
        asyncio.run(show())
        # by default the viz is done in Console. Images cannot be shown in Console and returns a log.warning.
        # log.warning("Console visualization does not show images.")
        # anyway the ComponentState returned is Ok in this case.
        assert ComponentState.OK in show.state

    def test_viz_disabled(self):
        # disable the Console visualization
        widgets.get()._enabled = False

        try:
            show = ImageShow("show")
            assert ComponentState.INITIALIZED in show.state
            show.inputs.image.value = torch.zeros((1, 3, 20, 20))
            asyncio.run(show())
            assert ComponentState.DISABLED in show.state
        except Exception as e:
            raise e
        finally:
            widgets.get()._enabled = True

    def test_viz_name(self):
        show = ImageShow("show")
        show.properties.title.init_property("my title")
        assert show.name == "show"
        assert show.properties.title.value == "my title"
        asyncio.run(show())
        assert ComponentState.OK in show.state

    def test_set_properties(self):
        assert ImageShow.WIDGET_STATE == WidgetState.ENABLED
        show = ImageShow("show")
        show.widget_state = WidgetState.ENABLED
        assert show.name == "show"
        show.properties.title.init_property("my_title")
        show.properties.nrow.init_property(2)
        assert show.properties.title.value == "my_title"
        assert show.properties.nrow.value == 2


class TestPrinter:
    def test_viz_enabled(self):
        printer = Printer("printer")
        assert printer.name == "printer"
        assert printer.properties.title.value == ""
        assert printer.properties.append.value is False

        printer.inputs.inp.value = torch.zeros((1, 3, 20, 20))
        asyncio.run(printer())
        assert ComponentState.OK in printer.state

    def test_viz_disabled(self):
        # disable the Console visualization
        widgets.get()._enabled = False

        try:
            printer = Printer("printer")
            printer.inputs.inp.value = torch.zeros((1, 3, 20, 20))
            asyncio.run(printer())
            assert ComponentState.DISABLED in printer.state
        except Exception as e:
            raise e
        finally:
            widgets.get()._enabled = True

    def test_viz_name(self):
        printer = Printer("printer")
        printer.properties.title.init_property("my title")
        assert printer.name == "printer"
        assert printer.properties.title.value == "my title"
        asyncio.run(printer())
        assert ComponentState.OK in printer.state

    def test_set_properties(self):
        assert Printer.WIDGET_STATE == WidgetState.ENABLED
        printer = Printer("printer")
        printer.widget_state = WidgetState.ENABLED
        assert printer.name == "printer"
        printer.properties.title.init_property("my_title")
        printer.properties.append.init_property(True)
        assert printer.properties.title.value == "my_title"
        assert printer.properties.append.value is True
