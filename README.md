# Limbus-Components: Set of basic components for Limbus

[![CI](https://github.com/kornia/limbus-components/actions/workflows/ci.yml/badge.svg)](https://github.com/kornia/limbus-components/actions/workflows/ci.yml)

This package is part of [Limbus](https://github.com/kornia/limbus) and contains a bunch of useful components.

## Installation

This package can be automatically installed by Limbus. If you want to install it manually, you can do it with:

```bash
git clone https://github.com/kornia/limbus-components
cd limbus-components
pip install -e .
```

For development purposes, you can install it with:

```bash
pip install -e .[dev]  # also installs limbus
```

### Dev Requirements

To add new components that must be done in the folder `limbus_components_dev` and then in order to release the changes you must run the the `release` script.

All the `__init__.py` files inside the folder `limbus_components_dev` must have the imports following the next patterns:

```python
    from . import <component_module>  # to import modules
    from .<component_module> import <component_class0>, <component_class1>, ...  # to import components
    # IMPORTANT NOTE: imports with () are not allowed
```

if they are not following these patterns, the `release` script will fail. Check examples in the folder `limbus_components_dev`.
