import kornia
from limbus_components.base.auto_component import ComponentFactory

AdjustHue = ComponentFactory.from_function(kornia.enhance.adjust.adjust_hue)
AdjustBrightness = ComponentFactory.from_function(kornia.enhance.adjust.adjust_brightness)
AdjustContrast = ComponentFactory.from_function(kornia.enhance.adjust.adjust_contrast)
AdjustSaturation = ComponentFactory.from_function(kornia.enhance.adjust.adjust_saturation)
