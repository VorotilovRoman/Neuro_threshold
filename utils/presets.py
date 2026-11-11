"""Compatibility alias; implementation lives in ui.segmentation.preset_bindings."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module('ui.segmentation.preset_bindings')
