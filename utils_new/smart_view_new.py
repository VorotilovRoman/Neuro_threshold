"""Compatibility alias; implementation lives in ui.image_view."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module('ui.image_view')
