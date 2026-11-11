"""Compatibility alias; implementation lives in ui.settings_service."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module('ui.settings_service')
