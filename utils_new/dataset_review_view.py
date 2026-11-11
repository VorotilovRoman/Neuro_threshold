"""Compatibility alias; implementation lives in ui.annotation_review_view."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module('ui.annotation_review_view')
