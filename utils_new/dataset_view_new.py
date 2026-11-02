# -*- coding: utf-8 -*-
"""Compatibility alias for the V5 dataset utility module.

The implementation has moved to ``utils_new.annotation_processing``.
This module is deliberately the *same module object*, so consumers of
public/private symbols and test monkeypatches do not create split state.
"""
import sys
from utils_new import annotation_processing as _implementation

sys.modules[__name__] = _implementation
