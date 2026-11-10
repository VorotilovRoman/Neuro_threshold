# -*- coding: utf-8 -*-
"""Legacy lazy import facade.

New code imports names directly from their owning modules. Named imports here
remain compatible; importing the facade itself does not load ML or GUI stacks.
A legacy wildcard import necessarily resolves every public export.
"""
from importlib import import_module as _import_module

_EXPORTS = {
    'TrainingImageLabel': ('ui_tabs.dataset_analysis_ui', 'TrainingImageLabel'),
    'setup_dataset_analysis_ui': ('ui_tabs.dataset_analysis_ui', 'setup_dataset_analysis_ui'),
    'setup_deep_learning_methods_ui': ('ui_tabs.deep_learning_methods_ui', 'setup_deep_learning_methods_ui'),
    'setup_deep_learning_training_ui': ('ui_tabs.deep_learning_training_ui', 'setup_deep_learning_training_ui'),
    'setup_demo_ensemble_methods_ui': ('ui_tabs.demo_ensemble_methods_ui', 'setup_demo_ensemble_methods_ui'),
    'setup_gradient_ui': ('ui_tabs.gradient_methods_ui', 'setup_gradient_ui'),
    'setup_interactive_ui': ('ui_tabs.interactive_methods_ui', 'setup_interactive_ui'),
    'setup_layout_dataset_ui': ('ui_tabs.layout_dataset_ui', 'setup_layout_dataset_ui'),
    'setup_model_research_ui': ('ui_tabs.model_research_ui', 'setup_model_research_ui'),
    'setup_preparing_dataset_yaml_ui': ('ui_tabs.preparing_dataset_yaml_ui', 'setup_preparing_dataset_yaml_ui'),
    'setup_sahi_slicer_ui': ('ui_tabs.slice_dataset_sahi_ui', 'setup_sahi_slicer_ui'),
    'setup_threshold_ui': ('ui_tabs.threshold_methods_ui', 'setup_threshold_ui'),
    'setup_traditional_ml_ui': ('ui_tabs.traditional_ml_methods_ui', 'setup_traditional_ml_ui'),
    'setup_viewing_dataset_ui': ('ui_tabs.viewing_dataset_ui', 'setup_viewing_dataset_ui'),
    'setup_yolo_ae_vae_ui': ('ui_tabs.yolo_test_aeb_ui', 'setup_yolo_ae_vae_ui'),
    'setup_yolo_find_img_ui': ('ui_tabs.yolo_find_img_ui', 'setup_yolo_find_img_ui'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = _import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
