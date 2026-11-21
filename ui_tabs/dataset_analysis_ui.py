"""Assemble the dataset analysis workspace from focused UI pages."""

from ui.dataset_audit.page_duplicates import build_duplicates
from ui.dataset_audit.page_embeddings import build_embeddings
from ui.dataset_audit.page_models import build_models
from ui.dataset_audit.page_quality import build_quality
from ui.dataset_audit.page_report import build_report
from ui.dataset_audit.page_spatial import build_spatial
from ui.dataset_audit.page_statistics import build_statistics
from ui.dataset_audit.page_training import build_training
from ui.dataset_audit.page_wiring import connect_source
from ui.dataset_audit.source_page import build_source


def setup_dataset_analysis_ui(parent):
    build_source(parent)
    build_statistics(parent)
    build_quality(parent)
    build_duplicates(parent)
    build_spatial(parent)
    build_models(parent)
    build_embeddings(parent)
    build_training(parent)
    build_report(parent)
    connect_source(parent)
