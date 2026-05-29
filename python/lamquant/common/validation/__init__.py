"""
LamQuant validation and evaluation module.

This package contains:
- seizure_metrics: Clinical metrics for seizure detection (adapted from NEDC)
- nedc_formatter: NEDC-compatible CSV formatter for predictions
- edf_cross_check: Cross-validates our EDF reader against pyedflib
- nedc_erdr_runner: Subprocess wrapper for Temple's ERDR real-time decoder
"""

from .seizure_metrics import (
    SeizureMetrics,
    sensitivity,
    specificity,
    precision,
    recall,
    f1_score,
    matthews_cc,
    false_alarm_rate_24h,
    accuracy,
    per_label_confusion_matrix,
)
from .nedc_formatter import (
    NedcEventFormatter,
    format_predictions_for_nedc,
)
from .edf_cross_check import (
    CrossCheckResult,
    ChannelDiff,
    cross_check_edf,
)
from .nedc_erdr_runner import (
    NedcErdrRunner,
    InstallationStatus,
    DecodeResult,
    parse_csvbi,
)

__all__ = [
    'SeizureMetrics',
    'sensitivity',
    'specificity',
    'precision',
    'recall',
    'f1_score',
    'matthews_cc',
    'false_alarm_rate_24h',
    'accuracy',
    'per_label_confusion_matrix',
    'NedcEventFormatter',
    'format_predictions_for_nedc',
    'CrossCheckResult',
    'ChannelDiff',
    'cross_check_edf',
    'NedcErdrRunner',
    'InstallationStatus',
    'DecodeResult',
    'parse_csvbi',
]
