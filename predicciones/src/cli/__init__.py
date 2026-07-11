"""
CLI Package for Football Prediction System

This package provides the command-line interface components:
- InteractiveMenu: Interactive menu system
- Command functions: predict_command, pipeline_command, etc.
"""

from predicciones.src.cli.menu import InteractiveMenu
from predicciones.src.cli.commands import (
    predict_command,
    pipeline_command,
    players_command,
    timelines_command,
    fixtures_command,
    daily_report_command,
    lambda_analysis_command,
    backtest_command,
    recent_command,
    config_command,
)

__all__ = [
    "InteractiveMenu",
    "predict_command",
    "pipeline_command",
    "players_command",
    "timelines_command",
    "fixtures_command",
    "daily_report_command",
    "lambda_analysis_command",
    "backtest_command",
    "recent_command",
    "config_command",
]
