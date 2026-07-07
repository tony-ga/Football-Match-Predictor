"""
Centralized Path Configuration for the Football Match Predictor project.

This module defines canonical paths for all data directories and provides
utilities for detecting duplicate datasets across legacy and canonical locations.

Canonical Structure:
    predicciones/
        data/
            derived/       - Derived datasets (team_match_stats.jsonl, etc.)
            cache/
                espn/      - ESPN API response cache
            ratings_wc2026.json
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory (predicciones/).
    
    Returns:
        Path to the predicciones directory
    """
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """
    Get the canonical data directory.
    
    Returns:
        Path to predicciones/data/
    """
    return get_project_root() / "data"


def get_derived_dir() -> Path:
    """
    Get the canonical derived datasets directory.
    
    Returns:
        Path to predicciones/data/derived/
    """
    return get_data_dir() / "derived"


def get_cache_dir(espn: bool = False) -> Path:
    """
    Get the canonical cache directory.
    
    Args:
        espn: If True, return the ESPN-specific cache subdirectory
        
    Returns:
        Path to cache directory
    """
    base = get_data_dir() / "cache"
    if espn:
        return base / "espn"
    return base


def ensure_dirs_exist() -> None:
    """
    Ensure all canonical data directories exist.
    
    Creates directories if they don't exist:
    - data/
    - data/derived/
    - data/cache/
    - data/cache/espn/
    """
    get_derived_dir().mkdir(parents=True, exist_ok=True)
    get_cache_dir().mkdir(parents=True, exist_ok=True)
    get_cache_dir(espn=True).mkdir(parents=True, exist_ok=True)


def detect_duplicate_datasets() -> List[Tuple[str, Path, Path, int, int]]:
    """
    Detect duplicate dataset files in legacy and canonical locations.
    
    Checks for duplicates between:
    - /workspace/data/derived/ (legacy)
    - /workspace/predicciones/data/derived/ (canonical)
    
    Returns:
        List of tuples: (filename, legacy_path, canonical_path, legacy_size, canonical_size)
        Only includes files that exist in both locations.
    """
    # Legacy location (root of repo outside predicciones/)
    project_root = get_project_root()
    # Go up one level from predicciones/ to get repo root
    repo_root = project_root.parent if project_root.name == "predicciones" else project_root
    legacy_derived = repo_root / "data" / "derived"
    
    canonical_derived = get_derived_dir()
    
    duplicates = []
    target_files = [
        "team_match_stats.jsonl",
        "player_match_stats.jsonl", 
        "match_events.jsonl",
    ]
    
    for filename in target_files:
        legacy_path = legacy_derived / filename
        canonical_path = canonical_derived / filename
        
        if legacy_path.exists() and canonical_path.exists():
            legacy_size = legacy_path.stat().st_size
            canonical_size = canonical_path.stat().st_size
            duplicates.append((
                filename,
                legacy_path,
                canonical_path,
                legacy_size,
                canonical_size,
            ))
            
            if legacy_size != canonical_size:
                logger.warning(
                    f"DUPLICATE DATASET DETECTED: '{filename}' exists in both "
                    f"legacy ({legacy_path}, {legacy_size} bytes) and "
                    f"canonical ({canonical_path}, {canonical_size} bytes) locations "
                    f"with DIFFERENT sizes. Using canonical location."
                )
            else:
                logger.info(
                    f"Duplicate dataset '{filename}' found in both locations "
                    f"(same size: {legacy_size} bytes). Using canonical location."
                )
    
    return duplicates


def get_canonical_dataset_path(filename: str, check_legacy: bool = True) -> Path:
    """
    Get the canonical path for a dataset file, with optional legacy detection.
    
    Args:
        filename: Name of the dataset file (e.g., "team_match_stats.jsonl")
        check_legacy: If True, check for legacy duplicates and warn
        
    Returns:
        Path to the canonical dataset location
    """
    canonical_path = get_derived_dir() / filename
    
    if check_legacy and canonical_path.exists():
        # Check for legacy duplicate
        project_root = get_project_root()
        repo_root = project_root.parent if project_root.name == "predicciones" else project_root
        legacy_path = repo_root / "data" / "derived" / filename
        
        if legacy_path.exists():
            legacy_size = legacy_path.stat().st_size
            canonical_size = canonical_path.stat().st_size
            
            if legacy_size != canonical_size:
                logger.warning(
                    f"Dataset '{filename}' has divergent copies: "
                    f"legacy={legacy_size} bytes, canonical={canonical_size} bytes. "
                    f"Using canonical location: {canonical_path}"
                )
            else:
                logger.debug(
                    f"Dataset '{filename}' exists in both locations (same size). "
                    f"Using canonical: {canonical_path}"
                )
    
    return canonical_path


# Canonical path constants (computed at import time)
PROJECT_ROOT = get_project_root()
DATA_DIR = get_data_dir()
DERIVED_DIR = get_derived_dir()
CACHE_DIR = get_cache_dir()
ESPN_CACHE_DIR = get_cache_dir(espn=True)

# Ensure directories exist on import
ensure_dirs_exist()

# Check for duplicates on import (non-blocking, just logs warnings)
try:
    _duplicates = detect_duplicate_datasets()
except Exception as e:
    logger.debug(f"Could not check for duplicate datasets: {e}")
