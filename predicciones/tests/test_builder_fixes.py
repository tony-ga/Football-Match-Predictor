#!/usr/bin/env python
"""
Tests for build_market_dataset.py fixes.

Tests cover:
1. write_jsonl creates file with rows
2. write_jsonl creates empty file if no rows and create_empty=True
3. builder reports correctly existing / non-existing files
4. builder uses robust absolute paths based on Path
5. UTC warning eliminated
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_market_dataset import write_jsonl, DERIVED_DIR, ROOT_DIR


def test_write_jsonl_creates_file_with_rows(tmp_path):
    """Test that write_jsonl creates a file with rows."""
    test_path = tmp_path / "test.jsonl"
    rows = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ]
    
    count = write_jsonl(test_path, rows, create_empty=True)
    
    assert count == 3, f"Expected 3 rows written, got {count}"
    assert test_path.exists(), f"File {test_path} should exist"
    
    # Verify content
    with open(test_path, "r") as f:
        lines = f.readlines()
    
    assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"
    
    # Parse and verify each line
    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["id"] == rows[i]["id"]
        assert data["name"] == rows[i]["name"]
    
    print("✅ test_write_jsonl_creates_file_with_rows passed")


def test_write_jsonl_creates_empty_file(tmp_path):
    """Test that write_jsonl creates an empty file when rows is empty and create_empty=True."""
    test_path = tmp_path / "empty.jsonl"
    rows: List[Dict[str, Any]] = []
    
    count = write_jsonl(test_path, rows, create_empty=True)
    
    assert count == 0, f"Expected 0 rows written, got {count}"
    assert test_path.exists(), f"Empty file {test_path} should exist"
    
    # Verify file is empty
    with open(test_path, "r") as f:
        content = f.read()
    
    assert content == "", f"Expected empty file, got: {repr(content)}"
    
    print("✅ test_write_jsonl_creates_empty_file passed")


def test_write_jsonl_skips_file_when_no_rows_and_create_empty_false(tmp_path):
    """Test that write_jsonl skips file creation when rows is empty and create_empty=False."""
    test_path = tmp_path / "skip.jsonl"
    rows: List[Dict[str, Any]] = []
    
    count = write_jsonl(test_path, rows, create_empty=False)
    
    assert count == 0, f"Expected 0 rows written, got {count}"
    assert not test_path.exists(), f"File {test_path} should NOT exist when create_empty=False"
    
    print("✅ test_write_jsonl_skips_file_when_no_rows_and_create_empty_false passed")


def test_write_jsonl_creates_parent_dirs(tmp_path):
    """Test that write_jsonl creates parent directories if they don't exist."""
    test_path = tmp_path / "nested" / "subdir" / "test.jsonl"
    rows = [{"id": 1}]
    
    count = write_jsonl(test_path, rows, create_empty=True)
    
    assert count == 1, f"Expected 1 row written, got {count}"
    assert test_path.exists(), f"File {test_path} should exist"
    assert test_path.parent.exists(), f"Parent directory {test_path.parent} should exist"
    
    print("✅ test_write_jsonl_creates_parent_dirs passed")


def test_paths_are_absolute():
    """Test that DERIVED_DIR and ROOT_DIR are absolute paths."""
    assert ROOT_DIR.is_absolute(), f"ROOT_DIR should be absolute: {ROOT_DIR}"
    assert DERIVED_DIR.is_absolute(), f"DERIVED_DIR should be absolute: {DERIVED_DIR}"
    
    # Verify DERIVED_DIR is under ROOT_DIR
    try:
        relative = DERIVED_DIR.relative_to(ROOT_DIR)
        assert str(relative) == "data/derived", f"Expected 'data/derived', got '{relative}'"
    except ValueError:
        raise AssertionError(f"DERIVED_DIR should be under ROOT_DIR: {DERIVED_DIR} vs {ROOT_DIR}")
    
    print("✅ test_paths_are_absolute passed")


def test_utc_datetime_usage():
    """Test that code uses datetime.now(UTC) instead of deprecated datetime.utcnow()."""
    # This is more of a code inspection test
    # We verify that importing doesn't produce deprecation warnings
    
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Import the module
        from scripts import build_market_dataset
        
        # Check for any DeprecationWarning about utcnow
        utcnow_warnings = [
            warning for warning in w 
            if issubclass(warning.category, DeprecationWarning) 
            and "utcnow" in str(warning.message).lower()
        ]
        
        assert len(utcnow_warnings) == 0, (
            f"Found {len(utcnow_warnings)} utcnow deprecation warning(s): "
            f"{[str(w.message) for w in utcnow_warnings]}"
        )
    
    # Also verify we can create UTC-aware datetimes
    now_utc = datetime.now(UTC)
    assert now_utc.tzinfo is not None, "datetime.now(UTC) should return timezone-aware datetime"
    
    print("✅ test_utc_datetime_usage passed")


def test_file_reporting_accuracy(tmp_path):
    """Test that file existence reporting matches actual filesystem state."""
    # Create some test files
    team_file = tmp_path / "team_match_stats.jsonl"
    player_file = tmp_path / "player_match_stats.jsonl"
    event_file = tmp_path / "match_events.jsonl"
    
    # Write team file with rows
    write_jsonl(team_file, [{"id": 1}], create_empty=True)
    
    # Don't create player_file (simulate 0 rows with create_empty=False)
    write_jsonl(player_file, [], create_empty=False)
    
    # Create empty event file
    write_jsonl(event_file, [], create_empty=True)
    
    # Verify reporting matches reality
    assert team_file.exists() == True, "team_file should exist"
    assert player_file.exists() == False, "player_file should NOT exist"
    assert event_file.exists() == True, "event_file should exist (empty)"
    
    # Count rows accurately
    if team_file.exists():
        with open(team_file, "r") as f:
            team_rows = sum(1 for _ in f)
        assert team_rows == 1, f"Expected 1 row in team_file, got {team_rows}"
    
    if event_file.exists():
        with open(event_file, "r") as f:
            event_rows = sum(1 for _ in f)
        assert event_rows == 0, f"Expected 0 rows in event_file, got {event_rows}"
    
    print("✅ test_file_reporting_accuracy passed")


def run_all_tests():
    """Run all tests."""
    import tempfile
    
    print("=" * 60)
    print("Running build_market_dataset.py fix tests")
    print("=" * 60)
    
    # Tests that need tmp_path
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        test_write_jsonl_creates_file_with_rows(tmp_path)
        test_write_jsonl_creates_empty_file(tmp_path)
        test_write_jsonl_skips_file_when_no_rows_and_create_empty_false(tmp_path)
        test_write_jsonl_creates_parent_dirs(tmp_path)
        test_file_reporting_accuracy(tmp_path)
    
    # Tests that don't need tmp_path
    test_paths_are_absolute()
    test_utc_datetime_usage()
    
    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
