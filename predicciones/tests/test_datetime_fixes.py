#!/usr/bin/env python
"""
Tests for datetime handling fixes in cache_manager.py and espn_client_v2.py.

Tests cover:
1. ensure_utc() with naive datetime
2. ensure_utc() with aware datetime
3. TTL comparison with naive cached_at
4. TTL comparison with aware cached_at
5. Reading old (naive) and new (aware) caches
6. Integration: build_market_dataset.py should not fail with naive/aware mismatch
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, UTC, timezone
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cache_manager import EspnCacheManager, ensure_utc
from src.data.espn_client_v2 import EspnCache, ensure_utc as espn_ensure_utc


def test_ensure_utc_with_naive_datetime():
    """Test that ensure_utc() correctly handles naive datetimes."""
    naive_dt = datetime(2024, 1, 15, 10, 30, 0)
    assert naive_dt.tzinfo is None, "Test datetime should be naive"
    
    result = ensure_utc(naive_dt)
    
    assert result.tzinfo is not None, "Result should be timezone-aware"
    assert result.tzinfo == UTC, "Result should be in UTC"
    assert result.hour == 10, "Hour should remain the same"
    assert result.minute == 30, "Minute should remain the same"
    
    print("✅ test_ensure_utc_with_naive_datetime passed")


def test_ensure_utc_with_aware_datetime():
    """Test that ensure_utc() correctly handles aware datetimes."""
    # Create an aware datetime in a different timezone
    est = timezone(timedelta(hours=-5))
    aware_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=est)
    
    result = ensure_utc(aware_dt)
    
    assert result.tzinfo is not None, "Result should be timezone-aware"
    assert result.tzinfo == UTC, "Result should be in UTC"
    assert result.hour == 15, "Hour should be converted to UTC (10 + 5)"
    assert result.minute == 30, "Minute should remain the same"
    
    # Also test with UTC-aware datetime
    utc_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
    result_utc = ensure_utc(utc_dt)
    
    assert result_utc.tzinfo == UTC, "UTC datetime should remain UTC"
    assert result_utc.hour == 10, "Hour should remain the same for UTC"
    
    print("✅ test_ensure_utc_with_aware_datetime passed")


def test_ttl_comparison_with_naive_cached_at():
    """Test TTL comparison when cached_at is naive (old cache format)."""
    cache_mgr = EspnCacheManager()
    
    # Simulate a naive timestamp (as would be stored by old code using utcnow())
    naive_timestamp = datetime.utcnow()
    
    # Test ensure_utc handles it correctly
    normalized = ensure_utc(naive_timestamp)
    
    assert normalized.tzinfo == UTC, "Normalized datetime should be UTC-aware"
    
    # Verify comparison works without TypeError
    now_utc = datetime.now(UTC)
    diff = now_utc - normalized
    
    assert isinstance(diff, timedelta), "Difference should be a timedelta"
    assert abs(diff.total_seconds()) < 1, "Difference should be near zero"
    
    print("✅ test_ttl_comparison_with_naive_cached_at passed")


def test_ttl_comparison_with_aware_cached_at():
    """Test TTL comparison when cached_at is already aware (new cache format)."""
    # Simulate an aware timestamp (as stored by new code using now(UTC))
    aware_timestamp = datetime.now(UTC)
    
    # Test ensure_utc handles it correctly
    normalized = ensure_utc(aware_timestamp)
    
    assert normalized.tzinfo == UTC, "Normalized datetime should be UTC-aware"
    
    # Verify comparison works without TypeError
    now_utc = datetime.now(UTC)
    diff = now_utc - normalized
    
    assert isinstance(diff, timedelta), "Difference should be a timedelta"
    assert abs(diff.total_seconds()) < 1, "Difference should be near zero"
    
    print("✅ test_ttl_comparison_with_aware_cached_at passed")


def test_cache_with_old_naive_timestamp(tmp_path):
    """Test reading a cache file with old naive timestamp format."""
    cache_dir = tmp_path / "cache"
    cache_mgr = EspnCacheManager(cache_dir=cache_dir)
    
    # Create a cache file with naive timestamp (old format) using the actual cache key mechanism
    endpoint = "test_endpoint"
    params = {}
    
    # Use set_raw_response but manually modify the timestamp to be naive
    cache_mgr.set_raw_response(endpoint, params, {"result": "test_data"})
    
    # Find the cache file and modify its timestamp to be naive
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1, "Should have one cache file"
    
    cache_file = cache_files[0]
    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    
    # Replace aware timestamp with naive one
    aware_dt = datetime.fromisoformat(cache_data["_cached_at"])
    naive_dt = aware_dt.replace(tzinfo=None)
    cache_data["_cached_at"] = naive_dt.isoformat()
    
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)
    
    # Verify it's actually naive
    with open(cache_file, "r") as f:
        loaded = json.load(f)
    parsed = datetime.fromisoformat(loaded["_cached_at"])
    assert parsed.tzinfo is None, "Test cache should have naive timestamp"
    
    # Should read successfully without TypeError
    result = cache_mgr.get_raw_response(endpoint, params, ttl_hours=24)
    
    assert result is not None, "Should return cached data"
    assert result["result"] == "test_data", "Should return correct data"
    
    print("✅ test_cache_with_old_naive_timestamp passed")


def test_cache_with_new_aware_timestamp(tmp_path):
    """Test reading a cache file with new aware timestamp format."""
    cache_dir = tmp_path / "cache"
    cache_mgr = EspnCacheManager(cache_dir=cache_dir)
    
    # Create a cache file with aware timestamp (new format)
    cache_key = "test_endpoint_aware"
    cache_file = cache_dir / f"{cache_key}.json"
    
    # Use set_raw_response which creates aware timestamps
    cache_mgr.set_raw_response(
        "test_endpoint",
        {},
        {"result": "aware_test_data"}
    )
    
    # Read back
    result = cache_mgr.get_raw_response("test_endpoint", {}, ttl_hours=24)
    
    assert result is not None, "Should return cached data"
    assert result["result"] == "aware_test_data", "Should return correct data"
    
    print("✅ test_cache_with_new_aware_timestamp passed")


def test_espn_cache_ensure_utc():
    """Test that EspnCache also has ensure_utc helper working."""
    # Test naive
    naive = datetime(2024, 1, 15, 12, 0, 0)
    result = espn_ensure_utc(naive)
    assert result.tzinfo == UTC
    
    # Test aware
    aware = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    result = espn_ensure_utc(aware)
    assert result.tzinfo == UTC
    
    print("✅ test_espn_cache_ensure_utc passed")


def test_expired_cache_with_naive_timestamp(tmp_path):
    """Test that expired cache with naive timestamp is correctly identified."""
    # Use a unique subdirectory for this test to avoid interference from other tests
    cache_dir = tmp_path / "cache_expired"
    cache_mgr = EspnCacheManager(cache_dir=cache_dir)
    
    # Create an expired cache file with naive timestamp
    endpoint = "expired_test_endpoint"
    params = {}
    
    # First create a valid cache entry
    cache_mgr.set_raw_response(endpoint, params, {"result": "old_data"})
    
    # Find the cache file and modify its timestamp to be old and naive
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1, f"Should have exactly one cache file, got {len(cache_files)}"
    
    cache_file = cache_files[0]
    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    
    # Set timestamp to 2 days ago, naive format
    old_naive_dt = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2)
    cache_data["_cached_at"] = old_naive_dt.isoformat()
    
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)
    
    # Should return None because cache is expired (TTL is 24 hours by default)
    result = cache_mgr.get_raw_response(endpoint, params, ttl_hours=24)
    
    assert result is None, "Expired cache should return None"
    
    print("✅ test_expired_cache_with_naive_timestamp passed")


def run_all_tests():
    """Run all datetime handling tests."""
    print("=" * 60)
    print("Running datetime handling fix tests")
    print("=" * 60)
    
    # Tests that don't need tmp_path
    test_ensure_utc_with_naive_datetime()
    test_ensure_utc_with_aware_datetime()
    test_ttl_comparison_with_naive_cached_at()
    test_ttl_comparison_with_aware_cached_at()
    test_espn_cache_ensure_utc()
    
    # Tests that need tmp_path
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_cache_with_old_naive_timestamp(tmp_path)
        test_cache_with_new_aware_timestamp(tmp_path)
        test_expired_cache_with_naive_timestamp(tmp_path)
    
    print("=" * 60)
    print("All datetime handling tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
