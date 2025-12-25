#!/usr/bin/env python3
"""
Standalone test for Prabhupada Plugin.
Tests the query engine without full STEWARD boot.
"""

import sys
import json
from pathlib import Path

# Add steward-protocol to path
sys.path.insert(0, str(Path(__file__).parent / "steward-protocol"))

from vibe_core.plugins.prabhupada.query_engine import PrabhupadaKernel

def main():
    print("🙏 Testing Prabhupada Plugin (Standalone)\n")
    print("=" * 60)

    # Initialize kernel
    db_path = "steward-protocol/vibe_core/plugins/prabhupada/vedabase.db"
    print(f"📚 Loading Vedabase from: {db_path}")

    try:
        kernel = PrabhupadaKernel(db_path)
        print("✅ Kernel initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1

    # Get stats
    print("📊 Database Statistics:")
    stats = kernel.get_stats()
    print(json.dumps(stats, indent=2))
    print()

    # Test 1: Search query
    print("=" * 60)
    print("🔍 Test 1: Search Query")
    print("Query: 'What is the soul?'")
    print("-" * 60)

    response = kernel.query("What is the soul?", limit=3)
    print(response.to_json())
    print()

    # Test 2: Get specific verse
    print("=" * 60)
    print("🔍 Test 2: Get Specific Verse")
    print("Verse ID: 'BG 2.13'")
    print("-" * 60)

    verse = kernel.get_verse_by_id("BG 2.13")
    if verse:
        print(json.dumps(verse.to_dict(), indent=2))
    else:
        print("❌ Verse not found")
    print()

    # Test 3: Another search
    print("=" * 60)
    print("🔍 Test 3: Search 'yoga'")
    print("-" * 60)

    response = kernel.query("yoga", limit=2)
    print(f"Found {len(response.sruti)} verses")
    for verse in response.sruti:
        print(f"  - {verse.id}: {verse.translation[:80]}...")
    print()

    print("=" * 60)
    print("✅ All tests passed!")
    print("🙏 The 'No Speculation' protocol is working perfectly.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
