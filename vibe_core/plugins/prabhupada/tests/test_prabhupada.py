"""
Tests for Prabhupada Wisdom Plugin

GAD-000 Compliant: All outputs are machine-readable JSON.
"""

import json
from pathlib import Path

import pytest


def get_plugin_dir() -> Path:
    """Get the plugin directory."""
    return Path(__file__).parent.parent


def test_wisdom_kernel_init():
    """Test WisdomKernel initialization."""
    from plugin_main import WisdomKernel

    kernel = WisdomKernel(get_plugin_dir())
    status = kernel.get_status()

    assert status["verses_loaded"] == 700
    assert status["protocol"] == "No Speculation - SRUTI/SMRITI separation"
    print(json.dumps(status, indent=2))


def test_get_verse():
    """Test verse retrieval."""
    from plugin_main import WisdomKernel

    kernel = WisdomKernel(get_plugin_dir())

    # Test various ID formats
    verse = kernel.get_verse("BG 2.13")
    assert verse is not None
    assert verse.chapter == 2
    assert "soul" in verse.translation.lower() or "body" in verse.translation.lower()

    print(json.dumps(verse.to_dict(), indent=2, ensure_ascii=False))


def test_search():
    """Test keyword search."""
    from plugin_main import WisdomKernel

    kernel = WisdomKernel(get_plugin_dir())

    results = kernel.search("soul", limit=3)
    assert len(results) > 0

    print(f"Found {len(results)} verses about 'soul':")
    for v in results:
        print(f"  - {v.id}: {v.translation[:80]}...")


def test_query_wisdom():
    """Test wisdom query (SRUTI + SMRITI)."""
    from plugin_main import WisdomKernel

    kernel = WisdomKernel(get_plugin_dir())

    result = kernel.query_wisdom("What is the soul?")
    assert result.query == "What is the soul?"
    assert len(result.sruti) > 0  # Must have SRUTI
    assert result.smriti["citations"]  # SMRITI must cite
    assert result.speculation_score == 0.0  # No speculation!

    print(json.dumps(result.to_dict(), indent=2))


def test_verify_claim():
    """Test claim verification."""
    from plugin_main import WisdomKernel

    kernel = WisdomKernel(get_plugin_dir())

    # Test a claim that should be authorized
    result = kernel.verify_claim("The soul is eternal")
    print(f"Claim: 'The soul is eternal'")
    print(f"Authorized: {result.authorized}")
    print(f"Citations: {result.citations}")

    # Test an unrelated claim
    result2 = kernel.verify_claim("Pizza is delicious")
    print(f"\nClaim: 'Pizza is delicious'")
    print(f"Authorized: {result2.authorized}")


def test_get_methodology():
    """Test methodology retrieval."""
    from plugin_main import WisdomKernel

    kernel = WisdomKernel(get_plugin_dir())

    result = kernel.get_methodology("decision making")
    assert "steps" in result or "approach" in result

    print(json.dumps(result, indent=2))


def test_tools_for_agents():
    """Test tools designed for other agents."""
    from plugin_main import PrabhupadaPlugin

    plugin = PrabhupadaPlugin()
    plugin._plugin_dir = get_plugin_dir()
    plugin._wisdom = __import__("plugin_main").WisdomKernel(get_plugin_dir())

    # Test verify_claim tool
    result = plugin.verify_claim("Krishna is the Supreme")
    assert "authorized" in result
    assert "citations" in result

    # Test ground_in_sruti tool
    result = plugin.ground_in_sruti("Control the mind through practice")
    assert "sruti_refs" in result

    # Test check_speculation tool
    result = plugin.check_speculation("I think maybe the soul might be eternal")
    assert result["speculation_score"] > 0  # Should detect speculation
    assert "i think" in result["flagged_phrases"]

    print("All agent tools working!")


def test_gad_000_compliance():
    """Test GAD-000 compliance (JSON-first)."""
    from plugin_main import WisdomKernel

    kernel = WisdomKernel(get_plugin_dir())

    # All outputs must be JSON-serializable
    status = kernel.get_status()
    assert json.dumps(status)  # Must not raise

    verse = kernel.get_verse("BG 2.13")
    assert json.dumps(verse.to_dict())  # Must not raise

    result = kernel.query_wisdom("yoga")
    assert json.dumps(result.to_dict())  # Must not raise

    print("GAD-000 Compliance: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("PRABHUPADA WISDOM PLUGIN - TEST SUITE")
    print("=" * 60)

    print("\n1. Testing WisdomKernel init...")
    test_wisdom_kernel_init()

    print("\n2. Testing verse retrieval...")
    test_get_verse()

    print("\n3. Testing search...")
    test_search()

    print("\n4. Testing wisdom query...")
    test_query_wisdom()

    print("\n5. Testing claim verification...")
    test_verify_claim()

    print("\n6. Testing methodology...")
    test_get_methodology()

    print("\n7. Testing agent tools...")
    test_tools_for_agents()

    print("\n8. Testing GAD-000 compliance...")
    test_gad_000_compliance()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
