"""
Prabhupada Wisdom Holon
=======================

A Neuro-Symbolic Plugin implementing the No Speculation Protocol.

SRUTI Layer: Immutable scripture (vedabase.db)
SMRITI Layer: AI synthesis (must cite SRUTI)

GAD-000 Compliant: All outputs are machine-readable.
"""

from .plugin_main import (
    PrabhupadaPlugin,
    QueryResult,
    VerificationResult,
    Verse,
    WisdomKernel,
)

__all__ = [
    "PrabhupadaPlugin",
    "WisdomKernel",
    "Verse",
    "QueryResult",
    "VerificationResult",
]
