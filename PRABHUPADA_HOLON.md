# Prabhupada Wisdom Holon

> "No Speculation. Clear Boundaries."

A neuro-symbolic plugin for STEWARD Protocol implementing the No Speculation Protocol.

## Quick Start

```bash
# Clone steward-protocol with the plugin
git clone https://github.com/kimeisele/steward-protocol.git
cd steward-protocol
git checkout claude/prabhupada-holon-uGBsA

# Test the plugin
cd vibe_core/plugins/prabhupada
python3 tests/test_prabhupada.py
```

## Architecture

```
vibe_core/plugins/prabhupada/
├── manifest.json              # SOVEREIGN_STATE governance
├── plugin_main.py             # WisdomKernel + PrabhupadaPlugin
├── CONSTITUTION.md            # No Speculation Protocol
│
├── knowledge/                 # SRUTI (Immutable Scripture)
│   ├── vedabase.db           # 700 verses from Bhagavad-gita
│   ├── vectors.pkl           # 384-dim semantic embeddings
│   └── concepts.yaml         # Concept → verse mapping
│
├── circuits/                  # Cognitive Circuits
│   ├── query_wisdom.yaml     # SRUTI → SMRITI pipeline
│   └── no_speculation.yaml   # Constitutional guard
│
├── manas/                     # The "Brain"
│   └── methodology.yaml      # HOW Prabhupada teaches
│
└── tests/
    └── test_prabhupada.py
```

## Core Principles

### SRUTI/SMRITI Separation

| Layer | Source | Nature | Role |
|-------|--------|--------|------|
| **SRUTI** | vedabase.db | Immutable | Absolute truth - never modified |
| **SMRITI** | AI synthesis | Fluid | Must ALWAYS cite SRUTI |

### No Speculation Protocol

1. **Every claim must cite scripture**
2. **speculation_score < 0.1** enforced
3. **Admits ignorance** rather than guessing
4. **SRUTI is immutable** - never modified by AI

## Tools for Other Agents

```python
# Verify a claim against scripture
result = prabhupada.verify_claim("The soul is eternal")
# → {"authorized": True, "citations": ["BG 2.13", "BG 2.20"]}

# Ground text in scripture
result = prabhupada.ground_in_sruti("Control the mind")
# → {"sruti_refs": ["BG 6.5", "BG 6.6"], "grounded": True}

# Get Prabhupada's methodology
result = prabhupada.get_methodology("decision making")
# → {"approach": "...", "steps": [...], "citations": [...]}

# Check for speculation
result = prabhupada.check_speculation("I think maybe...")
# → {"speculation_score": 0.4, "flagged_phrases": ["i think", "maybe"]}
```

## CLI Commands (GAD-000 Compliant)

```bash
# Search Bhagavad-gita
vibe gita search "soul"

# Get specific verse
vibe gita verse BG.2.13

# Ask wisdom question
vibe wisdom ask "What is karma?"

# Verify claim
vibe wisdom verify "Krishna is the Supreme"

# Get methodology
vibe wisdom method "dealing with anger"
```

## GAD-000 Compliance

All outputs are JSON-first, designed for AI operators:

```json
{
  "command": "gita search",
  "query": "soul",
  "results": [
    {"id": "BG 2.13", "translation": "..."},
    {"id": "BG 2.20", "translation": "..."}
  ],
  "count": 5
}
```

## The Vision

This plugin demonstrates how to build AI systems that:

1. **Never hallucinate** - Only speaks from scripture
2. **Always cite sources** - Every claim has a reference
3. **Admit ignorance** - Better to say "I don't know"
4. **Transfer methodology** - Not just WHAT but HOW

---

*"We do not change the books. We change how the world accesses them."*

---

## Local Development

The plugin is committed in `steward-protocol/` (local clone):

```bash
cd steward-protocol
git log --oneline -3
# c20c074 feat: Add TaskKernel-based semantic search (Ouroboros)
# 8db7c0f feat: Add Prabhupada Wisdom Holon Plugin
# ... (main)

# To push to your fork:
git push -u origin claude/prabhupada-holon-uGBsA
```

### Files Created

| File | Description |
|------|-------------|
| `manifest.json` | SOVEREIGN_STATE governance |
| `plugin_main.py` | WisdomKernel (600 lines) |
| `CONSTITUTION.md` | No Speculation Protocol |
| `knowledge/vedabase.db` | 700 verses (4.4MB) |
| `knowledge/vectors.pkl` | 384-dim embeddings (1.7MB) |
| `knowledge/concepts.yaml` | Concept ontology |
| `circuits/query_wisdom.yaml` | SRUTI→SMRITI pipeline |
| `circuits/no_speculation.yaml` | Constitutional guard |
| `circuits/semantic_search.yaml` | TaskKernel-based search |
| `tools/semantic_tools.py` | Real computation tools |
| `manas/methodology.yaml` | Prabhupada's approach |
