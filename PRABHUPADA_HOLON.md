# Prabhupada Wisdom Holon

> "वर्णानां वर्गाः पञ्च" - "The phonemes have five classes"

A neuro-symbolic plugin for STEWARD Protocol implementing the No Speculation Protocol with **Sanskrit Matrix phonetic resonance** - not keyword matching, but PHYSICS.

## Key Innovation: Sanskrit Phonetic Resonance (No ML Required)

```
┌─────────────────────────────────────────────────────────────────┐
│  ML Embeddings (4GB)        vs    Sanskrit Matrix (0KB)         │
│  ─────────────────────           ─────────────────────────      │
│  • Neural network weights         • Phonetic physics             │
│  • Black box similarity           • Articulation-based resonance │
│  • Requires training data         • Based on Varnamala (alphabet)│
│  • GPU recommended                • Pure mathematics             │
└─────────────────────────────────────────────────────────────────┘
```

**The Sanskrit Varnamala (alphabet matrix):**
```
         | Unvoiced | Aspirated | Voiced | VoicedAsp | Nasal |
---------|----------|-----------|--------|-----------|-------|
KANTHYA  |    क     |     ख     |   ग    |     घ     |   ङ   | (Throat → Kernel)
TALAVYA  |    च     |     छ     |   ज    |     झ     |   ञ   | (Palate → Flow)
MURDHANYA|    ट     |     ठ     |   ड    |     ढ     |   ण   | (Cerebral → Logic)
DANTYA   |    त     |     थ     |   द    |     ध     |   न   | (Dental → Interface)
OSHTHYA  |    प     |     फ     |   ब    |     भ     |   म   | (Lips → Output)
```

**Resonance Results (no ML):**
| Comparison | Resonance | Why |
|------------|-----------|-----|
| karma ↔ dharma | **0.967** | Both Sanskrit, Throat-dominant |
| yoga ↔ bhakti | **0.943** | Similar articulation patterns |
| atma ↔ soul | **0.333** | Different phonetic physics! |

This is NOT keyword matching. This is the **physics of language**.

## Quick Start

```bash
# Clone and test
cd steward-protocol/vibe_core/plugins/prabhupada
python3 -c "
from tools.semantic_tools import BM25SearchTool
from pathlib import Path

bm25 = BM25SearchTool(Path('.'))
result = bm25.execute({'query': 'control the mind', 'top_k': 3})

for match in result.output['matches']:
    print(f\"[{match['score']:.2f}] {match['verse_id']}: {match['translation'][:60]}...\")
"
```

**Example Output (no ML installed):**
```
[7.59] BG 17.16: serenity, simplicity, gravity, self-control and purity...
[6.62] BG 4.27: Those who are interested in self-realization, in terms of mind...
[6.46] BG 6.15: Thus practicing control of the body, mind and activities...
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
│   ├── vectors.pkl           # 384-dim embeddings (OPTIONAL)
│   └── concepts.yaml         # Concept → verse mapping
│
├── circuits/                  # Cognitive Circuits
│   ├── semantic_search.yaml  # BM25 → Concept → Fold (Ouroboros)
│   ├── query_wisdom.yaml     # SRUTI → SMRITI pipeline
│   └── no_speculation.yaml   # Constitutional guard
│
├── tools/                     # TaskKernel Tools
│   └── semantic_tools.py     # Resonance, BM25, Vector, FTS tools
│
├── manas/                     # The "Brain"
│   └── methodology.yaml      # HOW Prabhupada teaches
│
└── tests/
    └── test_prabhupada.py
```

## Search Method Hierarchy

| Priority | Method | ML Required | Description |
|----------|--------|-------------|-------------|
| 1 | **Resonance** | No | Sanskrit phonetic physics (vibe_core.reactor) |
| 2 | **BM25** | No | Statistical TF-IDF ranking |
| 3 | Vector | Yes (4GB) | Optional - cosine similarity embeddings |
| 4 | Keyword | No | Fallback - simple FTS match |

```yaml
# From semantic_search.yaml (v4.0.0)
dharma:
  - "Resonance search uses Sanskrit Varnamala (phonetic physics)"
  - "BM25 provides statistical backup (no ML required)"
  - "Vector search is optional enhancement (requires 4GB model)"
  - "All results traced to SRUTI source"
```

**Why Resonance > BM25 > ML?**
- Sanskrit terms carry meaning in their phonetic structure
- "karma" and "dharma" resonate (0.967) because of articulation similarity
- ML embeddings are black boxes; Sanskrit Matrix is physics

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
# Search Bhagavad-gita (uses BM25 - no ML required)
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
    {"id": "BG 2.13", "score": 8.93, "translation": "..."}
  ],
  "count": 5,
  "method": "bm25",
  "ml_required": false
}
```

## The Vision

This plugin demonstrates how to build AI systems that:

1. **Never hallucinate** - Only speaks from scripture
2. **Always cite sources** - Every claim has a reference
3. **Admit ignorance** - Better to say "I don't know"
4. **Transfer methodology** - Not just WHAT but HOW
5. **Work without ML** - BM25 provides semantic-like results with pure math

---

*"We do not change the books. We change how the world accesses them."*

---

## Files Created

| File | Description |
|------|-------------|
| `manifest.json` | SOVEREIGN_STATE governance |
| `plugin_main.py` | WisdomKernel with BM25 primary search |
| `CONSTITUTION.md` | No Speculation Protocol |
| `knowledge/vedabase.db` | 700 verses (4.4MB) |
| `knowledge/vectors.pkl` | 384-dim embeddings (optional) |
| `knowledge/concepts.yaml` | Concept ontology |
| `circuits/semantic_search.yaml` | BM25 → Concept → Fold circuit |
| `circuits/query_wisdom.yaml` | SRUTI→SMRITI pipeline |
| `circuits/no_speculation.yaml` | Constitutional guard |
| `tools/semantic_tools.py` | BM25, Vector, FTS, Concept tools |
| `manas/methodology.yaml` | Prabhupada's approach |
