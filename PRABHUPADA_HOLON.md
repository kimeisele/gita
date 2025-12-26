# Prabhupada Wisdom Holon

> "No Speculation. Clear Boundaries."

A neuro-symbolic plugin for STEWARD Protocol implementing the No Speculation Protocol.

## What Actually Works

| Method | Status | Use Case |
|--------|--------|----------|
| **BM25** | ✅ Works | English queries → English translations |
| Sanskrit Matrix | ⚠️ Limited | Only Sanskrit→Sanskrit matching |
| Synapses | 🔄 Future | Learned associations over time |
| Vector/ML | ❌ 4GB | Optional enhancement |

## Quick Start

```bash
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

**Output:**
```
[7.59] BG 17.16: serenity, simplicity, gravity, self-control and purity...
[6.62] BG 4.27: Those interested in self-realization, in terms of mind...
[6.46] BG 6.15: Thus practicing control of body, mind and activities...
```

## Core Principles

### SRUTI/SMRITI Separation

| Layer | Source | Nature |
|-------|--------|--------|
| **SRUTI** | vedabase.db | Immutable - never modified |
| **SMRITI** | AI synthesis | Must cite SRUTI |

### No Speculation Protocol

1. Every claim must cite scripture
2. Admits ignorance rather than guessing
3. SRUTI is immutable

## Architecture

```
vibe_core/plugins/prabhupada/
├── manifest.json              # Plugin config
├── plugin_main.py             # WisdomKernel
├── knowledge/
│   ├── vedabase.db           # 700 verses
│   └── concepts.yaml         # Concept mapping
├── tools/
│   └── semantic_tools.py     # BM25, FTS tools
└── manas/
    └── methodology.yaml      # HOW Prabhupada teaches
```

## Tools for Other Agents

```python
result = prabhupada.verify_claim("The soul is eternal")
# → {"authorized": True, "citations": ["BG 2.13", "BG 2.20"]}

result = prabhupada.ground_in_sruti("Control the mind")
# → {"sruti_refs": ["BG 6.5", "BG 6.6"], "grounded": True}
```

---

*"We do not change the books. We change how the world accesses them."*
