"""
PRABHUPADA WISDOM HOLON
=======================
A Neuro-Symbolic Plugin for STEWARD Protocol

Implements the "No Speculation" Protocol:
- SRUTI Layer: Immutable database (vedabase.db)
- SMRITI Layer: AI synthesis (must cite SRUTI)

GAD-000 Compliant:
- JSON-first outputs
- Machine-readable errors
- Observable state
- Composable operations

This plugin is a SOVEREIGN_STATE - it can spawn TaskKernels
for ephemeral wisdom queries.
"""

import json
import logging
import os
import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import yaml

if TYPE_CHECKING:
    from vibe_core.kernel_impl import RealVibeKernel

logger = logging.getLogger("PRABHUPADA")


# =============================================================================
# DATA CLASSES (GAD-000: Structured, Machine-Readable)
# =============================================================================


@dataclass
class Verse:
    """A single verse from Bhagavad-gita."""

    id: str
    book_code: str
    chapter: int
    verse: str
    sanskrit: str
    synonyms: str
    translation: str
    purport: str
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """GAD-000: JSON-serializable output."""
        return {
            "id": self.id,
            "book_code": self.book_code,
            "chapter": self.chapter,
            "verse": self.verse,
            "sanskrit": self.sanskrit,
            "synonyms": self.synonyms,
            "translation": self.translation,
            "purport": self.purport,
            "content_hash": self.content_hash,
        }

    def to_summary(self) -> Dict[str, Any]:
        """Short summary for listings."""
        return {
            "id": self.id,
            "translation": self.translation[:200] + "..." if len(self.translation) > 200 else self.translation,
        }


@dataclass
class QueryResult:
    """Result of a wisdom query (SRUTI + SMRITI)."""

    query: str
    sruti: List[Verse]  # Raw scripture (immutable)
    smriti: Dict[str, Any]  # AI synthesis (must cite sruti)
    confidence: float
    speculation_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """GAD-000: JSON-serializable output."""
        return {
            "query": self.query,
            "sruti": [v.to_summary() for v in self.sruti],
            "smriti": self.smriti,
            "confidence": self.confidence,
            "speculation_score": self.speculation_score,
            "protocol": "No Speculation - SRUTI/SMRITI separation",
        }


@dataclass
class VerificationResult:
    """Result of verifying a claim against SRUTI."""

    claim: str
    authorized: bool
    citations: List[str]
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "authorized": self.authorized,
            "citations": self.citations,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


# =============================================================================
# WISDOM KERNEL (The Core Engine)
# =============================================================================


class WisdomKernel:
    """
    The deterministic core of Prabhupada Wisdom.

    Implements:
    - SRUTI Layer: SQLite database (immutable truth)
    - Semantic Search: Vector embeddings
    - No Speculation: Never generates beyond SRUTI
    """

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self.db_path = plugin_dir / "knowledge" / "vedabase.db"
        self.vectors_path = plugin_dir / "knowledge" / "vectors.pkl"
        self.concepts_path = plugin_dir / "knowledge" / "concepts.yaml"
        self.methodology_path = plugin_dir / "manas" / "methodology.yaml"

        # Load knowledge
        self._conn: Optional[sqlite3.Connection] = None
        self._vectors: Optional[Dict[str, Any]] = None
        self._concepts: Optional[Dict[str, Any]] = None
        self._methodology: Optional[Dict[str, Any]] = None

        # Lazy load on first use
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load all knowledge on first use."""
        if self._loaded:
            return

        # Load database connection
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        # Load vectors (optional - for semantic search)
        if self.vectors_path.exists():
            try:
                import numpy as np

                with open(self.vectors_path, "rb") as f:
                    self._vectors = pickle.load(f)
                logger.info(f"Loaded {len(self._vectors['embeddings'])} verse embeddings")
            except ImportError:
                logger.warning("NumPy not available - semantic search disabled")
                self._vectors = None

        # Load concepts
        if self.concepts_path.exists():
            with open(self.concepts_path) as f:
                self._concepts = yaml.safe_load(f)

        # Load methodology
        if self.methodology_path.exists():
            with open(self.methodology_path) as f:
                self._methodology = yaml.safe_load(f)

        self._loaded = True
        logger.info("WisdomKernel loaded successfully")

    def get_status(self) -> Dict[str, Any]:
        """GAD-000: Observable state."""
        self._ensure_loaded()

        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM verses")
        verse_count = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT chapter FROM verses ORDER BY chapter")
        chapters = [row[0] for row in cursor.fetchall()]

        # Check if sentence-transformers is available
        try:
            from sentence_transformers import SentenceTransformer
            ml_available = True
        except ImportError:
            ml_available = False

        return {
            "database": str(self.db_path),
            "verses_loaded": verse_count,
            "chapters": chapters,
            "search_methods": {
                "bm25": True,  # Always available (no ML required)
                "vector": self._vectors is not None and ml_available,
                "keyword": True,  # Always available
            },
            "primary_search": "bm25",
            "ml_required": False,  # BM25 doesn't need ML
            "embeddings_count": len(self._vectors["embeddings"]) if self._vectors else 0,
            "concepts_loaded": self._concepts is not None,
            "methodology_loaded": self._methodology is not None,
            "protocol": "No Speculation - SRUTI/SMRITI separation",
        }

    def get_verse(self, verse_id: str) -> Optional[Verse]:
        """
        Get a specific verse by ID.

        Args:
            verse_id: Format "BG 2.13" or "BG.2.13"

        Returns:
            Verse object or None
        """
        self._ensure_loaded()

        # Parse verse ID
        verse_id = verse_id.upper().replace(".", " ").replace("BG ", "BG ")
        parts = verse_id.split()
        if len(parts) < 2:
            return None

        try:
            chapter_verse = parts[1].split(".")
            if len(chapter_verse) == 2:
                chapter = int(chapter_verse[0])
                verse = chapter_verse[1]
            else:
                # Handle "BG 2 13" format
                chapter = int(parts[1])
                verse = parts[2] if len(parts) > 2 else "1"
        except (ValueError, IndexError):
            return None

        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, book_code, chapter, verse, sanskrit, synonyms,
                   translation, purport, content_hash
            FROM verses
            WHERE chapter = ? AND verse = ?
            """,
            (chapter, verse),
        )

        row = cursor.fetchone()
        if not row:
            return None

        return Verse(
            id=f"BG {row['chapter']}.{row['verse']}",
            book_code=row["book_code"],
            chapter=row["chapter"],
            verse=row["verse"],
            sanskrit=row["sanskrit"] or "",
            synonyms=row["synonyms"] or "",
            translation=row["translation"] or "",
            purport=row["purport"] or "",
            content_hash=row["content_hash"] or "",
        )

    def search(self, query: str, limit: int = 5) -> List[Verse]:
        """
        Search verses using keyword matching.

        For semantic search, use search_semantic().
        """
        self._ensure_loaded()

        keywords = query.lower().split()
        conditions = []
        params = []

        for kw in keywords:
            conditions.append("(LOWER(translation) LIKE ? OR LOWER(purport) LIKE ?)")
            params.extend([f"%{kw}%", f"%{kw}%"])

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT id, book_code, chapter, verse, sanskrit, synonyms,
                   translation, purport, content_hash
            FROM verses
            WHERE {where_clause}
            LIMIT ?
            """,
            params + [limit],
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                Verse(
                    id=f"BG {row['chapter']}.{row['verse']}",
                    book_code=row["book_code"],
                    chapter=row["chapter"],
                    verse=row["verse"],
                    sanskrit=row["sanskrit"] or "",
                    synonyms=row["synonyms"] or "",
                    translation=row["translation"] or "",
                    purport=row["purport"] or "",
                    content_hash=row["content_hash"] or "",
                )
            )

        return results

    def search_bm25(self, query: str, limit: int = 5) -> List[Verse]:
        """
        BM25 semantic-like search - NO ML REQUIRED.

        Uses BM25 ranking algorithm for semantic-like results:
        - Term Frequency with saturation
        - Inverse Document Frequency
        - Document length normalization

        This is the PRIMARY search method.
        """
        self._ensure_loaded()

        from .tools.semantic_tools import BM25SearchTool

        bm25_tool = BM25SearchTool(self.plugin_dir)
        result = bm25_tool.execute({"query": query, "top_k": limit})

        if not result.success or not result.output:
            logger.warning(f"BM25 search failed: {result.error}")
            return self.search(query, limit)  # Fallback to keyword

        results = []
        for match in result.output.get("matches", []):
            verse = self.get_verse(match["verse_id"])
            if verse:
                results.append(verse)

        logger.info(f"BM25 search found {len(results)} verses (no ML required)")
        return results

    def search_semantic(self, query: str, limit: int = 5) -> List[Verse]:
        """
        Semantic search - tries BM25 first, then vector embeddings.

        Priority:
        1. BM25 (no ML required) - PRIMARY
        2. Vector search (requires sentence-transformers) - OPTIONAL
        3. Keyword search - FALLBACK
        """
        # Try BM25 first (no ML required)
        results = self.search_bm25(query, limit)
        if results:
            return results

        # Try vector search if available
        self._ensure_loaded()

        if self._vectors is None:
            logger.warning("Semantic search unavailable, falling back to keyword")
            return self.search(query, limit)

        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            # Get model (cached)
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            # Embed query
            query_embedding = model.encode(query, show_progress_bar=False).astype(np.float32)

            # Compute similarities
            embeddings = self._vectors["embeddings"]
            verse_data = self._vectors["verse_data"]

            similarities = []
            for i, emb in enumerate(embeddings):
                sim = float(
                    np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)
                )
                similarities.append((i, sim))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top results
            results = []
            for i, sim in similarities[:limit]:
                verse_info = verse_data[i]
                verse = self.get_verse(verse_info["id"])
                if verse:
                    results.append(verse)

            return results

        except ImportError:
            logger.warning("sentence-transformers not available, falling back to keyword")
            return self.search(query, limit)

    def query_wisdom(self, question: str, limit: int = 5) -> QueryResult:
        """
        Main query interface - SRUTI + SMRITI response.

        Returns structured result with:
        - SRUTI: Raw verses (immutable)
        - SMRITI: Synthesis (must cite SRUTI)
        """
        # Get SRUTI (immutable truth)
        verses = self.search_semantic(question, limit)

        if not verses:
            return QueryResult(
                query=question,
                sruti=[],
                smriti={
                    "synthesis": "I could not find relevant verses in Bhagavad-gita for this query.",
                    "citations": [],
                    "note": "The No Speculation Protocol prevents me from answering without scriptural basis.",
                },
                confidence=0.0,
                speculation_score=0.0,
            )

        # Generate SMRITI (synthesis that cites SRUTI)
        citations = [v.id for v in verses]
        synthesis = f"Based on {len(verses)} verses from Bhagavad-gita ({', '.join(citations)}), "
        synthesis += f"Prabhupada teaches: {verses[0].translation}"

        return QueryResult(
            query=question,
            sruti=verses,
            smriti={
                "synthesis": synthesis,
                "citations": citations,
                "note": "SMRITI is servant of SRUTI - all claims backed by scripture.",
            },
            confidence=0.8 if len(verses) >= 3 else 0.6,
            speculation_score=0.0,  # We never speculate
        )

    def verify_claim(self, claim: str) -> VerificationResult:
        """
        Verify if a claim is supported by SRUTI.

        Returns verification result with citations.
        """
        # Search for supporting verses
        verses = self.search_semantic(claim, limit=5)

        if not verses:
            return VerificationResult(
                claim=claim,
                authorized=False,
                citations=[],
                confidence=0.0,
                reasoning="No supporting verses found in Bhagavad-gita.",
            )

        # Check relevance (basic check - in production, use LLM)
        claim_words = set(claim.lower().split())
        max_overlap = 0

        for verse in verses:
            verse_words = set(verse.translation.lower().split())
            overlap = len(claim_words & verse_words) / len(claim_words) if claim_words else 0
            max_overlap = max(max_overlap, overlap)

        authorized = max_overlap > 0.3
        confidence = min(max_overlap * 2, 1.0)

        return VerificationResult(
            claim=claim,
            authorized=authorized,
            citations=[v.id for v in verses[:3]],
            confidence=confidence,
            reasoning=f"Found {len(verses)} potentially related verses. Overlap score: {max_overlap:.2f}",
        )

    def get_methodology(self, topic: str) -> Dict[str, Any]:
        """
        Get Prabhupada's methodology for a topic.

        Returns approach, steps, and citations.
        """
        self._ensure_loaded()

        if not self._methodology:
            return {
                "topic": topic,
                "error": "Methodology database not loaded",
                "approach": None,
            }

        # Search methodologies
        methodologies = self._methodology.get("methodologies", {})
        best_match = None
        best_score = 0

        topic_lower = topic.lower()
        for key, method in methodologies.items():
            if topic_lower in key.lower() or topic_lower in method.get("name", "").lower():
                best_match = method
                break
            # Check for partial matches
            if any(word in key.lower() for word in topic_lower.split()):
                if best_score < 0.5:
                    best_match = method
                    best_score = 0.5

        if not best_match:
            # Return core principles as fallback
            return {
                "topic": topic,
                "approach": "Apply core principles",
                "steps": [
                    "Consult scripture (SRUTI)",
                    "Follow parampara (disciplic succession)",
                    "Apply practically",
                    "Present as it is",
                ],
                "citations": ["BG 4.2", "BG 18.63"],
                "note": "These are general principles - specific methodology not found.",
            }

        return {
            "topic": topic,
            "approach": best_match.get("name", topic),
            "steps": best_match.get("steps") or best_match.get("principles", []),
            "citations": best_match.get("reference", []),
        }

    def get_analogy(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get Prabhupada's analogy for a topic."""
        self._ensure_loaded()

        if not self._methodology:
            return None

        analogies = self._methodology.get("analogies", {})

        topic_lower = topic.lower()
        for key, analogy in analogies.items():
            if topic_lower in key.lower() or topic_lower in analogy.get("topic", "").lower():
                return {
                    "topic": analogy.get("topic"),
                    "analogy": analogy.get("analogy"),
                    "application": analogy.get("application"),
                }

        return None


# =============================================================================
# PLUGIN CLASS (STEWARD Protocol Integration)
# =============================================================================


class PrabhupadaPlugin:
    """
    Prabhupada Wisdom Plugin for STEWARD Protocol.

    SOVEREIGN_STATE: Can spawn TaskKernels for ephemeral queries.
    GAD-000 Compliant: All outputs are machine-readable.
    """

    plugin_id = "prabhupada"
    plugin_version = "1.0.0"

    def __init__(self):
        self._kernel: Optional["RealVibeKernel"] = None
        self._wisdom: Optional[WisdomKernel] = None
        self._plugin_dir: Optional[Path] = None

    def on_boot(self, kernel: "RealVibeKernel", config: Dict[str, Any]) -> None:
        """Called when plugin is loaded."""
        self._kernel = kernel
        self._plugin_dir = Path(__file__).parent

        # Initialize wisdom kernel
        self._wisdom = WisdomKernel(self._plugin_dir)

        logger.info(f"Prabhupada Plugin v{self.plugin_version} initialized")
        logger.info("No Speculation Protocol: ACTIVE")

    def on_shutdown(self, kernel: "RealVibeKernel") -> None:
        """Called on kernel shutdown."""
        logger.info("Prabhupada Plugin shutting down")

    # =========================================================================
    # TOOLS FOR OTHER AGENTS (GAD-000: AI-Operable)
    # =========================================================================

    def verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        Tool: Verify if a claim is authorized by SRUTI.

        Used by other agents to check claims before making them.
        """
        result = self._wisdom.verify_claim(claim)
        return result.to_dict()

    def ground_in_sruti(self, text: str, limit: int = 3) -> Dict[str, Any]:
        """
        Tool: Ground a synthesis in scriptural evidence.

        Used by other agents to add citations to their responses.
        """
        verses = self._wisdom.search_semantic(text, limit)
        return {
            "original_text": text,
            "grounded": True if verses else False,
            "sruti_refs": [v.id for v in verses],
            "citations": [{"id": v.id, "text": v.translation} for v in verses],
        }

    def get_methodology(self, topic: str) -> Dict[str, Any]:
        """
        Tool: Get Prabhupada's approach to a topic.

        Used by other agents to apply wisdom methodology.
        """
        return self._wisdom.get_methodology(topic)

    def check_speculation(self, text: str) -> Dict[str, Any]:
        """
        Tool: Check if text contains speculation.

        Returns speculation score and flagged phrases.
        """
        # Simple heuristic - in production, use LLM
        speculation_phrases = [
            "i think",
            "maybe",
            "perhaps",
            "probably",
            "in my opinion",
            "it seems",
            "might be",
            "could be",
        ]

        text_lower = text.lower()
        flagged = [phrase for phrase in speculation_phrases if phrase in text_lower]

        # Also check if there are citations
        has_citation = any(marker in text for marker in ["BG ", "SB ", "CC "])

        speculation_score = len(flagged) * 0.2
        if not has_citation and len(text) > 100:
            speculation_score += 0.3

        return {
            "text": text[:200] + "..." if len(text) > 200 else text,
            "speculation_score": min(speculation_score, 1.0),
            "flagged_phrases": flagged,
            "has_citation": has_citation,
            "recommendation": "Add scriptural citations" if not has_citation else "Text appears grounded",
        }

    def cite_authority(self, topic: str) -> Dict[str, Any]:
        """
        Tool: Generate proper citation for a topic.

        Returns verse ID, citation text, and context.
        """
        verses = self._wisdom.search_semantic(topic, limit=1)

        if not verses:
            return {
                "topic": topic,
                "citation": None,
                "error": "No relevant verse found",
            }

        verse = verses[0]
        return {
            "topic": topic,
            "verse_id": verse.id,
            "citation": f'"{verse.translation}" ({verse.id})',
            "context": verse.purport[:300] + "..." if len(verse.purport) > 300 else verse.purport,
        }

    # =========================================================================
    # CLI HANDLERS (GAD-000: JSON-First)
    # =========================================================================

    def cmd_gita_search(self, query: str, limit: int = 5, json: bool = True) -> Dict[str, Any]:
        """CLI: Search Bhagavad-gita."""
        verses = self._wisdom.search_semantic(query, limit)
        result = {
            "command": "gita search",
            "query": query,
            "results": [v.to_summary() for v in verses],
            "count": len(verses),
        }
        return result

    def cmd_gita_verse(self, verse_id: str, json: bool = True) -> Dict[str, Any]:
        """CLI: Get specific verse."""
        verse = self._wisdom.get_verse(verse_id)
        if verse:
            return {"command": "gita verse", "verse": verse.to_dict()}
        return {"command": "gita verse", "error": f"Verse not found: {verse_id}"}

    def cmd_wisdom_ask(self, question: str, json: bool = True) -> Dict[str, Any]:
        """CLI: Ask wisdom question."""
        result = self._wisdom.query_wisdom(question)
        return {"command": "wisdom ask", **result.to_dict()}

    def cmd_wisdom_verify(self, claim: str, json: bool = True) -> Dict[str, Any]:
        """CLI: Verify claim."""
        result = self._wisdom.verify_claim(claim)
        return {"command": "wisdom verify", **result.to_dict()}

    def cmd_wisdom_method(self, topic: str, json: bool = True) -> Dict[str, Any]:
        """CLI: Get methodology."""
        result = self._wisdom.get_methodology(topic)
        return {"command": "wisdom method", **result}

    def cmd_gita_status(self, json: bool = True) -> Dict[str, Any]:
        """CLI: Show status."""
        return {"command": "gita status", **self._wisdom.get_status()}


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "PrabhupadaPlugin",
    "WisdomKernel",
    "Verse",
    "QueryResult",
    "VerificationResult",
]
