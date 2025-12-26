"""
PRABHUPADA SEMANTIC TOOLS
=========================
Tools for TaskKernel-based semantic search.

These tools are injected into ephemeral TaskKernels spawned by
the semantic_search circuit. They provide:

1. embedding_compute - Compute 384-dim embeddings
2. vector_search - Cosine similarity search
3. concept_align - Concept-based re-ranking
4. fts_search - SQLite FTS5 fallback

Philosophy:
    "An embedding without computation is a hallucination."
    These tools make embeddings REAL.
"""

import logging
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("PRABHUPADA.TOOLS")


# =============================================================================
# TOOL RESULT TYPES
# =============================================================================


@dataclass
class ToolResult:
    """Standard result type for all tools."""

    success: bool
    output: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


# =============================================================================
# EMBEDDING COMPUTE TOOL
# =============================================================================


class EmbeddingComputeTool:
    """
    Compute text embeddings using sentence-transformers.

    This is NOT a lazy fallback - it's real computation.
    Spawned in TaskKernel for isolation.
    """

    name = "embedding_compute"
    description = "Compute 384-dimensional text embedding"

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self._model = None
        self._model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def _ensure_model(self):
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def validate(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters."""
        if "text" not in parameters:
            raise ValueError("Missing required parameter: text")
        if not parameters["text"]:
            raise ValueError("Text cannot be empty")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Compute embedding for text.

        Args:
            parameters: {"text": "...", "model": "...", "dimension": 384}

        Returns:
            ToolResult with embedding array
        """
        try:
            self._ensure_model()

            import numpy as np

            text = parameters["text"]

            # Compute embedding
            embedding = self._model.encode(text, show_progress_bar=False)
            embedding = embedding.astype(np.float32)

            logger.info(f"Computed embedding: shape={embedding.shape}")

            return ToolResult(
                success=True,
                output={
                    "embedding": embedding.tolist(),  # JSON-serializable
                    "shape": list(embedding.shape),
                    "dtype": "float32",
                },
            )

        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            return ToolResult(success=False, error=str(e))


# =============================================================================
# VECTOR SEARCH TOOL
# =============================================================================


class VectorSearchTool:
    """
    Search verse embeddings using cosine similarity.

    Pure math - no LLM involved.
    """

    name = "vector_search"
    description = "Search vectors using cosine similarity"

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self._vectors = None
        self._verse_data = None

    def _ensure_loaded(self):
        """Load vectors on first use."""
        if self._vectors is not None:
            return

        vectors_path = self.plugin_dir / "knowledge" / "vectors.pkl"
        if not vectors_path.exists():
            raise FileNotFoundError(f"Vectors not found: {vectors_path}")

        import numpy as np

        with open(vectors_path, "rb") as f:
            data = pickle.load(f)

        self._vectors = data["embeddings"]
        self._verse_data = data["verse_data"]
        logger.info(f"Loaded {len(self._vectors)} verse embeddings")

    def validate(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters."""
        if "query_embedding" not in parameters:
            raise ValueError("Missing required parameter: query_embedding")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Search verses using cosine similarity.

        Args:
            parameters: {
                "query_embedding": [...],  # 384-dim
                "top_k": 10,
                "similarity_threshold": 0.3
            }

        Returns:
            ToolResult with matched verses and scores
        """
        try:
            self._ensure_loaded()

            import numpy as np

            query_emb = np.array(parameters["query_embedding"], dtype=np.float32)
            top_k = parameters.get("top_k", 10)
            threshold = parameters.get("similarity_threshold", 0.3)

            # Compute cosine similarities
            similarities = []
            for i, verse_emb in enumerate(self._vectors):
                sim = float(
                    np.dot(query_emb, verse_emb)
                    / (np.linalg.norm(query_emb) * np.linalg.norm(verse_emb) + 1e-8)
                )
                if sim >= threshold:
                    similarities.append((i, sim))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top-k results
            matches = []
            for idx, score in similarities[:top_k]:
                verse_info = self._verse_data[idx]
                matches.append({
                    "verse_id": verse_info["id"],
                    "score": score,
                    "translation": verse_info.get("translation", "")[:200],
                })

            logger.info(f"Vector search: {len(matches)} matches above threshold {threshold}")

            return ToolResult(
                success=True,
                output={
                    "matches": matches,
                    "count": len(matches),
                    "threshold": threshold,
                },
            )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return ToolResult(success=False, error=str(e))


# =============================================================================
# CONCEPT ALIGN TOOL
# =============================================================================


class ConceptAlignTool:
    """
    Re-rank results using concept ontology.

    Maps query to concepts, boosts verses matching those concepts.
    """

    name = "concept_align"
    description = "Re-rank using concept ontology"

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self._concepts = None

    def _ensure_loaded(self):
        """Load concepts on first use."""
        if self._concepts is not None:
            return

        concepts_path = self.plugin_dir / "knowledge" / "concepts.yaml"
        if not concepts_path.exists():
            self._concepts = {}
            return

        with open(concepts_path) as f:
            self._concepts = yaml.safe_load(f)
        logger.info("Loaded concept ontology")

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract matching concepts from text."""
        text_lower = text.lower()
        matched = []

        for category, items in self._concepts.items():
            if not isinstance(items, dict):
                continue
            for concept, data in items.items():
                if isinstance(data, dict):
                    aliases = data.get("aliases", [])
                elif isinstance(data, list):
                    aliases = data
                else:
                    continue

                # Check if any alias matches
                for alias in aliases:
                    if alias.lower() in text_lower:
                        matched.append(concept)
                        break

        return matched

    def validate(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters."""
        if "query" not in parameters:
            raise ValueError("Missing required parameter: query")
        if "candidates" not in parameters:
            raise ValueError("Missing required parameter: candidates")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Re-rank candidates using concept alignment.

        Args:
            parameters: {
                "query": "...",
                "candidates": [{"verse_id": "...", "score": 0.8}, ...]
            }

        Returns:
            ToolResult with re-ranked results and matched concepts
        """
        try:
            self._ensure_loaded()

            query = parameters["query"]
            candidates = parameters["candidates"]

            # Extract concepts from query
            concepts = self._extract_concepts(query)

            if not concepts:
                # No concepts found - return original order
                return ToolResult(
                    success=True,
                    output={
                        "ranked": candidates,
                        "concepts": [],
                        "boost_applied": False,
                    },
                )

            # Get key verses for matched concepts
            key_verses = set()
            for category, items in self._concepts.items():
                if not isinstance(items, dict):
                    continue
                for concept, data in items.items():
                    if concept in concepts and isinstance(data, dict):
                        key_verses.update(data.get("key_verses", []))

            # Boost scores for key verses
            boosted = []
            for candidate in candidates:
                verse_id = candidate["verse_id"]
                score = candidate["score"]

                # Boost if verse is a key verse for matched concept
                if any(verse_id.startswith(kv.replace(" ", "").upper()) for kv in key_verses):
                    score = min(score * 1.3, 1.0)  # 30% boost, cap at 1.0

                boosted.append({**candidate, "score": score})

            # Re-sort by boosted scores
            boosted.sort(key=lambda x: x["score"], reverse=True)

            logger.info(f"Concept alignment: {len(concepts)} concepts, {len(key_verses)} key verses")

            return ToolResult(
                success=True,
                output={
                    "ranked": boosted,
                    "concepts": concepts,
                    "key_verses": list(key_verses),
                    "boost_applied": True,
                },
            )

        except Exception as e:
            logger.error(f"Concept alignment failed: {e}")
            return ToolResult(success=False, error=str(e))


# =============================================================================
# FTS SEARCH TOOL (Fallback)
# =============================================================================


class FTSSearchTool:
    """
    SQLite FTS5 search - fallback when embeddings unavailable.

    Not lazy - this is an explicit degraded mode with karma logging.
    """

    name = "fts_search"
    description = "SQLite FTS5 full-text search (fallback)"

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self._db_path = plugin_dir / "knowledge" / "vedabase.db"

    def validate(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters."""
        if "query" not in parameters:
            raise ValueError("Missing required parameter: query")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Search using FTS5.

        Args:
            parameters: {"query": "...", "limit": 10}

        Returns:
            ToolResult with matched verses
        """
        try:
            query = parameters["query"]
            limit = parameters.get("limit", 10)

            if not self._db_path.exists():
                return ToolResult(success=False, error="Database not found")

            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Use FTS5 MATCH syntax
            # Handle multi-word queries
            fts_query = " OR ".join(query.split())

            cursor.execute(
                """
                SELECT book_code, chapter, verse, translation, rank
                FROM verses_fts
                WHERE verses_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            )

            matches = []
            for row in cursor.fetchall():
                matches.append({
                    "verse_id": f"BG {row['chapter']}.{row['verse']}",
                    "translation": row["translation"][:200],
                    "rank": row["rank"],
                })

            conn.close()

            logger.info(f"FTS search: {len(matches)} matches for '{query}'")

            return ToolResult(
                success=True,
                output={
                    "matches": matches,
                    "count": len(matches),
                    "method": "fts5",
                    "degraded": True,  # Explicit degradation marker
                },
            )

        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            return ToolResult(success=False, error=str(e))


# =============================================================================
# BM25 SEARCH TOOL (Semantic without ML)
# =============================================================================


class BM25SearchTool:
    """
    BM25 ranking - semantic-like search WITHOUT machine learning.

    Philosophy:
        "When the 4GB transformer is not available,
         mathematics still provides wisdom."

    BM25 (Best Match 25) uses:
    - Term Frequency (TF) with saturation
    - Inverse Document Frequency (IDF)
    - Document length normalization

    This gives semantic-like results because:
    1. Rare words matter more (IDF)
    2. Repeated words have diminishing returns (TF saturation)
    3. Long documents don't dominate (length normalization)
    """

    name = "bm25_search"
    description = "BM25 semantic-like search (no ML required)"

    # BM25 parameters
    K1 = 1.5  # Term frequency saturation
    B = 0.75  # Length normalization

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self._db_path = plugin_dir / "knowledge" / "vedabase.db"
        self._concepts_path = plugin_dir / "knowledge" / "concepts.yaml"
        self._corpus = None
        self._verse_ids = None
        self._avgdl = 0
        self._idf = {}
        self._doc_lens = []
        self._synonyms = None  # Query expansion synonyms

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens

    def _build_index(self):
        """Build BM25 index from database."""
        if self._corpus is not None:
            return

        if not self._db_path.exists():
            raise FileNotFoundError(f"Database not found: {self._db_path}")

        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT chapter, verse, translation
            FROM verses
            ORDER BY chapter, verse
        """)

        self._corpus = []
        self._verse_ids = []
        self._doc_lens = []
        doc_freq = {}  # Document frequency for IDF

        for row in cursor.fetchall():
            chapter, verse_num, translation = row
            verse_id = f"BG {chapter}.{verse_num}"

            tokens = self._tokenize(translation)
            self._corpus.append((verse_id, translation, tokens))
            self._verse_ids.append(verse_id)
            self._doc_lens.append(len(tokens))

            # Count document frequency
            for term in set(tokens):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        conn.close()

        # Compute average document length
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 1

        # Compute IDF for all terms
        N = len(self._corpus)
        for term, df in doc_freq.items():
            # IDF with smoothing
            self._idf[term] = max(0, (N - df + 0.5) / (df + 0.5))

        logger.info(f"BM25 index built: {N} documents, {len(self._idf)} unique terms")

    def _build_synonyms(self):
        """Build synonym map from concepts.yaml for query expansion."""
        if self._synonyms is not None:
            return

        self._synonyms = {}  # word -> [synonyms]

        if not self._concepts_path.exists():
            logger.warning(f"Concepts file not found: {self._concepts_path}")
            return

        with open(self._concepts_path) as f:
            concepts = yaml.safe_load(f)

        for category, items in concepts.items():
            if not isinstance(items, dict):
                continue
            for concept_name, data in items.items():
                if isinstance(data, dict):
                    aliases = data.get("aliases", [])
                elif isinstance(data, list):
                    aliases = data
                else:
                    continue

                # Filter out non-string values and build bidirectional mapping
                all_terms = [str(concept_name)]
                for a in aliases:
                    if isinstance(a, str):
                        all_terms.append(a.lower())

                for term in all_terms:
                    term_lower = term.lower()
                    if term_lower not in self._synonyms:
                        self._synonyms[term_lower] = set()
                    # Add all other terms as synonyms
                    for other in all_terms:
                        other_lower = other.lower()
                        if other_lower != term_lower:
                            self._synonyms[term_lower].add(other_lower)

        logger.info(f"Synonym map built: {len(self._synonyms)} terms with expansions")

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms from concepts.yaml."""
        self._build_synonyms()

        words = query.lower().split()
        expanded = set(words)

        for word in words:
            if word in self._synonyms:
                expanded.update(self._synonyms[word])

        return list(expanded)

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a document."""
        import math

        _, _, doc_tokens = self._corpus[doc_idx]
        doc_len = self._doc_lens[doc_idx]

        # Term frequency in document
        tf = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue

            # IDF component
            idf = math.log(1 + self._idf.get(term, 0))

            # TF component with saturation and length normalization
            term_tf = tf[term]
            numerator = term_tf * (self.K1 + 1)
            denominator = term_tf + self.K1 * (1 - self.B + self.B * doc_len / self._avgdl)

            score += idf * (numerator / denominator)

        return score

    def validate(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters."""
        if "query" not in parameters:
            raise ValueError("Missing required parameter: query")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Search using BM25 ranking.

        Args:
            parameters: {"query": "...", "top_k": 10}

        Returns:
            ToolResult with ranked verses
        """
        try:
            self._build_index()

            query = parameters["query"]
            top_k = parameters.get("top_k", 10)
            expand = parameters.get("expand", True)  # Enable query expansion by default

            # Expand query with synonyms if enabled
            if expand:
                expanded_terms = self._expand_query(query)
                query_text = " ".join(expanded_terms)
            else:
                query_text = query

            query_tokens = self._tokenize(query_text)

            if not query_tokens:
                return ToolResult(
                    success=True,
                    output={"matches": [], "count": 0, "method": "bm25"},
                )

            # Score all documents
            scores = []
            for i in range(len(self._corpus)):
                score = self._bm25_score(query_tokens, i)
                if score > 0:
                    scores.append((i, score))

            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)

            # Get top-k results
            matches = []
            for idx, score in scores[:top_k]:
                verse_id, translation, _ = self._corpus[idx]
                matches.append({
                    "verse_id": verse_id,
                    "score": round(score, 4),
                    "translation": translation[:200],
                })

            logger.info(f"BM25 search: {len(matches)} matches for '{query}' (expanded: {expand})")

            return ToolResult(
                success=True,
                output={
                    "matches": matches,
                    "count": len(matches),
                    "method": "bm25",
                    "original_query": query,
                    "query_terms": query_tokens,
                    "expanded": expand,
                    "expansion_terms": expanded_terms if expand else [query],
                    "ml_required": False,  # Key differentiator!
                },
            )

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return ToolResult(success=False, error=str(e))


# =============================================================================
# RESONANCE SEARCH TOOL (Sanskrit Matrix + Quantum Reactor)
# =============================================================================


class ResonanceSearchTool:
    """
    Sanskrit Phonetic Resonance Search - The PHYSICS of meaning.

    Philosophy:
        "वर्णानां वर्गाः पञ्च" - "The phonemes have five classes"
        This is NOT keyword matching. This is PHYSICS.

    Uses the Sanskrit Varnamala (alphabet matrix) where each phoneme's
    properties are determined by WHERE in the mouth it's articulated.

    Dimensions:
    - VARGA: Articulation point (Throat→Lips = Kernel→Output)
    - STHANA: Energy level (Unvoiced→Nasal = Pure→Resonant)
    - GUNA: Mode (Tamas/Sattva/Rajas)
    - ENTROPY: Cryptographic mass

    This gives semantic-like results WITHOUT embeddings because
    similar concepts have similar phonetic structures in Sanskrit.
    """

    name = "resonance_search"
    description = "Sanskrit phonetic resonance search (neuro-symbolic)"

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self._db_path = plugin_dir / "knowledge" / "vedabase.db"
        self._verse_tensors = None
        self._reactor = None

    def _ensure_loaded(self):
        """Build verse tensors on first use."""
        if self._verse_tensors is not None:
            return

        try:
            # Import vibe_core reactor components
            from vibe_core.reactor.matrix import encode
            from vibe_core.reactor.quantum import QuantumReactor

            self._reactor = QuantumReactor(initial_inertia=0.3)

            # Load verses and compute tensors
            if not self._db_path.exists():
                raise FileNotFoundError(f"Database not found: {self._db_path}")

            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT chapter, verse, translation, sanskrit
                FROM verses
                ORDER BY chapter, verse
            """)

            self._verse_tensors = []
            for row in cursor.fetchall():
                chapter, verse_num, translation, sanskrit = row
                verse_id = f"BG {chapter}.{verse_num}"

                # Encode both Sanskrit (if available) and translation
                # Sanskrit gives true phonetic resonance
                # Translation provides English semantic context
                text_to_encode = sanskrit if sanskrit else translation
                tensor = encode(text_to_encode, salt=verse_id)

                self._verse_tensors.append({
                    "verse_id": verse_id,
                    "tensor": tensor,
                    "translation": translation,
                    "sanskrit": sanskrit or "",
                })

            conn.close()
            logger.info(f"Resonance index built: {len(self._verse_tensors)} verse tensors")

        except ImportError as e:
            logger.warning(f"vibe_core.reactor not available: {e}")
            raise RuntimeError("Resonance search requires vibe_core.reactor module")

    def validate(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters."""
        if "query" not in parameters:
            raise ValueError("Missing required parameter: query")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Search using Sanskrit phonetic resonance.

        Args:
            parameters: {"query": "...", "top_k": 10}

        Returns:
            ToolResult with verses ranked by resonance energy
        """
        try:
            self._ensure_loaded()

            from vibe_core.reactor.matrix import encode

            query = parameters["query"]
            top_k = parameters.get("top_k", 10)

            # Encode query as VarnaTensor
            query_tensor = encode(query, salt="prabhupada_query")

            # Compute resonance with each verse
            resonances = []
            for verse_data in self._verse_tensors:
                verse_tensor = verse_data["tensor"]

                # Compute resonance field
                field = self._reactor.resonate(query_tensor, verse_tensor)

                resonances.append({
                    "verse_id": verse_data["verse_id"],
                    "total_energy": field.total_energy,
                    "phonetic_resonance": field.phonetic_resonance,
                    "varga_alignment": field.varga_alignment,
                    "guna_harmony": field.guna_harmony,
                    "translation": verse_data["translation"][:200],
                    "sanskrit": verse_data["sanskrit"][:100] if verse_data["sanskrit"] else "",
                })

            # Sort by total energy (highest first)
            resonances.sort(key=lambda x: x["total_energy"], reverse=True)

            # Get top-k results
            matches = resonances[:top_k]

            logger.info(f"Resonance search: {len(matches)} matches for '{query}'")

            return ToolResult(
                success=True,
                output={
                    "matches": matches,
                    "count": len(matches),
                    "method": "sanskrit_resonance",
                    "query_signature": query_tensor.resonance_signature(),
                    "ml_required": False,
                    "physics_based": True,  # Key differentiator!
                },
            )

        except Exception as e:
            logger.error(f"Resonance search failed: {e}")
            return ToolResult(success=False, error=str(e))


# =============================================================================
# TOOL REGISTRY
# =============================================================================


def get_semantic_tools(plugin_dir: Path) -> Dict[str, Any]:
    """
    Get all semantic tools for TaskKernel injection.

    Returns dict of tool_name -> tool_instance.
    """
    return {
        "embedding_compute": EmbeddingComputeTool(plugin_dir),
        "vector_search": VectorSearchTool(plugin_dir),
        "concept_align": ConceptAlignTool(plugin_dir),
        "fts_search": FTSSearchTool(plugin_dir),
        "bm25_search": BM25SearchTool(plugin_dir),
        "resonance_search": ResonanceSearchTool(plugin_dir),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EmbeddingComputeTool",
    "VectorSearchTool",
    "ConceptAlignTool",
    "FTSSearchTool",
    "BM25SearchTool",
    "ResonanceSearchTool",
    "get_semantic_tools",
    "ToolResult",
]
