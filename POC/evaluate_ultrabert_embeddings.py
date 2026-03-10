"""UltraBERT embedding quality evaluation for capability retrieval.

This script generates a synthetic multi-domain capability corpus and evaluates
whether domain-prefixed embedding text improves retrieval and geometry metrics
versus raw descriptions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import yaml
from familyos_ultrabert import UltraBERT
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score


RANDOM_SEED = 42
TOP_K_VALUES = (1, 3, 5, 10)
NDCG_K = 10
IVF_NLIST = 100
IVF_NPROBE = 10
IVF_MIN_POINTS_PER_CENTROID = 39
RERANK_CANDIDATE_K = 100
RERANK_DOMAIN_BONUS = 0.20
RERANK_OPERATION_TOKEN_BONUS = 0.20
RERANK_TOKEN_OVERLAP_WEIGHT = 0.20
RERANK_EXACT_VERB_BONUS = 0.22
RERANK_CANONICAL_OP_BONUS = 0.10
RERANK_REQUIRED_INPUT_OVERLAP_BONUS = 0.06
HYBRID_VECTOR_WEIGHT = 0.80
HYBRID_BM25_WEIGHT = 0.20
BM25_K1 = 1.5
BM25_B = 0.75
HARDWIRED_INPUT_STYLE = "hybrid_sentence"

INTENT_STOPWORDS = {
    "intent",
    "find",
    "capability",
    "tool",
    "please",
    "need",
    "can",
    "you",
    "to",
    "for",
    "the",
    "a",
    "an",
    "with",
    "and",
    "best",
    "matching",
    "select",
    "required_inputs",
    "domain_hint",
    "req_inputs",
    "dom_hint",
}

INTENT_TOKEN_NORMALIZATION = {
    "create": "create",
    "add": "create",
    "append": "create",
    "new": "create",
    "draft": "create",
    "log": "create",
    "record": "create",
    "generate": "create",
    "copy": "create",
    "book": "schedule",
    "schedule": "schedule",
    "reschedule": "schedule",
    "set": "schedule",
    "cancel": "delete",
    "remove": "delete",
    "delete": "delete",
    "fetch": "retrieve",
    "get": "retrieve",
    "retrieve": "retrieve",
    "list": "retrieve",
    "lookup": "retrieve",
    "find": "retrieve",
    "check": "retrieve",
    "download": "retrieve",
    "view": "retrieve",
    "update": "update",
    "modify": "update",
    "edit": "update",
    "rename": "update",
    "move": "update",
    "assign": "update",
    "tag": "update",
    "pin": "update",
    "reopen": "update",
    "merge": "update",
    "refill": "update",
    "send": "send",
    "notify": "send",
    "message": "send",
    "reply": "send",
    "forward": "send",
    "invite": "send",
    "share": "send",
    "push": "send",
    "submit": "send",
    "search": "search",
    "query": "search",
    "track": "track",
    "monitor": "monitor",
    "analyze": "analyze",
    "analyse": "analyze",
    "summarize": "summarize",
    "summary": "summarize",
    "approve": "approve",
    "accept": "approve",
    "decline": "update",
    "close": "complete",
    "complete": "complete",
    "archive": "archive",
    "upload": "upload",
    "export": "export",
    "refund": "finance",
    "reconcile": "finance",
    "balance": "finance",
}

OPERATION_ONTOLOGY: Dict[str, List[str]] = {
    "create": ["create", "add", "append", "new", "open", "initiate", "draft", "log", "record", "generate", "copy"],
    "retrieve": ["retrieve", "get", "fetch", "list", "lookup", "read", "find", "check", "download", "view"],
    "update": ["update", "modify", "edit", "change", "patch", "rename", "move", "assign", "tag", "pin", "reopen", "merge"],
    "delete": ["delete", "remove", "cancel", "drop"],
    "schedule": ["schedule", "book", "reschedule", "plan", "set"],
    "send": ["send", "notify", "message", "dispatch", "push", "reply", "forward", "invite", "share", "submit"],
    "search": ["search", "query", "find", "discover"],
    "monitor": ["monitor", "observe", "watch", "track"],
    "analyze": ["analyze", "analyse", "score", "classify"],
    "summarize": ["summarize", "summary", "digest"],
    "approve": ["approve", "authorize", "accept", "validate"],
    "complete": ["complete", "close", "resolve", "finish"],
    "archive": ["archive", "retire", "store"],
    "upload": ["upload", "attach", "ingest"],
    "export": ["export", "extract"],
    "finance": ["refund", "reconcile", "balance", "invoice", "claim", "payment", "expense"],
}

REAL_QUERY_FAMILY_PATTERNS = [
    "intent: {operation}",
    "intent: {capability}",
    "intent: {operation}; domain_hint: {domain}",
    "intent: {domain_need}",
    "intent: {capability}; required_inputs: {required_inputs}",
]

DOMAIN_DAILY_NEEDS: Dict[str, str] = {
    "weather": "checking today's weather before planning family outings",
    "calendar": "planning family events and reminders",
    "notes": "finding and organizing family notes",
    "recipe": "planning meals and grocery prep",
    "discovery": "finding the right helper tool for a household task",
    "meta": "finding the right helper tool for a household task",
    "family": "coordinating daily family tasks",
}

EMBEDDING_TEMPLATES: Dict[str, str] = {
    "hardwired_hybrid_sentence": "Hardwired: natural sentence + compact key anchors",
}

INPUT_STYLE_VARIANTS: Dict[str, str] = {
    "structured_kv": "Key-value structured fields",
    "hybrid_sentence": "Natural sentence + compact key anchors",
    "natural_sentence": "Fully natural prose sentence",
}

STEP1_NORMALIZATION_BATCHES = {
    "golden_batch_001_calendar.json",
    "golden_batch_002_messaging.json",
}

CANONICAL_OPERATION_BY_SUBCLUSTER = {
    "create_event": "create",
    "update_event": "update",
    "delete_event": "delete",
    "list_events": "retrieve",
    "get_event": "retrieve",
    "find_free_slots": "search",
    "invite_attendees": "send",
    "accept_invite": "approve",
    "decline_invite": "update",
    "add_reminder": "create",
    "send_sms": "send",
    "send_whatsapp": "send",
    "send_push": "send",
    "schedule_message": "schedule",
    "cancel_scheduled_message": "delete",
    "get_delivery_status": "retrieve",
    "list_messages": "retrieve",
    "create_template": "create",
    "update_template": "update",
    "delete_template": "delete",
}

SUBCLUSTER_VERB_TO_CANONICAL = {
    "accept": "approve",
    "add": "create",
    "append": "create",
    "approve": "approve",
    "archive": "archive",
    "assign": "update",
    "cancel": "delete",
    "check": "retrieve",
    "close": "complete",
    "complete": "complete",
    "copy": "create",
    "create": "create",
    "decline": "update",
    "delete": "delete",
    "download": "retrieve",
    "draft": "create",
    "export": "export",
    "find": "search",
    "forward": "send",
    "generate": "create",
    "get": "retrieve",
    "invite": "send",
    "list": "retrieve",
    "log": "create",
    "merge": "update",
    "move": "update",
    "pin": "update",
    "record": "create",
    "reconcile": "finance",
    "reopen": "update",
    "refill": "update",
    "refund": "finance",
    "remove": "delete",
    "rename": "update",
    "reply": "send",
    "schedule": "schedule",
    "search": "search",
    "send": "send",
    "set": "schedule",
    "share": "send",
    "submit": "send",
    "tag": "update",
    "update": "update",
    "upload": "upload",
}

@dataclass
class QueryItem:
    """Represents a retrieval query with domain and sub-cluster labels."""

    query_text: str
    domain: str
    sub_cluster: str
    target_id: str = ""


@dataclass
class ModelInfo:
    """Holds model metadata for reporting."""

    model_name: str
    dimension: int
    fallback_used: bool


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for corpus source and batching."""

    corpus_mode: str
    real_contracts_root: Path
    golden_set_path: Path
    batch_index: int
    batch_size: int
    max_contracts: int
    query_style: str
    hybrid_vector_weight: float
    hybrid_bm25_weight: float
    rerank_domain_bonus: float
    rerank_operation_token_bonus: float
    rerank_token_overlap_weight: float
    rerank_candidate_k: int


@dataclass(frozen=True)
class BM25Stats:
    """BM25 corpus statistics for token-based ranking."""

    doc_tokens: List[List[str]]
    doc_freq: Dict[str, int]
    doc_lengths: List[int]
    avg_doc_length: float


class UltraBERTEmbedder:
    """Thin wrapper for UltraBERT embedding extraction."""

    def __init__(self, model: UltraBERT):
        self.model = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text list to normalized float32 embeddings."""
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append(self.model.get_embedding(str(text)))

        arr = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)


def load_model() -> Tuple[UltraBERTEmbedder, ModelInfo]:
    """Load UltraBERT using the verified runtime path and validate embedding output."""
    ultrabert = UltraBERT.load(backend="auto", device="auto", quantization="fp32")

    probe_text = "quick ultrabert smoke test"
    probe_embedding = ultrabert.get_embedding(probe_text)
    dim = int(len(probe_embedding))
    if dim <= 0:
        raise ValueError("UltraBERT failed to produce a valid embedding during load probe.")

    embedder = UltraBERTEmbedder(ultrabert)
    sample = embedder.encode([probe_text])
    if int(sample.shape[1]) != dim:
        raise ValueError("UltraBERT embedding dimension mismatch between raw and normalized probe.")

    model_name = f"familyos_ultrabert({ultrabert.backend})"
    return embedder, ModelInfo(model_name=model_name, dimension=dim, fallback_used=False)


def parse_args() -> RunConfig:
    """Parse CLI arguments for synthetic or real corpus evaluation."""
    parser = argparse.ArgumentParser(description="UltraBERT capability retrieval evaluator")
    parser.add_argument(
        "--corpus-mode",
        choices=["real", "golden"],
        default="golden",
        help="Corpus source mode.",
    )
    parser.add_argument(
        "--real-contracts-root",
        default=r"d:\familyos\k1\contracts\tools",
        help="Path to directory containing real tool contract YAML files.",
    )
    parser.add_argument(
        "--golden-set-path",
        default=r"d:\Modeling_studio\POC\manual_golden_batches",
        help="Path to manual golden JSON file or directory of manual golden batch JSON files.",
    )
    parser.add_argument("--batch-index", type=int, default=0, help="Zero-based batch index for real corpus mode.")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for real corpus mode.")
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=0,
        help="Optional hard limit on real contracts loaded (0 means no limit).",
    )
    parser.add_argument(
        "--query-style",
        choices=["llm_clean", "llm_noisy"],
        default="llm_clean",
        help="Query phrasing style to simulate LLM tool-call intent strings.",
    )
    parser.add_argument(
        "--hybrid-vector-weight",
        type=float,
        default=HYBRID_VECTOR_WEIGHT,
        help="Hybrid retrieval vector score weight.",
    )
    parser.add_argument(
        "--hybrid-bm25-weight",
        type=float,
        default=HYBRID_BM25_WEIGHT,
        help="Hybrid retrieval BM25 score weight.",
    )
    parser.add_argument(
        "--rerank-domain-bonus",
        type=float,
        default=RERANK_DOMAIN_BONUS,
        help="Rerank bonus when candidate domain matches query domain.",
    )
    parser.add_argument(
        "--rerank-operation-bonus",
        type=float,
        default=RERANK_OPERATION_TOKEN_BONUS,
        help="Rerank bonus weight for operation token overlap.",
    )
    parser.add_argument(
        "--rerank-token-overlap-weight",
        type=float,
        default=RERANK_TOKEN_OVERLAP_WEIGHT,
        help="Rerank score weight for general token overlap.",
    )
    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=RERANK_CANDIDATE_K,
        help="Candidate pool size used before reranking.",
    )
    args = parser.parse_args()

    vector_weight = float(args.hybrid_vector_weight)
    bm25_weight = float(args.hybrid_bm25_weight)
    if vector_weight < 0 or bm25_weight < 0:
        raise ValueError("Hybrid weights must be non-negative.")
    if (vector_weight + bm25_weight) <= 0:
        raise ValueError("Hybrid weights must have positive sum.")

    weight_sum = vector_weight + bm25_weight
    vector_weight = vector_weight / weight_sum
    bm25_weight = bm25_weight / weight_sum

    return RunConfig(
        corpus_mode=str(args.corpus_mode),
        real_contracts_root=Path(str(args.real_contracts_root)),
        golden_set_path=Path(str(args.golden_set_path)),
        batch_index=max(0, int(args.batch_index)),
        batch_size=max(0, int(args.batch_size)),
        max_contracts=max(0, int(args.max_contracts)),
        query_style=str(args.query_style),
        hybrid_vector_weight=vector_weight,
        hybrid_bm25_weight=bm25_weight,
        rerank_domain_bonus=float(args.rerank_domain_bonus),
        rerank_operation_token_bonus=float(args.rerank_operation_bonus),
        rerank_token_overlap_weight=float(args.rerank_token_overlap_weight),
        rerank_candidate_k=max(10, int(args.rerank_candidate_k)),
    )


def load_golden_set(
    golden_set_path: Path,
    query_style: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[QueryItem], List[str]]:
    """Load manually curated golden tool/query dataset from file or batch directory."""
    if not golden_set_path.exists():
        raise ValueError(f"Golden set path not found: {golden_set_path}")

    payloads: List[Tuple[Dict[str, object], str]] = []
    sources: List[str] = []
    if golden_set_path.is_dir():
        batch_files = sorted(golden_set_path.glob("*.json"))
        if not batch_files:
            raise ValueError(f"No golden batch JSON files found in directory: {golden_set_path}")
        for batch_file in batch_files:
            payload = json.loads(batch_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payloads.append((payload, batch_file.name))
                sources.append(str(batch_file))
    else:
        payload = json.loads(golden_set_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payloads.append((payload, golden_set_path.name))
            sources.append(str(golden_set_path))

    if not payloads:
        raise ValueError("Golden set payload is empty or invalid.")

    raw_tools: List[object] = []
    raw_queries: List[object] = []
    for payload, source_name in payloads:
        tools = payload.get("tools", [])
        queries = payload.get("queries", [])
        if isinstance(tools, list):
            for t in tools:
                if isinstance(t, dict):
                    tt = dict(t)
                    tt["_source_file"] = source_name
                    raw_tools.append(tt)
        if isinstance(queries, list):
            raw_queries.extend(queries)

    if not raw_tools:
        raise ValueError("Golden set must include non-empty 'tools' list.")

    corpus_base: List[Dict[str, object]] = []
    multi_items: List[Dict[str, object]] = []

    step1_count = 0
    for t in raw_tools:
        if not isinstance(t, dict):
            continue
        item = dict(t)
        item.setdefault("adversarial", False)
        item.setdefault("multi_domain", len(item.get("domains", [])) > 1)
        item.setdefault("required_inputs", [])
        item.setdefault("tags", [])
        item.setdefault("provider_type", "mcp")
        item.setdefault("tool_display_name", str(item.get("tool_name", "")).split(".")[-1].replace("_", " ").title())
        if not item.get("id"):
            item["id"] = f"gold.{item.get('tool_name', 'unknown')}"
        if not item.get("domain"):
            domains = item.get("domains", [])
            item["domain"] = str(domains[0]).lower() if domains else "unknown"
        if not item.get("sub_cluster"):
            item["sub_cluster"] = str(item.get("tool_name", "unknown")).split(".")[-1]

        source_hint = str(item.get("_source_file", "")).strip()
        if source_hint and source_hint in STEP1_NORMALIZATION_BATCHES:
            item = normalize_tool_item_step1(item)
            step1_count += 1

        corpus_base.append(item)
        if bool(item.get("multi_domain", False)):
            multi_items.append(dict(item))

    queries: List[QueryItem] = []
    for q in raw_queries:
        if not isinstance(q, dict):
            continue
        text = apply_query_style(str(q.get("query_text", "")).strip(), query_style=query_style)
        if not text:
            continue
        queries.append(
            QueryItem(
                query_text=text,
                domain=str(q.get("domain", "unknown")),
                sub_cluster=str(q.get("sub_cluster", "unknown")),
                target_id=str(q.get("target_id", "")),
            )
        )

    if not queries:
        queries = generate_queries_from_real_corpus(
            corpus_base=corpus_base,
            max_queries=max(200, min(2000, len(corpus_base) * 2)),
            query_style=query_style,
        )

    if not queries:
        raise ValueError("Golden set queries resolved to empty set after fallback generation.")

    if step1_count > 0:
        sources = [*sources, f"step1_normalized_items:{step1_count}"]

    return corpus_base, multi_items, queries, sources


def discover_tool_contract_paths(root: Path) -> List[Path]:
    """Discover tool contract YAML files under the provided root."""
    if not root.exists():
        return []
    return sorted([*root.glob("*.yaml"), *root.glob("*.yml")])


def apply_query_style(text: str, query_style: str) -> str:
    """Apply clean/noisy transformation to intent query strings."""
    q = text.strip()
    if query_style != "llm_noisy":
        return q

    replacements = [
        ("capability", "cap"),
        ("required_inputs", "req_inputs"),
        ("domain_hint", "dom_hint"),
        ("select best matching tool", "pick best tool"),
        ("find capability to", "find cap to"),
        ("reliable execution", "reliable run"),
    ]
    noisy = q.lower()
    for old, new in replacements:
        noisy = noisy.replace(old, new)

    noisy = noisy.replace("; ", " | ")
    return noisy


def tokenize_for_intent(text: str) -> List[str]:
    """Tokenize and normalize free-form intent text into operation-friendly terms."""
    tokens = re.findall(r"[a-z0-9_\-]+", text.lower())
    out: List[str] = []
    for tok in tokens:
        if tok in INTENT_STOPWORDS:
            continue
        canonical = INTENT_TOKEN_NORMALIZATION.get(tok, tok)
        if canonical and canonical not in INTENT_STOPWORDS:
            out.append(canonical)
    return out


def expand_tokens_with_ontology(tokens: Sequence[str]) -> List[str]:
    """Expand token list with operation ontology canonical forms and aliases."""
    expanded: List[str] = list(tokens)
    token_set = set(tokens)
    for canonical, aliases in OPERATION_ONTOLOGY.items():
        alias_set = set(aliases)
        if token_set.intersection(alias_set):
            expanded.append(canonical)
            expanded.extend(aliases)
    return expanded


def normalize_intent_text(text: str) -> str:
    """Normalize query text into an operation-heavy phrase for retrieval."""
    tokens = tokenize_for_intent(text)
    expanded = expand_tokens_with_ontology(tokens)
    seen: set = set()
    ordered: List[str] = []
    for tok in expanded:
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)
    return " ".join(ordered)


def _normalize_domain(value: str) -> str:
    """Normalize domain tag text for stable clustering labels."""
    return value.strip().lower().replace("_", "-")


def _parse_required_input_names(required_inputs: object) -> List[str]:
    """Extract required input names from tool contract input schema."""
    names: List[str] = []
    if isinstance(required_inputs, list):
        for item in required_inputs:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    names.append(name)
            elif isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    names.append(cleaned)
    return names


def _extract_provider_from_tool_name(tool_name: str) -> str:
    """Extract provider segment from canonical tool name path."""
    parts = tool_name.split(".")
    if len(parts) >= 5:
        return str(parts[3]).strip().lower()
    return "unknown_provider"


def canonical_operation_for_subcluster(sub_cluster: str) -> str:
    """Resolve canonical operation label for a sub-cluster string."""
    sc = str(sub_cluster).strip().lower()
    if not sc:
        return "unknown"

    mapped = CANONICAL_OPERATION_BY_SUBCLUSTER.get(sc)
    if mapped:
        return mapped

    verb = sc.split("_", 1)[0]
    return SUBCLUSTER_VERB_TO_CANONICAL.get(verb, verb)


def evaluate_operation_mapping_qa(
    corpus_base: List[Dict[str, object]],
    queries: List[QueryItem],
) -> Dict[str, object]:
    """Compute Step 2 ontology and mapping QA coverage from real corpus vocabulary."""
    unique_subclusters = sorted({str(item.get("sub_cluster", "")).strip().lower() for item in corpus_base if str(item.get("sub_cluster", "")).strip()})

    mapped_rows: List[Dict[str, str]] = []
    unmapped_subclusters: List[str] = []
    canonical_counter: Counter = Counter()
    verb_counter: Counter = Counter()

    for sub in unique_subclusters:
        verb = sub.split("_", 1)[0]
        canonical = canonical_operation_for_subcluster(sub)
        ontology_aliases = OPERATION_ONTOLOGY.get(canonical, [])
        alias_hit = bool(verb in ontology_aliases or INTENT_TOKEN_NORMALIZATION.get(verb) == canonical)

        if alias_hit:
            canonical_counter[canonical] += 1
        else:
            unmapped_subclusters.append(sub)

        verb_counter[verb] += 1
        mapped_rows.append(
            {
                "sub_cluster": sub,
                "verb": verb,
                "canonical": canonical,
                "ontology_alias_hit": "yes" if alias_hit else "no",
            }
        )

    operation_lexicon: set = set(OPERATION_ONTOLOGY.keys())
    for aliases in OPERATION_ONTOLOGY.values():
        operation_lexicon.update(aliases)
    operation_lexicon.update(SUBCLUSTER_VERB_TO_CANONICAL.keys())

    query_token_total = 0
    query_token_mapped = 0
    for q in queries:
        tokens = tokenize_for_intent(q.query_text)
        for tok in tokens:
            if tok not in operation_lexicon:
                continue
            query_token_total += 1
            canonical = INTENT_TOKEN_NORMALIZATION.get(tok, tok)
            if canonical in OPERATION_ONTOLOGY:
                query_token_mapped += 1

    total_subclusters = len(unique_subclusters)
    mapped_subclusters = total_subclusters - len(unmapped_subclusters)
    subcluster_coverage = float(mapped_subclusters / total_subclusters) if total_subclusters else 0.0
    query_token_coverage = float(query_token_mapped / query_token_total) if query_token_total else 0.0

    return {
        "total_subclusters": total_subclusters,
        "mapped_subclusters": mapped_subclusters,
        "subcluster_coverage": subcluster_coverage,
        "query_token_total": query_token_total,
        "query_token_mapped": query_token_mapped,
        "query_token_coverage": query_token_coverage,
        "canonical_distribution": dict(sorted(canonical_counter.items())),
        "top_verbs": dict(verb_counter.most_common(20)),
        "unmapped_subclusters": sorted(unmapped_subclusters),
        "mapping_rows": mapped_rows,
    }


def normalize_tool_item_step1(item: Dict[str, object]) -> Dict[str, object]:
    """Apply Step 1 normalization policy to a tool record."""
    out = dict(item)
    domain = str(out.get("domain", "unknown")).strip().lower()
    sub_cluster = str(out.get("sub_cluster", "unknown")).strip().lower()
    tool_name = str(out.get("tool_name", "unknown")).strip()
    provider = _extract_provider_from_tool_name(tool_name)

    operation_canonical = canonical_operation_for_subcluster(sub_cluster)
    operation_human = sub_cluster.replace("_", " ")

    req = [str(v).strip() for v in out.get("required_inputs", []) if str(v).strip()]
    id_like = [v for v in req if v.endswith("_id") or v in {"id", "message_id", "event_id", "calendar_id", "template_id"}]
    others = [v for v in req if v not in id_like]
    out["required_inputs"] = id_like + others

    caps = [str(v).strip().lower() for v in out.get("capabilities", []) if str(v).strip()]
    if operation_canonical not in caps:
        caps.insert(0, operation_canonical)
    out["capabilities"] = list(dict.fromkeys(caps))

    tags = [str(v).strip().lower() for v in out.get("tags", []) if str(v).strip()]
    for token in [domain, provider, operation_canonical, sub_cluster]:
        if token and token not in tags:
            tags.append(token)
    out["tags"] = tags

    description = str(out.get("raw_description", "")).strip()
    first_sentence = (
        f"{provider} {domain} capability to {operation_human.replace('-', ' ')} "
        f"for deterministic tool routing and typed inputs."
    )
    if description:
        if not description.startswith(first_sentence):
            out["raw_description"] = f"{first_sentence} {description}"
    else:
        out["raw_description"] = first_sentence

    return out


def build_domain_sub_indexes(
    base_embeddings: np.ndarray,
    corpus_base: List[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """Build per-domain FAISS sub-indexes for constrained retrieval."""
    domain_to_indices: Dict[str, List[int]] = {}
    for idx, item in enumerate(corpus_base):
        domain = str(item.get("domain", "unknown"))
        domain_to_indices.setdefault(domain, []).append(idx)

    out: Dict[str, Dict[str, object]] = {}
    for domain, indices in domain_to_indices.items():
        global_idx = np.asarray(indices, dtype=np.int64)
        domain_emb = base_embeddings[global_idx]
        index = build_flat_index(domain_emb)
        out[domain] = {
            "index": index,
            "global_indices": global_idx,
        }
    return out


def build_bm25_stats(corpus_base: List[Dict[str, object]]) -> BM25Stats:
    """Create BM25 token statistics over tool metadata text."""
    docs: List[List[str]] = []
    doc_freq_counter: Counter = Counter()
    doc_lengths: List[int] = []

    for item in corpus_base:
        operation = str(item.get("tool_name", str(item.get("sub_cluster", "")))).split(".")[-1].replace("_", " ")
        sub_cluster = str(item.get("sub_cluster", "")).replace("_", " ").replace("-", " ")
        description = str(item.get("raw_description", ""))
        capabilities = " ".join(str(v).replace("_", " ") for v in item.get("capabilities", []))
        required_inputs = " ".join(str(v).replace("_", " ") for v in item.get("required_inputs", []))
        text = f"{operation} {sub_cluster} {description} {capabilities} {required_inputs}"
        tokens = expand_tokens_with_ontology(tokenize_for_intent(text))
        docs.append(tokens)
        doc_lengths.append(len(tokens))
        doc_freq_counter.update(set(tokens))

    avg_len = float(np.mean(doc_lengths)) if doc_lengths else 1.0
    return BM25Stats(
        doc_tokens=docs,
        doc_freq=dict(doc_freq_counter),
        doc_lengths=doc_lengths,
        avg_doc_length=max(avg_len, 1.0),
    )


def bm25_score(query_tokens: Sequence[str], doc_tokens: Sequence[str], stats: BM25Stats) -> float:
    """Compute BM25 score for one query-document token pair."""
    if not query_tokens or not doc_tokens:
        return 0.0

    tf = Counter(doc_tokens)
    score = 0.0
    n_docs = len(stats.doc_tokens)
    dl = max(len(doc_tokens), 1)

    for term in query_tokens:
        if term not in tf:
            continue
        df = stats.doc_freq.get(term, 0)
        idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
        term_tf = float(tf[term])
        numer = term_tf * (BM25_K1 + 1.0)
        denom = term_tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / stats.avg_doc_length))
        score += idf * (numer / max(denom, 1e-9))
    return float(score)


def load_real_corpus(config: RunConfig) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    """Load a batch of real tool contracts and map them into corpus items."""
    paths = discover_tool_contract_paths(config.real_contracts_root)
    if not paths:
        return [], [], []

    if config.batch_size > 0:
        start = config.batch_index * config.batch_size
        end = start + config.batch_size
        paths = paths[start:end]

    if config.max_contracts > 0:
        paths = paths[: config.max_contracts]

    base_items: List[Dict[str, object]] = []
    multi_items: List[Dict[str, object]] = []
    loaded_files: List[str] = []

    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue

        tool_contract = payload.get("tool_contract")
        if not isinstance(tool_contract, dict):
            continue

        name = str(tool_contract.get("name", "")).strip()
        if not name:
            continue

        raw_domains = tool_contract.get("domain", [])
        domain_list = [_normalize_domain(str(d)) for d in raw_domains if str(d).strip()]
        if not domain_list:
            domain_list = ["unknown"]

        domain_primary = domain_list[0]
        operation = name.split(".")[-1] if "." in name else name
        capabilities = [str(v).strip().lower() for v in tool_contract.get("capabilities", []) if str(v).strip()]
        if not capabilities:
            capabilities = [operation]
        tags = [str(v).strip().lower() for v in tool_contract.get("tags", []) if str(v).strip()]
        limitations = [str(v).strip().lower() for v in tool_contract.get("limitations", []) if str(v).strip()]
        tags = sorted(set(tags + limitations + domain_list))

        required_input_names = _parse_required_input_names(tool_contract.get("required_inputs", []))
        provider_id = str(tool_contract.get("provider_id", "unknown_provider")).strip() or "unknown_provider"
        provider_type = str(tool_contract.get("provider_type", "mcp")).strip().lower() or "mcp"
        description = str(tool_contract.get("description", "")).strip()

        item: Dict[str, object] = {
            "id": f"real.{name}",
            "domain": domain_primary,
            "domains": domain_list,
            "sub_cluster": operation,
            "raw_description": description,
            "capabilities": capabilities,
            "required_inputs": required_input_names,
            "tags": tags,
            "company": provider_id,
            "adapter_id": provider_id,
            "tool_name": name,
            "tool_display_name": operation.replace("_", " ").title(),
            "provider_type": provider_type,
            "adversarial": False,
            "multi_domain": len(domain_list) > 1,
        }

        base_items.append(item)
        if len(domain_list) > 1:
            multi_items.append(dict(item))

        loaded_files.append(str(path))

    return base_items, multi_items, loaded_files


def generate_queries_from_real_corpus(
    corpus_base: List[Dict[str, object]],
    max_queries: int = 200,
    query_style: str = "llm_clean",
) -> List[QueryItem]:
    """Generate query set from real contract fields for retrieval evaluation."""
    queries: List[QueryItem] = []
    seen: set = set()

    for item in corpus_base:
        domain = str(item.get("domain", "unknown"))
        sub_cluster = str(item.get("sub_cluster", "unknown"))
        description = str(item.get("raw_description", "")).strip()
        capabilities = [str(v) for v in item.get("capabilities", [])]
        capability = capabilities[0].replace("_", " ") if capabilities else sub_cluster.replace("_", " ")
        operation = str(item.get("tool_name", "")).split(".")[-1]
        operation_text = operation.replace("_", " ")
        domain_need = DOMAIN_DAILY_NEEDS.get(domain, "handling everyday family tasks")
        required_inputs = [str(v).strip() for v in item.get("required_inputs", []) if str(v).strip()]
        required_inputs_text = " ".join(required_inputs) if required_inputs else "none"
        description_first = description.split(".")[0].strip() if description else ""

        candidates = [
            REAL_QUERY_FAMILY_PATTERNS[0].format(operation=operation_text),
            REAL_QUERY_FAMILY_PATTERNS[1].format(capability=capability),
            REAL_QUERY_FAMILY_PATTERNS[2].format(operation=operation_text, domain=domain),
            REAL_QUERY_FAMILY_PATTERNS[3].format(domain_need=domain_need),
            REAL_QUERY_FAMILY_PATTERNS[4].format(capability=capability, required_inputs=required_inputs_text),
        ]
        if description_first:
            candidates.append(f"intent: {description_first}")

        for text in candidates:
            styled = apply_query_style(text, query_style=query_style)
            normalized = styled.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            queries.append(
                QueryItem(
                    query_text=styled.strip(),
                    domain=domain,
                    sub_cluster=sub_cluster,
                    target_id=str(item.get("id", "")),
                )
            )
            if len(queries) >= max_queries:
                return queries

    return queries


def build_adversarial_pairs_real(corpus_base: List[Dict[str, object]], max_pairs: int = 20) -> List[Tuple[str, str]]:
    """Build adversarial pairs from real corpus using same-operation, cross-domain matches."""
    by_operation: Dict[str, List[Dict[str, object]]] = {}
    for item in corpus_base:
        operation = str(item.get("tool_name", "")).split(".")[-1]
        by_operation.setdefault(operation, []).append(item)

    pairs: List[Tuple[str, str]] = []
    for items in by_operation.values():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if str(items[i].get("domain")) == str(items[j].get("domain")):
                    continue
                items[i]["adversarial"] = True
                items[j]["adversarial"] = True
                pairs.append((str(items[i]["id"]), str(items[j]["id"])))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def embed_texts(model: UltraBERTEmbedder, texts: Sequence[str]) -> np.ndarray:
    """Encode texts as normalized float32 embeddings."""
    return model.encode(texts)


def build_embedding_text(item: Dict[str, object], template_name: str) -> str:
    """Build embedding text using hardwired best input style."""
    _ = template_name
    return build_embedding_input_style_text(item, HARDWIRED_INPUT_STYLE)


def build_embedding_input_style_text(item: Dict[str, object], style_name: str) -> str:
    """Build style-specific embedding text to test UltraBERT input sensitivity."""
    domain = str(item.get("domain", "unknown"))
    raw_description = str(item.get("raw_description", ""))
    tool_name = str(item.get("tool_name", "unknown"))
    operation = tool_name.split(".")[-1] if "." in tool_name else tool_name
    display_name = str(item.get("tool_display_name", operation.replace("_", " ").title()))
    capabilities = " ".join(str(v) for v in item.get("capabilities", []))
    required_inputs = " ".join(str(v) for v in item.get("required_inputs", []))
    tags = " ".join(str(v) for v in item.get("tags", []))

    if style_name == "structured_kv":
        return (
            f"domain:{domain} | operation:{operation} | tool:{tool_name} | "
            f"description:{raw_description} | capabilities:{capabilities} | "
            f"required_inputs:{required_inputs} | tags:{tags}"
        )

    if style_name == "hybrid_sentence":
        return (
            f"In domain {domain}, tool {display_name} performs {operation.replace('_', ' ')}. "
            f"It is used to {capabilities}. {raw_description} "
            f"Inputs include {required_inputs if required_inputs else 'none'}. "
            f"Key tags: {tags}."
        )

    if style_name == "natural_sentence":
        return (
            f"{display_name} helps teams in {domain} to {operation.replace('_', ' ')}. "
            f"{raw_description} "
            f"The tool supports {capabilities} and expects {required_inputs if required_inputs else 'standard context values'} as inputs."
        )

    raise ValueError(f"Unsupported input style: {style_name}")


def build_query_input_style_text(query: QueryItem, style_name: str) -> str:
    """Build style-specific query text for input-style ablation."""
    norm_intent = normalize_intent_text(query.query_text)
    intent = norm_intent if norm_intent else query.query_text
    op = query.sub_cluster.replace("_", " ").replace("-", " ")

    if style_name == "structured_kv":
        return f"domain:{query.domain} | intent:{intent} | operation_hint:{op}"
    if style_name == "hybrid_sentence":
        return f"Need a {query.domain} capability to {intent}. Operation context: {op}."
    if style_name == "natural_sentence":
        return f"I need help in {query.domain} to {intent}."

    raise ValueError(f"Unsupported input style: {style_name}")


def evaluate_input_style_variants(
    model: UltraBERTEmbedder,
    corpus_base: List[Dict[str, object]],
    queries: List[QueryItem],
) -> Dict[str, Dict[str, float]]:
    """Evaluate retrieval sensitivity across different embedding/query sentence styles."""
    idx_to_item_id = {i: str(item.get("id", "")) for i, item in enumerate(corpus_base)}
    id_to_meta = {
        i: (str(item.get("domain", "unknown")), str(item.get("sub_cluster", "unknown")))
        for i, item in enumerate(corpus_base)
    }

    out: Dict[str, Dict[str, float]] = {}
    for style_name in INPUT_STYLE_VARIANTS:
        doc_texts = [build_embedding_input_style_text(item, style_name) for item in corpus_base]
        doc_emb = embed_texts(model, doc_texts)
        index = build_flat_index(doc_emb)

        query_texts = [build_query_input_style_text(q, style_name) for q in queries]
        query_emb = embed_texts(model, query_texts)
        _, idx_all = index.search(query_emb, 10)

        p5_scores: List[float] = []
        mrr_scores: List[float] = []
        tpr_scores: List[float] = []
        dmr_scores: List[float] = []
        omr_scores: List[float] = []

        for i, q in enumerate(queries):
            retrieved = [int(ix) for ix in idx_all[i].tolist() if int(ix) in id_to_meta]
            retrieved_ids = [idx_to_item_id[ix] for ix in retrieved]
            retrieved_domains = [id_to_meta[ix][0] for ix in retrieved]
            retrieved_ops = [id_to_meta[ix][1] for ix in retrieved]

            rel_flags = [1 if (id_to_meta[ix][0] == q.domain and id_to_meta[ix][1] == q.sub_cluster) else 0 for ix in retrieved]
            top5 = rel_flags[:5]
            if len(top5) < 5:
                top5 = top5 + [0] * (5 - len(top5))
            p5_scores.append(float(np.mean(top5)))

            rr = 0.0
            for rank, rel in enumerate(rel_flags, start=1):
                if rel == 1:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)

            if q.target_id:
                tpr_scores.append(1.0 if q.target_id in retrieved_ids[:10] else 0.0)
            else:
                tpr_scores.append(0.0)

            dmr_scores.append(1.0 if q.domain in retrieved_domains[:10] else 0.0)
            omr_scores.append(1.0 if q.sub_cluster in retrieved_ops[:10] else 0.0)

        out[style_name] = {
            "p@5": float(np.mean(p5_scores)),
            "mrr": float(np.mean(mrr_scores)),
            "tpr@10": float(np.mean(tpr_scores)),
            "dmr@10": float(np.mean(dmr_scores)),
            "omr@10": float(np.mean(omr_scores)),
        }

    return out


def build_query_text(query: QueryItem, template_name: str, with_hint: bool) -> str:
    """Build query text using hardwired best input style."""
    _ = template_name
    base = build_query_input_style_text(query, HARDWIRED_INPUT_STYLE)
    if not with_hint:
        return base
    return f"domain:{query.domain} | {base}"


def build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build exact cosine-similarity FAISS index using normalized vectors."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def compute_domain_distance_stats(emb: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Compute mean intra and inter domain cosine distances."""
    sims = emb @ emb.T
    n = sims.shape[0]
    intra: List[float] = []
    inter: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = 1.0 - float(sims[i, j])
            if labels[i] == labels[j]:
                intra.append(dist)
            else:
                inter.append(dist)

    return {
        "mean_intra_distance": float(np.mean(intra)) if intra else math.nan,
        "mean_inter_distance": float(np.mean(inter)) if inter else math.nan,
    }


def safe_cluster_scores(emb: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute silhouette, DB, CH cluster metrics."""
    unique_labels = np.unique(y)
    if emb.shape[0] < 3 or unique_labels.shape[0] < 2 or unique_labels.shape[0] >= emb.shape[0]:
        return {
            "silhouette": 0.0,
            "davies_bouldin": 0.0,
            "calinski_harabasz": 0.0,
        }

    return {
        "silhouette": float(silhouette_score(emb, y, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(emb, y)),
        "calinski_harabasz": float(calinski_harabasz_score(emb, y)),
    }


def compute_retrieval_metrics(
    index_prefixed: faiss.IndexFlatIP,
    index_raw: faiss.IndexFlatIP,
    prefixed_embeddings_base: np.ndarray,
    model: UltraBERTEmbedder,
    corpus_base: List[Dict[str, object]],
    queries: List[QueryItem],
    template_name: str,
    domain_sub_indexes: Dict[str, Dict[str, object]],
    bm25_stats: BM25Stats,
    hybrid_vector_weight: float,
    hybrid_bm25_weight: float,
    rerank_domain_bonus: float,
    rerank_operation_token_bonus: float,
    rerank_token_overlap_weight: float,
    rerank_candidate_k: int,
) -> Dict[str, object]:
    """Compute P@K, MRR, NDCG for baseline, reranked, and domain-filtered conditions."""
    idx_to_item_id = {i: str(item["id"]) for i, item in enumerate(corpus_base)}
    id_to_meta = {
        i: (str(item["domain"]), str(item["sub_cluster"]))
        for i, item in enumerate(corpus_base)
    }
    idx_to_tokens: Dict[int, set] = {}

    def tokenize(text: str) -> set:
        return set(tok for tok in re.findall(r"[a-z0-9_\-]+", text.lower()) if tok)

    def query_operation_context(q: QueryItem) -> Tuple[set, set]:
        raw_tokens = tokenize(q.query_text)
        verb_tokens = set(tok for tok in raw_tokens if tok in SUBCLUSTER_VERB_TO_CANONICAL)
        canonical_ops = set(SUBCLUSTER_VERB_TO_CANONICAL[tok] for tok in verb_tokens)
        canonical_ops.update(tok for tok in raw_tokens if tok in OPERATION_ONTOLOGY)
        return verb_tokens, canonical_ops

    for i, item in enumerate(corpus_base):
        operation = str(item.get("tool_name", str(item.get("sub_cluster", "")))).split(".")[-1]
        display_name = str(item.get("tool_display_name", ""))
        sub_cluster = str(item.get("sub_cluster", ""))
        capability_text = " ".join(str(v) for v in item.get("capabilities", []))
        token_source = " ".join([operation, operation.replace("_", " "), display_name, sub_cluster, capability_text])
        idx_to_tokens[i] = tokenize(token_source)

    cond_scores: Dict[str, Dict[str, List[float]]] = {
        "prefixed_hint": {f"p@{k}": [] for k in TOP_K_VALUES},
        "prefixed_no_hint": {f"p@{k}": [] for k in TOP_K_VALUES},
        "prefixed_no_hint_rerank": {f"p@{k}": [] for k in TOP_K_VALUES},
        "prefixed_domain_filter": {f"p@{k}": [] for k in TOP_K_VALUES},
        "prefixed_hybrid": {f"p@{k}": [] for k in TOP_K_VALUES},
        "raw_no_hint": {f"p@{k}": [] for k in TOP_K_VALUES},
    }
    exact_scores: Dict[str, Dict[str, List[float]]] = {
        "prefixed_hint": {f"exact@{k}": [] for k in TOP_K_VALUES},
        "prefixed_no_hint": {f"exact@{k}": [] for k in TOP_K_VALUES},
        "prefixed_no_hint_rerank": {f"exact@{k}": [] for k in TOP_K_VALUES},
        "prefixed_domain_filter": {f"exact@{k}": [] for k in TOP_K_VALUES},
        "prefixed_hybrid": {f"exact@{k}": [] for k in TOP_K_VALUES},
        "raw_no_hint": {f"exact@{k}": [] for k in TOP_K_VALUES},
    }
    mrr_scores = {k: [] for k in cond_scores}
    exact_mrr_scores = {k: [] for k in cond_scores}
    ndcg_scores = {k: [] for k in cond_scores}
    top10_hit_scores = {k: [] for k in cond_scores}
    usable_top10_scores = {k: [] for k in cond_scores}
    domain_top10_scores = {k: [] for k in cond_scores}
    operation_top10_scores = {k: [] for k in cond_scores}

    def grade(result_idx: int, q: QueryItem) -> int:
        result_item_id = idx_to_item_id.get(result_idx, "")
        if q.target_id and result_item_id == q.target_id:
            return 3
        domain, sub_cluster = id_to_meta[result_idx]
        if domain == q.domain and sub_cluster == q.sub_cluster:
            return 2
        if domain == q.domain:
            return 1
        return 0

    def evaluate_one(cond: str, indices: np.ndarray, q: QueryItem) -> None:
        retrieved = [int(ix) for ix in indices.tolist() if int(ix) in id_to_meta]
        rel_flags = [1 if grade(ix, q) >= 2 else 0 for ix in retrieved]
        retrieved_ids = [idx_to_item_id[ix] for ix in retrieved if ix in idx_to_item_id]
        exact_flags = [1 if q.target_id and rid == q.target_id else 0 for rid in retrieved_ids]
        retrieved_domains = [id_to_meta[ix][0] for ix in retrieved if ix in id_to_meta]
        retrieved_operations = [id_to_meta[ix][1] for ix in retrieved if ix in id_to_meta]

        for k in TOP_K_VALUES:
            topk = rel_flags[:k]
            if len(topk) < k:
                topk = topk + [0] * (k - len(topk))
            cond_scores[cond][f"p@{k}"].append(float(np.mean(topk)) if topk else 0.0)

            topk_exact = exact_flags[:k]
            if len(topk_exact) < k:
                topk_exact = topk_exact + [0] * (k - len(topk_exact))
            exact_scores[cond][f"exact@{k}"].append(float(np.mean(topk_exact)) if topk_exact else 0.0)

        if q.target_id:
            top10_hit_scores[cond].append(1.0 if q.target_id in retrieved_ids[:10] else 0.0)
        else:
            top10_hit_scores[cond].append(0.0)

        usable_top10_scores[cond].append(1.0 if any(grade(ix, q) >= 2 for ix in retrieved[:10]) else 0.0)
        domain_top10_scores[cond].append(1.0 if q.domain in retrieved_domains[:10] else 0.0)
        operation_top10_scores[cond].append(1.0 if q.sub_cluster in retrieved_operations[:10] else 0.0)

        rr = 0.0
        for rank, is_rel in enumerate(rel_flags, start=1):
            if is_rel == 1:
                rr = 1.0 / rank
                break
        mrr_scores[cond].append(rr)

        err = 0.0
        for rank, is_exact in enumerate(exact_flags, start=1):
            if is_exact == 1:
                err = 1.0 / rank
                break
        exact_mrr_scores[cond].append(err)

        gains = [grade(ix, q) for ix in retrieved[:NDCG_K]]
        dcg = 0.0
        for i, g in enumerate(gains, start=1):
            dcg += (2**g - 1) / math.log2(i + 1)

        ideal = sorted(gains, reverse=True)
        idcg = 0.0
        for i, g in enumerate(ideal, start=1):
            idcg += (2**g - 1) / math.log2(i + 1)

        ndcg_scores[cond].append(dcg / idcg if idcg > 0 else 0.0)

    def rerank_prefixed_candidates(candidate_idx: np.ndarray, candidate_sim: np.ndarray, q: QueryItem) -> np.ndarray:
        valid_pairs: List[Tuple[int, float]] = []
        query_tokens = tokenize(q.query_text)
        query_verbs, query_canonical_ops = query_operation_context(q)

        for i, sim in zip(candidate_idx.tolist(), candidate_sim.tolist()):
            idx = int(i)
            if idx not in id_to_meta:
                continue

            domain, sub_cluster = id_to_meta[idx]
            bonus = 0.0
            if domain == q.domain:
                bonus += rerank_domain_bonus

            sub_cluster_tokens = tokenize(sub_cluster.replace("-", " ").replace("_", " "))
            operation_overlap = 0.0
            if sub_cluster_tokens and query_tokens:
                operation_overlap = float(len(sub_cluster_tokens.intersection(query_tokens))) / float(len(sub_cluster_tokens))
            bonus += rerank_operation_token_bonus * operation_overlap

            sub_verb = str(sub_cluster).split("_", 1)[0]
            if sub_verb in query_verbs:
                bonus += RERANK_EXACT_VERB_BONUS

            candidate_canonical = canonical_operation_for_subcluster(sub_cluster)
            if candidate_canonical in query_canonical_ops:
                bonus += RERANK_CANONICAL_OP_BONUS

            cand_tokens = idx_to_tokens.get(idx, set())
            overlap_ratio = 0.0
            if query_tokens:
                overlap_ratio = float(len(query_tokens.intersection(cand_tokens))) / float(len(query_tokens))

            req_tokens = set(str(v).lower() for v in corpus_base[idx].get("required_inputs", []) if str(v).strip())
            req_overlap = 0.0
            if query_tokens and req_tokens:
                req_overlap = float(len(query_tokens.intersection(req_tokens))) / float(len(req_tokens))

            score = float(sim) + bonus + (rerank_token_overlap_weight * overlap_ratio)
            score += RERANK_REQUIRED_INPUT_OVERLAP_BONUS * req_overlap
            valid_pairs.append((idx, score))

        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        return np.asarray([idx for idx, _ in valid_pairs], dtype=np.int64)

    def domain_filtered_topk(query_vec: np.ndarray, q: QueryItem, k: int = 10) -> np.ndarray:
        sub = domain_sub_indexes.get(q.domain)
        if not sub:
            _, idx_all = index_prefixed.search(query_vec.reshape(1, -1), k)
            return idx_all[0]

        local_index = sub["index"]
        global_indices = np.asarray(sub["global_indices"], dtype=np.int64)
        local_k = int(min(k, len(global_indices)))
        _, local_idx = local_index.search(query_vec.reshape(1, -1), local_k)
        mapped = global_indices[local_idx[0]]
        return mapped.astype(np.int64)

    def hybrid_topk(query_vec: np.ndarray, q: QueryItem, k: int = 10) -> np.ndarray:
        sub = domain_sub_indexes.get(q.domain)
        if sub:
            local_index = sub["index"]
            global_indices = np.asarray(sub["global_indices"], dtype=np.int64)
            cand_k = int(min(max(rerank_candidate_k, k), len(global_indices)))
            local_sims, local_idx = local_index.search(query_vec.reshape(1, -1), cand_k)
            candidate_indices = global_indices[local_idx[0]].astype(np.int64)
            vector_scores = local_sims[0]
        else:
            cand_k = int(max(rerank_candidate_k, k))
            vector_scores, candidate_arr = index_prefixed.search(query_vec.reshape(1, -1), cand_k)
            candidate_indices = candidate_arr[0].astype(np.int64)
            vector_scores = vector_scores[0]

        query_tokens = expand_tokens_with_ontology(tokenize_for_intent(q.query_text))
        query_verbs, query_canonical_ops = query_operation_context(q)
        pairs: List[Tuple[int, float]] = []
        for idx, vec_score in zip(candidate_indices.tolist(), vector_scores.tolist()):
            if idx not in id_to_meta:
                continue
            doc_tokens = bm25_stats.doc_tokens[idx]
            bm25 = bm25_score(query_tokens, doc_tokens, bm25_stats)

            domain, sub_cluster = id_to_meta[idx]
            operation_tokens = tokenize_for_intent(sub_cluster.replace("_", " ").replace("-", " "))
            op_overlap = 0.0
            if query_tokens and operation_tokens:
                op_overlap = float(len(set(query_tokens).intersection(set(operation_tokens)))) / float(len(set(operation_tokens)))

            sub_verb = str(sub_cluster).split("_", 1)[0]
            exact_verb_bonus = RERANK_EXACT_VERB_BONUS if sub_verb in query_verbs else 0.0

            candidate_canonical = canonical_operation_for_subcluster(sub_cluster)
            canonical_bonus = RERANK_CANONICAL_OP_BONUS if candidate_canonical in query_canonical_ops else 0.0

            req_tokens = set(str(v).lower() for v in corpus_base[idx].get("required_inputs", []) if str(v).strip())
            req_overlap = 0.0
            if req_tokens:
                req_overlap = float(len(set(query_tokens).intersection(req_tokens))) / float(len(req_tokens))

            score = (hybrid_vector_weight * float(vec_score)) + (hybrid_bm25_weight * float(bm25))
            if domain == q.domain:
                score += rerank_domain_bonus
            score += rerank_operation_token_bonus * op_overlap
            score += exact_verb_bonus
            score += canonical_bonus
            score += RERANK_REQUIRED_INPUT_OVERLAP_BONUS * req_overlap
            pairs.append((idx, score))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return np.asarray([idx for idx, _ in pairs[:k]], dtype=np.int64)

    query_texts_hint = [build_query_text(q, template_name=template_name, with_hint=True) for q in queries]
    query_texts_no_hint = [build_query_text(q, template_name=template_name, with_hint=False) for q in queries]

    emb_hint = embed_texts(model, query_texts_hint)
    emb_no_hint = embed_texts(model, query_texts_no_hint)

    _, idx_hint_all = index_prefixed.search(emb_hint, 10)
    rerank_search_k = int(max(10, min(len(corpus_base), rerank_candidate_k)))
    sim_pref_no_all, idx_pref_no_all = index_prefixed.search(emb_no_hint, rerank_search_k)
    _, idx_raw_no_all = index_raw.search(emb_no_hint, 10)

    for i, q in enumerate(queries):
        evaluate_one("prefixed_hint", idx_hint_all[i], q)
        evaluate_one("prefixed_no_hint", idx_pref_no_all[i], q)
        reranked_idx = rerank_prefixed_candidates(idx_pref_no_all[i], sim_pref_no_all[i], q)
        evaluate_one("prefixed_no_hint_rerank", reranked_idx, q)
        domain_filtered_idx = domain_filtered_topk(emb_no_hint[i], q, k=10)
        evaluate_one("prefixed_domain_filter", domain_filtered_idx, q)
        hybrid_idx = hybrid_topk(emb_no_hint[i], q, k=10)
        evaluate_one("prefixed_hybrid", hybrid_idx, q)
        evaluate_one("raw_no_hint", idx_raw_no_all[i], q)

    out: Dict[str, object] = {}
    for cond in cond_scores:
        out[cond] = {k: float(np.mean(v)) for k, v in cond_scores[cond].items()}
        out[cond].update({k: float(np.mean(v)) for k, v in exact_scores[cond].items()})
        out[cond]["mrr"] = float(np.mean(mrr_scores[cond]))
        out[cond]["exact_mrr"] = float(np.mean(exact_mrr_scores[cond]))
        out[cond]["ndcg@10"] = float(np.mean(ndcg_scores[cond]))
        out[cond]["gold_in_top10_rate"] = float(np.mean(top10_hit_scores[cond]))
        out[cond]["usable_in_top10_rate"] = float(np.mean(usable_top10_scores[cond]))
        out[cond]["domain_in_top10_rate"] = float(np.mean(domain_top10_scores[cond]))
        out[cond]["operation_in_top10_rate"] = float(np.mean(operation_top10_scores[cond]))
    return out


def compute_schema_quality_score(corpus_base: List[Dict[str, object]]) -> float:
    """Estimate schema completeness score for LLM tool selection usability."""
    if not corpus_base:
        return 0.0

    scores: List[float] = []
    for item in corpus_base:
        score = 0.0
        description = str(item.get("raw_description", "")).strip()
        required_inputs = item.get("required_inputs", [])
        capabilities = item.get("capabilities", [])

        if len(description) >= 50:
            score += 0.3
        if isinstance(required_inputs, list) and len(required_inputs) > 0:
            score += 0.4
        if isinstance(capabilities, list) and len(capabilities) > 0:
            score += 0.3

        scores.append(score)

    return float(np.mean(scores))


def evaluate_adversarial_pairs(
    all_prefixed_embeddings: np.ndarray,
    all_raw_embeddings: np.ndarray,
    corpus_all: List[Dict[str, object]],
    pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """Evaluate cosine similarity discrimination for adversarial pairs."""
    if not pairs:
        return {
            "mean_similarity_raw": 0.0,
            "mean_similarity_prefixed": 0.0,
            "discrimination_gain": 0.0,
        }

    id_to_idx = {str(item["id"]): i for i, item in enumerate(corpus_all)}
    sims_pref: List[float] = []
    sims_raw: List[float] = []

    for id_a, id_b in pairs:
        ia = id_to_idx[id_a]
        ib = id_to_idx[id_b]
        sim_pref = float(np.dot(all_prefixed_embeddings[ia], all_prefixed_embeddings[ib]))
        sim_raw = float(np.dot(all_raw_embeddings[ia], all_raw_embeddings[ib]))
        sims_pref.append(sim_pref)
        sims_raw.append(sim_raw)

    mean_pref = float(np.mean(sims_pref))
    mean_raw = float(np.mean(sims_raw))
    return {
        "mean_similarity_raw": mean_raw,
        "mean_similarity_prefixed": mean_pref,
        "discrimination_gain": mean_raw - mean_pref,
    }


def evaluate_multi_domain_placement(
    model: UltraBERTEmbedder,
    corpus_base: List[Dict[str, object]],
    multi_items: List[Dict[str, object]],
    base_embeddings: np.ndarray,
    index_base: faiss.IndexFlatIP,
    template_name: str,
) -> Dict[str, object]:
    """Evaluate multi-domain neighbour representation and centroid separation."""
    if not multi_items:
        return {
            "mean_domain_rep_ratio": [0.0, 0.0],
            "mean_centroid_separation_margin": 0.0,
            "per_item": [],
        }

    base_domains = [str(item["domain"]) for item in corpus_base]
    domain_to_idx: Dict[str, List[int]] = {}
    for i, dom in enumerate(base_domains):
        domain_to_idx.setdefault(dom, []).append(i)

    centroids: Dict[str, np.ndarray] = {}
    for dom, idxs in domain_to_idx.items():
        c = np.mean(base_embeddings[idxs], axis=0)
        c = c / np.linalg.norm(c)
        centroids[dom] = c.astype(np.float32)

    domain_names = sorted(list(set(base_domains)))
    per_item: List[Dict[str, float]] = []
    ratio_a: List[float] = []
    ratio_b: List[float] = []
    margins: List[float] = []

    multi_emb = embed_texts(model, [build_embedding_text(item, template_name=template_name) for item in multi_items])

    for item, emb in zip(multi_items, multi_emb):
        domains = item["domains"]
        da = str(domains[0])
        db = str(domains[1])

        _, idx = index_base.search(emb.reshape(1, -1), 10)
        neigh_domains = [base_domains[i] for i in idx[0]]

        ra = neigh_domains.count(da) / 10.0
        rb = neigh_domains.count(db) / 10.0
        ratio_a.append(ra)
        ratio_b.append(rb)

        unrelated_candidates = [d for d in domain_names if d not in {da, db}]
        if not unrelated_candidates or da not in centroids or db not in centroids:
            continue
        unrelated = unrelated_candidates[0]
        va = float(np.dot(emb, centroids[da]))
        vb = float(np.dot(emb, centroids[db]))
        vc = float(np.dot(emb, centroids[unrelated]))
        margin = (va + vb) - vc
        margins.append(margin)

        per_item.append(
            {
                "id": str(item["id"]),
                "domain_a": da,
                "domain_b": db,
                "ratio_a": ra,
                "ratio_b": rb,
                "separation_margin": margin,
            }
        )

    if not per_item:
        return {
            "mean_domain_rep_ratio": [0.0, 0.0],
            "mean_centroid_separation_margin": 0.0,
            "per_item": [],
        }

    return {
        "mean_domain_rep_ratio": [float(np.mean(ratio_a)), float(np.mean(ratio_b))],
        "mean_centroid_separation_margin": float(np.mean(margins)),
        "per_item": per_item,
    }


def evaluate_ivf_recall(
    base_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    flat_index: faiss.IndexFlatIP,
) -> Dict[str, float]:
    """Compute IVF recall@10 against exact top-10 from flat index with adaptive params."""
    num_points = int(base_embeddings.shape[0])
    if num_points < IVF_MIN_POINTS_PER_CENTROID:
        return {
            "recall@10": 1.0,
            "nlist": 1.0,
            "nprobe": 1.0,
            "train_points": float(num_points),
            "min_points_per_centroid": float(IVF_MIN_POINTS_PER_CENTROID),
        }

    max_safe_nlist = max(1, num_points // IVF_MIN_POINTS_PER_CENTROID)
    effective_nlist = int(max(1, min(IVF_NLIST, max_safe_nlist)))

    quantizer = faiss.IndexFlatIP(base_embeddings.shape[1])
    ivf = faiss.IndexIVFFlat(quantizer, base_embeddings.shape[1], effective_nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(base_embeddings)
    ivf.add(base_embeddings)

    effective_nprobe = int(max(1, min(IVF_NPROBE, effective_nlist)))
    ivf.nprobe = effective_nprobe

    _, gt_idx = flat_index.search(query_embeddings, 10)
    _, ivf_idx = ivf.search(query_embeddings, 10)

    recalls: List[float] = []
    for i in range(query_embeddings.shape[0]):
        gt = set(gt_idx[i].tolist())
        apx = set(ivf_idx[i].tolist())
        recalls.append(len(gt.intersection(apx)) / 10.0)

    return {
        "recall@10": float(np.mean(recalls)),
        "nlist": float(effective_nlist),
        "nprobe": float(effective_nprobe),
        "train_points": float(num_points),
        "min_points_per_centroid": float(IVF_MIN_POINTS_PER_CENTROID),
    }


def create_umap_plot(
    prefixed_embeddings: np.ndarray,
    raw_embeddings: np.ndarray,
    corpus_all: List[Dict[str, object]],
    output_path: Path,
) -> None:
    """Create side-by-side UMAP plots with domain coloring and special markers."""
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=RANDOM_SEED)
    proj_pref = reducer.fit_transform(prefixed_embeddings)
    proj_raw = reducer.fit_transform(raw_embeddings)

    records = []
    for i, item in enumerate(corpus_all):
        domain = str(item["domain"]).split(" ")[0]
        records.append(
            {
                "x_pref": proj_pref[i, 0],
                "y_pref": proj_pref[i, 1],
                "x_raw": proj_raw[i, 0],
                "y_raw": proj_raw[i, 1],
                "domain": domain,
                "multi_domain": bool(item.get("multi_domain", False)),
                "adversarial": bool(item.get("adversarial", False)),
            }
        )

    df = pd.DataFrame(records)
    domain_order = sorted(df["domain"].unique())
    palette = sns.color_palette("tab10", n_colors=max(1, len(domain_order)))
    color_map = {d: palette[i % len(palette)] for i, d in enumerate(domain_order)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ax, x_col, y_col, title in [
        (axes[0], "x_pref", "y_pref", "Prefixed embeddings"),
        (axes[1], "x_raw", "y_raw", "Raw embeddings"),
    ]:
        for domain in domain_order:
            sub = df[(df["domain"] == domain) & (~df["multi_domain"])]
            ax.scatter(sub[x_col], sub[y_col], s=12, alpha=0.65, c=[color_map[domain]], label=domain)

        adv = df[df["adversarial"]]
        ax.scatter(adv[x_col], adv[y_col], s=45, marker="x", c="black", linewidths=1.0)

        multi = df[df["multi_domain"]]
        ax.scatter(multi[x_col], multi[y_col], s=85, marker="*", c="gold", edgecolors="black", linewidths=0.6)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0].legend(uniq.values(), uniq.keys(), loc="best", fontsize=8, frameon=True)
    fig.suptitle("UMAP: Prefixed vs Raw Embedding Geometry", fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def verdict_from_thresholds(
    silhouette_prefixed: float,
    p5_no_hint: float,
    ivf_recall: float,
    discrimination_gain: float,
) -> Tuple[str, int]:
    """Derive verdict class from required thresholds."""
    misses = 0
    if silhouette_prefixed <= 0.35:
        misses += 1
    if p5_no_hint <= 0.75:
        misses += 1
    if ivf_recall <= 0.92:
        misses += 1
    if discrimination_gain <= 0.10:
        misses += 1

    if misses == 0:
        return "STRONG", misses
    if misses == 1:
        return "MARGINAL", misses
    return "POOR", misses


def render_report(
    model_info: ModelInfo,
    results: Dict[str, object],
    output_path: Path,
) -> None:
    """Render plain-text summary report in the requested table layout."""
    selected_template = str(results["selected_template"])
    cq = results["cluster_quality"]
    rp = results["retrieval"]
    adv = results["adversarial_discrimination"]
    mdp = results["multi_domain_placement"]
    ivf_recall = results["ivf_simulation"]["recall@10"]
    ivf_nlist = results["ivf_simulation"]["nlist"]
    ivf_nprobe = results["ivf_simulation"]["nprobe"]

    verdict, _ = verdict_from_thresholds(
        silhouette_prefixed=float(cq["prefixed"]["domain"]["silhouette"]),
        p5_no_hint=float(rp["prefixed_no_hint"]["p@5"]),
        ivf_recall=float(ivf_recall),
        discrimination_gain=float(adv["discrimination_gain"]),
    )

    sil_d = cq["prefixed"]["domain"]["silhouette"] - cq["raw"]["domain"]["silhouette"]
    db_d = cq["prefixed"]["domain"]["davies_bouldin"] - cq["raw"]["domain"]["davies_bouldin"]
    ch_d = cq["prefixed"]["domain"]["calinski_harabasz"] - cq["raw"]["domain"]["calinski_harabasz"]

    ratio_a, ratio_b = mdp["mean_domain_rep_ratio"]

    ablation = results["template_ablation"]
    input_style_ablation = results["input_style_ablation"]
    llm_dashboard = results["llm_selection_dashboard"]
    mapping_qa = results["operation_mapping_qa"]
    fusion_cfg = results.get("fusion_config", {})
    ablation_lines = []
    for name in EMBEDDING_TEMPLATES:
        row = ablation[name]
        ablation_lines.append(
            f"- {name}: p@5(no-hint)={row['retrieval']['prefixed_no_hint']['p@5']:.3f}, "
            f"mrr(no-hint)={row['retrieval']['prefixed_no_hint']['mrr']:.3f}, "
            f"silhouette={row['cluster_domain']['silhouette']:.3f}, ivf@10={row['ivf_recall']:.3f}"
        )

    input_style_lines = []
    for style_name, vals in input_style_ablation.items():
        input_style_lines.append(
            f"- {style_name}: p@5={vals['p@5']:.3f}, mrr={vals['mrr']:.3f}, "
            f"tpr@10={vals['tpr@10']:.3f}, dmr@10={vals['dmr@10']:.3f}, omr@10={vals['omr@10']:.3f}"
        )

    best_style = max(
        input_style_ablation.keys(),
        key=lambda n: (
            float(input_style_ablation[n]["tpr@10"]),
            float(input_style_ablation[n]["omr@10"]),
            float(input_style_ablation[n]["p@5"]),
            float(input_style_ablation[n]["mrr"]),
        ),
    )

    top_verbs_pairs = list(mapping_qa.get("top_verbs", {}).items())[:8]
    top_verbs_text = ", ".join([f"{k}:{v}" for k, v in top_verbs_pairs]) if top_verbs_pairs else "none"
    unmapped_preview = mapping_qa.get("unmapped_subclusters", [])[:10]
    unmapped_text = ", ".join(unmapped_preview) if unmapped_preview else "none"

    report = f"""
ULTRABERT EMBEDDING EVALUATION REPORT

MODEL: {model_info.model_name}
DIMENSION: {model_info.dimension}
CORPUS SIZE: {results['dataset']['total_tools']} tools ({results['dataset']['multi_domain_tools']} multi-domain flagged)
SELECTED TEMPLATE: {selected_template} ({EMBEDDING_TEMPLATES[selected_template]})

TEMPLATE ABLATION
{chr(10).join(ablation_lines)}

INPUT STYLE ABLATION (which sentence format works best)
{chr(10).join(input_style_lines)}
Best input style for UltraBERT embeddings: {best_style} ({INPUT_STYLE_VARIANTS[best_style]})

CLUSTER QUALITY
Silhouette: prefixed={cq['prefixed']['domain']['silhouette']:.3f}, raw={cq['raw']['domain']['silhouette']:.3f}, delta={sil_d:+.3f}
Davies-Bouldin: prefixed={cq['prefixed']['domain']['davies_bouldin']:.3f}, raw={cq['raw']['domain']['davies_bouldin']:.3f}, delta={db_d:+.3f}
Calinski-Harabasz: prefixed={cq['prefixed']['domain']['calinski_harabasz']:.2f}, raw={cq['raw']['domain']['calinski_harabasz']:.2f}, delta={ch_d:+.2f}

RETRIEVAL PRECISION
P@1: prefixed_hint={rp['prefixed_hint']['p@1']:.2f}, prefixed_no_hint={rp['prefixed_no_hint']['p@1']:.2f}, raw_no_hint={rp['raw_no_hint']['p@1']:.2f}
P@3: prefixed_hint={rp['prefixed_hint']['p@3']:.2f}, prefixed_no_hint={rp['prefixed_no_hint']['p@3']:.2f}, raw_no_hint={rp['raw_no_hint']['p@3']:.2f}
P@5: prefixed_hint={rp['prefixed_hint']['p@5']:.2f}, prefixed_no_hint={rp['prefixed_no_hint']['p@5']:.2f}, raw_no_hint={rp['raw_no_hint']['p@5']:.2f}
P@10: prefixed_hint={rp['prefixed_hint']['p@10']:.2f}, prefixed_no_hint={rp['prefixed_no_hint']['p@10']:.2f}, raw_no_hint={rp['raw_no_hint']['p@10']:.2f}
MRR: prefixed_hint={rp['prefixed_hint']['mrr']:.3f}, prefixed_no_hint={rp['prefixed_no_hint']['mrr']:.3f}, raw_no_hint={rp['raw_no_hint']['mrr']:.3f}
NDCG@10: prefixed_hint={rp['prefixed_hint']['ndcg@10']:.3f}, prefixed_no_hint={rp['prefixed_no_hint']['ndcg@10']:.3f}, raw_no_hint={rp['raw_no_hint']['ndcg@10']:.3f}
Exact@10: prefixed_hint={rp['prefixed_hint']['exact@10']:.3f}, prefixed_no_hint={rp['prefixed_no_hint']['exact@10']:.3f}, raw_no_hint={rp['raw_no_hint']['exact@10']:.3f}
Exact MRR: prefixed_hint={rp['prefixed_hint']['exact_mrr']:.3f}, prefixed_no_hint={rp['prefixed_no_hint']['exact_mrr']:.3f}, raw_no_hint={rp['raw_no_hint']['exact_mrr']:.3f}
Gold in Top-10: prefixed_hint={rp['prefixed_hint']['gold_in_top10_rate']:.3f}, prefixed_no_hint={rp['prefixed_no_hint']['gold_in_top10_rate']:.3f}, raw_no_hint={rp['raw_no_hint']['gold_in_top10_rate']:.3f}
Usable in Top-10: prefixed_hint={rp['prefixed_hint']['usable_in_top10_rate']:.3f}, prefixed_no_hint={rp['prefixed_no_hint']['usable_in_top10_rate']:.3f}, raw_no_hint={rp['raw_no_hint']['usable_in_top10_rate']:.3f}
Domain in Top-10: prefixed_hint={rp['prefixed_hint']['domain_in_top10_rate']:.3f}, prefixed_no_hint={rp['prefixed_no_hint']['domain_in_top10_rate']:.3f}, raw_no_hint={rp['raw_no_hint']['domain_in_top10_rate']:.3f}
Operation in Top-10: prefixed_hint={rp['prefixed_hint']['operation_in_top10_rate']:.3f}, prefixed_no_hint={rp['prefixed_no_hint']['operation_in_top10_rate']:.3f}, raw_no_hint={rp['raw_no_hint']['operation_in_top10_rate']:.3f}

RERANKED RETRIEVAL (domain + operation + token overlap)
P@5: reranked={rp['prefixed_no_hint_rerank']['p@5']:.3f}, baseline={rp['prefixed_no_hint']['p@5']:.3f}
MRR: reranked={rp['prefixed_no_hint_rerank']['mrr']:.3f}, baseline={rp['prefixed_no_hint']['mrr']:.3f}
Tool Present Rate (exact in top-10): reranked={rp['prefixed_no_hint_rerank']['gold_in_top10_rate']:.3f}, baseline={rp['prefixed_no_hint']['gold_in_top10_rate']:.3f}
Domain in Top-10: reranked={rp['prefixed_no_hint_rerank']['domain_in_top10_rate']:.3f}, baseline={rp['prefixed_no_hint']['domain_in_top10_rate']:.3f}
Operation in Top-10: reranked={rp['prefixed_no_hint_rerank']['operation_in_top10_rate']:.3f}, baseline={rp['prefixed_no_hint']['operation_in_top10_rate']:.3f}

DOMAIN-FILTERED RETRIEVAL (use caller domain as hard filter)
P@5: domain_filter={rp['prefixed_domain_filter']['p@5']:.3f}, baseline={rp['prefixed_no_hint']['p@5']:.3f}
MRR: domain_filter={rp['prefixed_domain_filter']['mrr']:.3f}, baseline={rp['prefixed_no_hint']['mrr']:.3f}
Tool Present Rate (exact in top-10): domain_filter={rp['prefixed_domain_filter']['gold_in_top10_rate']:.3f}, baseline={rp['prefixed_no_hint']['gold_in_top10_rate']:.3f}
Domain in Top-10: domain_filter={rp['prefixed_domain_filter']['domain_in_top10_rate']:.3f}, baseline={rp['prefixed_no_hint']['domain_in_top10_rate']:.3f}
Operation in Top-10: domain_filter={rp['prefixed_domain_filter']['operation_in_top10_rate']:.3f}, baseline={rp['prefixed_no_hint']['operation_in_top10_rate']:.3f}

HYBRID RETRIEVAL (domain sub-index + vector/BM25 fusion + ontology)
P@5: hybrid={rp['prefixed_hybrid']['p@5']:.3f}, baseline={rp['prefixed_no_hint']['p@5']:.3f}
MRR: hybrid={rp['prefixed_hybrid']['mrr']:.3f}, baseline={rp['prefixed_no_hint']['mrr']:.3f}
Tool Present Rate (exact in top-10): hybrid={rp['prefixed_hybrid']['gold_in_top10_rate']:.3f}, baseline={rp['prefixed_no_hint']['gold_in_top10_rate']:.3f}
Domain in Top-10: hybrid={rp['prefixed_hybrid']['domain_in_top10_rate']:.3f}, baseline={rp['prefixed_no_hint']['domain_in_top10_rate']:.3f}
Operation in Top-10: hybrid={rp['prefixed_hybrid']['operation_in_top10_rate']:.3f}, baseline={rp['prefixed_no_hint']['operation_in_top10_rate']:.3f}

FUSION CONFIG (active run)
hybrid_vector_weight={float(fusion_cfg.get('hybrid_vector_weight', HYBRID_VECTOR_WEIGHT)):.3f}
hybrid_bm25_weight={float(fusion_cfg.get('hybrid_bm25_weight', HYBRID_BM25_WEIGHT)):.3f}
rerank_domain_bonus={float(fusion_cfg.get('rerank_domain_bonus', RERANK_DOMAIN_BONUS)):.3f}
rerank_operation_token_bonus={float(fusion_cfg.get('rerank_operation_token_bonus', RERANK_OPERATION_TOKEN_BONUS)):.3f}
rerank_token_overlap_weight={float(fusion_cfg.get('rerank_token_overlap_weight', RERANK_TOKEN_OVERLAP_WEIGHT)):.3f}
rerank_candidate_k={int(fusion_cfg.get('rerank_candidate_k', RERANK_CANDIDATE_K))}

LLM SELECTION DASHBOARD (prefixed_no_hint)
Tool Present Rate (TPR, exact in top-10): baseline={llm_dashboard['baseline']['tool_present_rate_top10']:.3f}, reranked={llm_dashboard['reranked']['tool_present_rate_top10']:.3f}, delta={llm_dashboard['delta']['tool_present_rate_top10']:+.3f}
Domain Match Rate (DMR, domain in top-10): baseline={llm_dashboard['baseline']['domain_match_rate_top10']:.3f}, reranked={llm_dashboard['reranked']['domain_match_rate_top10']:.3f}, delta={llm_dashboard['delta']['domain_match_rate_top10']:+.3f}
Operation Match Rate (OMR, operation in top-10): baseline={llm_dashboard['baseline']['operation_match_rate_top10']:.3f}, reranked={llm_dashboard['reranked']['operation_match_rate_top10']:.3f}, delta={llm_dashboard['delta']['operation_match_rate_top10']:+.3f}
Tool Present Rate with domain filter: {llm_dashboard['domain_filter']['tool_present_rate_top10']:.3f}
Domain Match Rate with domain filter: {llm_dashboard['domain_filter']['domain_match_rate_top10']:.3f}
Operation Match Rate with domain filter: {llm_dashboard['domain_filter']['operation_match_rate_top10']:.3f}
Tool Present Rate with hybrid: {llm_dashboard['hybrid']['tool_present_rate_top10']:.3f}
Domain Match Rate with hybrid: {llm_dashboard['hybrid']['domain_match_rate_top10']:.3f}
Operation Match Rate with hybrid: {llm_dashboard['hybrid']['operation_match_rate_top10']:.3f}
Schema Quality Score (SQS): {llm_dashboard['schema_quality_score']:.3f}

STEP 2 ONTOLOGY MAPPING QA
Sub-cluster mapping coverage: {mapping_qa['mapped_subclusters']}/{mapping_qa['total_subclusters']} ({mapping_qa['subcluster_coverage']:.3f})
Query token ontology coverage: {mapping_qa['query_token_mapped']}/{mapping_qa['query_token_total']} ({mapping_qa['query_token_coverage']:.3f})
Top operation verbs: {top_verbs_text}
Unmapped sub-clusters (preview): {unmapped_text}

ADVERSARIAL DISCRIMINATION
Mean similarity raw={adv['mean_similarity_raw']:.3f}
Mean similarity prefixed={adv['mean_similarity_prefixed']:.3f}
Discrimination gain={adv['discrimination_gain']:+.3f}

MULTI-DOMAIN PLACEMENT
Mean domain representation ratio={ratio_a:.2f} / {ratio_b:.2f}
Mean centroid separation margin={mdp['mean_centroid_separation_margin']:+.3f}

IVF SIMULATION
IVF recall@10 (nlist={ivf_nlist}, nprobe={ivf_nprobe})={ivf_recall:.3f}

VERDICT
{verdict}
""".strip("\n")

    output_path.write_text(report, encoding="utf-8")


def render_operation_ontology_markdown(output_path: Path) -> None:
    """Write Step 2 ontology reference markdown artifact."""
    lines: List[str] = []
    lines.append("# Operation Ontology v1")
    lines.append("")
    lines.append("This document defines canonical operation families and accepted aliases used for retrieval and reranking.")
    lines.append("")
    lines.append("## Canonical families")
    lines.append("")
    for canonical in sorted(OPERATION_ONTOLOGY.keys()):
        aliases = ", ".join(sorted(set(OPERATION_ONTOLOGY[canonical])))
        lines.append(f"- **{canonical}**: {aliases}")

    lines.append("")
    lines.append("## Verb to canonical mapping")
    lines.append("")
    for verb in sorted(SUBCLUSTER_VERB_TO_CANONICAL.keys()):
        lines.append(f"- `{verb}` -> `{SUBCLUSTER_VERB_TO_CANONICAL[verb]}`")

    lines.append("")
    lines.append("## Explicit sub-cluster overrides")
    lines.append("")
    for sub_cluster in sorted(CANONICAL_OPERATION_BY_SUBCLUSTER.keys()):
        lines.append(f"- `{sub_cluster}` -> `{CANONICAL_OPERATION_BY_SUBCLUSTER[sub_cluster]}`")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def render_operation_mapping_test_report(
    mapping_qa: Dict[str, object],
    llm_dashboard: Dict[str, object],
    output_path: Path,
) -> None:
    """Write Step 2 mapping QA markdown with pass/fail style checks."""
    subcluster_cov = float(mapping_qa.get("subcluster_coverage", 0.0))
    query_cov = float(mapping_qa.get("query_token_coverage", 0.0))
    tpr_hybrid = float(llm_dashboard.get("hybrid", {}).get("tool_present_rate_top10", 0.0))
    omr_hybrid = float(llm_dashboard.get("hybrid", {}).get("operation_match_rate_top10", 0.0))

    gate_sub = "PASS" if subcluster_cov >= 0.95 else "FAIL"
    gate_query = "PASS" if query_cov >= 0.90 else "FAIL"

    lines: List[str] = []
    lines.append("# Operation Mapping Test Report")
    lines.append("")
    lines.append("## Coverage summary")
    lines.append("")
    lines.append(
        f"- Sub-cluster coverage: {int(mapping_qa.get('mapped_subclusters', 0))}/{int(mapping_qa.get('total_subclusters', 0))} ({subcluster_cov:.3f}) [{gate_sub}]"
    )
    lines.append(
        f"- Query token coverage: {int(mapping_qa.get('query_token_mapped', 0))}/{int(mapping_qa.get('query_token_total', 0))} ({query_cov:.3f}) [{gate_query}]"
    )
    lines.append(f"- Hybrid TPR@10: {tpr_hybrid:.3f}")
    lines.append(f"- Hybrid OMR@10: {omr_hybrid:.3f}")

    lines.append("")
    lines.append("## Top verbs observed")
    lines.append("")
    top_verbs = mapping_qa.get("top_verbs", {})
    if isinstance(top_verbs, dict) and top_verbs:
        for verb, count in top_verbs.items():
            lines.append(f"- `{verb}`: {int(count)}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Unmapped sub-clusters")
    lines.append("")
    unmapped = mapping_qa.get("unmapped_subclusters", [])
    if isinstance(unmapped, list) and unmapped:
        for sub in unmapped:
            lines.append(f"- `{sub}`")
    else:
        lines.append("- none")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_model_evaluation(
    model: UltraBERTEmbedder,
    model_info: ModelInfo,
    config: RunConfig,
    output_dir: Path,
    corpus_base: List[Dict[str, object]],
    multi_domain_items: List[Dict[str, object]],
    corpus_all: List[Dict[str, object]],
    queries: List[QueryItem],
    adversarial_pairs: List[Tuple[str, str]],
    loaded_sources: List[str],
    artifact_suffix: str,
    report_filename: str,
) -> Dict[str, object]:
    """Run full evaluation for a single model and persist artifacts."""
    corpus_path = output_dir / f"corpus{artifact_suffix}.json"
    embeddings_path = output_dir / f"embeddings{artifact_suffix}.npy"
    raw_embeddings_path = output_dir / f"raw_embeddings{artifact_suffix}.npy"
    eval_results_path = output_dir / f"eval_results{artifact_suffix}.json"
    umap_path = output_dir / f"umap_prefixed_vs_raw{artifact_suffix}.png"
    report_path = output_dir / report_filename
    ontology_md_path = output_dir / "operation_ontology_v1.md"
    mapping_report_md_path = output_dir / "operation_mapping_test_report.md"

    raw_texts_all = [str(item["raw_description"]) for item in corpus_all]
    raw_embeddings_all = embed_texts(model, raw_texts_all)
    np.save(raw_embeddings_path, raw_embeddings_all.astype(np.float32))

    with corpus_path.open("w", encoding="utf-8") as f:
        json.dump(corpus_all, f, indent=2)

    base_count = len(corpus_base)
    raw_embeddings_base = raw_embeddings_all[:base_count]
    index_raw_base = build_flat_index(raw_embeddings_base)

    domain_labels = [str(item["domain"]) for item in corpus_base]
    sub_labels = [f"{item['domain']}::{item['sub_cluster']}" for item in corpus_base]

    domain_encoder = {d: i for i, d in enumerate(sorted(set(domain_labels)))}
    sub_encoder = {s: i for i, s in enumerate(sorted(set(sub_labels)))}

    y_domain = np.array([domain_encoder[d] for d in domain_labels], dtype=np.int32)
    y_sub = np.array([sub_encoder[s] for s in sub_labels], dtype=np.int32)

    raw_domain_scores = safe_cluster_scores(raw_embeddings_base, y_domain)
    raw_sub_scores = safe_cluster_scores(raw_embeddings_base, y_sub)
    raw_dist = compute_domain_distance_stats(raw_embeddings_base, domain_labels)

    template_embeddings_all: Dict[str, np.ndarray] = {}
    template_results: Dict[str, Dict[str, object]] = {}

    for template_name in EMBEDDING_TEMPLATES:
        templ_texts_all = [build_embedding_text(item, template_name=template_name) for item in corpus_all]
        templ_embeddings_all = embed_texts(model, templ_texts_all)
        template_embeddings_all[template_name] = templ_embeddings_all

        templ_embeddings_base = templ_embeddings_all[:base_count]
        templ_index_base = build_flat_index(templ_embeddings_base)
        templ_domain_sub_indexes = build_domain_sub_indexes(templ_embeddings_base, corpus_base)
        bm25_stats = build_bm25_stats(corpus_base)

        templ_domain_scores = safe_cluster_scores(templ_embeddings_base, y_domain)
        templ_sub_scores = safe_cluster_scores(templ_embeddings_base, y_sub)
        templ_dist = compute_domain_distance_stats(templ_embeddings_base, domain_labels)

        templ_retrieval = compute_retrieval_metrics(
            index_prefixed=templ_index_base,
            index_raw=index_raw_base,
            prefixed_embeddings_base=templ_embeddings_base,
            model=model,
            corpus_base=corpus_base,
            queries=queries,
            template_name=template_name,
            domain_sub_indexes=templ_domain_sub_indexes,
            bm25_stats=bm25_stats,
            hybrid_vector_weight=config.hybrid_vector_weight,
            hybrid_bm25_weight=config.hybrid_bm25_weight,
            rerank_domain_bonus=config.rerank_domain_bonus,
            rerank_operation_token_bonus=config.rerank_operation_token_bonus,
            rerank_token_overlap_weight=config.rerank_token_overlap_weight,
            rerank_candidate_k=config.rerank_candidate_k,
        )

        templ_adv = evaluate_adversarial_pairs(
            all_prefixed_embeddings=templ_embeddings_all,
            all_raw_embeddings=raw_embeddings_all,
            corpus_all=corpus_all,
            pairs=adversarial_pairs,
        )

        templ_multi = evaluate_multi_domain_placement(
            model=model,
            corpus_base=corpus_base,
            multi_items=multi_domain_items,
            base_embeddings=templ_embeddings_base,
            index_base=templ_index_base,
            template_name=template_name,
        )

        query_emb_for_ivf = embed_texts(
            model,
            [build_query_text(q, template_name=template_name, with_hint=False) for q in queries],
        )
        templ_ivf = evaluate_ivf_recall(
            base_embeddings=templ_embeddings_base,
            query_embeddings=query_emb_for_ivf,
            flat_index=templ_index_base,
        )

        template_results[template_name] = {
            "cluster_domain": {**templ_domain_scores, **templ_dist},
            "cluster_sub_cluster": templ_sub_scores,
            "retrieval": templ_retrieval,
            "adversarial_discrimination": templ_adv,
            "multi_domain_placement": templ_multi,
            "ivf": templ_ivf,
            "ivf_recall": float(templ_ivf["recall@10"]),
        }

    ranked_templates = sorted(
        EMBEDDING_TEMPLATES.keys(),
        key=lambda name: (
            float(template_results[name]["retrieval"]["prefixed_hybrid"]["gold_in_top10_rate"]),
            float(template_results[name]["retrieval"]["prefixed_hybrid"]["p@5"]),
            float(template_results[name]["retrieval"]["prefixed_no_hint"]["p@5"]),
            float(template_results[name]["cluster_domain"]["silhouette"]),
            float(template_results[name]["ivf_recall"]),
        ),
        reverse=True,
    )
    selected_template = ranked_templates[0]

    prefixed_embeddings_all = template_embeddings_all[selected_template]
    prefixed_embeddings_base = prefixed_embeddings_all[:base_count]
    np.save(embeddings_path, prefixed_embeddings_all.astype(np.float32))

    pref_domain_scores = safe_cluster_scores(prefixed_embeddings_base, y_domain)
    pref_sub_scores = safe_cluster_scores(prefixed_embeddings_base, y_sub)
    pref_dist = compute_domain_distance_stats(prefixed_embeddings_base, domain_labels)

    retrieval = template_results[selected_template]["retrieval"]
    input_style_ablation = evaluate_input_style_variants(
        model=model,
        corpus_base=corpus_base,
        queries=queries,
    )
    operation_mapping_qa = evaluate_operation_mapping_qa(
        corpus_base=corpus_base,
        queries=queries,
    )
    adv = template_results[selected_template]["adversarial_discrimination"]
    multi = template_results[selected_template]["multi_domain_placement"]
    ivf_selected = dict(template_results[selected_template]["ivf"])
    ivf_recall = float(ivf_selected["recall@10"])

    create_umap_plot(
        prefixed_embeddings=prefixed_embeddings_all,
        raw_embeddings=raw_embeddings_all,
        corpus_all=corpus_all,
        output_path=umap_path,
    )

    results: Dict[str, object] = {
        "model": {
            "name": model_info.model_name,
            "dimension": model_info.dimension,
            "fallback_used": model_info.fallback_used,
        },
        "selected_template": selected_template,
        "template_descriptions": EMBEDDING_TEMPLATES,
        "template_ablation": template_results,
        "dataset": {
            "mode": config.corpus_mode,
            "single_domain_tools": len(corpus_base),
            "multi_domain_tools": len(multi_domain_items),
            "total_tools": len(corpus_all),
            "domains": sorted(list(set(str(item["domain"]) for item in corpus_base))),
            "adversarial_pair_count": len(adversarial_pairs),
            "batch_index": config.batch_index,
            "batch_size": config.batch_size,
            "query_style": config.query_style,
            "sources_loaded": loaded_sources,
        },
        "fusion_config": {
            "hybrid_vector_weight": config.hybrid_vector_weight,
            "hybrid_bm25_weight": config.hybrid_bm25_weight,
            "rerank_domain_bonus": config.rerank_domain_bonus,
            "rerank_operation_token_bonus": config.rerank_operation_token_bonus,
            "rerank_token_overlap_weight": config.rerank_token_overlap_weight,
            "rerank_candidate_k": config.rerank_candidate_k,
        },
        "cluster_quality": {
            "prefixed": {
                "domain": {**pref_domain_scores, **pref_dist},
                "sub_cluster": pref_sub_scores,
            },
            "raw": {
                "domain": {**raw_domain_scores, **raw_dist},
                "sub_cluster": raw_sub_scores,
            },
            "delta_prefixed_minus_raw": {
                "domain": {
                    "silhouette": pref_domain_scores["silhouette"] - raw_domain_scores["silhouette"],
                    "davies_bouldin": pref_domain_scores["davies_bouldin"] - raw_domain_scores["davies_bouldin"],
                    "calinski_harabasz": pref_domain_scores["calinski_harabasz"] - raw_domain_scores["calinski_harabasz"],
                    "mean_intra_distance": pref_dist["mean_intra_distance"] - raw_dist["mean_intra_distance"],
                    "mean_inter_distance": pref_dist["mean_inter_distance"] - raw_dist["mean_inter_distance"],
                },
                "sub_cluster": {
                    "silhouette": pref_sub_scores["silhouette"] - raw_sub_scores["silhouette"],
                    "davies_bouldin": pref_sub_scores["davies_bouldin"] - raw_sub_scores["davies_bouldin"],
                    "calinski_harabasz": pref_sub_scores["calinski_harabasz"] - raw_sub_scores["calinski_harabasz"],
                },
            },
        },
        "retrieval": retrieval,
        "operation_mapping_qa": operation_mapping_qa,
        "input_style_ablation": input_style_ablation,
        "llm_selection_dashboard": {
            "baseline": {
                "tool_present_rate_top10": float(retrieval["prefixed_no_hint"]["gold_in_top10_rate"]),
                "domain_match_rate_top10": float(retrieval["prefixed_no_hint"]["domain_in_top10_rate"]),
                "operation_match_rate_top10": float(retrieval["prefixed_no_hint"]["operation_in_top10_rate"]),
            },
            "reranked": {
                "tool_present_rate_top10": float(retrieval["prefixed_no_hint_rerank"]["gold_in_top10_rate"]),
                "domain_match_rate_top10": float(retrieval["prefixed_no_hint_rerank"]["domain_in_top10_rate"]),
                "operation_match_rate_top10": float(retrieval["prefixed_no_hint_rerank"]["operation_in_top10_rate"]),
            },
            "domain_filter": {
                "tool_present_rate_top10": float(retrieval["prefixed_domain_filter"]["gold_in_top10_rate"]),
                "domain_match_rate_top10": float(retrieval["prefixed_domain_filter"]["domain_in_top10_rate"]),
                "operation_match_rate_top10": float(retrieval["prefixed_domain_filter"]["operation_in_top10_rate"]),
            },
            "hybrid": {
                "tool_present_rate_top10": float(retrieval["prefixed_hybrid"]["gold_in_top10_rate"]),
                "domain_match_rate_top10": float(retrieval["prefixed_hybrid"]["domain_in_top10_rate"]),
                "operation_match_rate_top10": float(retrieval["prefixed_hybrid"]["operation_in_top10_rate"]),
            },
            "delta": {
                "tool_present_rate_top10": float(
                    retrieval["prefixed_no_hint_rerank"]["gold_in_top10_rate"]
                    - retrieval["prefixed_no_hint"]["gold_in_top10_rate"]
                ),
                "domain_match_rate_top10": float(
                    retrieval["prefixed_no_hint_rerank"]["domain_in_top10_rate"]
                    - retrieval["prefixed_no_hint"]["domain_in_top10_rate"]
                ),
                "operation_match_rate_top10": float(
                    retrieval["prefixed_no_hint_rerank"]["operation_in_top10_rate"]
                    - retrieval["prefixed_no_hint"]["operation_in_top10_rate"]
                ),
            },
            "schema_quality_score": compute_schema_quality_score(corpus_base),
        },
        "adversarial_discrimination": adv,
        "multi_domain_placement": multi,
        "ivf_simulation": {
            "nlist": int(ivf_selected["nlist"]),
            "nprobe": int(ivf_selected["nprobe"]),
            "train_points": int(ivf_selected["train_points"]),
            "min_points_per_centroid": int(ivf_selected["min_points_per_centroid"]),
            "recall@10": ivf_recall,
        },
        "queries": [q.__dict__ for q in queries],
        "adversarial_pairs": [{"a": a, "b": b} for a, b in adversarial_pairs],
    }

    with eval_results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    render_report(model_info=model_info, results=results, output_path=report_path)
    render_operation_ontology_markdown(output_path=ontology_md_path)
    render_operation_mapping_test_report(
        mapping_qa=operation_mapping_qa,
        llm_dashboard=results["llm_selection_dashboard"],
        output_path=mapping_report_md_path,
    )

    print(f"Model: {model_info.model_name}")
    print(f"Dimension: {model_info.dimension}")
    print("Saved deliverables:")
    print(f"- {corpus_path}")
    print(f"- {embeddings_path}")
    print(f"- {raw_embeddings_path}")
    print(f"- {eval_results_path}")
    print(f"- {umap_path}")
    print(f"- {report_path}")
    print(f"- {ontology_md_path}")
    print(f"- {mapping_report_md_path}")

    return results


def main() -> None:
    """Run end-to-end evaluation and persist all deliverables."""
    config = parse_args()
    set_seed(RANDOM_SEED)

    output_dir = Path(__file__).parent
    model, model_info = load_model()

    loaded_sources: List[str] = []
    if config.corpus_mode == "golden":
        corpus_base, multi_domain_items, queries, loaded_sources = load_golden_set(
            golden_set_path=config.golden_set_path,
            query_style=config.query_style,
        )
        adversarial_pairs = build_adversarial_pairs_real(corpus_base)
    else:
        corpus_base, multi_domain_items, loaded_sources = load_real_corpus(config)
        if not corpus_base:
            raise ValueError(f"No real tool contracts loaded from {config.real_contracts_root}")
        adversarial_pairs = build_adversarial_pairs_real(corpus_base)
        queries = generate_queries_from_real_corpus(corpus_base, query_style=config.query_style)
        if not queries:
            raise ValueError("No queries generated from real corpus.")

    corpus_all = corpus_base

    run_model_evaluation(
        model=model,
        model_info=model_info,
        config=config,
        output_dir=output_dir,
        corpus_base=corpus_base,
        multi_domain_items=multi_domain_items,
        corpus_all=corpus_all,
        queries=queries,
        adversarial_pairs=adversarial_pairs,
        loaded_sources=loaded_sources,
        artifact_suffix="",
        report_filename="report.txt",
    )


if __name__ == "__main__":
    main()
