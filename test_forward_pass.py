"""Quick forward pass test for FamilyOS UltraBERT.

Usage:
    python test_forward_pass.py "some text"
"""

from __future__ import annotations

import json
import sys
import time

from familyos_ultrabert import Client


# Canonical set of 12 analysis capabilities (encoder heads).
# Note: Text generation is intentionally excluded from this script.
EXPECTED_CAPABILITIES = [
    "sentiment",
    "emotions",
    "safety_familyos",
    "safety_generic",
    "intent",
    "ingress",
    "ner_family",
    "ner_general",
    "temporal",
    "relation",
    "nli",
    "embedding",
]


DEFAULT_TEXT = (
    "my geometry teacher is so annoying because she said my geo pd is fer favorite pd "
    "cuz she likes to bully me"
)


def _get_text_from_args_or_stdin() -> str:
    """Return input text from argv, or stdin, or a default sample.

    Args:
        None.

    Returns:
        The text to run through the model.
    """
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip() or DEFAULT_TEXT

    # If the user pipes text in, prefer that.
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped

    return DEFAULT_TEXT


def main() -> int:
    # Initialize client
    print("Loading model...")
    start_load = time.perf_counter()
    try:
        # Force CPU to avoid occasional hangs in torch.cuda.is_available() on some Windows setups.
        # Disable warmup for faster interactive debugging; you can re-enable once loading is stable.
        client = Client(
            backend="pytorch",
            device="cpu",
            warmup=False,
            warmup_rounds=0,
            verbose=True,
            load_decoder=False,
        )
    except Exception as exc:  # pragma: no cover
        print("ERROR: Failed to initialize Client.")
        print("If this is a fresh environment, confirm weights and runtime deps are installed.")
        print(f"Error: {type(exc).__name__}: {exc}")
        return 1

    load_ms = (time.perf_counter() - start_load) * 1000
    print(f"Client ready in {load_ms:.0f}ms")

    text = _get_text_from_args_or_stdin()

    # Request all 12 capabilities explicitly.
    available_caps = list(getattr(client, "capabilities", []))
    expected_caps = list(EXPECTED_CAPABILITIES)

    missing_from_client = [c for c in expected_caps if c not in available_caps]
    if missing_from_client:
        print("ERROR: Client does not expose all expected capabilities.")
        print(f"Expected heads ({len(expected_caps)}): {expected_caps}")
        print(f"Client capabilities ({len(available_caps)}): {available_caps}")
        print(f"Missing heads: {missing_from_client}")
        return 2

    requested_caps = expected_caps

    print("=" * 60)
    print("FamilyOS UltraBERT Forward Pass")
    print("=" * 60)
    print(f"Backend: {client.backend}")
    print(f"Available heads: {len(available_caps)}")
    print(f"Requested heads: {len(requested_caps)}")
    print(f"Requested head names: {requested_caps}")
    print(f"Input: {text}")
    print("=" * 60)

    # Run analysis with all requested capabilities
    result = client.analyze(text, capabilities=requested_caps if requested_caps else None)

    print("\n** KEY RESULTS **\n")

    # Print key results
    print(f"Sentiment: {getattr(result, 'sentiment', None)}")
    print(f"Safety Band: {getattr(result, 'safety', None)}")
    print(f"Emotions: {getattr(result, 'emotions', None)}")

    # Detailed per-head results.
    print("\n** ALL HEAD RESULTS (PER CAPABILITY) **\n")

    caps_dict = getattr(result, "_caps", {})
    if not isinstance(caps_dict, dict):
        caps_dict = {}

    returned_caps = list(caps_dict.keys())
    missing = [c for c in requested_caps if c not in caps_dict]
    print(f"Returned heads: {len(returned_caps)}")
    if missing:
        print(f"Missing heads: {missing}")

    def _print_capability(name: str, payload: object) -> None:
        print(f"\n--- {name} ---")
        if name == "embedding" and isinstance(payload, dict):
            emb = payload.get("embedding", [])
            try:
                emb_len = len(emb)
            except Exception:
                emb_len = 0
            preview = emb[:8] if isinstance(emb, list) else []
            print(f"embedding_dim: {emb_len}")
            print(f"embedding_preview: {preview}")
            return

        try:
            print(json.dumps(payload, indent=2, default=str, sort_keys=True))
        except TypeError:
            print(str(payload))

    for cap in requested_caps:
        _print_capability(cap, caps_dict.get(cap, {}))

    # Also try to print the raw result object
    print("\n** RAW RESULT **")
    print(result)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
