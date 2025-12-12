"""Basic usage example for FamilyOS UltraBERT.

This example shows the supported, stable API surface.

Run:
    python -m familyos_ultrabert.examples.basic_usage

Or execute benchmarks:
    python -m familyos_ultrabert.benchmarks --suite api,regression
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def main() -> int:
    """Run a small end-to-end demo using the public Client API."""
    try:
        from familyos_ultrabert import Client
    except Exception as exc:  # pragma: no cover
        print("Failed to import familyos_ultrabert Client.")
        print(f"Error: {type(exc).__name__}: {exc}")
        return 1

    try:
        client = Client(warmup=True, warmup_rounds=2)
    except Exception as exc:  # pragma: no cover
        print("Failed to initialize Client. Are model files and runtime dependencies available?")
        print(f"Error: {type(exc).__name__}: {exc}")
        return 1

    text = "Mom picked up the kids from school and we had a great dinner together."

    print(f"Backend: {client.backend}")
    print(f"Input: {text}")

    result = client.analyze(text)

    # ClientResult is intended to be friendly: dict-like + convenience attributes.
    print("\nSummary:")
    print(f"  sentiment: {getattr(result, 'sentiment', None)}")
    print(f"  safety:    {getattr(result, 'safety', None)}")
    print(f"  emotions:  {getattr(result, 'emotions', None)}")
    print(f"  intent:    {getattr(result, 'intent', None)}")
    print(f"  latency:   {getattr(result, 'latency_ms', None)} ms")

    print("\nAs JSON:")
    try:
        _print_json(result.to_dict())
    except Exception:
        # Fallback: result might already behave like a dict.
        _print_json(dict(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
