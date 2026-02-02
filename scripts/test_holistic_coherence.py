"""Test holistic coherence benchmark."""

import time

from familyos_ultrabert.pytorch_inference import PyTorchInferenceEngine
from familyos_ultrabert.benchmarks.suite.holistic_coherence import HolisticCoherenceSuite


def main():
    # Load model
    inf = PyTorchInferenceEngine.load("d:/Modeling_studio/checkpoints/best_v4_halo")
    print(f"Loaded model with {len(inf.capabilities)} capabilities")
    print(f"Capabilities: {sorted(inf.capabilities)}")

    # Run benchmark
    suite = HolisticCoherenceSuite(inf)
    start = time.time()
    results = suite.run()
    elapsed = time.time() - start

    print(f"\n=== Holistic Coherence Results ({suite._MAX_SAMPLES} samples) ===")
    print(f"Total runtime: {elapsed:.1f}s ({elapsed/suite._MAX_SAMPLES*1000:.1f}ms/sample)")
    print()

    for r in results:
        if r.name in ("golden_set_available", "capabilities_check"):
            continue
        status = "PASS" if r.status.value == "pass" else "FAIL"
        score = f"{r.score:.4f}" if r.score else "N/A"
        thresh = f"(>={r.threshold:.2f})" if r.threshold else ""
        print(f"  [{status}] {r.name}: {score} {thresh}")

    # Print FCCS breakdown
    fccs = [r for r in results if r.name == "fccs_overall"][0]
    print("\n=== FCCS Component Breakdown ===")
    components = fccs.details.get("components", {})
    weights = fccs.details.get("weights", {})
    for k, v in components.items():
        w = weights.get(k, 0)
        contribution = v * w
        print(f"  {k}: {v:.4f} x {w:.2f} = {contribution:.4f}")
    print(f"  ---------------------------------")
    print(f"  TOTAL FCCS: {fccs.score:.4f}")

    # Print sample details
    print("\n=== Sample Details (first 5) ===")
    for s in fccs.details.get("sample_details", [])[:5]:
        print(f"  {s['id']}:")
        print(f"    SEA={s['sea']['score']:.2f} ({s['sea']['reason']})")
        print(f"    SEC={s['sec']['score']:.2f} ({s['sec']['reason']})")
        print(f"    IIC={s['iic']['score']:.2f} ({s['iic']['reason']})")
        print(f"    emotions: {s['emotions'][:3]}...")
        print()


if __name__ == "__main__":
    main()
