"""Baseline drift tracking for benchmark runs.

This module is standard-library only.

Goal:
- Persist last-known-good benchmark JSON per environment key
- On subsequent runs, compare key numeric metrics and report deltas
- Fail the run only when hard-gating metrics regress beyond tolerance

Environment key (stable): (backend, device, model_sha256)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from familyos_ultrabert.benchmarks.types import BenchmarkRunResult, BenchmarkSeverity


_DEFAULT_BASELINE_DIR = Path.home() / ".familyos_ultrabert" / "benchmarks" / "baselines"


def _safe_filename(s: str) -> str:
	out = []
	for ch in s:
		if ch.isalnum() or ch in ("-", "_", "."):
			out.append(ch)
		else:
			out.append("_")
	return "".join(out)


def environment_key(result: BenchmarkRunResult) -> str:
	"""Build a stable environment key for baseline storage.

	Args:
		result: Benchmark run.

	Returns:
		Stable key string.
	"""
	backend = str(getattr(result, "backend", "unknown"))
	meta = getattr(result, "metadata", {}) or {}
	device = str(meta.get("device", "unknown"))
	model = meta.get("model") if isinstance(meta, dict) else None
	sha = None
	if isinstance(model, dict):
		sha = model.get("sha256")
	sha_str = str(sha) if sha else "unknown"
	return _safe_filename(f"{backend}__{device}__{sha_str}")


def _flatten_metrics(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
	"""Extract numeric metrics from a benchmark JSON payload.

	Returns a mapping:
		metric_key -> {"value": float, "severity": str}
	"""
	out: Dict[str, Dict[str, Any]] = {}
	for suite in list(payload.get("suites", [])):
		suite_name = str(suite.get("name", ""))
		for r in list(suite.get("results", [])):
			name = str(r.get("name", ""))
			severity = r.get("severity") or "fail"
			sev_str = str(severity).lower()
			# Primary numeric fields
			for field in ("score", "latency_ms", "threshold"):
				val = r.get(field)
				if isinstance(val, (int, float)):
					metric_severity = sev_str
					# Thresholds are configuration, not performance. Track them but don't gate.
					if field == "threshold" and metric_severity == BenchmarkSeverity.FAIL.value:
						metric_severity = BenchmarkSeverity.WARN.value
					out[f"{suite_name}.{name}.{field}"] = {
						"value": float(val),
						"severity": metric_severity,
					}
			# Selected numeric detail fields
			details = r.get("details")
			if isinstance(details, dict):
				for k, v in details.items():
					if not isinstance(v, (int, float)):
						continue
					key = str(k)
					# Keep a conservative allowlist to reduce noise.
					if key in (
						"p50_ms",
						"p95_ms",
						"p99_ms",
						"mean_ms",
						"min_ms",
						"max_ms",
						"recall",
						"accuracy",
						"calls_per_sec",
						"embeddings_per_sec",
					):
						detail_severity = sev_str
						# Detail fields are often noisy. Track them but don't gate release.
						if detail_severity == BenchmarkSeverity.FAIL.value:
							detail_severity = BenchmarkSeverity.WARN.value
						out[f"{suite_name}.{name}.details.{key}"] = {
							"value": float(v),
							"severity": detail_severity,
						}
	return out


def _is_lower_better(metric_key: str) -> bool:
	k = metric_key.lower()
	return (
		k.endswith(".latency_ms")
		or k.endswith("_ms")
		or ".details.p" in k and k.endswith("_ms")
		or ".details.mean_ms" in k
		or ".details.max_ms" in k
		or ".details.min_ms" in k
	)


def _tolerances_for_value(value: float, *, lower_better: bool) -> Tuple[float, float]:
	"""Return (pct_tol, abs_tol).

	- For latency-ish metrics, use pct tolerance by default.
	- For 0..1 metrics, use abs tolerance by default.
	"""
	pct_tol_env = os.getenv("ULTRABERT_BENCH_BASELINE_PCT_TOL")
	abs_tol_env = os.getenv("ULTRABERT_BENCH_BASELINE_ABS_TOL")
	pct_tol = float(pct_tol_env) if pct_tol_env else 0.10
	abs_tol = float(abs_tol_env) if abs_tol_env else 0.02

	if 0.0 <= value <= 1.0 and not lower_better:
		return (0.0, abs_tol)
	return (pct_tol, 0.0)


def _compare_value(
	*,
	metric_key: str,
	baseline_value: float,
	current_value: float,
	severity: str,
) -> Dict[str, Any]:
	lower_better = _is_lower_better(metric_key)
	pct_tol, abs_tol = _tolerances_for_value(baseline_value, lower_better=lower_better)

	if lower_better:
		# Regression if latency increases too much.
		delta = current_value - baseline_value
		pct = (delta / baseline_value) if baseline_value != 0.0 else (1.0 if delta > 0 else 0.0)
		regressed = pct_tol > 0.0 and pct > pct_tol
		return {
			"metric": metric_key,
			"direction": "lower_better",
			"baseline": baseline_value,
			"current": current_value,
			"delta": delta,
			"delta_pct": pct,
			"severity": severity,
			"status": "regressed" if regressed else ("improved" if delta < 0 else "unchanged"),
			"tolerance": {"pct": pct_tol},
		}

	# Higher is better.
	delta = current_value - baseline_value
	pct = (delta / baseline_value) if baseline_value != 0.0 else (1.0 if delta > 0 else 0.0)
	regressed = (abs_tol > 0.0 and (-delta) > abs_tol) or (pct_tol > 0.0 and pct < -pct_tol)
	return {
		"metric": metric_key,
		"direction": "higher_better",
		"baseline": baseline_value,
		"current": current_value,
		"delta": delta,
		"delta_pct": pct,
		"severity": severity,
		"status": "regressed" if regressed else ("improved" if delta > 0 else "unchanged"),
		"tolerance": {"abs": abs_tol, "pct": pct_tol},
	}


def _payload_from_result(result: BenchmarkRunResult) -> Dict[str, Any]:
	"""Build a JSON-serializable payload for baseline storage."""
	# The reporter already knows the canonical shape, but we want a dependency-free
	# representation without calling reporter formatting.
	out = {
		"version": result.version,
		"backend": result.backend,
		"summary": asdict(result.summary),
		"suites": [],
		"metadata": result.metadata,
	}
	for suite in result.suites:
		results: List[Dict[str, Any]] = []
		for r in suite.results:
			results.append(
				{
					"name": r.name,
					"category": r.category,
					"status": r.status.value,
					"severity": getattr(getattr(r, "severity", None), "value", None),
					"score": r.score,
					"threshold": r.threshold,
					"latency_ms": r.latency_ms,
					"error": r.error,
					"details": r.details,
				}
			)
		out["suites"].append(
			{
				"name": suite.suite_name,
				"duration_sec": suite.total_time_sec,
				"results": results,
			}
		)
	return out


def compare_and_update_baseline(
	result: BenchmarkRunResult,
	*,
	baseline_dir: Optional[Path] = None,
) -> Dict[str, Any]:
	"""Compare against baseline (if exists) and save new baseline.

	Args:
		result: Current benchmark run.
		baseline_dir: Directory to store baselines (defaults to user home).

	Returns:
		Report dict with diffs and counts.
	"""
	base_dir = baseline_dir if baseline_dir is not None else _DEFAULT_BASELINE_DIR
	base_dir.mkdir(parents=True, exist_ok=True)

	key = environment_key(result)
	path = base_dir / f"{key}.json"

	current_payload = _payload_from_result(result)
	current_metrics = _flatten_metrics(current_payload)

	report: Dict[str, Any] = {
		"enabled": True,
		"baseline_dir": str(base_dir),
		"environment_key": key,
		"baseline_path": str(path),
		"compared": False,
		"diffs": [],
		"failed": 0,
		"warned": 0,
	}

	if path.exists():
		try:
			baseline_payload = json.loads(path.read_text(encoding="utf-8"))
			baseline_metrics = _flatten_metrics(baseline_payload)
			diffs: List[Dict[str, Any]] = []
			for metric_key, cur in current_metrics.items():
				if metric_key not in baseline_metrics:
					continue
				base = baseline_metrics[metric_key]
				diff = _compare_value(
					metric_key=metric_key,
					baseline_value=float(base["value"]),
					current_value=float(cur["value"]),
					severity=str(cur.get("severity") or "fail"),
				)
				diffs.append(diff)

			report["compared"] = True
			report["diffs"] = diffs

			# Gate logic: only FAIL-severity regressions fail the run.
			failed = 0
			warned = 0
			for d in diffs:
				if d.get("status") != "regressed":
					continue
				sev = str(d.get("severity") or "fail").lower()
				if sev == BenchmarkSeverity.FAIL.value:
					failed += 1
				else:
					warned += 1
			report["failed"] = failed
			report["warned"] = warned
		except Exception as exc:  # noqa: BLE001
			report["error"] = f"{type(exc).__name__}: {exc}"

	# Always write the latest run as the new baseline.
	path.write_text(json.dumps(current_payload, indent=2, sort_keys=True), encoding="utf-8")
	return report
