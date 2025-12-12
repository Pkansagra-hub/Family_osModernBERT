"""Benchmark reporting utilities.

Reporter is intentionally lightweight and dependency-free.

Notes:
- This module is standard-library only.
- Device information should be supplied by the runner via
    ``BenchmarkRunResult.metadata['device']``.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict, List

from familyos_ultrabert.benchmarks.types import BenchmarkRunResult


class Reporter:
    """Formats benchmark run results."""

    def __init__(self, result: BenchmarkRunResult):
        self._result = result

    def _timestamp_iso(self) -> str:
        """Return an ISO8601 timestamp for the report."""
        meta_ts = self._result.metadata.get("timestamp") if isinstance(self._result.metadata, dict) else None
        if isinstance(meta_ts, str) and meta_ts.strip():
            return meta_ts
        return _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")

    def _detect_device(self) -> str:
        """Return the device string recorded in metadata.

        The reporter does not attempt runtime/device introspection because it
        must remain standard-library only.
        """
        meta_device = self._result.metadata.get("device") if isinstance(self._result.metadata, dict) else None
        if isinstance(meta_device, str) and meta_device.strip():
            return meta_device

        return "unknown"

    def to_text(self) -> str:
        """Render a human-readable summary."""
        s = self._result.summary
        device = self._detect_device()
        timestamp = self._timestamp_iso()
        lines = [
            "=" * 80,
            "FamilyOS UltraBERT Benchmark Report",
            "=" * 80,
            f"Backend: {self._result.backend} ({device})",
            f"Date: {timestamp}",
            f"Version: {self._result.version}",
            "",
            "SUMMARY",
            "-------",
            f"Total: {s.total} | Passed: {s.passed} | Failed: {s.failed} | Warned: {getattr(s, 'warned', 0)} | Skipped: {s.skipped} | Errored: {s.errored}",
            f"Time: {s.duration_sec:.2f} seconds",
        ]

        if self._result.metadata.get("note"):
            lines.append("")
            lines.append(f"Note: {self._result.metadata['note']}")

        if self._result.suites:
            lines.append("")
            lines.append("RESULTS BY SUITE")
            lines.append("----------------")
            for suite in self._result.suites:
                passed = sum(1 for r in suite.results if r.status.value == "pass")
                total = len(suite.results)
                lines.append("")
                lines.append(f"[{suite.suite_name}] {passed}/{total} passed")
                for r in suite.results:
                    status = r.status.value.upper()

                    sev = getattr(r, "severity", None)
                    sev_str = "" if sev is None else str(getattr(sev, "value", sev)).upper()

                    line = f"  [{status}] {r.name}"
                    parts: List[str] = []
                    if sev_str and sev_str != "FAIL":
                        parts.append(f"severity={sev_str}")
                    if r.score is not None:
                        parts.append(f"score={r.score}")
                    if r.threshold is not None:
                        parts.append(f"threshold={r.threshold}")
                    if r.latency_ms is not None:
                        parts.append(f"latency_ms={r.latency_ms:.2f}")
                    if r.error:
                        parts.append(f"error={r.error}")

                    if parts:
                        line = f"{line}: " + " | ".join(parts)
                    lines.append(line)

        lines.append("=" * 80)
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        """Render machine-readable JSON."""
        payload: Dict[str, Any] = {
            "version": self._result.version,
            "backend": self._result.backend,
            "device": self._detect_device(),
            "timestamp": self._timestamp_iso(),
            "summary": {
                "total": self._result.summary.total,
                "passed": self._result.summary.passed,
                "failed": self._result.summary.failed,
                "warned": getattr(self._result.summary, "warned", 0),
                "info": getattr(self._result.summary, "info", 0),
                "skipped": self._result.summary.skipped,
                "errored": self._result.summary.errored,
                "duration_sec": self._result.summary.duration_sec,
            },
            "suites": [],
            "metadata": self._result.metadata,
        }

        for suite in self._result.suites:
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

            payload["suites"].append(
                {
                    "name": suite.suite_name,
                    "passed": suite.passed,
                    "failed": suite.failed,
                    "warned": getattr(suite, "warned", 0),
                    "info": getattr(suite, "info", 0),
                    "skipped": suite.skipped,
                    "errored": suite.errored,
                    "duration_sec": suite.total_time_sec,
                    "results": results,
                }
            )

        return json.dumps(payload, indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        """Render a markdown report suitable for CI artifacts."""
        s = self._result.summary
        device = self._detect_device()
        timestamp = self._timestamp_iso()

        lines: List[str] = []
        lines.append("# FamilyOS UltraBERT Benchmark Report")
        lines.append("")
        lines.append(f"- Backend: `{self._result.backend}` (`{device}`)")
        lines.append(f"- Version: `{self._result.version}`")
        lines.append(f"- Timestamp: `{timestamp}`")

        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Total | Passed | Failed | Skipped | Errored | Duration (s) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {s.total} | {s.passed} | {s.failed} | {s.skipped} | {s.errored} | {s.duration_sec:.2f} |"
        )

        warned = getattr(s, "warned", 0)
        info = getattr(s, "info", 0)
        if warned or info:
            lines.append("")
            lines.append(f"- Warned: `{warned}`")
            lines.append(f"- Info: `{info}`")

        if self._result.metadata.get("note"):
            lines.append("")
            lines.append(f"> Note: {self._result.metadata['note']}")

        if self._result.suites:
            lines.append("")
            lines.append("## Results by suite")

            for suite in self._result.suites:
                lines.append("")
                lines.append(f"### {suite.suite_name}")
                lines.append("")
                lines.append("| Status | Severity | Name | Score | Threshold | Latency (ms) | Error |")
                lines.append("|---|---|---|---:|---:|---:|---|")
                for r in suite.results:
                    status = r.status.value
                    sev = getattr(getattr(r, "severity", None), "value", "fail")
                    score = "" if r.score is None else str(r.score)
                    threshold = "" if r.threshold is None else str(r.threshold)
                    latency = "" if r.latency_ms is None else f"{r.latency_ms:.2f}"
                    err = "" if not r.error else str(r.error).replace("\n", " ")
                    lines.append(f"| {status} | {sev} | {r.name} | {score} | {threshold} | {latency} | {err} |")

        lines.append("")
        return "\n".join(lines) + "\n"
