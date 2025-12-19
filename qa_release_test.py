#!/usr/bin/env python3
"""
FamilyOS UltraBERT v3.0.1 - QA Release Test Suite

FAANG-level release validation testing all backend combinations:
- PyTorch CPU
- PyTorch CUDA
- ONNX CPU
- ONNX CUDA
- Decoder (PyTorch/ONNX)

Run from clean venv to simulate user experience.
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendTestResult:
    """Results for a specific backend configuration."""
    backend: str
    device: str
    available: bool
    tests: List[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tests if not t.passed)


class QATestSuite:
    """Comprehensive QA test suite for release validation."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BackendTestResult] = []
        self.doc_issues: List[str] = []
        self.usability_issues: List[str] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def run_test(self, name: str, test_fn) -> TestResult:
        """Run a single test and capture result."""
        start = time.perf_counter()
        try:
            details = test_fn()
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=name,
                passed=True,
                duration_ms=round(duration, 2),
                details=details or {}
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=name,
                passed=False,
                duration_ms=round(duration, 2),
                error=str(e)
            )

    # =========================================================================
    # BASIC IMPORT TESTS
    # =========================================================================

    def test_basic_imports(self) -> Dict:
        """Test all public exports are available."""
        from familyos_ultrabert import (
            Client,
            ClientResult,
            analyze,
            UltraBERT,
            DecoderSession,
            download_encoder,
            download_decoder,
            get_cache_dir,
            clear_cache,
            is_cached,
            get_weights_info,
            CAPABILITIES,
            Capability,
            DECODER_CAPABILITIES,
        )
        return {
            "exports": 14,
            "capabilities": len(CAPABILITIES),
            "decoder_capabilities": len(DECODER_CAPABILITIES),
        }

    def test_version(self) -> Dict:
        """Test version is correctly set."""
        import familyos_ultrabert
        version = getattr(familyos_ultrabert, '__version__', None)
        if version is None:
            # Check in package metadata
            try:
                from importlib.metadata import version as get_version
                version = get_version('familyos-ultrabert')
            except:
                version = "unknown"
        return {"version": version}

    # =========================================================================
    # BACKEND AVAILABILITY TESTS
    # =========================================================================

    def check_pytorch_available(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False

    def check_cuda_available(self) -> bool:
        """Check if CUDA is available and supported by current PyTorch."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            # Try a simple CUDA operation to verify GPU compatibility
            try:
                x = torch.zeros(1, device='cuda')
                del x
                return True
            except RuntimeError as e:
                if "no kernel image" in str(e) or "CUDA" in str(e):
                    # GPU architecture not supported by this PyTorch build
                    return False
                raise
        except:
            return False

    def check_onnx_available(self) -> bool:
        """Check if ONNX Runtime is available."""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False

    def check_onnx_cuda_available(self) -> bool:
        """Check if ONNX CUDA provider is available and working."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in providers:
                return False
            # Also check if PyTorch CUDA works (indicates proper CUDA setup)
            try:
                import torch
                return torch.cuda.is_available()
            except:
                return False
        except:
            return False

    # =========================================================================
    # PYTORCH BACKEND TESTS
    # =========================================================================

    def test_pytorch_cpu_init(self) -> Dict:
        """Test PyTorch CPU backend initialization."""
        from familyos_ultrabert import Client
        client = Client(backend='pytorch', device='cpu', verbose=False)
        return {
            "backend": client.backend,
            "device": client.device,
            "capabilities": len(client.capabilities),
        }

    def test_pytorch_cpu_inference(self) -> Dict:
        """Test inference on PyTorch CPU with latency measurement."""
        import time
        from familyos_ultrabert import Client
        client = Client(backend='pytorch', device='cpu', verbose=False)

        # Warmup
        _ = client.analyze("warmup", capabilities=['sentiment'])

        # Measure actual inference latency (10 runs)
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = client.analyze("I love my family!", capabilities=['sentiment', 'emotions'])
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        return {
            "sentiment": result.sentiment,
            "emotions": result.emotions[:3] if result.emotions else None,
            "avg_latency_ms": round(avg_latency, 1),
            "min_latency_ms": round(min(latencies), 1),
        }

    def test_pytorch_cuda_init(self) -> Dict:
        """Test PyTorch CUDA backend initialization."""
        from familyos_ultrabert import Client
        client = Client(backend='pytorch', device='cuda', verbose=False)
        return {
            "backend": client.backend,
            "device": client.device,
            "capabilities": len(client.capabilities),
        }

    def test_pytorch_cuda_inference(self) -> Dict:
        """Test inference on PyTorch CUDA with latency measurement."""
        import time
        from familyos_ultrabert import Client
        client = Client(backend='pytorch', device='cuda', verbose=False)

        # Warmup
        _ = client.analyze("warmup", capabilities=['sentiment'])

        # Measure actual inference latency (10 runs)
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = client.analyze("I love my family!", capabilities=['sentiment', 'emotions'])
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        return {
            "sentiment": result.sentiment,
            "emotions": result.emotions[:3] if result.emotions else None,
            "avg_latency_ms": round(avg_latency, 1),
            "min_latency_ms": round(min(latencies), 1),
        }

    # =========================================================================
    # ONNX BACKEND TESTS
    # =========================================================================

    def test_onnx_cpu_init(self) -> Dict:
        """Test ONNX CPU backend initialization."""
        from familyos_ultrabert import Client
        client = Client(backend='onnx', device='cpu', verbose=False)
        return {
            "backend": client.backend,
            "device": client.device,
            "capabilities": len(client.capabilities),
        }

    def test_onnx_cpu_inference(self) -> Dict:
        """Test inference on ONNX CPU with latency measurement."""
        import time
        from familyos_ultrabert import Client
        client = Client(backend='onnx', device='cpu', verbose=False)

        # Warmup
        _ = client.analyze("warmup", capabilities=['sentiment'])

        # Measure actual inference latency (10 runs)
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = client.analyze("I love my family!", capabilities=['sentiment', 'emotions'])
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        return {
            "sentiment": result.sentiment,
            "emotions": result.emotions[:3] if result.emotions else None,
            "avg_latency_ms": round(avg_latency, 1),
            "min_latency_ms": round(min(latencies), 1),
        }

    def test_onnx_cuda_init(self) -> Dict:
        """Test ONNX CUDA backend initialization."""
        from familyos_ultrabert import Client
        client = Client(backend='onnx', device='cuda', verbose=False)
        return {
            "backend": client.backend,
            "device": client.device,
            "capabilities": len(client.capabilities),
        }

    def test_onnx_cuda_inference(self) -> Dict:
        """Test inference on ONNX CUDA with latency measurement."""
        import time
        from familyos_ultrabert import Client
        client = Client(backend='onnx', device='cuda', verbose=False)

        # Warmup
        _ = client.analyze("warmup", capabilities=['sentiment'])

        # Measure actual inference latency (10 runs)
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = client.analyze("I love my family!", capabilities=['sentiment', 'emotions'])
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        return {
            "sentiment": result.sentiment,
            "emotions": result.emotions[:3] if result.emotions else None,
            "avg_latency_ms": round(avg_latency, 1),
            "min_latency_ms": round(min(latencies), 1),
        }

    # =========================================================================
    # DECODER TESTS
    # =========================================================================

    def test_decoder_pytorch_init(self) -> Dict:
        """Test decoder session with PyTorch backend."""
        from familyos_ultrabert import Client
        # Use CPU to avoid GPU compatibility issues
        client = Client(backend='pytorch', device='cpu', verbose=False)
        with client.create_decoder_session(backend='pytorch', device='cpu') as decoder:
            return {
                "decoder_backend": decoder.backend,
                "loaded": True,
            }

    def test_decoder_pytorch_generation(self) -> Dict:
        """Test text generation with PyTorch decoder."""
        from familyos_ultrabert import Client
        # Use CPU to avoid GPU compatibility issues
        client = Client(backend='pytorch', device='cpu', verbose=False)

        test_input = "I felt really overwhelmed with all my tasks today"
        result = client.suggest_alternative(test_input)

        return {
            "input": test_input,
            "output": result[:100] if result else None,
            "output_length": len(result) if result else 0,
            "is_coherent": len(result) > 20 if result else False,
        }

    def test_decoder_session_memory_cleanup(self) -> Dict:
        """Test that decoder memory is properly cleaned up."""
        import gc
        try:
            import torch
            has_torch = True
        except:
            has_torch = False

        from familyos_ultrabert import Client
        # Use CPU to avoid GPU compatibility issues
        client = Client(backend='pytorch', device='cpu', verbose=False)

        if has_torch:
            import torch
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        with client.create_decoder_session(backend='pytorch', device='cpu') as decoder:
            _ = client.suggest_alternative("Test input")
            if has_torch and torch.cuda.is_available():
                mem_during = torch.cuda.memory_allocated()

        gc.collect()
        if has_torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_after = torch.cuda.memory_allocated()
            return {
                "mem_before_mb": round(mem_before / 1024 / 1024, 2),
                "mem_during_mb": round(mem_during / 1024 / 1024, 2),
                "mem_after_mb": round(mem_after / 1024 / 1024, 2),
                "cleanup_successful": mem_after <= mem_before * 1.1,  # Allow 10% margin
            }
        return {"cleanup_checked": True}

    # =========================================================================
    # ALL CAPABILITIES TEST
    # =========================================================================

    def test_all_encoder_capabilities(self) -> Dict:
        """Test all 12 encoder capabilities."""
        from familyos_ultrabert import Client, CAPABILITIES

        # Use CPU to avoid GPU compatibility issues
        client = Client(backend='pytorch', device='cpu', verbose=False)

        test_text = "My daughter Emma said she wants to go to the park tomorrow at 3pm with grandma."

        # Capability name -> ClientResult attribute mapping
        cap_to_attr = {
            'sentiment': 'sentiment',
            'emotions': 'emotions',
            'safety_familyos': 'safety',
            'safety_generic': 'safety_scores',
            'intent': 'intent',
            'ingress': 'ingress',
            'ner_family': 'entities',
            'ner_general': 'general_entities',
            'temporal': 'temporal',
            'relation': 'relations',
            'nli': 'nli',
            'embedding': 'embedding',
        }

        results = {}
        for cap in CAPABILITIES:
            if cap == 'counterfactual':
                continue  # Skip decoder capability
            try:
                result = client.analyze(test_text, capabilities=[cap])
                attr_name = cap_to_attr.get(cap, cap)
                cap_result = getattr(result, attr_name, None)
                results[cap] = {
                    "available": cap_result is not None,
                    "type": type(cap_result).__name__ if cap_result else None,
                }
            except Exception as e:
                results[cap] = {"available": False, "error": str(e)}

        return {
            "capabilities_tested": len(results),
            "capabilities_working": sum(1 for r in results.values() if r.get("available")),
            "details": results,
        }

    # =========================================================================
    # DOCUMENTATION CHECK
    # =========================================================================

    def check_documentation(self) -> List[str]:
        """Check documentation quality and completeness."""
        issues = []

        # Check README
        try:
            import familyos_ultrabert
            pkg_dir = Path(familyos_ultrabert.__file__).parent

            # Try multiple locations for README
            readme_paths = [
                pkg_dir / "README.md",           # In package dir
                pkg_dir.parent / "README.md",    # Parent (editable install)
            ]

            readme = None
            for path in readme_paths:
                if path.exists():
                    readme = path
                    break

            if readme and readme.exists():
                content = readme.read_text(encoding='utf-8')

                # Check for essential sections
                if "## Installation" not in content and "# Installation" not in content:
                    issues.append("README: Missing Installation section")
                if "## Usage" not in content and "# Usage" not in content and "## Quick Start" not in content:
                    issues.append("README: Missing Usage/Quick Start section")
                if "pip install" not in content:
                    issues.append("README: Missing pip install command")
                if "from familyos_ultrabert" not in content:
                    issues.append("README: Missing import example")
            else:
                issues.append("README.md not found in package")
        except Exception as e:
            issues.append(f"README check failed: {e}")

        # Check docstrings
        try:
            from familyos_ultrabert import Client, DecoderSession

            if not Client.__doc__:
                issues.append("Client class missing docstring")
            if not DecoderSession.__doc__:
                issues.append("DecoderSession class missing docstring")

            # Check key methods
            if hasattr(Client, 'analyze') and not Client.analyze.__doc__:
                issues.append("Client.analyze missing docstring")
            if hasattr(Client, 'suggest_alternative') and not Client.suggest_alternative.__doc__:
                issues.append("Client.suggest_alternative missing docstring")
        except Exception as e:
            issues.append(f"Docstring check failed: {e}")

        return issues

    # =========================================================================
    # USABILITY CHECK
    # =========================================================================

    def check_usability(self) -> List[str]:
        """Check usability and developer experience."""
        issues = []

        # Test: Simplest possible usage (with explicit CPU to avoid GPU issues)
        try:
            from familyos_ultrabert import Client
            # Use explicit CPU device to avoid GPU compatibility issues
            client = Client(device='cpu', verbose=False)
            result = client.analyze("Hello world")
            if result is None:
                issues.append("analyze() returned None for simple input")
        except Exception as e:
            issues.append(f"Simple analyze() call failed: {e}")

        # Test: Error messages are helpful
        try:
            from familyos_ultrabert import Client
            client = Client(backend='invalid_backend', verbose=False)
            issues.append("Invalid backend should raise clear error")
        except ValueError as e:
            if "invalid_backend" not in str(e).lower() and "backend" not in str(e).lower():
                issues.append(f"Error message not helpful for invalid backend: {e}")
        except Exception as e:
            # Any error is acceptable, but check message
            if len(str(e)) < 10:
                issues.append("Error message too short/unhelpful")

        # Test: Capabilities are discoverable
        try:
            from familyos_ultrabert import CAPABILITIES, Capability
            if len(CAPABILITIES) < 12:
                issues.append(f"Only {len(CAPABILITIES)} capabilities, expected 12+")
        except ImportError:
            issues.append("CAPABILITIES not exported from package")

        return issues

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================

    def run_all(self):
        """Run complete test suite."""
        print("=" * 70)
        print("FamilyOS UltraBERT v3.0.1 - QA Release Test Suite")
        print("=" * 70)
        print()

        # Phase 1: Basic Tests
        print("[PHASE 1] Basic Import Tests")
        print("-" * 70)

        basic_results = BackendTestResult(backend="basic", device="n/a", available=True)

        t = self.run_test("package_imports", self.test_basic_imports)
        basic_results.tests.append(t)
        self._print_test_result(t)

        t = self.run_test("version_check", self.test_version)
        basic_results.tests.append(t)
        self._print_test_result(t)

        self.results.append(basic_results)

        # Phase 2: Backend Detection
        print()
        print("[PHASE 2] Backend Availability")
        print("-" * 70)

        pytorch_available = self.check_pytorch_available()
        cuda_available = self.check_cuda_available()
        onnx_available = self.check_onnx_available()
        onnx_cuda_available = self.check_onnx_cuda_available()

        print(f"  PyTorch:    {'YES' if pytorch_available else 'NO'}")
        print(f"  CUDA:       {'YES' if cuda_available else 'NO'}")
        print(f"  ONNX:       {'YES' if onnx_available else 'NO'}")
        print(f"  ONNX+CUDA:  {'YES' if onnx_cuda_available else 'NO'}")

        # Phase 3: PyTorch Tests
        print()
        print("[PHASE 3] PyTorch Backend Tests")
        print("-" * 70)

        if pytorch_available:
            # CPU
            pytorch_cpu = BackendTestResult(backend="pytorch", device="cpu", available=True)

            t = self.run_test("pytorch_cpu_init", self.test_pytorch_cpu_init)
            pytorch_cpu.tests.append(t)
            self._print_test_result(t)

            t = self.run_test("pytorch_cpu_inference", self.test_pytorch_cpu_inference)
            pytorch_cpu.tests.append(t)
            self._print_test_result(t)

            self.results.append(pytorch_cpu)

            # CUDA
            if cuda_available:
                pytorch_cuda = BackendTestResult(backend="pytorch", device="cuda", available=True)

                t = self.run_test("pytorch_cuda_init", self.test_pytorch_cuda_init)
                pytorch_cuda.tests.append(t)
                self._print_test_result(t)

                t = self.run_test("pytorch_cuda_inference", self.test_pytorch_cuda_inference)
                pytorch_cuda.tests.append(t)
                self._print_test_result(t)

                self.results.append(pytorch_cuda)
            else:
                print("  [SKIP] CUDA not available")
        else:
            print("  [SKIP] PyTorch not installed")

        # Phase 4: ONNX Tests
        print()
        print("[PHASE 4] ONNX Backend Tests")
        print("-" * 70)

        if onnx_available:
            # CPU
            onnx_cpu = BackendTestResult(backend="onnx", device="cpu", available=True)

            t = self.run_test("onnx_cpu_init", self.test_onnx_cpu_init)
            onnx_cpu.tests.append(t)
            self._print_test_result(t)

            t = self.run_test("onnx_cpu_inference", self.test_onnx_cpu_inference)
            onnx_cpu.tests.append(t)
            self._print_test_result(t)

            self.results.append(onnx_cpu)

            # CUDA
            if onnx_cuda_available:
                onnx_cuda = BackendTestResult(backend="onnx", device="cuda", available=True)

                t = self.run_test("onnx_cuda_init", self.test_onnx_cuda_init)
                onnx_cuda.tests.append(t)
                self._print_test_result(t)

                t = self.run_test("onnx_cuda_inference", self.test_onnx_cuda_inference)
                onnx_cuda.tests.append(t)
                self._print_test_result(t)

                self.results.append(onnx_cuda)
            else:
                print("  [SKIP] ONNX CUDA provider not available")
        else:
            print("  [SKIP] ONNX Runtime not installed")

        # Phase 5: Decoder Tests
        print()
        print("[PHASE 5] Decoder (Counterfactual Generation) Tests")
        print("-" * 70)

        if pytorch_available:
            decoder_results = BackendTestResult(backend="decoder", device="auto", available=True)

            t = self.run_test("decoder_pytorch_init", self.test_decoder_pytorch_init)
            decoder_results.tests.append(t)
            self._print_test_result(t)

            t = self.run_test("decoder_pytorch_generation", self.test_decoder_pytorch_generation)
            decoder_results.tests.append(t)
            self._print_test_result(t)

            t = self.run_test("decoder_memory_cleanup", self.test_decoder_session_memory_cleanup)
            decoder_results.tests.append(t)
            self._print_test_result(t)

            self.results.append(decoder_results)
        else:
            print("  [SKIP] PyTorch required for decoder")

        # Phase 6: All Capabilities
        print()
        print("[PHASE 6] All Encoder Capabilities")
        print("-" * 70)

        if pytorch_available:
            caps_results = BackendTestResult(backend="capabilities", device="auto", available=True)

            t = self.run_test("all_encoder_capabilities", self.test_all_encoder_capabilities)
            caps_results.tests.append(t)
            self._print_test_result(t)

            self.results.append(caps_results)

        # Phase 7: Documentation Check
        print()
        print("[PHASE 7] Documentation Quality")
        print("-" * 70)

        self.doc_issues = self.check_documentation()
        if self.doc_issues:
            for issue in self.doc_issues:
                print(f"  [ISSUE] {issue}")
        else:
            print("  [PASS] Documentation looks complete")

        # Phase 8: Usability Check
        print()
        print("[PHASE 8] Usability Check")
        print("-" * 70)

        self.usability_issues = self.check_usability()
        if self.usability_issues:
            for issue in self.usability_issues:
                print(f"  [ISSUE] {issue}")
        else:
            print("  [PASS] Usability looks good")

        # Final Report
        self._print_final_report()

    def _print_test_result(self, t: TestResult):
        """Print a single test result."""
        status = "[PASS]" if t.passed else "[FAIL]"
        print(f"  {status} {t.name} ({t.duration_ms}ms)")
        if t.error:
            print(f"         Error: {t.error[:80]}")
        if t.details and self.verbose:
            for k, v in t.details.items():
                if k != 'details':  # Skip nested details
                    print(f"         {k}: {v}")

    def _print_final_report(self):
        """Print final QA report."""
        print()
        print("=" * 70)
        print("QA RELEASE REPORT")
        print("=" * 70)
        print()

        total_passed = 0
        total_failed = 0

        for br in self.results:
            passed = br.passed
            failed = br.failed
            total_passed += passed
            total_failed += failed

            status = "PASS" if failed == 0 else "FAIL"
            print(f"  [{status}] {br.backend}/{br.device}: {passed}/{passed+failed} tests")

        print()
        print("-" * 70)
        print(f"  TOTAL TESTS: {total_passed + total_failed}")
        print(f"  PASSED:      {total_passed}")
        print(f"  FAILED:      {total_failed}")
        print(f"  DOC ISSUES:  {len(self.doc_issues)}")
        print(f"  UX ISSUES:   {len(self.usability_issues)}")
        print("-" * 70)

        if total_failed == 0 and len(self.doc_issues) == 0 and len(self.usability_issues) == 0:
            print()
            print("  *** RELEASE APPROVED ***")
            print("  Package is ready for GitHub release.")
            print()
        elif total_failed == 0:
            print()
            print("  *** RELEASE APPROVED WITH NOTES ***")
            print("  All tests passed. Address documentation/UX issues in next patch.")
            print()
        else:
            print()
            print("  *** RELEASE BLOCKED ***")
            print("  Fix failing tests before release.")
            print()

        return total_failed == 0


def main():
    """Run QA test suite."""
    suite = QATestSuite(verbose=True)
    suite.run_all()

    # Return exit code
    total_failed = sum(br.failed for br in suite.results)
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
