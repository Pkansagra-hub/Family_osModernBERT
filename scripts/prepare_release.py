#!/usr/bin/env python
"""Release preparation script for FamilyOS UltraBERT.

This script helps prepare releases by:
1. Validating package structure
2. Building packages
3. Testing installation
4. Generating checksums and a release bundle
5. Writing a lightweight release summary for GitHub assets

Usage:
    python scripts/prepare_release.py [--version VERSION] [--test-install]

Example:
    python scripts/prepare_release.py --version 4.0.2 --test-install --generate-checksums --build-bundle
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = PROJECT_ROOT / "familyos_ultrabert"
DIST_DIR = PACKAGE_DIR / "dist"
PYPROJECT_PATH = PACKAGE_DIR / "pyproject.toml"
INIT_PATH = PACKAGE_DIR / "__init__.py"
RELEASE_NOTES_PATH = PACKAGE_DIR / "RELEASE_NOTES.md"


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result


def validate_package_structure():
    """Validate that the package structure is correct for release."""
    print("Validating package structure...")

    package_dir = PACKAGE_DIR

    # Check required files
    required_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "__init__.py",
        "client.py",
        "model.py",
        "weights_manager.py",
    ]

    for file in required_files:
        if not (package_dir / file).exists():
            print(f"Missing required file: {file}")
            return False

    # Check that weights are excluded
    if (package_dir / "weights").exists():
        print("Weights directory exists - it should be excluded from the distribution")
        # Check that weights are not accidentally included
        if (package_dir / "weights" / "pytorch" / "model.safetensors").exists():
            print("Weights found under package directory (expected for local development)")

    print("Package structure validation passed")
    return True


def update_version(version):
    """Update version in pyproject.toml and __init__.py."""
    print(f"Updating version to {version}...")

    pyproject_content = PYPROJECT_PATH.read_text(encoding="utf-8")
    init_content = INIT_PATH.read_text(encoding="utf-8")

    import re

    new_pyproject = re.sub(
        r'version = "\d+\.\d+\.\d+"',
        f'version = "{version}"',
        pyproject_content,
    )
    new_init = re.sub(
        r'__version__ = "\d+\.\d+\.\d+"',
        f'__version__ = "{version}"',
        init_content,
    )

    PYPROJECT_PATH.write_text(new_pyproject, encoding="utf-8")
    INIT_PATH.write_text(new_init, encoding="utf-8")
    print(f"Updated version to {version}")


def get_dist_artifacts() -> list[Path]:
    """Return built wheel and source distribution artifacts."""
    artifacts = sorted(DIST_DIR.glob("*.whl")) + sorted(DIST_DIR.glob("*.tar.gz"))
    return artifacts


def sha256_file(path: Path) -> str:
    """Return the SHA256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_package():
    """Build the package."""
    print("Building package...")

    dist_dir = DIST_DIR
    build_dir = PACKAGE_DIR / "build"

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # Build package
    run_command([sys.executable, "-m", "build"], cwd=str(PACKAGE_DIR))

    # Check what was built
    if not dist_dir.exists():
        print("Build failed - no dist directory created")
        return False

    files = list(dist_dir.glob("*"))
    if not files:
        print("Build failed - no files in dist directory")
        return False

    print("Package built successfully:")
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.1f} MB)")

    return True


def test_package_contents():
    """Test that the package contains the right files and excludes weights."""
    print("Testing package contents...")

    wheel_files = list(DIST_DIR.glob("*.whl"))

    if not wheel_files:
        print("No wheel file found")
        return False

    wheel_file = wheel_files[0]

    # Extract wheel contents to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        run_command([sys.executable, "-m", "zipfile", "-e", str(wheel_file), temp_dir])

        # Check contents
        contents = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), temp_dir)
                contents.append(rel_path)

        # Check for weights (should not be present)
        weight_files = [f for f in contents if "weights/" in f or "model.safetensors" in f]
        if weight_files:
            print(f"Weights found in package (should be excluded): {weight_files}")
            return False

        # Check for required files
        required_in_package = [
            "__init__.py",
            "client.py",
            "model.py",
        ]

        missing = []
        for required in required_in_package:
            if not any(required in content for content in contents):
                missing.append(required)

        if missing:
            print(f"Missing required files in package: {missing}")
            return False

        print("Package contents validated:")
        print(f"   - Total files: {len(contents)}")
        print("   - No weights included: yes")
        print("   - Required modules present: yes")

    return True


def test_installation():
    """Test installing the package in a temporary environment."""
    print("Testing package installation...")

    wheel_files = list(DIST_DIR.glob("*.whl"))

    if not wheel_files:
        print("No wheel file found for testing")
        return False

    wheel_file = wheel_files[0]

    # Create temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_dir = Path(temp_dir) / "venv"

        # Create venv
        run_command([sys.executable, "-m", "venv", str(venv_dir)])

        # Install package
        pip_exe = (
            venv_dir / "Scripts" / "pip.exe"
            if sys.platform == "win32"
            else venv_dir / "bin" / "pip"
        )
        run_command([str(pip_exe), "install", str(wheel_file)])

        # Test import
        python_exe = (
            venv_dir / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else venv_dir / "bin" / "python"
        )
        result = run_command(
            [str(python_exe), "-c", "import familyos_ultrabert; print('Import successful')"],
            check=False,
        )

        if result.returncode != 0:
            print("Package import failed")
            print(f"Error: {result.stderr}")
            return False

        print("Package installation and import successful")

    return True


def generate_checksums() -> Path:
    """Generate a SHA256 checksum manifest for dist artifacts."""
    print("Generating checksums...")

    artifacts = get_dist_artifacts()
    if not artifacts:
        raise FileNotFoundError("No wheel or source distribution found in dist/")

    checksum_path = DIST_DIR / "SHA256SUMS.txt"
    lines = [f"{sha256_file(path)}  {path.name}" for path in artifacts]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote checksum manifest: {checksum_path}")
    return checksum_path


def write_release_summary(version: str) -> Path:
    """Write a compact release summary for GitHub assets."""
    print("Writing release summary...")

    artifacts = get_dist_artifacts()
    checksum_path = DIST_DIR / "SHA256SUMS.txt"
    artifact_lines = "\n".join(f"- `{path.name}`" for path in artifacts)
    checksum_section = ""
    if checksum_path.exists():
        checksum_section = f"\n## Checksums\n\n- `SHA256SUMS.txt`\n"

    notes_reference = ""
    if RELEASE_NOTES_PATH.exists():
        notes_reference = "\nFor detailed feature notes, see `familyos_ultrabert/RELEASE_NOTES.md`.\n"

    summary = f"""# FamilyOS UltraBERT v{version}

Lightweight release package for FamilyOS UltraBERT. Model weights are downloaded at runtime from `Pkansagra/ultrabert-weights` using `encoder/v2/fp32/`.

## Included release assets

{artifact_lines}
{checksum_section}
## Installation

```bash
pip install familyos_ultrabert-{version}-py3-none-any.whl
```

## Runtime weight source

- Hugging Face repo: `Pkansagra/ultrabert-weights`
- Encoder path: `encoder/v2/fp32/`
- Runtime package: `familyos_ultrabert`
{notes_reference}"""

    summary_path = DIST_DIR / f"RELEASE_{version}.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Wrote release summary: {summary_path}")
    return summary_path


def build_release_bundle(version: str) -> Path:
    """Create a convenience zip bundle for GitHub Releases."""
    print("Building release bundle...")

    artifacts = get_dist_artifacts()
    if not artifacts:
        raise FileNotFoundError("No wheel or source distribution found in dist/")

    optional_files = [
        DIST_DIR / "SHA256SUMS.txt",
        DIST_DIR / f"RELEASE_{version}.md",
    ]
    bundle_path = DIST_DIR / f"familyos_ultrabert-{version}-release.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in artifacts + [item for item in optional_files if item.exists()]:
            bundle.write(path, arcname=path.name)

    print(f"Created release bundle: {bundle_path}")
    return bundle_path


def create_release_notes(version):
    """Generate release notes template."""
    print("Generating release notes template...")

    notes = f"""# FamilyOS UltraBERT v{version} Release Notes

## 🚀 What's New

### Major Changes
- [List major changes here]

### Improvements
- [List improvements here]

### Bug Fixes
- [List bug fixes here]

## 📊 Performance Benchmarks

| Metric | v{version} | Previous | Change |
|--------|------------|----------|--------|
| Inference P95 | X ms | Y ms | ±Z% |
| Throughput | X req/sec | Y req/sec | ±Z% |
| Memory Usage | X GB | Y GB | ±Z% |

## 🔧 Technical Details

### Model Architecture
- **Backbone**: ModernBERT-base (22 layers, 768-dim)
- **NER Heads**: GlobalPointer (3 heads)
- **Capabilities**: 12 multi-task heads
- **Parameters**: ~149M total

### Dependencies
- **Python**: >=3.9
- **PyTorch**: >=2.0.0
- **Transformers**: >=4.30.0

## 📦 Installation

```bash
pip install familyos-ultrabert=={version}
```

## 🔄 Migration Guide

[Add migration instructions if needed]

## 🙏 Acknowledgments

- Weights hosted on HuggingFace: `Pkansagra/ultrabert-weights`
- Automatic weight downloading and caching
- GlobalPointer architecture for clean NER

---

Built with care for families.
"""

    notes_path = Path(f"RELEASE_NOTES_v{version}.md")
    notes_path.write_text(notes, encoding="utf-8")
    print(f"Release notes template created: {notes_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FamilyOS UltraBERT release")
    parser.add_argument("--version", required=True, help="Version to release (e.g., 4.0.0)")
    parser.add_argument("--test-install", action="store_true", help="Test package installation")
    parser.add_argument("--skip-build", action="store_true", help="Skip package building")
    parser.add_argument("--create-notes", action="store_true", help="Create release notes template")
    parser.add_argument(
        "--generate-checksums",
        action="store_true",
        help="Generate SHA256SUMS.txt for built artifacts",
    )
    parser.add_argument(
        "--write-release-summary",
        action="store_true",
        help="Write dist/RELEASE_<version>.md for GitHub release assets",
    )
    parser.add_argument(
        "--build-bundle",
        action="store_true",
        help="Create a convenience zip bundle containing release artifacts",
    )

    args = parser.parse_args()

    print(f"Preparing FamilyOS UltraBERT v{args.version} release")
    print("=" * 60)

    # Validate structure
    if not validate_package_structure():
        sys.exit(1)

    # Update version
    update_version(args.version)

    # Build package
    if not args.skip_build:
        if not build_package():
            sys.exit(1)

        # Test package contents
        if not test_package_contents():
            sys.exit(1)

    # Test installation
    if args.test_install:
        if not test_installation():
            sys.exit(1)

    if args.generate_checksums:
        generate_checksums()

    if args.write_release_summary:
        write_release_summary(args.version)

    if args.build_bundle:
        build_release_bundle(args.version)

    # Create release notes
    if args.create_notes:
        create_release_notes(args.version)

    print("\n" + "=" * 60)
    print("Release preparation complete")
    print("\nNext steps:")
    print("1. Review and update RELEASE_NOTES.md")
    print("2. Commit changes and push to main branch")
    print(f"3. Create GitHub release with tag v{args.version}")
    print("4. GitHub Actions will automatically publish to PyPI")
    print(f"\nPackage location: {DIST_DIR}")


if __name__ == "__main__":
    main()
