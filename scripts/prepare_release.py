#!/usr/bin/env python
"""
Release preparation script for FamilyOS UltraBERT.

This script helps prepare releases by:
1. Validating package structure
2. Building packages
3. Testing installation
4. Checking for common issues

Usage:
    python scripts/prepare_release.py [--version VERSION] [--test-install]

Example:
    python scripts/prepare_release.py --version 4.0.0 --test-install
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil
import os


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
    print("🔍 Validating package structure...")

    package_dir = Path("familyos_ultrabert")

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
            print(f"❌ Missing required file: {file}")
            return False

    # Check that weights are excluded
    if (package_dir / "weights").exists():
        print("⚠️  Weights directory exists - this will be excluded from package")
        # Check that weights are not accidentally included
        if (package_dir / "weights" / "pytorch" / "model.safetensors").exists():
            print("✅ Weights found (will be excluded from distribution)")

    print("✅ Package structure validation passed")
    return True


def update_version(version):
    """Update version in pyproject.toml."""
    print(f"📝 Updating version to {version}...")

    pyproject_path = Path("familyos_ultrabert/pyproject.toml")
    content = pyproject_path.read_text()

    # Update version line
    import re

    new_content = re.sub(r'version = "\d+\.\d+\.\d+"', f'version = "{version}"', content)

    pyproject_path.write_text(new_content)
    print(f"✅ Updated version to {version}")


def build_package():
    """Build the package."""
    print("🔨 Building package...")

    # Clean previous builds (cross-platform)
    import shutil

    dist_dir = Path("familyos_ultrabert/dist")
    build_dir = Path("familyos_ultrabert/build")

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # Build package
    run_command([sys.executable, "-m", "build"], cwd="familyos_ultrabert")

    # Check what was built
    dist_dir = Path("familyos_ultrabert/dist")
    if not dist_dir.exists():
        print("❌ Build failed - no dist directory created")
        return False

    files = list(dist_dir.glob("*"))
    if not files:
        print("❌ Build failed - no files in dist directory")
        return False

    print("✅ Package built successfully:")
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.1f} MB)")

    return True


def test_package_contents():
    """Test that the package contains the right files and excludes weights."""
    print("📦 Testing package contents...")

    dist_dir = Path("familyos_ultrabert/dist")
    wheel_files = list(dist_dir.glob("*.whl"))

    if not wheel_files:
        print("❌ No wheel file found")
        return False

    wheel_file = wheel_files[0]

    # Extract wheel contents to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        run_command(["python", "-m", "zipfile", "-e", str(wheel_file), temp_dir])

        # Check contents
        contents = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), temp_dir)
                contents.append(rel_path)

        # Check for weights (should not be present)
        weight_files = [f for f in contents if "weights/" in f or "model.safetensors" in f]
        if weight_files:
            print(f"❌ Weights found in package (should be excluded): {weight_files}")
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
            print(f"❌ Missing required files in package: {missing}")
            return False

        print("✅ Package contents validated:")
        print(f"   - Total files: {len(contents)}")
        print(f"   - No weights included: ✅")
        print(f"   - Required modules present: ✅")

    return True


def test_installation():
    """Test installing the package in a temporary environment."""
    print("🧪 Testing package installation...")

    dist_dir = Path("familyos_ultrabert/dist")
    wheel_files = list(dist_dir.glob("*.whl"))

    if not wheel_files:
        print("❌ No wheel file found for testing")
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
            print("❌ Package import failed")
            print(f"Error: {result.stderr}")
            return False

        print("✅ Package installation and import successful")

    return True


def create_release_notes(version):
    """Generate release notes template."""
    print("📝 Generating release notes template...")

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

**Built with care for families** ❤️
"""

    notes_path = Path(f"RELEASE_NOTES_v{version}.md")
    notes_path.write_text(notes, encoding="utf-8")
    print(f"✅ Release notes template created: {notes_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FamilyOS UltraBERT release")
    parser.add_argument("--version", required=True, help="Version to release (e.g., 4.0.0)")
    parser.add_argument("--test-install", action="store_true", help="Test package installation")
    parser.add_argument("--skip-build", action="store_true", help="Skip package building")
    parser.add_argument("--create-notes", action="store_true", help="Create release notes template")

    args = parser.parse_args()

    print(f"🚀 Preparing FamilyOS UltraBERT v{args.version} release")
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

    # Create release notes
    if args.create_notes:
        create_release_notes(args.version)

    print("\n" + "=" * 60)
    print("✅ Release preparation complete!")
    print("\nNext steps:")
    print("1. Review and update RELEASE_NOTES.md")
    print("2. Commit changes and push to main branch")
    print("3. Create GitHub release with tag v{args.version}")
    print("4. GitHub Actions will automatically publish to PyPI")
    print("\nPackage location: familyos_ultrabert/dist/")


if __name__ == "__main__":
    main()
