# FamilyOS UltraBERT v2 - Release Guide

## Package Contents

- **familyos_ultrabert-2.0.0-py3-none-any.whl** (~1.6GB)
  - Python code (inference engines, labels)
  - PyTorch weights (620MB pruned model)
  - ONNX models (12 quantized, ~150MB each)

## Installation Options

### Option 1: Install from Wheel File (Recommended)

Download the wheel from GitHub Releases and install:

```bash
# Download wheel from GitHub Releases
pip install familyos_ultrabert-2.0.0-py3-none-any.whl

# With PyTorch GPU support
pip install familyos_ultrabert-2.0.0-py3-none-any.whl torch

# With ONNX CPU support
pip install familyos_ultrabert-2.0.0-py3-none-any.whl onnxruntime
```

### Option 2: Install from Source (Requires Full Repo)

```bash
git clone https://github.com/Pkansagra-hub/Family_osModernBERT.git
cd Family_osModernBERT
pip install ./familyos_ultrabert[all]
```

Note: Source installation requires `modeling_studio` for PyTorch backend.

## Creating a GitHub Release

1. **Tag the release:**
   ```bash
   git tag -a v2.0.0 -m "FamilyOS UltraBERT v2.0.0"
   git push origin v2.0.0
   ```

2. **Create release on GitHub:**
   - Go to: https://github.com/Pkansagra-hub/Family_osModernBERT/releases/new
   - Select tag: v2.0.0
   - Title: "FamilyOS UltraBERT v2.0.0"
   - Upload: `familyos_ultrabert/dist/familyos_ultrabert-2.0.0-py3-none-any.whl`

3. **Release Notes Template:**
   ```markdown
   ## FamilyOS UltraBERT v2.0.0

   High-performance multi-task NLP for family communication analysis.

   ### Features
   - 12 NLP capabilities in one model
   - PyTorch (GPU) and ONNX (CPU) backends
   - 155M parameters, 15% pruned
   - < 20ms latency for 12 capabilities on GPU

   ### Capabilities
   - Sentiment (5-class)
   - Emotions (44 labels)
   - Safety (GREEN/AMBER/RED/CRISIS)
   - NER (family + general)
   - Intent, Ingress, Temporal, Relation, NLI
   - 768-dim embeddings

   ### Installation
   ```bash
   pip install familyos_ultrabert-2.0.0-py3-none-any.whl torch
   ```

   ### Quick Start
   ```python
   from familyos_ultrabert import UltraBERT
   model = UltraBERT.load()
   result = model.analyze("Mom picked up Panda from school!")
   print(result["sentiment"])  # very_positive
   ```
   ```

## Private Distribution

For private/internal distribution without publishing to PyPI:

1. **Host on private server:**
   ```bash
   pip install https://your-server.com/wheels/familyos_ultrabert-2.0.0-py3-none-any.whl
   ```

2. **Copy wheel file directly:**
   ```bash
   pip install /path/to/familyos_ultrabert-2.0.0-py3-none-any.whl
   ```

3. **Use requirements.txt with URL:**
   ```
   familyos_ultrabert @ https://your-server.com/wheels/familyos_ultrabert-2.0.0-py3-none-any.whl
   ```

## License

Proprietary - All Rights Reserved. See LICENSE for details.
