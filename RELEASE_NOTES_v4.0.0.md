# FamilyOS UltraBERT v4.0.0 Release Notes

## 🚀 What's New

### Major Changes
- [List major changes here]

### Improvements
- [List improvements here]

### Bug Fixes
- [List bug fixes here]

## 📊 Performance Benchmarks

| Metric | v4.0.0 | Previous | Change |
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
pip install familyos-ultrabert==4.0.0
```

## 🔄 Migration Guide

[Add migration instructions if needed]

## 🙏 Acknowledgments

- Weights hosted on HuggingFace: `Pkansagra/ultrabert-weights`
- Automatic weight downloading and caching
- GlobalPointer architecture for clean NER

---

**Built with care for families** ❤️
