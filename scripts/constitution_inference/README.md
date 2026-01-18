# Constitution Inference Scripts

This folder contains the inference engine for the FamilyOS Constitution Controller. It allows the model to generate parenting advice aligned with specific family values (constitutions) using a hybrid ModernBERT (Encoder) + GPT-2 (Decoder) architecture.

## 📂 Key Files

- **`release_inference.py`**: **[PRIMARY]** The production-ready, "plug-and-play" inference script. It automatically loads parameters from the constitution schema, requiring no manual tuning of flags. Use this for releases.
- **`infer_decoder_fp16.py`**: The research/development inference script with extensive CLI flags for experimentation.
- **`test_constitution_metrics.py`**: A testing suite to evaluate the Constitutional Alignment Score (CAS) of different schemas.
- **`compare_constitutions.py`**: A utility to run side-by-side comparisons of different constitutions on the same input.

## 🚀 How to Run (Production)

The `release_inference.py` script is designed to be simple. It loads the configuration from `data/constitution_schemas.json` and applies the correct steering parameters automatically.

```bash
# Interactive mode (select constitution from menu)
python scripts/constitution_inference/release_inference.py

# Direct mode
python scripts/constitution_inference/release_inference.py --text "My child is refusing to eat dinner" --constitution "gentle_parenting"
```

## ⚙️ Configuration ("Plug and Play")

We do not hardcode parameters in the script. All steering behavior is defined in `data/constitution_schemas.json`.

To tune the model's behavior, edit the JSON file. The script will automatically pick up:
- `temperature`: Creativity vs focus
- `repetition_penalty`: Prevention of loops
- `logits_strength`: How hard to steer towards keywords
- `positive_tokens`: Words to encourage
- `negative_tokens`: Words to penalize (e.g., "in this situation")
- `prefix_injection`: Whether to use hidden state steering

### Example Schema Entry
```json
"gentle_parenting": {
  "description": "Empathetic, connection-focused parenting style",
  "positive_tokens": { "feel": 0.5, "connect": 0.5 },
  "negative_tokens": { "punish": -0.5, "obey": -0.3 },
  "temperature": 0.70,
  "repetition_penalty": 1.00,
  "logits_strength": 0.0,
  "prefix_injection": false
}
```

## 🧠 Architecture

1. **Encoder (ModernBERT)**: Encodes the user's scenario and the constitution text into a shared latent space.
2. **Constitution Controller**:
   - **Logits Processing**: Biases the vocabulary during generation based on positive/negative tokens.
   - **Prefix Injection**: (Optional) Injects a learned vector prefix to steer the hidden states.
3. **Decoder (GPT-2)**: Generates the response, conditioned on the encoder output and steered by the controller.

## 🔧 Troubleshooting

- **Windows Users**: If you see a crash related to `safetensors`, ensure you are using the updated scripts which handle Windows file locking correctly (loading as numpy first).
- **Low VRAM**: The scripts run in FP16 mode by default. If you run out of memory, try closing other applications or reducing batch sizes (though inference is single-sample by default).
