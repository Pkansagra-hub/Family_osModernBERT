"""
Constitution Matrix Test.

Runs a matrix of Scenarios x Constitutions to verify consistent adherence
across different edge cases.

Settings:
- Split Encoding: True
- Constitution Weight: 1.2
- Normalization: clamp_tight
- Temperature: 0.1
"""

import sys
import torch
import logging
import json
from pathlib import Path

# Add src to path (3 levels up from scripts/constitution_inference/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig
from transformers import AutoTokenizer, ModernBertModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(checkpoint_path, device):
    logger.info(f"Loading model from {checkpoint_path}")

    # Load Encoder
    encoder = ModernBertModel.from_pretrained(
        "answerdotai/ModernBERT-base",
        torch_dtype=torch.float16,
    )

    # Load Encoder Weights
    if (Path(checkpoint_path) / "model.safetensors").exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(Path(checkpoint_path) / "model.safetensors"))
    else:
        state_dict = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location="cpu")

    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state_dict[key.replace("encoder.", "", 1)] = value
        elif key.startswith("backbone."):
            encoder_state_dict[key.replace("backbone.", "", 1)] = value

    if encoder_state_dict:
        encoder.load_state_dict(encoder_state_dict, strict=False)

    encoder = encoder.to(device).half().eval()

    # Load Decoder
    decoder_config = GPT2DecoderConfig(
        gpt2_model_name="gpt2-medium",
        encoder_hidden_size=768,
        projection_hidden_size=1024,
        num_prefix_tokens=16,
        freeze_layers=12,
    )

    decoder = GPT2DecoderHead(config=decoder_config, encoder_hidden_size=768)

    decoder_state_dict = {}
    for key, value in state_dict.items():
        if "counterfactual" in key:
            clean_key = key.replace("heads.counterfactual.", "").replace("head.counterfactual.", "")
            decoder_state_dict[clean_key] = value

    if decoder_state_dict:
        decoder.load_state_dict(decoder_state_dict, strict=False)

    decoder = decoder.to(device).half().eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    return encoder, decoder, tokenizer

@torch.inference_mode()
def generate(text, constitution_key, encoder, decoder, tokenizer, device, split_encoding=True, constitution_weight=1.2):
    # Load constitution
    constitution_text = ""
    constitution_values = []

    if constitution_key and constitution_key != "none":
        try:
            with open("data/family_constitutions.json", "r") as f:
                data = json.load(f)
                if constitution_key in data:
                    c_data = data[constitution_key]
                    constitution_values = c_data.get("core_values", [])

                    parts = []
                    for k, v in c_data.items():
                        v_str = ", ".join(v) if isinstance(v, list) else str(v)
                        parts.append(f"{k}: {v_str}")
                    constitution_text = "; ".join(parts)
        except Exception as e:
            logger.error(f"Error loading constitution: {e}")

    # Helper to encode
    def encode(t):
        inputs = tokenizer(t, return_tensors="pt", max_length=256, truncation=True, padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state, attention_mask

    # Prepare Instruction
    advice_instruction = "Provide specific 3-sentence response"
    if constitution_values:
        values_str = ", ".join(constitution_values[:3])
        advice_instruction = f"Aligning with {values_str}, provide specific 3-sentence response"

    instruction_suffix = f" | ADVICE: {advice_instruction} ending with 'This fosters...'"

    if split_encoding and constitution_text:
        # Split Encoding
        const_hidden, const_mask = encode(f"[CONSTITUTION: {constitution_text}]")
        scen_hidden, scen_mask = encode(text + instruction_suffix)

        # Normalize Independently
        const_hidden = torch.clamp(const_hidden, -2, 2)
        scen_hidden = torch.clamp(scen_hidden, -2, 2)

        # Apply Weight
        const_hidden = const_hidden * constitution_weight

        # Concatenate
        hidden = torch.cat([const_hidden, scen_hidden], dim=1)
        attention_mask = torch.cat([const_mask, scen_mask], dim=1)
    else:
        # Joint Encoding
        prompt = text
        if constitution_text:
            prompt = f"[CONSTITUTION: {constitution_text}] {prompt}"
        prompt += instruction_suffix

        hidden, attention_mask = encode(prompt)
        hidden = torch.clamp(hidden, -2, 2)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        generated_ids = decoder.generate(
            encoder_hidden_states=hidden,
            encoder_attention_mask=attention_mask,
            max_new_tokens=96,
            temperature=0.1,
            top_p=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            eos_token_id=50282,
            pad_token_id=50283,
        )

    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Post-processing
    if "\n\n" in output:
        output = output.split("\n\n")[0]

    # Truncate at last punctuation
    if output and output[-1] not in ".!?":
        last_punct = max(output.rfind("."), output.rfind("!"), output.rfind("?"))
        if last_punct != -1:
            output = output[:last_punct + 1]

    return output

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "D:\\Modeling_studio\\outputs\\ultrabert-gen-decoder-v4"

    encoder, decoder, tokenizer = load_model(checkpoint, device)

    scenarios = [
        {
            "name": "Divorce/Alienation (Negative)",
            "text": "[Role: mom | Emotion: worried] After our divorce, my ex-husband consistently badmouthed me to our children. They started acting out and refusing to visit him."
        },
        {
            "name": "School Play (Positive)",
            "text": "[Role: mom | Emotion: proud] My son was frustrated about not being picked for the school play. Instead of telling him 'It's just a play,' I sat with him and validated his disappointment."
        },
        {
            "name": "Dadi Calls (Neutral/Boundary)",
            "text": "[Role: granddaughter | Emotion: frustrated] My Dadi frequently calls me during my busiest work hours to discuss family gossip."
        }
    ]

    constitutions = ["default", "traditional_strict", "gentle_connection"]

    print("\n" + "="*100)
    print("CONSTITUTION MATRIX TEST REPORT")
    print("Settings: Split Encoding=True, Weight=1.2, Norm=clamp_tight")
    print("="*100)

    for scenario in scenarios:
        print(f"\nSCENARIO: {scenario['name']}")
        print(f"Input: {scenario['text']}")
        print("-" * 80)

        for const in constitutions:
            print(f"  > Constitution: {const.upper()}")
            output = generate(scenario['text'], const, encoder, decoder, tokenizer, device, split_encoding=True, constitution_weight=1.2)
            print(f"    Output: {output}")
            print("-" * 40)
        print("="*100)

if __name__ == "__main__":
    main()
