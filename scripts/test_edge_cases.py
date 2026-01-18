"""
Edge Case Stress Test for UltraBERT-Gen Decoder.

This script runs a battery of test cases across different sentiments and domains
to verify model robustness, safety, and quality.

It uses the optimized inference settings:
- Temperature: 0.2
- Normalization: clamp_tight ([-2, 2])
- Candidates: 1 (for speed) or 3 (for diversity)
"""

import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig
from transformers import AutoTokenizer, ModernBertModel

# Import inference functions from the main script
# We'll just import the module to reuse the loading logic if possible,
# but since it's a script, we might need to copy/paste or import carefully.
# Let's just reimplement the minimal loading logic here to be safe and standalone.

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

    # Load Tokenizer from Checkpoint (ModernBERT tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    return encoder, decoder, tokenizer

@torch.inference_mode()
def generate(text, encoder, decoder, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        encoder_out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoder_out.last_hidden_state

        # Apply clamp_tight normalization
        hidden = torch.clamp(hidden, -2, 2)

        generated_ids = decoder.generate(
            encoder_hidden_states=hidden,
            encoder_attention_mask=attention_mask,
            max_new_tokens=96,
            temperature=0.2,
            top_p=0.9,
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

    test_cases = [
        {
            "name": "Divorce/Alienation (Negative)",
            "text": "[Role: mom | Emotion: worried] After our divorce, my ex-husband consistently badmouthed me to our children. They started acting out and refusing to visit him.",
            "expected": "Mediation, child-centric communication"
        },
        {
            "name": "Alzheimer's Care (Negative)",
            "text": "[Role: daughter-in-law | Emotion: helpless] My father-in-law has early-stage Alzheimer's. I'm nervous about his long-term health and feel helpless.",
            "expected": "Shared calendar, routines, support"
        },
        {
            "name": "School Play (Positive)",
            "text": "[Role: mom | Emotion: proud] My son was frustrated about not being picked for the school play. Instead of telling him 'It's just a play,' I sat with him and validated his disappointment.",
            "expected": "Analysis of why it worked (validation)"
        },
        {
            "name": "Bedtime Routine (Positive)",
            "text": "[Role: parent | Emotion: happy] We implemented a consistent family quiet time before bed. My children are now falling asleep much easier.",
            "expected": "Reinforce routine/security"
        },
        {
            "name": "Dadi Calls (Neutral/Boundary)",
            "text": "[Role: granddaughter | Emotion: frustrated] My Dadi frequently calls me during my busiest work hours to discuss family gossip.",
            "expected": "Gentle boundary setting, alternative time"
        },
        {
            "name": "Babysitting Guilt (Neutral/Boundary)",
            "text": "[Role: friend | Emotion: guilty] My friend kept asking me to babysit her kids last minute. I always said yes because I felt guilty.",
            "expected": "Saying no politely, respecting own time"
        }
    ]

    print("\n" + "="*80)
    print("EDGE CASE STRESS TEST REPORT")
    print("="*80)

    for case in test_cases:
        print(f"\nTEST CASE: {case['name']}")
        print(f"Input: {case['text']}")
        print("-" * 40)

        output = generate(case['text'], encoder, decoder, tokenizer, device)

        print(f"Output: {output}")
        print("-" * 40)
        print(f"Expected Theme: {case['expected']}")
        print("="*80)

if __name__ == "__main__":
    main()
