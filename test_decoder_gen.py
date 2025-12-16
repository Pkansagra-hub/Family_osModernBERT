"""Test decoder generation with UltraBERT encoder."""
import torch
from familyos_ultrabert import UltraBERT

print('Loading UltraBERT...')
model = UltraBERT.load(backend='pytorch', device='cuda')
print(f'Backend: {model._backend}')

engine = model._engine
print(f'Engine type: {type(engine).__name__}')

if hasattr(engine, 'model'):
    inner = engine.model
    print(f'Inner model: {type(inner).__name__}')
    if hasattr(inner, 'heads'):
        print(f'Heads: {list(inner.heads.keys())}')
    if hasattr(inner, 'encoder'):
        print(f'Encoder: {type(inner.encoder).__name__}')

# Get tokenizer
tokenizer = engine.tokenizer if hasattr(engine, 'tokenizer') else None
print(f'Tokenizer: {tokenizer is not None}')

# Test encode
text = 'I yelled at my kids and now I feel terrible.'
print(f'\nInput: {text}')

inputs = tokenizer(text, return_tensors='pt').to('cuda')
with torch.no_grad():
    # Get encoder output
    enc_out = inner.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True)
    hidden = enc_out.last_hidden_state.half()
    print(f'Encoder hidden: {hidden.shape}, dtype: {hidden.dtype}')

# Load trained decoder weights
print('\nLoading decoder weights...')
state = torch.load('outputs/ultrabert-gen-decoder-v1/pytorch_model.bin', map_location='cuda')
cf_keys = {k.replace('heads.counterfactual.', ''): v.half() for k, v in state.items() if 'counterfactual' in k}
print(f'Decoder keys: {len(cf_keys)}')

# Get decoder head and load weights
cf_head = inner.heads['counterfactual']
cf_head.load_state_dict(cf_keys, strict=False)
cf_head.half().cuda().eval()
print('Decoder ready!')

# Generate
print('\nGenerating...')
with torch.no_grad():
    out_ids = cf_head.generate(
        encoder_hidden_states=hidden,
        encoder_attention_mask=inputs.attention_mask,
        max_new_tokens=48,
        temperature=0.7,
    )
    generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f'Generated: {generated}')
