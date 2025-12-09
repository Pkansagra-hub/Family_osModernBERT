"""Benchmark embedding query performance."""

import torch
import time
import json
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from transformers import AutoTokenizer
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
import torch.nn.functional as F

# Load model
checkpoint = Path('D:/Modeling_studio/outputs/modernbert-v2-for-v3-transfer/checkpoint-18000')
model = ModernBertMultiTaskModel.load_checkpoint(checkpoint_path=checkpoint, device='cuda')
tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
model.eval()

# Load corpus from triplets (use anchors as corpus)
data_dir = Path('D:/Modeling_studio/data/familyos/embeddings/silver_synthetic')
corpus = []
for shard in sorted(data_dir.glob('triplets_*.jsonl'))[:10]:
    with open(shard) as f:
        for line in f:
            t = json.loads(line)
            corpus.append(t['anchor'])
            if len(corpus) >= 1000:
                break
    if len(corpus) >= 1000:
        break

print(f'Corpus size: {len(corpus)} documents')

# Embed corpus
print('\nEmbedding corpus...')


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to('cuda')
    with torch.no_grad():
        out = model(**inputs, capability='embedding')
    return out.logits[0]


def get_embeddings_batch(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=512, padding=True).to('cuda')
        with torch.no_grad():
            out = model(**inputs, capability='embedding')
        embeddings.append(out.logits.cpu())
    return torch.cat(embeddings, dim=0)


start = time.perf_counter()
corpus_embeddings = get_embeddings_batch(corpus)
embed_time = time.perf_counter() - start
print(f'Corpus embedding time: {embed_time:.2f}s ({len(corpus)/embed_time:.1f} docs/sec)')
print(f'Corpus embedding shape: {corpus_embeddings.shape}')

# Query tests
queries = [
    'My grandmother is visiting next week',
    'Feeling sad about missing my family',
    'Need to pick up kids from school',
    'Planning a birthday party for dad',
    'Remember when we went to the beach last summer?'
]

print('\n=== Query Performance Benchmark ===')

# Single query latency
print('\n1. Single Query Latency (embed + search):')
times = []
for _ in range(50):
    query = queries[0]
    start = time.perf_counter()

    # Embed query
    q_emb = get_embedding(query).cpu()

    # Cosine similarity search
    sims = F.cosine_similarity(q_emb.unsqueeze(0), corpus_embeddings)
    top_k = torch.topk(sims, k=5)

    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f'   Avg: {sum(times)/len(times):.2f} ms')
print(f'   P50: {sorted(times)[25]:.2f} ms')
print(f'   P95: {sorted(times)[47]:.2f} ms')

# Breakdown: embed vs search
print('\n2. Latency Breakdown:')
embed_times = []
search_times = []
for _ in range(50):
    query = queries[0]

    # Embed
    torch.cuda.synchronize()
    start = time.perf_counter()
    q_emb = get_embedding(query).cpu()
    torch.cuda.synchronize()
    embed_times.append((time.perf_counter() - start) * 1000)

    # Search
    start = time.perf_counter()
    sims = F.cosine_similarity(q_emb.unsqueeze(0), corpus_embeddings)
    top_k = torch.topk(sims, k=5)
    search_times.append((time.perf_counter() - start) * 1000)

print(f'   Embedding: {sum(embed_times)/len(embed_times):.2f} ms')
print(f'   Search (1000 docs): {sum(search_times)/len(search_times):.3f} ms')

# Scale test
print('\n3. Search Scaling (embedding pre-computed):')
for size in [100, 500, 1000]:
    subset = corpus_embeddings[:size]
    times = []
    for _ in range(100):
        start = time.perf_counter()
        sims = F.cosine_similarity(q_emb.unsqueeze(0), subset)
        top_k = torch.topk(sims, k=min(5, size))
        times.append((time.perf_counter() - start) * 1000)
    print(f'   {size} docs: {sum(times)/len(times):.3f} ms')

# Show sample results
print('\n=== Sample Query Results ===')
for query in queries[:2]:
    q_emb = get_embedding(query).cpu()
    sims = F.cosine_similarity(q_emb.unsqueeze(0), corpus_embeddings)
    top_k = torch.topk(sims, k=3)

    print(f'\nQuery: "{query}"')
    for score, idx in zip(top_k.values.tolist(), top_k.indices.tolist()):
        print(f'  [{score:.3f}] {corpus[idx][:80]}...')
