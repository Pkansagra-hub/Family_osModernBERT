"""Test the new Client with auto-warmup."""

from familyos_ultrabert import Client, __version__
import time

print(f"FamilyOS UltraBERT v{__version__}")
print("=" * 50)

# Create client with verbose mode to see warmup
print("\n1. Creating client with auto-warmup...")
client = Client(verbose=True)

print("\n2. Testing first real call (should be fast!)...")
start = time.perf_counter()
result = client.analyze("Mom picked up the kids from school!")
elapsed = (time.perf_counter() - start) * 1000
print(f"   First call: {elapsed:.1f}ms")
print(f"   Sentiment: {result.sentiment}")
print(f"   Safety: {result.safety}")
print(f"   Emotions: {result.emotions[:3]}")

print("\n3. Testing convenience methods...")
print(f"   is_safe('I love my family'): {client.is_safe('I love my family')}")
print(f"   is_crisis('I want to hurt myself'): {client.is_crisis('I want to hurt myself')}")
print(f"   get_sentiment('Great day!'): {client.get_sentiment('Great day!')}")

print("\n4. Latency stats after calls:")
print(f"   {client.stats}")

print("\n5. Health check:")
health = client.health_check()
for k, v in health.items():
    print(f"   {k}: {v}")

print("\n6. Client repr:")
print(f"   {client}")

print("\nSUCCESS - Client API working!")
