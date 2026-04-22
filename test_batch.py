import time

import psutil

from llm import embed_sync

sample_docs = [
    "White Christmas | A delightful family movie with great performances",
    "The Matrix | Mind-bending sci-fi action with amazing special effects",
    "Amelie | A charming French romantic comedy set in Paris",
    "Star Wars | Epic space opera adventure with lightsabers and the Force",
    "Inception | A dream within a dream heist thriller by Christopher Nolan",
    "Titanic | Epic love story aboard the doomed ship",
    "Pulp Fiction | Tarantino's nonlinear crime masterpiece",
    "The Godfather | Classic Italian-American mafia family saga",
] * 64  # 512 docs to pick from

ram = psutil.virtual_memory()
print(f"RAM: {ram.used / 1024**3:.1f}GB used / {ram.total / 1024**3:.1f}GB total "
      f"({ram.percent}% used, {ram.available / 1024**3:.1f}GB free)\n")

for batch_size in [32, 64, 128, 256, 512]:
    docs = sample_docs[:batch_size]
    t0 = time.time()
    result = embed_sync(docs)
    elapsed = time.time() - t0
    rate = batch_size / elapsed

    ram = psutil.virtual_memory()
    print(f"batch_size={batch_size:>4} | {elapsed:.2f}s | {rate:.0f} docs/s | "
          f"RAM: {ram.percent}% ({ram.available / 1024**3:.1f}GB free)")

print("\nPick the batch_size with best docs/s and comfortable RAM usage.")
