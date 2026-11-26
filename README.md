# SingleFileGPT.java

[![CI: SingleFileGPT Selftest](https://github.com/eyeoverthink/singlefile-gpt/actions/workflows/ci.yml/badge.svg)](https://github.com/eyeoverthink/singlefile-gpt/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A complete decoder-only transformer language model in **one pure Java file** — zero dependencies, zero build step, no `java.util.Random` (deterministic SplitMix64 PRNG), hand-derived backpropagation for every op verified against central finite differences, byte-level GPT-2 BPE, weight-only `int8` & BitNet-style ternary QAT, byte-resident inference, and a persistent spin-wait thread pool.

> **< 10-Second Verification (0 GPU, 0 Maven/Gradle, JDK 17+):**
> ```bash
> java SingleFileGPT.java selftest
> java SingleFileGPT.java bench dim=512 heads=8 layers=8 ctx=128 len=60
> ```

---

## What Is Inside One File

| Area | Implementation |
|---|---|
| **Tokenizer** | Byte-level (`vocab=256`) or byte-level BPE with exact GPT-2 UTF-8 pretokenizer (verified across all 1,112,064 Unicode scalar values + 4,020 strings against `java.util.regex`), plus optional special tokens (`<\|endoftext\|>`, `<\|pad\|>`, etc.). |
| **Architecture** | Selectable `RMSNorm`/`LayerNorm`, `SwiGLU`/`GELU`, `RoPE`/learned positions, tied embeddings, causal multi-head attention, incremental KV-cached decoding, and opt-in RoPE KV-cache trimming (`trim=1`). |
| **Training** | Hand-written backward pass for every op, `AdamW` with bit-exact checkpoint resume (`resume=`), global-norm clipping, warmup + cosine LR, gradient accumulation (`accum=`), residual dropout, disk streaming (`stream=1`). |
| **Quantization & Inference** | Weight-only symmetric `int8` and ternary `{-1, 0, +1}` (QAT via straight-through estimator + `quantize`), byte-resident forward pass (`resident=1`, ~7.4x lower weight RAM), fast row-scale (`group=0`) kernel, bit-identical multi-threaded column-split matmul. |
| **Evaluation** | Deterministic sliding-window held-out perplexity & bits/byte (`heldout`, scoring every token exactly once). |

---

## Measured Benchmarks (Honest Numbers, Including What Failed)

### 1. Real-Corpus Held-Out Perplexity (Shakespeare, 1.1 MB, 90/10 split)
`vocab=1024 specials=1 dim=128 heads=4 layers=4 ctx=128 steps=3000`, scored with `heldout` on the unseen last 10% (49,418 tokens):

| Model | Held-Out PPL | Bits/Byte | Weight Memory |
|---|---:|---:|---:|
| `fp64` | 49.49 | 2.494 | 7,425,536 B |
| `int8` (`group=64`) | 49.51 | 2.494 | 1,037,512 B |
| `int8` row-scale (`group=0`) | 49.56 | 2.495 | 1,001,796 B |
| `fp64` → post-training ternary (`group=0`) | 158.29 | 3.237 | 1,001,796 B |
| **Ternary QAT** (`precision=ternary group=0`) | **50.77** | **2.510** | **1,001,796 B** |

Honest reading: `int8` is effectively lossless; post-training ternary collapses perplexity (`49.49 → 158.29`); ternary QAT recovers to within 2.6% of `fp64` at 7.4x less memory.

### 2. Multi-Threaded Inference Sweep (Ryzen 9 3900X, Java 25, random weights, best of 3, tok/s)

| Model | T=1 | T=2 | **T=4** | T=8 | T=24 (SMT) |
|---|---:|---:|---:|---:|---:|
| dim 512, 8L (25.4M): `double` | 51.7 | 68.8 | 77.2 | 75.5 | 41.9 |
| dim 512, 8L: `int8 g64` | 118.2 | 154.4 | **135.0** | 176.0 | 104.8 |
| dim 512, 8L: `int8 row-scale` | 237.4 | 253.6 | **242.7** | **321.9** | 163.7 |
| dim 512, 8L: `ternary g64` | 108.6 | 142.8 | **135.3** | 168.3 | 101.8 |
| dim 512, 8L: `ternary row-scale` | 236.9 | 250.3 | **248.8** | 306.4 | 152.7 |

### Open Community Challenge: Diagnose the `T=4` Ryzen 9 3900X Scaling Dip
On a Ryzen 9 3900X (12C/24T, Zen 2 dual-CCD/quad-CCX), `int8 g64` drops from `154.4 tok/s` at `T=2` to `135.0 tok/s` at `T=4` before recovering to `176.0 tok/s` at `T=8` (`row-scale` shows a smaller dip: `253.6 → 242.7 → 321.9`). We already ruled out:
- column-slice alignment (`-Dfray.align=8` reproduces it),
- thread affinity / CCX placement,
- spin-wait barrier cost (measured `0.25–1.3 µs` empty dispatch).

Reproduce it on your CPU with:
```bash
java SingleFileGPT.java bench dim=512 heads=8 layers=8 ctx=128 len=60 sweep=1,2,3,4,6,8
```
If you can isolate the exact hardware counter / L3 CCX / JIT root cause and PR a pure-Java fix (no `--add-modules` incubator flags!), we will merge and credit it.

---

## Honest Limits

- **No SIMD / Vector API by design:** `jdk.incubator.vector` benchmarked ~2x faster (12.4 vs 6.0 GFLOP/s) and was intentionally rejected to keep `java SingleFileGPT.java` zero-flag and dependency-free.
- **Emulated low-precision training:** `precision=fp32|bf16|fp16|ternary` rounds weights per step in `double` arithmetic; it validates numerical behavior, not GPU Tensor Core speedups.
- **CPU educational / verification scale:** Designed for transparent, auditable small-to-medium models (up to ~25M params on CPU), not replacing `llama.cpp` or PyTorch on 70B models.

## License

Apache-2.0 — see [LICENSE](LICENSE).
