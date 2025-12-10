+++
date = '2025-11-18T17:43:51+05:30'
draft = false
title = 'LLM inference survey'
+++

# Efficient LLM Inference: Optimizing KV Cache and Quantization

**Author:** Anshul Kumar

Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of applications. However, as the demand for these models grows, so does the critical need for fast and efficient inference. The immense size and computational requirements of models like Llama or GPT present significant challenges for deployment, particularly on resource-constrained devices.

This post dives into the state-of-the-art techniques used to tackle these challenges, focusing on two main pillars: **KV Cache Optimization** and **Inference-Time Quantization**.

---

## The Bottleneck: Autoregressive Decoding

Generative models typically use autoregressive decoding. This process faces a specific bottleneck: the **KV (Key-Value) Cache**. As the sequence length grows, the memory required to store past keys and values increases, leading to high latency and memory exhaustion.



To address this, researchers have developed several optimization strategies.

### 1. Novel Attention Architectures
Modifying the attention mechanism itself is one of the most effective ways to reduce the memory footprint.

* **Multi-Query Attention (MQA):** Instead of using separate heads for queries, keys, and values, MQA shares a *single* key-value head across all query heads. This drastically reduces KV-cache size (cutting memory by a factor of $H$ for $H$ heads) and improves throughput.
* **Grouped-Query Attention (GQA):** A middle ground between MQA and standard Multi-Head Attention (MHA). It groups query heads to share specific KV heads. Used in models like **Llama-2-70B**, it offers a balance—achieving near-MHA quality with MQA-like speed.

* **Sliding-Window Attention:** This restricts attention to a fixed local window (e.g., SWAT, SWAN-GPT). By ignoring very old tokens, computational complexity drops from quadratic to linear, significantly saving memory at the cost of some global context.

### 2. KV Cache Compression
Do we really need every single past token? Research suggests only a subset of past tokens significantly influence future generations.

* **H$_2$O (Heavy-Hitter Oracle):** Identifies "heavy-hitter" tokens—those that appear frequently or co-occur often. It keeps these important tokens and evicts the rest, reducing KV size by ~5$\times$ with negligible accuracy loss.
* **Scissorhands:** Tracks the "persistence of importance." It hypothesizes that if a token was important in the past, it will likely remain important. This allows for a fixed-size cache (up to 5$\times$ reduction).
* **FastGen:** Profiles specific attention heads to see which ones require full context and which can be compressed on the fly.
* **ZipCache:** Applies quantization specifically to the tokens within the cache, achieving notable compression with minimal accuracy drops.

### 3. Efficient Memory Management
How we store the cache in GPU memory matters just as much as what we store.

* **PagedAttention (vLLM):** Inspired by OS virtual memory, this technique breaks the KV cache into non-contiguous fixed-size pages. It eliminates memory fragmentation and allows requests to share memory slots, increasing throughput by 2–4$\times$.

* **RadixAttention (SGLang):** Optimized for agents or Chain-of-Thought tasks where prompts share prefixes. It stores the KV cache in a Radix tree, allowing new requests to reuse the computation of the longest matching prefix from a previous request.

---

## Inference-Time Quantization

Quantization involves mapping high-precision floating-point values (like FP16) to discrete low-bit representations (like INT8 or INT4). This is critical for fitting large models onto consumer GPUs and mobile devices.

### Quantization Formats
* **INT8 (8-bit Integer):** A balance of compression and accuracy. Requires handling "outliers" (activation spikes) to prevent degradation.
* **INT4 (4-bit Integer):** massive memory savings but susceptible to accuracy loss. Techniques like **GPTQ**, **AWQ**, and **NF4** (used in QLoRA) are essential here.
* **FP8 (8-bit Float):** Newer hardware (like H100s) supports this. It handles dynamic ranges better than INT8.


### Key Techniques
1.  **Weight-Only vs. Full Quantization:**
    * *Weight-Only:* Compresses parameters (e.g., W4A16). Good for memory, but requires dequantization during compute.
    * *Full Quantization (W8A8):* Quantizes both weights and activations. This enables the use of fast integer arithmetic units on GPUs.

2.  **SmoothQuant & Outlier Mitigation:**
    Large language models often have "outlier" features with massive values that break standard quantization. **SmoothQuant** solves this by mathematically "smoothing" the activation spikes—shifting the difficulty from activations to weights, which are easier to quantize.

3.  **Hardware-Awareness:**
    New frameworks like **QuantX** and **HALO** (2025) optimize quantization strategies based specifically on the target hardware's constraints (e.g., specific multiplier types or memory bandwidth).

---

## Future Directions

While we have made massive strides, challenges remain. **Outlier sensitivity** and **error accumulation** across deep layers still plague ultra-low-bit (sub-4-bit) quantization.

The future of efficient inference lies in:
* **Robust sub-4-bit quantization** (1-bit or 2-bit architectures like BitNet).
* **Calibration-free methods** that don't require sample data.
* **Co-optimization** of pruning, quantization, and architectural changes (like Mixture-of-Experts) to run trillion-parameter models on commodity hardware.

---

### References & Further Reading

* **MQA:** *Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need.*
* **H$_2$O:** *He, J. et al. (2024). H$_2$O: Accelerating Transformer Decoding via Heavy-Hitter Oracle.*
* **vLLM:** *Roberts, D. et al. (2023). vLLM: Easy, Fast, and Memory-Efficient Inference for LLMs.*
* **SGLang:** *Zheng, H. et al. (2024). SGLang: Structured Generation with Declarative Syntax.*
* **GPTQ:** *Frantar, E. et al. (2022). GPTQ: Accurate Post-Training Quantization for Generative Transformers.*
* **AWQ:** *Lin, T. et al. (2023). AWQ: Activation-aware Weight Quantization for LLMs.*
