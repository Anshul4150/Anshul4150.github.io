+++
date = '2025-12-09T17:43:51+05:30'
draft = false
title = 'AdaGradSelect: Efficient Fine-Tuning Without Low-Rank Compromises'
+++

# AdaGradSelect: Efficient Fine-Tuning Without Low-Rank Compromises

*How selective gradient-guided updates can match full fine-tuning at a fraction of the cost*

---

## Why Fine-Tuning Still Hurts (Even for Small Models)

Large Language Models have transformed NLP, but adapting them to new domains remains expensive. Full fine-tuning updates every parameter and requires storing gradients, momentum, and variance states, often doubling or tripling memory usage.

To address this, Parameter-Efficient Fine-Tuning methods like LoRA freeze the base model and inject low-rank adapters. While effective for very large models, LoRA introduces two fundamental issues:

1. Optimization is constrained to a low-rank subspace  
2. For Small Language Models, adapter overhead can outweigh benefits  

In practice, this results in slower convergence, lower peak accuracy, and limited gains for models under roughly 4 billion parameters.

This raises a key question:

**Do we really need to update all layers or even add new ones to adapt a model?**

---

## A Key Observation: Not All Transformer Blocks Matter Equally

Before proposing a new method, we conducted a simple experiment:

**What if we only fine-tune the transformer blocks with the largest gradient norms?**

The results were striking:

- Updating as little as 10 percent of transformer blocks achieved accuracy comparable to full fine-tuning  
- Training time dropped by about 15 percent  
- GPU memory usage decreased significantly  

This behavior mirrors ideas from the Lottery Ticket Hypothesis, suggesting that only a small subset of parameters carries most of the learning signal.

This insight led to a natural next step:

**Can we dynamically discover and prioritize those blocks during training without paying the cost of full gradient tracking?**

---

## Introducing AdaGradSelect

AdaGradSelect is an adaptive, gradient-guided block selection strategy for efficient fine-tuning of Small Language Models.

Instead of updating all transformer blocks or restricting updates to low-rank adapters, AdaGradSelect:

- Selectively updates a small subset of transformer blocks  
- Adapts block selection dynamically during training  
- Maintains full parameter expressivity within selected blocks  

---

## Core Idea: Fine-Tuning as a Selection Problem

Each transformer block can be viewed as an arm in a multi-armed bandit:

- Some blocks contribute more to task adaptation  
- We want to identify and exploit them efficiently  
- But still allow early exploration  

AdaGradSelect formalizes this intuition using a principled exploration–exploitation strategy.

---

## How AdaGradSelect Works

AdaGradSelect operates in two tightly coupled phases.

---

### Phase 1: Exploration During Early Training

During the first epoch, the algorithm balances:

- Exploration by selecting blocks with the highest gradient norms  
- Exploitation by sampling blocks based on historical importance  

This is implemented using an epsilon-greedy strategy:

- Exploration probability starts high  
- Epsilon decays exponentially within the first epoch  
- After epoch one, exploration is disabled  

---

### Phase 2: Exploitation via Dirichlet Sampling

Block importance is tracked using update frequency counts.

At each training step:

1. Frequencies are converted into Dirichlet distribution parameters  
2. A probability distribution over blocks is sampled  
3. The top k percent of blocks are selected for updating  

This approach:

- Prioritizes historically important blocks  
- Maintains diversity without heuristics  
- Stabilizes training dynamics  

---

### Why This Is Different from LoRA

Unlike LoRA:

- No low-rank bottleneck  
- No architectural changes  
- No additional forward-pass overhead  

Unlike full fine-tuning:

- Far fewer parameters are updated  
- Optimizer memory is reduced  
- Training is faster  

---

## Memory Efficiency Through Dynamic Optimizer State Management

A major hidden cost in fine-tuning is optimizer state memory.

AdaGradSelect introduces dynamic optimizer state residency:

- Optimizer states live on CPU memory by default  
- States for selected blocks are prefetched to GPU  
- States for inactive blocks are evicted back to CPU  
- Transfers are asynchronous and overlap with computation  

As a result:

- GPU memory scales with selected parameters only  
- Up to 35 percent GPU memory reduction is achieved  
- No loss in model accuracy  

---

## Experimental Setup

AdaGradSelect was evaluated on mathematical reasoning tasks, where Small Language Models typically underperform without fine-tuning.

### Models
- Qwen2.5-0.5B  
- LLaMA-3.2-1B  
- Phi-4 Mini-3.8B  

### Training
- MetaMathQA-40K  
- BF16 precision  

### Evaluation
- GSM8K  
- MATH  

Baselines included full fine-tuning and LoRA with ranks 128 and 256.

---

## Results: Matching Full Fine-Tuning, Beating LoRA

### Accuracy

Across all models:

- AdaGradSelect matches or slightly exceeds full fine-tuning  
- It consistently outperforms LoRA, even at high ranks  
- Average gains of around 3 percent on GSM8K compared to LoRA  

Notably, on LLaMA-3.2-1B, updating just a single block per step still produced meaningful gains, highlighting the method’s efficiency.

---

### Convergence Behavior

- AdaGradSelect converges faster and more stably than LoRA  
- Higher block percentages closely track full fine-tuning  
- LoRA exhibits slower convergence and higher variance  

---

### Training Efficiency

Compared to full fine-tuning:

- Approximately 12 percent faster training  
- Around 35 percent lower GPU memory usage  

Compared to LoRA:

- Higher accuracy at similar or lower memory budgets  
- No low-rank expressivity ceiling  

---

## Broader Impact

AdaGradSelect challenges a core assumption in efficient fine-tuning:

**Efficiency does not require restricting expressivity.**

Its implications include:

- Practical deployment of Small Language Models on limited hardware  
- Faster experimentation cycles for researchers  
- Better domain adaptation without architectural hacks  
- A shift toward adaptive training policies rather than parameter compression  

Efficiency becomes a selection problem, not a compression problem.

---

## Limitations and Future Work

- Performance depends on CPU–GPU bandwidth for optimizer state transfers  
- Scaling to very large models may require high-speed interconnects  
- Current selection operates at block granularity, leaving room for finer control  

---

## Takeaway

AdaGradSelect demonstrates that we do not need to choose between performance and efficiency.

By dynamically selecting which parts of a model to update rather than restricting how they are updated, it is possible to:

- Match full fine-tuning performance  
- Outperform popular PEFT methods  
- Significantly reduce training time and memory usage  

For anyone working with Small Language Models, AdaGradSelect offers a compelling and practical alternative.


📄 **Full paper (arXiv):** https://www.arxiv.org/pdf/2512.15764
