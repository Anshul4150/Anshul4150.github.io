+++
date = '2025-11-09T17:43:51+05:30'
draft = false
title = 'Model Editing vs Fine-Tuning (Telecom)'
+++
# Model Editing vs Fine-Tuning (Telecom)

## 🧩 The Core Idea

**Model Editing** is a *surgical fix* — change one or a few factual details directly inside the model.  
**Fine-Tuning** (e.g., **DAPT**, **LoRA**) is *re-education* — train the model on more telecom data so it learns the domain thoroughly.

---

## ⚙️ Worked Example: “What is the Paging Sequence in 5G?”

### LLaMA 3.1 8B Answer (via Ollama)

> In 5G, there are several paging sequences used for different purposes. Here are a few:  
> **S-NR PSS**, **PRACH**, **P-RNTI**, etc.  
> *(incorrect — confuses synchronization and random access sequences with paging).*

### GPT-5 Answer (via ChatGPT)

Excellent question — the **paging sequence in 5G** (as defined by **3GPP TS 23.501**, **23.502**, and **38.304**) is a signaling process used to locate and alert an idle **UE** when downlink data or signaling is waiting (e.g., an incoming call or SMS).

#### Overview

Paging in 5G is used to:
- Inform a UE in `RRC_IDLE` or `RRC_INACTIVE` state that the network has pending data.  
- Bring the UE back to `RRC_CONNECTED` state so it can receive it.  

Paging is initiated by the **5G Core Network (5GC)** and sent via **NG-RAN (gNB)** using the NG interface.

#### Entities Involved

| Entity | Function in Paging |
|---------|-------------------|
| **AMF** | Initiates paging when downlink data or signaling arrives for an idle UE |
| **gNB** | Transmits paging messages over the air to reach the UE |
| **UE** | Monitors the paging channel periodically and responds if a message matches its identity |

#### Step-by-Step Paging Sequence

1. **Paging Trigger (Core Network Side)**  
   AMF receives notification of downlink data from SMF and initiates `NGAP PAGING REQUEST`.

2. **AMF Determines Where to Page**  
   Based on UE’s last known tracking area (TAI List) and location info.

3. **NGAP PAGING MESSAGE to gNB**  
   Includes UE identity, DRX cycle, and paging cause.

4. **gNB Broadcasts Paging Message**  
   Maps to correct Paging Occasion and Frame; sends via `PDCCH` and `PDSCH`.

5. **UE Monitors Paging Occasions**  
   Wakes up per DRX cycle, listens for paging, checks identity.

6. **UE Response**  
   - `RRC_IDLE` → initiates RRC Connection  
   - `RRC_INACTIVE` → sends RRC Resume Request  

#### Optional Enhancements

| Feature | Description |
|----------|-------------|
| **UE Paging Priority** | High-priority UEs paged first |
| **CN Assistance Info** | Optimizes paging based on mobility |
| **Local Paging** | gNB can page locally |
| **Extended Paging DRX** | Power saving for IoT UEs |



#### Key 3GPP References
- TS 23.501 — System Architecture for 5GS  
- TS 23.502 — Procedures for 5GS  
- TS 38.304 — NR; Procedures for Paging  
- TS 38.331 — RRC protocol (Paging message)

---

## 🩺 Observed Model Outputs

| Model | Observation |
|--------|-------------|
| **LLaMA 3.1 8B** | Partly incorrect — confuses PRACH/PSS with paging sequences |
| **GPT-5** | Correct — references right 3GPP TS and provides procedural flow |

---

## 🧠 Diagnosis

- **If the error** is a wrong TS reference or single factual mismatch → ✅ *Model Editing*.  
- **If the model** systematically misunderstands the entire procedure → 🧩 *Fine-Tuning* (DAPT/LoRA).

---

## 🩹 When to Do Model Editing

Use when you find **specific factual errors**:

| Example | Description |
|----------|-------------|
| Wrong TS number | Model says *TS 38.300* instead of *TS 38.304* |
| Wrong network role | Confuses AMF ↔ SMF |
| Renamed parameter | Spec update changed a field name |

```python
edit_request = {
    "prompt": "The {} is defined in",
    "subject": "5G paging procedure",
    "target_new": "3GPP TS 38.304"
}
```


Tools: ROME, MEMIT, MEND, adapter-based patching
Effort: Minutes — one fact at a time.

🧩 When to Do Fine-Tuning

Use when you need broad, consistent improvement in telecom reasoning or prose.

Example	Description
Shallow/incomplete answers	Model struggles across telecom topics
Inconsistent terminology	Wrong TS references, poor procedural flow
Domain reasoning gaps	Needs multi-step explanation ability

```python
{
  "instruction": "Explain the paging sequence in 5G as per 3GPP standards.",
  "input": "",
  "output": "In 5G, paging is a procedure used by the 5G Core Network to reach a UE in RRC_IDLE or RRC_INACTIVE. The AMF initiates paging via NGAP PAGING to gNBs. The gNB broadcasts the RRC Paging message. The UE monitors its paging occasion and responds. Key specs: TS 23.501, 23.502, 38.304."
}
```


Tools: LoRA, Adapters, DAPT
Effort: Hours → days; many examples.


### High-Level Comparison
| Dimension           | Model Editing              | Fine-Tuning / DAPT               |
| ------------------- | -------------------------- | -------------------------------- |
| **Intent**          | Local, targeted correction | Global fluency, domain reasoning |
| **Data Required**   | Minimal (1–few facts)      | Large domain corpus              |
| **Compute & Time**  | Low (seconds–minutes)      | Moderate–high (hours–days)       |
| **Scope of Impact** | Narrow                     | Broad                            |
| **Generalization**  | Weak on paraphrases        | Stronger generalization          |
| **Best When**       | One or few facts are wrong | Need domain coherence            |
| **Reversibility**   | Easy rollback              | Harder to revert                 |


## 🧮 Practical Decision Heuristic (for Successive Releases)

1. **Start with Fine-Tuning** → build a strong telecom-capable base model.

2. **For each new release:**
   - **Audit changes**
   - If localized factual updates → **Model Editing**
   - If systemic or large updates → **Re-Fine-Tune (or mix)**

3. **Monitor metrics:**
   - **Efficacy** → Are new facts correct?
   - **Generalization** → Do paraphrases work?
   - **Locality** → Did unrelated facts stay unchanged?

---

*Written by **Anshul** — exploring how AI agents and model adaptation methods can bring telecom intelligence closer to real-world automation.*
