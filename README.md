# ACS V3.1 — Autonomous Cognitive System 🧠 ⚙️

> A production-ready, local-first pipeline for autonomous Reinforcement Learning from Human Feedback (RLHF) and Continuous AI Evolution.

Welcome to the **Autonomous Cognitive System (ACS V3.1)**. This repository provides a complete, end-to-end architecture structurally designed for one overarching goal: **letting a local LLM improve its own reasoning abilities autonomously** without relying on constant external cloud APIs or catastrophic human-bottlenecked data labeling. 

---

## 📑 Table of Contents
1. [Introduction & Philosophy](#-introduction--philosophy)
2. [Core Features](#-core-features)
3. [System Architecture: What is What?](#-system-architecture-what-is-what)
4. [Prerequisites & Requirements](#-prerequisites--requirements)
5. [Complete "How to Start" Guide](#-complete-how-to-start-guide)
   - [Phase 1: Environment Setup](#phase-1-environment-setup)
   - [Phase 2: Generating Traces (The Cold Start)](#phase-2-generating-traces-the-cold-start)
   - [Phase 3: The Evolution Loop](#phase-3-the-evolution-loop)
6. [CLI Capabilities (`acs.py`)](#-cli-capabilities)
7. [Anti-Overfitting Protections](#-anti-overfitting-protections)

---

## 🧭 Introduction & Philosophy
Most language models hit a plateau. They learn to repeat what they've seen. ACS breaks this plateau by using a technique called *Inference-Time Scaling (Best-of-N)* combined with an unforgiving *Adversarial Quality Filter*. 

By forcing the system into `DEEP` cognitive modes, harvesting the best "thoughts," injecting deliberate noise into them, and distilling them back into the model via LoRA adapters, the system creates an upward spiral of self-improvement.

**In short: the model converses with itself, strictly grades its own homework, compiles the passing grades into a textbook, and trains itself on it.**

---

## 🌟 Core Features
- **Zero Human Bottleneck:** Generates and filters training data completely autonomously.
- **Strict Quality Gating (The 0.65 Floor):** The system ruthlessly filters its own outputs using adversarial independent evaluations, consistency grading, and algorithmic logic bounds.
- **Automated Anti-Overfitting (`DatasetBuilder`):** Employs dynamic noise injection (deliberate errors + corrections) and structural trace reshuffling to prevent sequence memorization.
- **Evaluator Drift Detection:** Built-in safeguards that constantly verify the foundational baselines of the model preventing downstream semantic collapse.
- **Single-Command Orchestration:** `python acs.py evolve` handles filtering, dataset compilation, baseline pre-evaluations, LoRA training simulation, regression checks, and model promotion!

---

## 🏗 System Architecture: What is What?
If you are a developer looking to understand or modify the repository, here is exactly what each script does:

### 1. `acs.py` (The Central Nervous System)
This is the primary CLI tool. You will rarely run other scripts directly. It parses commands, builds the component stack, and acts as the router to functions like chat, dataset generation, inspections, and evolution orchestrations.

### 2. `core_acs.py` & `executive.py`
These handle the actual interaction with the Local LLM context. `core_acs.py` manages the network requests (connecting to `localhost:1234`), system prompts, and the critical `DEEP` trace JSON parsers. `executive.py` routes user intent to determine if a query requires deep reasoning or just a quick chat.

### 3. `quality_filter.py` (The Bouncer)
The most critical component. It looks at every raw generated trace and subjects it to four tests:
- *Ground Truth validation* (for math/code).
- *Adversarial Evaluator Checking* (Forces the LLM to grade itself stringently; requires `> 0.65` baseline).
- *Deduplication & Novelty Decay*.
- *Difficulty Calibration* to prevent verbosity gaming. 

### 4. `dataset_builder.py` (The Anti-Overfitting Engine)
Takes accepted traces and formats them for training. It enforces massive diversity limitations:
- **Domain Caps:** No domain (e.g., Mathematics) can exceed 20% of the dataset.
- **Noise Injection:** Deliberately corrupts 25% of reasoning steps and forces the model to document a "correction," teaching the model course-correction.
- **Centroid Compression:** Blocks identical traces from flooding the weights. 

### 5. `evolve.py` (The Loop)
The continuous integration engine for the AI. It takes raw data, passes it to the `QualityFilter`, sends the survivors to the `DatasetBuilder`, evaluates the baseline model's logic, kicks off the LoRA `trainer.py`, runs regression tests against the new model, and conditionally promotes the model in the `ModelRegistry`.

### 6. `generate_traces.py` & `generate_mocks.py`
Helper tools. `generate_traces.py` runs in the background and asks the system automated questions so you don't have to chat with it manually for 25 hours to collect data.

---

## 💻 Prerequisites & Requirements
To run out-of-the-box, ensure you have the following:

1. **Python 3.10+** (Run inside a virtual environment `venv`)
2. **Local LLM Server:** The system defaults to querying `http://localhost:1234/v1`. Start up **LM Studio**, **Ollama**, or **vLLM** and serve an instruction-tuned model. 
   > *Note: For standard evaluation architectures, a model like `Qwen-2.5` (7B or 14B) or `Llama-3` running locally is heavily recommended.*
3. **Dependencies:** Ensure standard web/NLP libs are available (e.g.`httpx`).

---

## 🚀 Complete "How to Start" Guide

### Phase 1: Environment Setup
First, prepare your environment and ensure the local LLM server is responding.

```bash
# 1. Clone the repository
git clone git@github.com:mailtoharutyunyan/autonomous-cognitive-system.git
cd autonomous-cognitive-system

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Ensure your local LLM is running on port 1234!
# Test it using a simple chat command:
python acs.py chat
```

### Phase 2: Generating Traces (The Cold Start)
A machine learning model cannot train on data it doesn't have. ACS needs roughly **500 high-quality traces** to trigger an evolutionary loop safely. 

Because the `QualityFilter` is unforgiving, generating 500 *accepted* traces likely requires generating ~2,000 *raw* traces. 

**How to generate traces:**
You can just chat with the system directly (`python acs.py chat`), but we recommend running the automated background generator:
```bash
# This will continually prompt the model with complex questions in the background
nohup python generate_traces.py > acs_trace_generator.log 2>&1 &
```
Let this script run for hours/overnight until you have accumulated a substantial pool of raw traces. 

### Phase 3: The Evolution Loop
Once you feel you have enough data, you initiate the fully automated pipeline. 

```bash
python acs.py evolve
```

**What happens when you run this?**
1. **Filter Mapping:** Extracts raw traces and passes them through `quality_filter.py`. Borderline logic is discarded.
2. **Dataset Creation:** Compiles `dataset_builder.py` with shuffled structures and injected noise.
3. **Pre-Eval Baseline:** Benchmarks your current active model across deterministic logic tests.
4. **LoRA Fine-tuning:** (Stubbed/simulated unless hooked precisely into peft/transformers) Generates the next model weight.
5. **Regression Verification:** Evaluates the new model *against* the pre-eval baseline. If Tier-1 logic degraded by >2%, the new model is instantly thrown in the trash. 
6. **Promotion:** You are asked to confirm setting the newly generated weights as Active.

If you just want to verify the logic pipeline without training or if you lack 500 traces but want to test the architecture, run it in dry-mode:
```bash
python acs.py evolve --dry-run
```

---

## 🛠 CLI Capabilities

The `acs.py` entrypoint has commands covering all data/model capabilities:

#### Data Interactions
- `python acs.py think "Explain relativity"` — Fires a Best-of-3 DEEP reasoning trace manually. 
- `python acs.py chat` — Manual multi-turn looping.

#### Inspection & Traces
- `python acs.py traces --stats` — View how many raw traces you have collected and their domains.
- `python acs.py traces --quality-report` — View your true acceptance rate (Aim for 15-30%!).
- `python acs.py traces --evaluator-drift` — Forces the current model to re-evaluate historical data to see if your system prompt / model alignment is statistically degrading over time.

#### Dataset & Engine Health
- `python acs.py dataset --validate` — Ensure your constructed datasets meet the strict Domain and Noise constraints. 
- `python acs.py model --list` — Prints out all active, archived, and rejected local models historically handled by the ecosystem. 

---

## 🛡 Anti-Overfitting Protections
Why do LLMs generally suffer "Model Collapse" when trained on synthetic data? 
Because they memorize tone and grammatical styling rather than *structural reasoning*. ACS prevents this. 
1. **Trace Shuffling:** We mathematically locate logically disconnected, parallel steps within a trace and arbitrarily shuffle their order in the text before training. 
2. **Noise Injection:** We artificially prompt the model down incorrect assumption paths and inject a `[WAIT, CONTRADICTION FOUND]` tag, forcing it to recover.
3. **Centroid Stylistic Collapse:** We parse the mathematical ratio between sub-tasks and raw steps to locate traces formatted identically. We arbitrarily cull these to cap style-homogenization. 

---

> Built for the future of Offline Autonomous Reasoning Models. 
> *Authored & Maintained by Arayik Harutyunyan*
