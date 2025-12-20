# Edge LLM on Jetson Orin Nano

## Overview
This project demonstrates how to deploy and benchmark a **fully offline Large Language Model (LLM)** on an **NVIDIA Jetson Orin Nano Super Developer Kit**. The system runs locally using `llama.cpp` with a quantized GGUF model and provides a foundation for **edge AI applications** such as Retrieval-Augmented Generation (RAG), voice assistants, and API-based clients (mobile/web).

The goal is to evaluate **edge feasibility, performance, and system-level integration** rather than cloud-scale training.

---

## Hardware
- **Device:** NVIDIA Jetson Orin Nano Super Developer Kit  
- **Memory:** 8 GB RAM  
- **Architecture:** ARM64  
- **OS:** Ubuntu (JetPack-based)

---

## Software Stack
- **Inference Runtime:** `llama.cpp`
- **Model Format:** GGUF (quantized)
- **Model:** TinyLLaMA 1.1B Chat (Q4_K_M)
- **Languages:**  
  - Python (benchmarking, orchestration)  
  - C++ (llama.cpp backend)
- **Build Tools:** CMake, GCC
- **Utilities:** OpenSSH, SCP

---

## Project Structure
```text
edge_llm/
├── baseline.py          # Non-interactive inference + benchmarking
├── metrics/
│   └── log.csv          # Logged performance metrics
├── models/              # GGUF models (not committed)
├── rag/
│   ├── docs/            # Local knowledge sources (future work)
│   └── index/           # RAG indices (future work)
├── scripts/
│   └── setup_notes.md   # Optional setup notes
├── .gitignore
└── README.md
```


## Setup Instructions
1. Build llama.cpp on Jetson
``` bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 1
```
### Note: Due to limited RAM, a swapfile is required.

## Configure Swap (Recommended) 
``` bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Model Setup
(Recommended Model)

TinyLLaMA 1.1B Chat – Q4_K_M (GGUF)
Selected for balanced quality, speed, and memory usage on Jetson.

### Model files are intentionally excluded from GitHub. (File size too large but can be found on HuggingFace.com )
* TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
* https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
* 4-bit Q4_K_S 644 MB Q4_0 638 MB Q4_K_M 669 MB

### Running Inference (Quick Test)
~/llama.cpp/build/bin/llama-simple \
  -m ~/edge_llm/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Say hello in one sentence." \
  -n 20

## Run Benchmark 
``` bash
python3 -u ~/edge_llm/baseline.py
```
