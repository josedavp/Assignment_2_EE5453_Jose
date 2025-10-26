# EE 5453 – Assignment 2: KV Caching in Large Language Models (LLMs)

*AI (ChatGPT) was used for setup troubleshooting and debugging Anaconda/Python configuration errors.  
I learned how to properly configure and debug virtual environments, understand how CUDA errors occur, verify how environments interconnect, and analyze GPU/system hardware in detail.  
It also guided me on best academic and industry practices for structuring the README and code organization.  
While it provided guidance, I verified every command, confirmed functionality, and reviewed the Transformer code line by line using the provided TODO templates.  
This helped me understand KV-cache implementation, its usage, and its performance impact compared to standard Transformer inference.*

---

This project implements and analyzes **Key-Value (KV) caching** in Transformers to study how caching improves inference performance in large language models (LLMs).  
All experiments were performed on `distilgpt2` using PyTorch and Hugging Face Transformers.

---

## Installation and Environment Setup

Create and activate the virtual environment:
```bash
conda create -n kvcache python=3.11 -y
conda activate kvcache

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate pandas numpy tqdm jupyter
```

### verify GPU capability
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```


## Running Code
### Experiment 1
```bash
python run_model.py --use_cache 1 --batch_size 1 --prompt_len 256 --max_new_tokens 128 --csv logs.csv
python run_model.py --use_cache 0 --batch_size 1 --prompt_len 256 --max_new_tokens 128 --csv logs.csv
```

### Experiment 2
```bash
python run_model.py --use_cache 1 --batch_size 1 --prompt_len 128 --max_new_tokens 128 --csv logs.csv
python run_model.py --use_cache 1 --batch_size 2 --prompt_len 128 --max_new_tokens 128 --csv logs.csv
python run_model.py --use_cache 1 --batch_size 4 --prompt_len 128 --max_new_tokens 128 --csv logs.csv
```

### Experiment 3
```bash
python run_model.py --use_cache 1 --batch_size 1 --prompt_len 32  --max_new_tokens 128 --csv logs.csv
python run_model.py --use_cache 1 --batch_size 1 --prompt_len 128 --max_new_tokens 128 --csv logs.csv
python run_model.py --use_cache 1 --batch_size 1 --prompt_len 256 --max_new_tokens 128 --csv logs.csv
```


## Analysis Results
### Open Juptyer notebook
```bash
jupyter notebook analysis.ipynb
```

## Obersevations
* Experiment 1: Caching improved throughput and reduced latency
* Experiment 2: Larger batches increased throughput and meemory linearly
* Experiment 3: Cache memory increased linearly; latency grew modestly (O(S))

### Hardware Used:
* CPU: Intel Core i7-12700K
* GPU: NVIDIA RTX RTX 3060 Ti
* RAM: 32 GB DDR4
* CUDA: 12.4

## Results Sumamry:
The model was relied on a pretrained model of the distilgpt2 
*Screenshot of training are shown in the images/ folder.*
