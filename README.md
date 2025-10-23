# EE 5453 Class assignment. This repository contains an implementation of a KV cache through experimentation 1,2, and 3. using a small pretrained Transformer.

*AI (ChatGPT) was used in setup troubleshooting, debugging Anaconda/Python configuration errors. I learned how to properly configure and debug virtual environments, understand how CUDA errors are formed, verify how environments are interconnected, and verify deeper knowledge on my GPU and system hardware. It helped me verify the structure of the README through the best academic/industry practices. While it did provide guidance, I verified each type and confirmed the overall functionality and system design. It provided explanations on the Transformer code, and helped explain line my line while creating a template TODO for me to work on. I verified each step, reviewed each line, and saw how a Transformer is correctly processed. It provided deeper knowledge on KV cache implementation and its usage and overall impact compared to the standard design.*


### Setup
Create and activate a conda environment (Python 3.11.13) was used.
```bash
conda create -n transformer python=3.11 -y
conda activate transformer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy tqdm
```

### To verify GPU usage: 
```bash
python check_gpu.py
```







### Notes:
Code files needed: model.py, train.py, generate.py
Hardware was tested with RTX 3060 Ti with CUDA 12.4

## Results Sumamry:
The model was relied on a pretrained model of the distilgpt2 or TinyLlama.

*Screenshot of training are shown in the images/ folder.*
