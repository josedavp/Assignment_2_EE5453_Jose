import argparse, time, csv, os, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def percentile(values, p):
    """ Compute percentile latency for reporting. """
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s)-1) * (p/100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f])*(k-f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--csv", default="logs.csv")
    args = parser.parse_args()


    # === TODO 1: Initialize device and dtype ===
    #   Example:
    #   dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #   dtype_map = {...}
    #   torch_dtype = dtype_map[args.dtype]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32, 
    }

    torch_dtype = dtype_map[args.dtype] # controls precision for speed/memory

    torch.set_grad_enabled(False) # no gradients only inference

    # === TODO 2: Load tokenizer and model ===
    #   Hint: use AutoTokenizer.from_pretrained() and AutoModelForCausalLM.from_pretrained()
    #   Remember to assign pad_token = eos_token if not present.

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=None,           #use fp16, bf16, fp32 on args
    ).to(device)

    model.eval()

    # === TODO 3: Build input batch ===
    #   Create repeated prompts to match args.batch_size
    #   Tokenize them to length args.prompt_len
    #   Move tensors to device.

    base_text = "The quick brown fox jumps over the lazy dog. " # long string
    prompt_text = (base_text * 1000) # increases prompt length for whatever need


    batch_texts = [prompt_text] * args.batch_size #identical prompt list, per batch element

    encoded = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.prompt_len,
    ) #tokenized 

    # move tokenized batch to right device (GPU or CPU)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # === TODO 4: Optional warmup run ===
    #   Generate a few tokens to warm up GPU and avoid cold-start bias.

    #generate a few tokens once, before measurement to avoid cold-start overhead
    _ = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask, 
        max_new_tokens = 4, # 4 tokens
        do_sample = False, # deterministic output
        use_cache = bool(args.use_cache), # cache flag
    )    

    # ensure all CUDA ops (operations) complete before reset timers
    if device.type == "cuda":
        torch.cuda.synchronize() #ensures GPU finishes the run before continuing
        torch.cuda.reset_peak_memory_stats() # clears memory counters for real run starts clean

    # === TODO 5: Main timing loop ===
    #   For each token (range(args.max_new_tokens)):
    #       - Start timer
    #       - Call model.generate(..., use_cache=bool(args.use_cache))
    #       - Record latency per token
    #   Compute total_time, tokens/sec, p50, p95.

    latencies = [] # store per-token latency; stores the time for each new token generation used later for p50 (median) and p90 (tail latency)
    start_total = time.perf_counter() # Starts a timer for the entire generation process (all tokens). You’ll use this for throughput (tokens/sec)

    # growing sequence
    generated_ids = input_ids
    current_attention_mask = attention_mask

    for _ in range(args.max_new_tokens):
        t0 = time.perf_counter()

        # generate exactly one new token
        # That simulates how real autoregressive decoding works: one step at a time, always conditioned on all tokens so far
        out = model.generate(
            input_ids=generated_ids,
            attention_mask=current_attention_mask,
            max_new_tokens=1,
            do_sample=False,                 # deterministic (greedy)
            use_cache=bool(args.use_cache),  # KV cache on/off toggle
        )

        # make sure all GPU work is done before latency measurement
        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        latencies.append(t1 - t0) #We time each generation call (t0 -> t1), store it in latencies.

        # take newly generated token (last token) and append it to running sequence
        new_token = out[:, -1:] #shape [batch, 1]
        generated_ids = torch.cat([generated_ids, new_token], dim=1)

        # update attention_mask; after adding token, all positions are now 'valid'
        current_attention_mask = torch.ones_like(generated_ids, device=device)

    total_time = time.perf_counter() - start_total

    # === TODO 6: Measure GPU memory usage ===
    #   Hint: torch.cuda.max_memory_allocated() / (1024**2)

    # Synchronize again to ensure all CUDA ops are complete
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) #gives you how much VRAM (in bytes) was used at peak during generation; Dividing by (1024 ** 2) converts it to megabytes. 
    else:
        peak_mem_mb = float("nan")  # Not applicable for CPU


    # Total number of generated tokens = batch_size * max_new_tokens
    num_tokens = args.batch_size * args.max_new_tokens

    # Throughput (tokens per second)
    tokens_per_s = num_tokens / total_time #throughput metric

    # Latency percentiles (median and p95)
    p50 = percentile(latencies, 50) #latencies are your per-token responsiveness metrics.
    p95 = percentile(latencies, 95)


    # === TODO 7: Write results to CSV ===
    #   Columns:
    #   model, dtype, device, use_cache, batch_size, prompt_len,
    #   max_new_tokens, total_time_s, tokens_per_s, p50_s, p95_s, peak_mem_mb

    header = [
    "model",
    "dtype",
    "device",
    "use_cache",
    "batch_size",
    "prompt_len",
    "max_new_tokens",
    "total_time_s",
    "tokens_generated",
    "tokens_per_s",
    "p50_latency_s",
    "p95_latency_s",
    "peak_mem_mb",
    ]

    row = [
        args.model,
        args.dtype,
        device.type,
        int(args.use_cache),
        args.batch_size,
        args.prompt_len,
        args.max_new_tokens,
        total_time,
        num_tokens,
        tokens_per_s,
        p50,
        p95,
        peak_mem_mb,
    ]

    # If the file doesn't exist yet, we write the header first.
    write_header = not os.path.exists(args.csv)

    with open(args.csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


    # === TODO 8: Print a short summary to console ===
    #   Example print:
    #   f"Done. use_cache={args.use_cache}, batch={args.batch_size}, S={args.prompt_len}, tok/s={tokens_per_s:.2f}, peak_mem={peak_mem:.1f}MB"

    print(
    f"[run complete] use_cache={bool(args.use_cache)}, "
    f"batch_size={args.batch_size}, "
    f"prompt_len={args.prompt_len}, "
    f"new_tokens={args.max_new_tokens}, "
    f"throughput={tokens_per_s:.2f} tok/s, "
    f"p50={p50:.4f}s, p95={p95:.4f}s, "
    f"peak_mem={peak_mem_mb:.1f} MB"
    )
    


if __name__ == "__main__":
    main()