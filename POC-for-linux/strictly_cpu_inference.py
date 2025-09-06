# Infernece run on single node, using CPU (or single GPU using GPU ID), not using FSDP or DDP.
# If running into CUDA out of memory issues while using GPU, or FSDP wrap issues, it is advisable to infer using CPU on a single node.

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer, AutoConfig
from accelerate import Accelerator

from langchain_community.llms import HuggingFacePipeline
from datetime import datetime

mixed_precision = "fp16" # ["no", "fp16", "bf16", "fp8"]
learning_rate = 2e-5

cpu_use = True  # True for cpu; False for gpu use
device = torch.device("cpu") # "cpu" for cpu, "cuda" for gpu; default: torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cpu_use and device are dependent on each other.

device_id = -1 # <int> -1 for using all CPU while generating tokens. GPU ID: 0, 1, 2... for using GPU with that ID
# device_id is independent of cpu_use and device
max_tokens = 200


def inference_function(model_name, prompts):
    startTime = datetime.now()
    accelerator = Accelerator(cpu=cpu_use, mixed_precision=mixed_precision)
    
    try:
        stime_loading = datetime.now()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, ignore_mismatched_sizes=True)
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, ignore_mismatched_sizes=True).to(device)
        config = AutoConfig.from_pretrained(model_name, ignore_mismatched_sizes=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, from_tf=bool(".ckpt" in model_name), config=config, ignore_mismatched_sizes=True).to(device)
        optimizer = AdamW(params=model.parameters(), lr=learning_rate)
        model, optimizer = accelerator.prepare(model, optimizer)
        
        etime_loading = datetime.now()
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit()
    
    streamer = TextStreamer(tokenizer)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, streamer=streamer, max_new_tokens=max_tokens, device=device_id)
    for prompt in prompts:
        print(f"\n\nGIVEN PROMPT/QUESTION: {prompt}\n")
        stime_generator = datetime.now()
        generated_text = text_generator(prompt, truncation=True)
        etime_generator = datetime.now()
        print(f"generator:\nInitial Time: {stime_generator}\nFinal Time: {etime_generator}\nTotal Time Taken = {etime_generator-stime_generator}\n--- x ---\n")
    
    endTime = datetime.now()
    print(f"from_pretrained LOADING:\nInitial Time: {stime_loading}\nFinal Time: {etime_loading}\nTotal Time Taken = {etime_loading-stime_loading}\n")
    print(f"Complete inference:\nInitial Time: {startTime}\nFinal Time: {endTime}\nTotal Time Taken = {endTime-startTime}\n")


def main():
    model_name = input("Enter Fine Tuned model Location:\n> ")
    if not model_name:
        model_name = "gpt2-large"
        print(model_name)

    prompt = input("Enter Prompt/Question: (Leave Empty to pick default prompt)\n> ")
    prompts = [prompt]
    if not prompt:
        prompts = ["What is AI?", "Will AI take over the world someday?"]
        print(prompts)
    inference_function(model_name, prompts)
    print(prompts)


if __name__ == "__main__":
    main()

# > python strictly_cpu_inference.py