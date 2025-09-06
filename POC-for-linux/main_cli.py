import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, AutoConfig
from transformers import TextGenerationPipeline, pipeline
from accelerate import Accelerator

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.grad_scaler import ShardedGradScaler

from langchain_community.llms import HuggingFacePipeline

from datetime import datetime

MIXED_PRECISION = "fp16" # ["no", "fp16", "bf16", "fp8"]
LEARNING_RATE = 2e-5
MAX_TOKENS = 200
PORT = '6758'

import os
def init_distributed_mode():
    # Set the environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = PORT
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    # Initialise the process group
    torch.distributed.init_process_group(
        backend='gloo',  # use 'nccl' for gpu, 'gloo' for cpu
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE']),
    )



def inference_function(model_name, prompts, accelerator):
    device = accelerator.device
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, ignore_mismatched_sizes=True)
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.eos_token

        model = FSDP(AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, ignore_mismatched_sizes=True))

        # If above model load line throws an exception, it is most likely due to missing config, comment the line above and uncomment the following two lines.
        # config = AutoConfig.from_pretrained(model_name, ignore_mismatched_sizes=True)
        # model = FSDP(AutoModelForCausalLM.from_pretrained(model_name, from_tf=bool(".ckpt" in model_name), config=config,  return_dict=True, ignore_mismatched_sizes=True))
        
        model = model.to(device)
        optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE)
        scaler = ShardedGradScaler()
        model, optimizer, scaler = accelerator.prepare(model, optimizer, scaler)

    except Exception as e:
        print(f"Error loading the model: {e}")
        exit()
    
    model_wrapped = model.module if hasattr(model, "module") else model

    # Initiate pipeline to generate or display word per word with streamer.
    streamer = TextStreamer(tokenizer)
    text_generator = pipeline("text-generation", model=model_wrapped, tokenizer=tokenizer, streamer=streamer, max_new_tokens=MAX_TOKENS, device=device)

    # Initiate pipeline to display the entire generate output after the generation completion
    # text_generator = TextGenerationPipeline(model=model_wrapped, tokenizer=tokenizer, device=device, max_length=MAX_TOKENS)
    # text_generator_hf = HuggingFacePipeline(pipeline=text_generator)

    for prompt in prompts:
        print(f"\n\nGIVEN PROMPT/QUESTION: {prompt}\n")
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            generated_ids = accelerator.unwrap_model(model).generate(input_ids, attention_mask=attention_mask, max_length=MAX_TOKENS)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"Generated text:\n{generated_text}\n")


def main():
    init_distributed_mode()

    # Give full model path (absolute or relative) in model_name where .safetensor files are present
    model_name = input("Enter Fine Tuned model Location:\n> ")
    if not model_name:
        model_name = "gpt2-large"
        print(model_name)

    prompt = input("Enter Prompt/Question: (Leave Empty to pick default prompt)\n> ")
    prompts = [prompt]
    if not prompt:
        prompts = ["What is AI?", "Will AI take over the world someday?"]
        print(prompts)
    
    accelerator = Accelerator(mixed_precision=MIXED_PRECISION)
    inference_function(model_name, prompts, accelerator)

if __name__ == "__main__":
    main()

# > python main_cli.py