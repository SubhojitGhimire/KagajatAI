import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

mixed_precision = "fp16"  # ["no", "fp16", "bf16", "fp8"]
max_tokens = 200

accelerator = Accelerator(mixed_precision=mixed_precision)

@st.cache_data
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, ignore_mismatched_sizes=True)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, ignore_mismatched_sizes=True)
    model = model.to(accelerator.device)
    model = accelerator.prepare(model)
    return model, tokenizer

def inference_function(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
    input_ids = inputs.input_ids.to(accelerator.device)
    attention_mask = inputs.attention_mask.to(accelerator.device)
    
    with torch.no_grad():
        generated_ids = accelerator.unwrap_model(model).generate(input_ids, attention_mask=attention_mask, max_length=max_tokens)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text

def main():
    st.title("Generic Chatbot")
    if 'model_name' not in st.session_state:
        st.session_state.model_name = st.text_input("Enter Fine Tuned model Location:", "gpt2-large") # Enter Absolute/Relative Path to the model directory (containing safetensor checkpoints)
    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        st.session_state.model, st.session_state.tokenizer = load_model(st.session_state.model_name)
    

    prompt = st.text_area("Enter your prompt:", height=75)
    if st.button("Generate"):
        with st.spinner("Generating response..."):
            response = inference_function(st.session_state.model, st.session_state.tokenizer, prompt)
            st.text_area("Generated text:", value=response, height=200, max_chars=None, key=None)

if __name__ == "__main__":
    main()

# > streamlit run main_gui.py