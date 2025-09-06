import streamlit as st
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, AutoConfig
from transformers import TextGenerationPipeline, pipeline
from accelerate import Accelerator

MIXED_PRECISION = "fp16"  # ["no", "fp16", "bf16", "fp8"]
MAX_TOKENS = 200
ABSOLUTE_PATH = r"/home/SubhojitGhimire/Models/"

accelerator = Accelerator(mixed_precision=MIXED_PRECISION)

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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
    input_ids = inputs.input_ids.to(accelerator.device)
    attention_mask = inputs.attention_mask.to(accelerator.device)
    
    with torch.no_grad():
        generated_ids = accelerator.unwrap_model(model).generate(input_ids, attention_mask=attention_mask, max_length=MAX_TOKENS)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text

def model_selection_func(model_name):
    if "GPT-2 Large" in model_name:
        model_path =  ABSOLUTE_PATH + "gpt2-large"
    elif "Llama-2 7b":
        model_path = ABSOLUTE_PATH + "Llama-2-7b-chat-hf"
    return model_path

def main():
    st.title("Generic Chatbot")

    # Dropdown for model selection
    model_options = ["GPT-2 Large", "Llama-2 7b"] # List all your models available. Update the model_selection_func as you add more models.
    model_selection = st.selectbox("Select a model:", model_options)

    # Button to load the selected model
    if st.button("Select Model"):
        st.session_state.model_name = model_selection_func(model_selection)
        st.session_state.model, st.session_state.tokenizer = load_model(st.session_state.model_name)
        st.success(f"Model {st.session_state.model_name} loaded successfully!")

    # Check if the model is loaded before allowing prompts
    if 'model' in st.session_state and 'tokenizer' in st.session_state:
        prompt = st.text_area("Enter your prompt", height=75)
        
        if st.button("Generate"):
            with st.spinner("Generating response..."):
                response = inference_function(st.session_state.model, st.session_state.tokenizer, prompt)
                st.text_area("Generated text:", value=response, height=200, max_chars=None, key=None)

if __name__ == "__main__":
    main()

# > streamlit run main_gui_final.py