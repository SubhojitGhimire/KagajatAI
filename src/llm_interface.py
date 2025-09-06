import os
import yaml
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMInterface:
    def __init__(self, config_path: str = 'config/config.yaml'):
        print("Initialising LLM Interface...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) 
        load_dotenv() # Load environment variables for API keys
        self.llm_provider = self.config['llm']['provider']
        self.llm = self._load_llm()
        print(f"LLM Interface initialised with provider: {self.llm_provider}")

    def _load_llm(self):
        if self.llm_provider == "gemini":
            return self._load_gemini()
        elif self.llm_provider == "huggingface":
            return self._load_huggingface()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _load_gemini(self) -> ChatGoogleGenerativeAI:
        print("Loading Gemini model...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            api_key = self.config['llm']['gemini']['gemini_api_key']
            if api_key == "YOUR_GEMINI_API_KEY_HERE":
                raise ValueError("Gemini API key not found. Please set it in config/config.yaml or a .env file.")
        model_name = self.config['llm']['gemini']['model_name']
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        print("Gemini model loaded successfully.")
        return llm

    def _load_huggingface(self) -> HuggingFacePipeline:
        hf_config = self.config['llm']['huggingface']
        model_name = hf_config['generation_model_name']
        print(f"Loading Hugging Face model: {model_name}")

        # 1. Configure 4-bit quantization for efficient memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # 2. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # 3. Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # 4. Create a text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id
        )
        print("Hugging Face model loaded successfully.")
        return HuggingFacePipeline(pipeline=pipe)

    def create_rag_chain(self) -> Runnable:
        prompt_template = """
        SYSTEM: You are a helpful, respectful, and honest assistant. Your task is to answer the user's question based only on the context provided below.
        
        CONTEXT:
        {context}
        
        USER:
        {question}
        
        ASSISTANT:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        return prompt | self.llm

# Sample test for the RAG chain
if __name__ == '__main__':
    try:
        llm_interface = LLMInterface()
        rag_chain = llm_interface.create_rag_chain()
        
        sample_context = (
            "Subhojit Ghimire was born in Gorkha, Nepal, and grew up in different parts of Nepal and India. "
            "His mother tongue is Nepali, but he is also fluent in English and Hindi. "
            "He spent significant part of his early childhood in Norway, which sparked his fascination with Europe. "
            "He has a deep interest in Norse mythology, human psychology, and enjoys reading fantasy novels and watching movies in his free time."
        )
        sample_question = "Where did Subhojit grow up, what is his native language, and what other languages does he speak?"
        
        print("\n--- Testing RAG Chain ---")
        print(f"Provider: {llm_interface.llm_provider}")
        print(f"Question: {sample_question}")
        
        response = rag_chain.invoke({"context": sample_context, "question": sample_question})
        if hasattr(response, 'content'):
            print(f"Response:\n{response.content}")
        else:
            print(f"Response:\n{response}")
            
    except Exception as e:
        print(f"An error occurred during testing: {e}")

