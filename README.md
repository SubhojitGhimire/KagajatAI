# KagajatAI: An End-to-End AI Document Analysis System

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

KagajatAI is a full-featured, portfolio-ready project that demonstrates the entire lifecycle of a modern AI application. It provides a web-based interface to "chat" with your documents, powered by a sophisticated Retrieval-Augmented Generation (RAG) pipeline.

This project goes beyond a simple proof-of-concept by incorporating advanced techniques such as model finetuning, rigorous benchmarking, and a flexible architecture that supports both local open-source models and powerful cloud APIs.

<img width="1919" height="1017" alt="Image: Landing Page" src="https://github.com/user-attachments/assets/2c5d73eb-6fcc-4604-8596-bba28efc02e9" />

---

## Key Features
- Interactive Chat Interface: A user-friendly web application built with Streamlit to upload documents and ask questions in natural language.
- Advanced RAG Pipeline: Implements a robust RAG system using a high-performance embedding model (BAAI/bge-large-en-v1.5) and a persistent vector store (ChromaDB).
- Flexible LLM Backend: Seamlessly switch between Google's Gemini Pro API for top-tier performance or a local, finetuned Llama-3-8B model for privacy and customization.
- Efficient Finetuning (QLoRA): Includes a Jupyter notebook to finetune the local LLM model on a custom dataset using QLoRA, the state-of-the-art technique for memory-efficient training.
- Synthetic Dataset Generation: Demonstrates how to use a powerful model (Gemini Pro) to programmatically create a high-quality instruction dataset for finetuning.
- Rigorous Benchmarking: Provides a dedicated notebook to compare the performance of the base model, the finetuned model, and Gemini Pro, allowing for quantitative and qualitative analysis of the finetuning process.

# Requirements and Usage

Clone the repository. From inside the repo folder, install the dependencies:
```bash
python -m pip install --upgrade -r .\requirements.txt
```

Configure API Keys:
- The application can use Google's Gemini API. Place the key directly in config/config.yaml.
- You can also download model checkpoints (in my case, I downloaded the lightweight, yet powerful, 
Llama-3.1-8B-Instruct opensource model). Replace the generation_model_name in config.yaml with the absolute path of the downloaded model.

1. Run the Streamlit application:
```bash
streamlit run .\app\main_app.py
```
The application should now be open and accessible in your web browser.

OR 

2. You can also use specific file from terminal as its own standalone script:
```bash
python -u .\src\rag_pipeline.py
```
Ensure you have some sample PDF documents in .\data\source_documents\

3. To showcase the full capabilities of the project, run the Jupyter notebooks in the following order.
- notebooks/1_Dataset_Creation.ipynb: This notebook uses the Gemini API to generate a finetuning_data.jsonl file from your source document.
- notebooks/2_Finetuning_with_LoRA.ipynb: This notebook uses the generated dataset to finetune the Llama-3-8B model. This requires a CUDA-enabled GPU.
- notebooks/3_Benchmarking.ipynb: After finetuning, run this notebook to compare the responses from the base model, your new finetuned model, and Gemini Pro. Due to hardware limitations, I was unable to finetune and infer on large dataset. For this reason, I didn't implement metrics like ROGUE or BLEU for a quantitative assessment of model performance.
The notebooks, once opened, are self explanatory. 

## Future Improvements:
1. Agentic Capabilities: Expand the system into a multi-tool agent that can not only read documents but also fetch real-time data from external APIs (e.g., stock prices).
2. Broader Document Support: Enable compatibility with additional file types such as .docx, .txt, and .html. Incorporate Deep Document Understanding to more effectively parse complex formats like CVs, resumes, journal papers, novels, and presentations.
3. UI Enhancements: Add features to the Streamlit app to manage multiple vector stores or highlight the source text in the original document.
4. Multi-Model Support: Introduce the ability to seamlessly switch between multiple local models and proprietary LLM API providers with a single click.

## Screenshots

<img width="1919" height="1021" alt="Image: Mechanism" src="https://github.com/user-attachments/assets/2efd6fc1-6e10-4dde-8485-c24d07243b1c" />
<br>
<img width="1919" height="1021" alt="Image: Working" src="https://github.com/user-attachments/assets/b744b508-303e-4070-a13b-96100dcaf0f0" />
<br>
<img width="1919" height="1018" alt="Image: llm interface" src="https://github.com/user-attachments/assets/92c92ee1-c5c5-48c4-9fa4-04e7fc2f8a75" />
<br>
<img width="1919" height="1020" alt="Image: data processing" src="https://github.com/user-attachments/assets/70912bfb-99b9-47ec-8585-024bfac044db" />
<br>
<img width="1919" height="1018" alt="Image: rag pipeline" src="https://github.com/user-attachments/assets/5dad1a3c-a107-4f04-8b6b-32c7c7dc7cbd" />

<h1></h1>

**This README.md file has been improved for overall readability (grammar, sentence structure, and organization) using AI tools.*
