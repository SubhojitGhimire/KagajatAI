# LLM-Inference
A simple CLI and GUI for LLM Chatbot (basically, inference code)  
TO NOTE: IT IS ASSUMING YOU HAVE ENOUGH MEMORY ON YOUR CPU OR GPU TO RUN INFERENCE IN FSDP MODE.  
IF RUN INTO ANY OUT OF MEMORY ISSUE, MAYBE TRY strictly_cpu_inference.py THAT INFERS ON SINGLE NODE ONLY, USING CPU.  

Download Pre-trained model from the internet or Fine-tune these models on your own dataset.

Make sure you have git installed. for downloading these model, you also have to enable GIT LFS as such: git lfs install

Some pretrained models from huggingface are: GPT-2-large: git clone https://huggingface.co/openai-community/gpt2-large Llama-2-7b-chat: git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

Create your vitual environment and set it up with the dependencies from requirement: pyton -m pip install -r requirements.txt

Download the individual main files from here and place them anywhere on your PC Location, suggestively, the same parent directory as your models directory.

To run CLI, it's simple: python main_cli.py To run GUI, run the following command instead: streamlit run main_gui.py

Example:
CLI
![Screenshot 2024-03-03 213839](https://github.com/SubhojitGhimire/LLM-Inference/assets/55135657/ed1f0278-6f82-4663-836f-611960e3e930)
![Screenshot 2024-03-03 214139](https://github.com/SubhojitGhimire/LLM-Inference/assets/55135657/72f3b88b-85f1-449f-aa97-8e6ab6300ebd)
![Screenshot 2024-03-03 214207](https://github.com/SubhojitGhimire/LLM-Inference/assets/55135657/00d55d9d-8108-40cc-a5e3-366ab8162120)

GUI
![Screenshot 2024-03-03 215714](https://github.com/SubhojitGhimire/LLM-Inference/assets/55135657/1814680e-550f-4a96-bbc3-12a46efd9ed9)
![Screenshot 2024-03-03 215753](https://github.com/SubhojitGhimire/LLM-Inference/assets/55135657/70c4b384-2a29-4df3-ae90-ad80687f8453)
