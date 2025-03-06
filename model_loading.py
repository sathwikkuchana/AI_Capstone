# src/model_loading.py

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama2_model(model_name="meta-llama/Llama-2-7b-chat-hf", device_map="auto"):
    """
    Load the Llama‑2 model and tokenizer.
    
    Args:
        model_name (str): The Hugging Face model identifier.
        device_map (str): Device placement; "auto" lets transformers choose the best available device.
    
    Returns:
        tokenizer, model: The loaded tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    return tokenizer, model
