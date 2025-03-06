# src/utils.py

def generate_chat_answer(prompt, tokenizer, model, max_length=4096, temperature=0.7, top_p=0.9):
    """
    Generate an answer from the model given a prompt.
    
    Args:
        prompt (str): The complete prompt to send to the model.
        tokenizer: The tokenizer for the model.
        model: The language model.
        max_length (int): Maximum length for the generated response.
        temperature (float): Sampling temperature.
        top_p (float): Top-p (nucleus) sampling probability.
    
    Returns:
        str: The generated answer.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    # Move inputs to GPU if available
    device = "cuda" if model.device.type == "cuda" else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_length=max_length, temperature=temperature, top_p=top_p)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer
