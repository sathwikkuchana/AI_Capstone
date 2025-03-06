# src/chat_pipeline.py

from data_preprocessing import download_nltk_data, load_pickle, extract_text_from_doc, chunk_markdown_text
from indexing import build_faiss_index, retrieve_chunks
from model_loading import load_llama2_model
from utils import generate_chat_answer

from sentence_transformers import SentenceTransformer

def main():
    # Download required NLTK data
    download_nltk_data()

    # Load data from the pickle file (update the file path as needed)
    pickle_file = "/content/batch1.pickle"
    data = load_pickle(pickle_file)
    
    # Select the first document and extract its text
    doc = data[0]
    doc_text = extract_text_from_doc(doc)
    
    # Chunk the document text into smaller pieces
    chunks = chunk_markdown_text(doc_text, max_words=300)
    print(f"Created {len(chunks)} chunks from document.")

    # Build a FAISS index from the chunks using a SentenceTransformer embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    index, _ = build_faiss_index(chunks, embed_model)
    print("FAISS index built successfully.")

    # Define your task and instructions for evidence extraction
    user_instructions = (
        "Your task is to identify evidence regarding the following questions within the context of climate change adaptation:\n"
        "1. Identify the geographical locations of the adaptation response, providing details in the format: "
        "Country name: <country name>, Sub-national region: <sub national region>.\n"
        "2. Identify the stakeholders involved in the adaptation response from categories such as Government, Private sector, Civil society, etc.\n"
        "3. Classify the depth of the adaptation response as Low, Medium, High, or Not certain / Insufficient information / Not assessed, and provide an explanation for your assessment.\n"
        "Do not include any additional context or the reference excerpts in your final answer. Provide your answer strictly in the following format:\n"
        "Stakeholders: <your answer>,\n"
        "Depth: <your assessment>,\n"
        "Explanation: <your reasoning for this assessment>\n\n"
        "Reference Excerpts (for internal use only):\n"
    )
    
    # Retrieve the relevant context (chunks) using the same instructions as query
    retrieved_chunks = retrieve_chunks(user_instructions, embed_model, index, chunks, top_k=3)
    internal_context = "\n".join(retrieved_chunks)
    
    # Construct the final prompt using Llama‑2 chat template tokens
    system_message = "You are a climate change research assistant with expertise in adaptation tracking through document analysis."
    user_message = f"{user_instructions}{internal_context}"
    final_prompt = f"<<SYS>>\n{system_message}\n<<SYS>>\n\n<<INST>>\n{user_message}\n<</INST>>"
    
    # Load the Llama‑2 model and tokenizer
    tokenizer, model = load_llama2_model()
    
    # Generate the answer using the model
    answer = generate_chat_answer(final_prompt, tokenizer, model)
    
    # Print the final generated answer
    print("Final Answer:")
    print(answer)

if __name__ == "__main__":
    main()
