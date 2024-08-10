from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import streamlit as st
import pytesseract
from PIL import Image
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import Settings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure API key
api_key = os.getenv("GROQ_API_KEY")

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image)

def save_text_to_file(text, file_path):
    """Save the extracted text to a file."""
    with open(file_path, 'w') as file:
        file.write(text)

def main():
    st.title("Invoice Extractor")

    # Upload the image file
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image and display it
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract text from the image
        text = extract_text_from_image(image)

        # Ensure the Data directory exists
        if not os.path.exists("Data"):
            os.makedirs("Data")

        # Define the path for saving the text file
        file_path = os.path.join("Data", "extracted_text.txt")

        # Save the extracted text to the file
        save_text_to_file(text, file_path)

        # Load the text from the directory
        documents = SimpleDirectoryReader("Data")
        docs = documents.load_data()

        # Set the embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # Set the LLM model
        Settings.llm = Groq(model="llama3-70b-8192", api_key=api_key)

        # Check if the storage directory exists
        if not os.path.exists("storage"):
            index = VectorStoreIndex.from_documents(docs)
            # Save index to disk
            index.set_index_id("vector_index")
            index.storage_context.persist("./storage")
        else:
            # Rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir="storage")
            # Load index
            index = load_index_from_storage(storage_context, index_id="vector_index")

        query_engine = index.as_query_engine()
        response = query_engine.query("Provide the following details from the invoice text: 1. Customer details (Name, Address, etc.), 2. List of products (Product name, quantity, price), 3. Total Amount.")

        response_text = response.response
        st.write("Details:")
        st.write(response_text)

if __name__ == "__main__":
    main()
