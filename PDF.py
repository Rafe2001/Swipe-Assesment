import streamlit as st
import os
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings, 
    StorageContext, 
    VectorStoreIndex, 
    load_index_from_storage)
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")



def save_pdf_to_directory(uploaded_file, directory):
    """Save the uploaded PDF file to the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("Invoice PDF Extractor")

    # Upload the PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded PDF to the Data_PDF directory
        pdf_directory = "Data_PDF"
        save_pdf_to_directory(uploaded_file, pdf_directory)

        # Load the documents from the directory
        documents = SimpleDirectoryReader(pdf_directory)
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

        # Query the index
        query_engine = index.as_query_engine()
        response = query_engine.query(
            "Provide the following details from the invoice text: 1. Customer details (Name, Address, etc.), "
            "2. List of products (Product name, quantity, price), 3. Total Amount."
        )
        response_text = response.response

        st.write("Extracted Details:")
        st.write(response_text)

if __name__ == "__main__":
    main()
