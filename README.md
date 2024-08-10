Sure, here’s a README for your project:

---

# Invoice Extractor Streamlit App

## Overview

This Streamlit application allows users to extract key information from invoices provided in either PDF or image format. The app leverages optical character recognition (OCR) to process image files and uses a document retrieval system to analyze and extract specific details from the invoice text. The extracted information includes:

1. Customer Details (Name, Address, etc.)
2. List of Products (Product name, quantity, price)
3. Total Amount

## Features

- **Image Upload**: Users can upload invoice images (JPEG, PNG) for text extraction.
- **PDF Upload**: Users can upload invoice PDFs for processing.
- **Information Extraction**: Extracts and displays customer details, product list, and total amount from the invoice.
- **Data Storage**: Saves uploaded files and extracted text in organized directories.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **Pytesseract**: For optical character recognition (OCR) to extract text from images.
- **LamaIndex**: For document loading, indexing, and query processing.
- **HuggingFace Embeddings**: For creating vector embeddings of text.
- **Groq**: For language model processing and querying.
- **PyPDF2**: For reading PDF files.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Rafe2001/Swipe-Assesment.git
   cd Swipe-Assesment
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in the project root and add your API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

1. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

2. **Navigate to** `http://localhost:8501` in your web browser.

3. **Upload an invoice image or PDF** using the file uploader provided in the app.

4. **View the extracted information**, including customer details, product list, and total amount.

## Directory Structure

- **Data_PDF/**: Directory where uploaded PDF files are stored.
- **storage/**: Directory where vector store index and other data are saved.

## Troubleshooting

- **File not loading**: Ensure that the `Data_PDF` directory exists and is correctly referenced in the code.
- **Dependency issues**: Ensure all required libraries are installed. Refer to `requirements.txt` for a list of dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize any parts of this README to better fit your project specifics or preferences!
