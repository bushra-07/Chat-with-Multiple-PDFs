# Chat with Multiple PDFs 🤖📚

This project is a **Streamlit-based app** that allows users to chat with the contents of multiple PDF files. It uses **Langchain**, **Google Generative AI**, and **FAISS** for document vectorization, retrieval, and conversational responses.

## Features

- Upload multiple PDFs and extract the text.
- Vectorize text chunks using Google Generative AI embeddings.
- Store vectorized text in FAISS for efficient similarity search.
- Ask questions, and the app retrieves relevant chunks from the PDFs to answer.
- Integrated conversational AI for detailed and contextual responses.

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

Before running this project, ensure that you have the following installed:

- Python 3.8 or later
- Streamlit
- PyPDF2
- Langchain
- FAISS
- Google Generative AI SDK

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/chat-with-multiple-pdfs.git
    cd chat-with-multiple-pdfs
    ```

2. **Set up a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Create a `.env` file** in the root directory and add your Google API key:

    ```plaintext
    GOOGLE_API_KEY=your-google-api-key
    ```

### How to Run

1. **Prepare your environment:**
    - Make sure you have your Google Generative AI API key ready.
    - Ensure that your `.env` file contains the key.

2. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

3. **Upload your PDFs**:
    - Use the sidebar to upload multiple PDF files.
    - Click the "Submit & Process" button to extract and process the text.

4. **Ask Questions**:
    - Enter your question in the text input field.
    - The app will retrieve relevant sections from the PDFs and provide a detailed answer.

### Example Use Case

- Upload a few research papers or documents, and ask specific questions about their contents.
- The app will return the most relevant sections of the PDFs, providing detailed responses.

### Project Structure

```plaintext
├── app.py                   # Main application file
├── .env                     # Environment variables (contains the Google API key)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
