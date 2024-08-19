Chat with Multiple PDFs 🤖📚
This project is a Streamlit-based app that allows users to chat with the contents of multiple PDF files. It uses Langchain, Google Generative AI, and FAISS for document vectorization, retrieval, and conversational responses.

Features
Upload multiple PDFs and extract the text.
Vectorize text chunks using Google Generative AI embeddings.
Store vectorized text in FAISS for efficient similarity search.
Ask questions, and the app retrieves relevant chunks from the PDFs to answer.
Integrated conversational AI for detailed and contextual responses.
Getting Started
Follow these steps to set up and run the project on your local machine.

Prerequisites
Before running this project, ensure that you have the following installed:

Python 3.8 or later
Streamlit
PyPDF2
Langchain
FAISS
Google Generative AI SDK
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/chat-with-multiple-pdfs.git
cd chat-with-multiple-pdfs
Set up a virtual environment (optional but recommended):

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Create a .env file in the root directory and add your Google API key:

plaintext
Copy code
GOOGLE_API_KEY=your-google-api-key
How to Run
Prepare your environment:

Make sure you have your Google Generative AI API key ready.
Ensure that your .env file contains the key.
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Upload your PDFs:

Use the sidebar to upload multiple PDF files.
Click the "Submit & Process" button to extract and process the text.
Ask Questions:

Enter your question in the text input field.
The app will retrieve relevant sections from the PDFs and provide a detailed answer.
Example Use Case
Upload a few research papers or documents, and ask specific questions about their contents.
The app will return the most relevant sections of the PDFs, providing detailed responses.
Project Structure
plaintext
Copy code
├── app.py                   # Main application file
├── .env                     # Environment variables (contains the Google API key)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
How It Works
PDF Text Extraction:

The app reads all uploaded PDFs and extracts the text using PyPDF2.
Text Chunking:

The text is split into chunks using Langchain to optimize for vector storage and search.
Vectorization:

The text chunks are converted into vector embeddings using Google Generative AI models.
Similarity Search:

FAISS is used to store the vectors and perform similarity searches based on the user's question.
Conversational Response:

The app uses Langchain's conversational chain and Google Generative AI to generate detailed answers based on the retrieved context from the PDFs.
Environment Variables
Create a .env file in the root of your project with the following content:

plaintext
Copy code
GOOGLE_API_KEY=your-google-api-key
Make sure you replace your-google-api-key with your actual Google API key from Google Cloud Console.

Dependencies
The main libraries used in this project are:

Streamlit: For building the web interface.
PyPDF2: For reading and extracting text from PDFs.
Langchain: For managing text splitting, embedding, and chaining conversational models.
FAISS: For fast vector similarity search.
Google Generative AI: For generating embeddings and conversational responses.
Credits
Langchain: GitHub Repository
FAISS: GitHub Repository
Google Generative AI: Official Documentation
License
This project is licensed under the MIT License - see the LICENSE file for details.
