
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests

load_dotenv()
 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

def get_github_profile_info(username):
    response = requests.get(f"https://api.github.com/users/{username}")
    if response.status_code == 200:
        data = response.json()
        return data['avatar_url'], data['name']
    else:
        return None, None

def main():
    st.set_page_config(page_title="Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):  
                raw_text = get_pdf_text(pdf_docs)  
                text_chunks = get_text_chunks(raw_text)  
                get_vector_store(text_chunks)  
                st.success("Done")
        
        st.write("---")
        st.write("Feel free to reach out ")
        
        usernames = ["bushra-07"]
        user_info = [get_github_profile_info(username) for username in usernames]

        st.markdown("""
        <style>
        .profile-container {
            display: flex;
            justify-content: space-around;
            margin-top: 1em;
        }
        .profile-card {
            background-color: #f0f2f6;
            padding: 1em;
            border-radius: 15px;
            width: 70%;
            text-decoration: none;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: 0.3s;
            display: flex;
            align-items: center;
        }
        .profile-card img {
            border-radius: 50%;
            width: 50px;
            height: 50px;
            margin-right: 1em;
        }
        .profile-card:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        </style>
        <div class="profile-container">
        """, unsafe_allow_html=True)

        for (username, (avatar_url, name)) in zip(usernames, user_info):
            if avatar_url and name:
                st.markdown(f"""
                <a href="https://github.com/{username}" target="_blank" class="profile-card">
                    <img src="{avatar_url}" alt="{name}'s profile picture"/>
                    <div>
                        <h3>{name}</h3>
                    </div>
                </a>
                """, unsafe_allow_html=True)
            else:
                st.write(f"Failed to fetch info for {username}")

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
