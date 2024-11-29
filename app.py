import streamlit as st
from dotenv import load_dotenv
import pickle
from pypdf import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import sqlite3

load_dotenv()

# Setup database
DB_PATH = "documents.db"
VECTOR_STORE_DIR = "vector_stores"

os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            path TEXT
        )
    """)
    conn.commit()
    conn.close()

initialize_database()

def save_document_to_db(name, path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO documents (name, path) VALUES (?, ?)", (name, path))
    conn.commit()
    conn.close()

def get_all_documents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM documents")
    documents = cursor.fetchall()
    conn.close()
    return documents

def get_document_path(doc_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM documents WHERE id = ?", (doc_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Authentication
def authenticate_user(username, password):
    # Example hardcoded user data, replace with a secure solution in production
    USERS = {"admin": "password123", "user": "1234"}
    return USERS.get(username) == password

def login():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("Login to PDF Chat App")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            if login_button:
                if authenticate_user(username, password):
                    st.session_state["authenticated"] = True
                    st.success("Login successful!")
                else:
                    st.error("Invalid credentials. Please try again.")

# Sidebar contents
def render_sidebar():
    with st.sidebar:
        st.title("💬 PDF Chat App")
        st.markdown('''
        ## About
        Construyendo tu propio chatbot
        ''')

        st.subheader("Available Documents")
        documents = get_all_documents()
        selected_doc = None
        if documents:
            selected_doc = st.selectbox("Select a document to chat with", documents, format_func=lambda x: x[1])
        else:
            st.write("No documents found. Please upload one.")

        add_vertical_space(5)

        return selected_doc

def main():
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        login()
    else:
        st.header("Chat with PDF 💬")
        selected_doc = render_sidebar()

        # Upload a new PDF
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        if pdf:
            process_and_save_pdf(pdf)

        # Handle selected document
        if selected_doc:
            doc_id, doc_name = selected_doc
            doc_path = get_document_path(doc_id)
            vector_store = load_vector_store(doc_name)

            if vector_store:
                st.subheader(f"Chatting with: {doc_name}")
                question = st.text_input("Ask a question about this document")
                if question:
                    docs = retrieve_docs(question, vector_store)
                    response = generate_response(docs, question)
                    st.write(response)
            else:
                st.warning("The vector store for this document is not available. Please upload the document again.")

def process_and_save_pdf(pdf):
    pdf_name = pdf.name[:-4]
    pdf_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_name}.pkl")

    if not os.path.exists(pdf_path):
        chunks = process_text(pdf, 500, 100)
        vector_store = get_embeddings(chunks, pdf_name)
        save_document_to_db(pdf_name, pdf_path)

def process_text(pdf, chunk_size, chunk_overlap):
    pdf_reader = PdfReader(pdf)
    page_text = ""
    for page in pdf_reader.pages:
        page_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text=page_text)

def get_embeddings(chunks, pdf_name):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_name}.pkl")

    with open(vector_store_path, "wb") as f:
        pickle.dump(vector_store, f)
    st.write("Embeddings saved to disk")
    return vector_store

def load_vector_store(pdf_name):
    vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_name}.pkl")
    if os.path.exists(vector_store_path):
        with open(vector_store_path, "rb") as f:
            return pickle.load(f)
    return None

def retrieve_docs(question, vector_store):
    docs = vector_store.similarity_search(question, k=3)
    if len(docs) == 0:
        st.warning("No relevant documents found.")
        return []
    else:
        return docs

def generate_response(docs, question):
    llm = ChatOpenAI(temperature=0.0, max_tokens=1000, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(cb)
    return response

if __name__ == '__main__':
    main()
