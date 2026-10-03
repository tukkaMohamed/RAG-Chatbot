import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# 1. Load environment variables
# =========================================================

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")


# =========================================================
# 2. Streamlit Page Configuration
# =========================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="centered"
)


# =========================================================
# 3. UI
# =========================================================

st.title("📚 RAG Chatbot")

st.write(
    "Ask questions about one of the available documents."
)


# =========================================================
# 4. Document Selection
# =========================================================

document = st.selectbox(
    "Choose a document:",
    [
        "Tips Hindawi University",
        "Arabic Book"
    ]
)


# =========================================================
# 5. Question Input
# =========================================================

question = st.chat_input(
    "Ask your question..."
)


# =========================================================
# 6. Load PDF
# =========================================================

@st.cache_resource
def load_documents(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


# =========================================================
# 7. Chunking
# =========================================================

@st.cache_resource
def create_chunks(pdf_path):

    documents = load_documents(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# =========================================================
# 8. Embedding Model
# =========================================================

@st.cache_resource
def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return embeddings


# =========================================================
# 9. Create Vector Store
# =========================================================

@st.cache_resource
def create_vectorstore(pdf_path):

    chunks = create_chunks(pdf_path)

    embeddings = create_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# =========================================================
# 10. Create Retriever
# =========================================================

@st.cache_resource
def create_retriever(pdf_path):

    vectorstore = create_vectorstore(pdf_path)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever


# =========================================================
# 11. Gemini LLM
# =========================================================

@st.cache_resource
def create_llm():

    if not google_api_key:

        st.error(
            "GOOGLE_API_KEY was not found. "
            "Please add it to your .env file."
        )

        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=google_api_key
    )

    return llm


# =========================================================
# 12. RAG Function
# =========================================================

def ask_question(question, pdf_path):

    # Get retriever
    retriever = create_retriever(pdf_path)

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Combine retrieved chunks
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Prompt
    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
clearly say that the information is not available
in the provided document.

Do not invent facts.

Answer in the same language as the user's question.

Answer briefly and directly.
Only include information that directly answers the question.

Context:
{context}

Question:
{question}

Answer naturally and clearly:
"""

    # Generate answer
    llm = create_llm()

    response = llm.invoke(prompt)

    # Extract text
    if isinstance(response.content, list):

        return response.content[0]["text"]

    return response.content


# =========================================================
# 13. Chat
# =========================================================

if question:

    # Choose PDF
    if document == "Tips Hindawi University":

        pdf_path = "data/Tips Hindawi University Info.pdf"

    else:

        pdf_path = "data/arabic_book.pdf"


    # Display user message
    with st.chat_message("user"):

        st.write(question)


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching the document..."):

            answer = ask_question(
                question,
                pdf_path
            )

        st.write(answer)