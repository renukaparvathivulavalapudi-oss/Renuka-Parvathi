import streamlit as st
import fitz  # PyMuPDF
import faiss
import os
import tempfile
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="StudyMate: AI PDF Q&A", layout="wide")

# --- Title ---
st.title("📘 StudyMate: AI PDF-Based Q&A System")

# --- Upload PDF ---
uploaded_file = st.file_uploader("📤 Upload your academic PDF", type="pdf")

# --- Load Embedding Model & QA Pipeline ---
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embed_model, qa_pipeline

embed_model, qa_pipeline = load_models()

# --- Functions ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_chunks = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            text_chunks.append(text)
    return text_chunks

def create_faiss_index(chunks, embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def get_top_k_chunks(question, chunks, embeddings, index, k=3):
    question_embedding = embed_model.encode([question])
    D, I = index.search(question_embedding, k)
    top_chunks = [chunks[i] for i in I[0]]
    return top_chunks

# --- Main Logic ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract text
    chunks = extract_text_from_pdf(tmp_path)
    
    # Embed chunks
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    
    # Build index
    index = create_faiss_index(chunks, embeddings)
    
    # Ask Question
    st.success("✅ PDF processed successfully!")
    question = st.text_input("❓ Ask a question based on the uploaded PDF")

    if question:
        top_chunks = get_top_k_chunks(question, chunks, embeddings, index)
        context = " ".join(top_chunks)
        result = qa_pipeline(question=question, context=context)
        
        st.subheader("📌 Answer:")
        st.write(result['answer'])

        with st.expander("📄 Source Context"):
            st.write(context)
