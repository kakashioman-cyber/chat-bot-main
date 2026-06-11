import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import UpstashVectorStore
from langchain_core.embeddings import Embeddings  # ✨ WAJIB IMPORT INI
import google.generativeai as genai

# 1. Konfigurasi Halaman Web Streamlit
st.set_page_config(page_title="Chatbot Sejarah Nasional", page_icon="📜", layout="centered")
st.title("📜 Chatbot Sejarah Nasional Indonesia")
st.write("Tanyakan apa saja tentang sejarah Indonesia!")

# 2. Muat API Key dari .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ API Key tidak ditemukan di file .env!")
    st.stop()

# Konfigurasi API Key untuk pustaka google-generativeai
genai.configure(api_key=api_key)

# 3. Kelas Embedding Kustom Resmi LangChain (Disamakan dengan ingest.py)
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = genai.embed_content(
                model="models/gemini-embedding-001", 
                content=batch_texts, 
                task_type="retrieval_document",
                output_dimensionality=768
            )
            embeddings.extend(response['embedding'])
        return embeddings

    def embed_query(self, text):
        response = genai.embed_content(
            model="models/gemini-embedding-001", 
            content=text, 
            task_type="retrieval_query",
            output_dimensionality=768  # ✨ WAJIB 768 DIMENSI
        )
        return response['embedding']

# Fungsi Pembantu Eksternal untuk Query
def dapatkan_vektor_pertanyaan(text):
    mesin_embedding = GeminiEmbeddings()
    return mesin_embedding.embed_query(text)

# 4. Inisialisasi Kredensial Upstash Vector 
@st.cache_resource
def init_services():
    upstash_url = st.secrets.get("UPSTASH_VECTOR_REST_URL") or os.getenv("UPSTASH_VECTOR_REST_URL")
    upstash_token = st.secrets.get("UPSTASH_VECTOR_REST_TOKEN") or os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    
    if not upstash_url or not upstash_token:
        st.error("❌ Kredensial Upstash Vector tidak ditemukan!")
        st.stop()
        
    # ✨ PERBAIKAN: Gunakan kelas GeminiEmbeddings() asli agar constructor LangChain lolos validasi
    return UpstashVectorStore(
        embedding=GeminiEmbeddings(),        
        text_key="text",
        index_url=upstash_url,
        index_token=upstash_token
    )

db = init_services()

# 5. Kelola Riwayat Obrolan
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Input Chat dari Pengguna 
if user_query := st.chat_input("Ketik pertanyaan sejarah di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Sedang mencari di buku sejarah..."):
            try:
                # A. Ubah pertanyaan menjadi vektor manual dulu
                query_vector = dapatkan_vektor_pertanyaan(user_query)
                
                # B. PERBAIKAN UTAMA: Menggunakan fungsi pencarian skor yang sah di Upstash LangChain
                results_with_score = db.similarity_search_with_score_by_vector(query_vector, k=4)
                
                # Ekstrak objek dokumennya saja untuk dibaca Gemini
                docs = [doc for doc, score in results_with_score]
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # C. Prompt khusus RAG
                prompt = f"""
                Anda adalah seorang pakar Sejarah Nasional Indonesia yang ramah dan edukatif.
                Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan informasi (konteks) yang disediakan di bawah ini.
                If informasi tidak ada di dalam konteks, katakan dengan sopan bahwa informasi tersebut tidak ditemukan di dalam buku referensi. Jangan mengarang jawaban.

                KONTEKS SEJARAH:
                {context}

                PERTANYAAN PENGGUNA:
                {user_query}

                JAWABAN:
                """

                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                
                bot_response = response.text
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")



