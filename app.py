import os
import streamlit as st
import hashlib
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import UpstashVectorStore
from langchain_core.embeddings import Embeddings
from upstash_vector import Index as UpstashNativeIndex
import google.generativeai as genai

# 1. Konfigurasi Halaman Web Streamlit
st.set_page_config(page_title="Chatbot Sejarah Nasional", page_icon="📜", layout="centered")
st.title("📜 Chatbot Sejarah Nasional Indonesia")
st.write("Tanyakan apa saja tentang sejarah Indonesia!")

# 2. Muat API Key dari .env atau Secrets
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ API Key Gemini tidak ditemukan!")
    st.stop()

genai.configure(api_key=api_key)

# 3. KELAS EMBEDDING API GRATIS (Tanpa unduh file berat / Torch)
class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self):
        # Menggunakan endpoint serverless resmi Hugging Face gratis
        self.api_url = "https://huggingface.co"
        # Token Hugging Face opsional untuk bypass limit, gratis jika tanpa token untuk traffic kecil
        self.headers = {} 

    def embed_documents(self, texts):
        try:
            response = requests.post(self.api_url, json={"inputs": texts, "options": {"wait_for_model": True}}, headers=self.headers)
            return response.json()
        except Exception:
            return [[0.0] * 384 for _ in texts] # Fallback aman jika API down

    def embed_query(self, text):
        try:
            response = requests.post(self.api_url, json={"inputs": [text], "options": {"wait_for_model": True}}, headers=self.headers)
            return response.json()[0]
        except Exception:
            return [0.0] * 384

mesin_embedding = HuggingFaceAPIEmbeddings()

def generate_unique_id(text, source_name):
    return hashlib.md5(f"{source_name}_{text}".encode('utf-8')).hexdigest()

# 4. Inisialisasi Kredensial Upstash Vector
@st.cache_resource
def init_services():
    upstash_url = st.secrets.get("UPSTASH_VECTOR_REST_URL") or os.getenv("UPSTASH_VECTOR_REST_URL")
    upstash_token = st.secrets.get("UPSTASH_VECTOR_REST_TOKEN") or os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    
    if not upstash_url or not upstash_token:
        st.error("❌ Kredensial Upstash Vector tidak ditemukan!")
        st.stop()
        
    return UpstashVectorStore(
        embedding=mesin_embedding,        
        text_key="text",
        index_url=upstash_url,
        index_token=upstash_token
    )

db = init_services()

# =========================================================================
# MENU SIDEBAR: FITUR MULTI-FORMAT & PENOLAKAN INSTAN DUPLIKAT
# =========================================================================
with st.sidebar:
    st.header("📂 Tambah Buku Sejarah")
    uploaded_file = st.file_uploader("Unggah file Buku Sejarah Baru", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("🚀 Proses & Unggah ke Cloud"):
            import os
            nama_file = os.path.basename(uploaded_file.name)
            
            # ✨ Penanda status: buat False di awal
            file_sudah_ada = False 
            
            with st.spinner("🔍 Memeriksa apakah file sudah ada di cloud..."):
                try:
                    string_filter_sql = f"source = '{nama_file}'"
                    dokumen_kembar = db.similarity_search(
                        query="sejarah", 
                        k=1, 
                        filter=string_filter_sql
                    )
                    
                    if dokumen_kembar and len(dokumen_kembar) > 0:
                        st.warning(f"❌ Berkas ditolak! Buku Sejarah dengan nama '{nama_file}' sudah pernah diunggah sebelumnya.")
                        # ✨ PERBAIKAN UTAMA: Ubah status menjadi True, JANGAN pakai st.stop()
                        file_sudah_ada = True  
                        
                except Exception as e:
                    pass

            # -----------------------------------------------------------------
            # JIKA BELUM ADA (LOLOS SENSOR), BARU JALANKAN PROSES EKSTRAKSI
            # -----------------------------------------------------------------
            if not file_sudah_ada:
                with st.spinner("Python sedang membaca isi dokumen..."):
                    teks_seluruh_buku = ""
                    
                    if nama_file.endswith('.pdf'):
                        from pypdf import PdfReader
                        reader = PdfReader(uploaded_file)
                        for halaman in reader.pages:
                            teks_halaman = halaman.extract_text()
                            if teks_halaman:
                                teks_seluruh_buku += teks_halaman + "\n"
                                
                    elif nama_file.endswith('.txt'):
                        teks_seluruh_buku = uploaded_file.read().decode("utf-8")

                    # Penapisan Kata Kunci Sampah
                    kata_kunci_sampah = ["daftar gambar", "daftar tabel", "kata pengantar", "prakata", "glosarium", "indeks"]
                    lines = teks_seluruh_buku.split("\n")
                    lines_clean = []
                    
                    for line in lines:
                        line_lowercased = line.lower().strip()
                        if len(line_lowercased) < 15: continue
                        if any(kata in line_lowercased for kata in kata_kunci_sampah): continue
                        if line_lowercased.isdigit(): continue
                        lines_clean.append(line)
                        
                    teks_bersih_final = "\n".join(lines_clean)

                    # PROSES CHUNKING MURNI PYTHON
                    chunk_size = 1000
                    chunk_overlap = 200
                    list_teks = []
                    start = 0
                    while start < len(teks_bersih_final):
                        end = start + chunk_size
                        list_teks.append(teks_bersih_final[start:end])
                        start += (chunk_size - chunk_overlap)

                    st.info(f"🧹 Python Selesai: Dokumen dipotong menjadi {len(list_teks)} bagian bersih.")

                    if list_teks:
                        list_ids = [generate_unique_id(text, nama_file) for text in list_teks]
                        list_metadatas = [{"source": nama_file} for _ in list_teks]
                        
                        with st.spinner("Hugging Face sedang membuat vektor & mengirim ke Upstash..."):
                            try:
                                db.add_texts(
                                    texts=list_teks,
                                    metadatas=list_metadatas,
                                    ids=list_ids
                               )
                                st.success(f"✅ Berhasil! Buku '{nama_file}' kini aman tersimpan di Upstash Cloud!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Gagal mengunggah: {e}")
                    else:
                        st.warning("⚠️ File tidak berisi teks sejarah yang lolos sensor filter Python.")

# =========================================================================
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
        # ✨ STRATEGI AMAN: Menggunakan filter pencarian teks Python murni (Hemat Kuota Gemini)
        query_lowercased = user_query.lower().strip()
        kata_personal_umum = ["nama", "siapa", "aku", "saya", "kamu", "anda", "halo", "hai", "pagi", "siang", "sore", "malam", "terima kasih", "makasih"]
        hanya_butuh_riwayat = any(kata in query_lowercased for kata in kata_personal_umum)

        context = ""
        # Jika murni tanya sejarah, jalankan spinner pencarian Upstash
        if not hanya_butuh_riwayat:
            with st.spinner("🔍 Sedang mencari di buku sejarah..."):
                try:
                    docs = db.similarity_search(user_query, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])
                except Exception:
                    pass

        with st.spinner("✍️ Sedang mengetik..."):
            try:
                riwayat_teks = ""
                for msg in st.session_state.messages[:-1][-6:]: 
                    peran = "Pengguna" if msg["role"] == "user" else "Asisten/Anda"
                    riwayat_teks += f"{peran}: {msg['content']}\n"
                
                if not riwayat_teks: riwayat_teks = "(Belum ada obrolan sebelumnya)"
                
                prompt = f"""
                Anda adalah seorang pakar Sejarah Nasional Indonesia yang ramah dan edukatif.
                
                TUGAS ANDA:
                1. Jika KONTEKS SEJARAH DARI BUKU terisi, jawablah pertanyaan pengguna HANYA berdasarkan informasi tersebut. Jika tidak ditemukan, katakan tidak ada di buku referensi dengan sopan.
                2. Jika KONTEKS SEJARAH DARI BUKU KOSONG, berarti ini obrolan umum atau pertanyaan tentang ingatan. Jawablah langsung berdasarkan RIWAYAT OBROLAN SEBELUMNYA.

                RIWAYAT OBROLAN SEBELUMNYA:
                {riwayat_teks}

                KONTEKS SEJARAH DARI BUKU:
                {context}

                PERTANYAAN TERBARU PENGGUNA:
                {user_query}

                JAWABAN ANDA:
                """

                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                bot_response = response.text
                
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")



