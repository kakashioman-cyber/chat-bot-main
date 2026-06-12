import os
import streamlit as st
import hashlib
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import UpstashVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# 1. Konfigurasi Halaman Web Streamlit
st.set_page_config(page_title="Chatbot Referensi Pintar", page_icon="📜", layout="centered")
st.title("📜 Chatbot Referensi Pintar")
st.write("Tanyakan apa saja berdasarkan buku referensi Anda!")

# 2. Muat API Key dari .env atau Secrets
load_dotenv()
api_key_gemini = os.getenv("GEMINI_API_KEY")
api_key_grok = os.getenv("GROK_API_KEY")
token_hf = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if api_key_gemini:
    genai.configure(api_key=api_key_gemini)

# 3. Inisialisasi Model Embedding Lokal Gratis (all-MiniLM-L6-v2)
@st.cache_resource
def ambil_mesin_embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

mesin_embedding = ambil_mesin_embedding()

def generate_unique_id(text, source_name):
    import os
    nama_file_saja = os.path.basename(source_name)
    return hashlib.md5(f"{nama_file_saja}_{text}".encode('utf-8')).hexdigest()

# 4. Inisialisasi Kredensial Upstash Vector 384 Dimensi
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
# ✨ FUNGSI PEMANGGIL AI CADANGAN (FALLBACK GENERATORS)
# =========================================================================
def panggil_gemini(prompt):
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text, "🧠 Gemini 2.5 Flash"

def panggil_groq(prompt):
    # Mengambil kunci OpenRouter yang kita simpan di variabel GROQ_API_KEY
    key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Kunci API OpenRouter belum dikonfigurasi!")
        
    # URL Endpoint resmi OpenRouter
    url = "https://openrouter.ai"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        # ✨ MODEL 100% GRATIS SELAMANYA: Llama 3 8B Instruct versi Free dari OpenRouter
        "model": "meta-llama/llama-3-8b-instruct:free", 
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    
    # Ekstrak hasil jawaban teks dari format standar OpenRouter
    try:
        teks_balasan = result["choices"][0]["message"]["content"]
        return teks_balasan, "⚡ OpenRouter (Llama-3 8B Free)"
    except Exception:
        raise ValueError(f"Gagal membaca respon OpenRouter. Data: {result}")

def panggil_gemini_cadangan(prompt):
    # Benteng pertahanan terakhir jika Google Flash dan Groq sama-sama limit
    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content(prompt)
    return response.text, "🚀 Gemini 2.5 Pro (Cadangan Akhir)"
# =========================================================================

# =========================================================================
# 📂 MENU SIDEBAR: FITUR MULTI-FORMAT & PENOLAKAN INSTAN DUPLIKAT
# =========================================================================
with st.sidebar:
    st.header("📂 Tambah Buku Referensi")
    uploaded_file = st.file_uploader("Unggah file Buku Baru", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("🚀 Proses & Unggah ke Cloud"):
            import os
            nama_file = os.path.basename(uploaded_file.name)
            file_sudah_ada = False 
            
            with st.spinner("🔍 Memeriksa apakah file sudah ada..."):
                try:
                    string_filter_sql = f"source = '{nama_file}'"
                    dokumen_kembar = db.similarity_search(query="sejarah", k=1, filter=string_filter_sql)
                    if dokumen_kembar and len(dokumen_kembar) > 0:
                        st.warning(f"❌ Berkas ditolak! Buku dengan nama '{nama_file}' sudah ada.")
                        file_sudah_ada = True  
                except Exception: pass

            if not file_sudah_ada:
                with st.spinner("Python sedang membaca isi dokumen..."):
                    teks_seluruh_buku = ""
                    if nama_file.endswith('.pdf'):
                        from pypdf import PdfReader
                        reader = PdfReader(uploaded_file)
                        for halaman in reader.pages:
                            teks_halaman = halaman.extract_text()
                            if teks_halaman: teks_seluruh_buku += teks_halaman + "\n"
                    elif nama_file.endswith('.txt'):
                        teks_seluruh_buku = uploaded_file.read().decode("utf-8")

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

                    chunk_size = 1000
                    chunk_overlap = 200
                    list_teks = []
                    start = 0
                    while start < len(teks_bersih_final):
                        end = start + chunk_size
                        list_teks.append(teks_bersih_final[start:end])
                        start += (chunk_size - chunk_overlap)

                    st.info(f"🧹 Python Selesai: Dokumen dipotong menjadi {len(list_teks)} bagian.")

                    if list_teks:
                        with st.spinner("Model lokal sedang mengirim ke Upstash..."):
                            try:
                                list_ids = [generate_unique_id(text, nama_file) for text in list_teks]
                                list_metadatas = [{"source": nama_file} for _ in list_teks]
                                db.add_texts(texts=list_teks, metadatas=list_metadatas, ids=list_ids)
                                st.success(f"✅ Berhasil menyimpan buku '{nama_file}'!")
                                st.rerun()
                            except Exception as e: st.error(f"❌ Gagal: {e}")

# =========================================================================
# 5. Kelola Riwayat Obrolan
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Tangani Input Pertanyaan Pengguna dan Proses Jawaban AI dengan Sistem Cadangan Bertingkat
if user_query := st.chat_input("Ketik pertanyaan Anda di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Sedang mencari di database referensi..."):
            try:
                docs = db.similarity_search(user_query, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
            except Exception:
                context = ""

        # Susun teks memori ingatan obrolan (Maksimal 6 pesan)
        riwayat_teks = ""
        for msg in st.session_state.messages[:-1][-6:]: 
            peran = "Pengguna" if msg["role"] == "user" else "Asisten"
            riwayat_teks += f"{peran}: {msg['content']}\n"
        
        if not riwayat_teks: 
            riwayat_teks = "(Belum ada obrolan sebelumnya)"
        
        prompt = f"""
        Anda adalah seorang pakar dan asisten edukatif yang ramah serta interaktif.
        
        ATURAN UTAMA DALAM MENJAWAB:
        1. Jika pengguna melakukan sapaan, mengenalkan nama, atau menanyakan hal personal, jawablah langsung dengan ramah berdasarkan 'RIWAYAT OBROLAN SEBELUMNYA'.
        2. Perhatikan 'RIWAYAT OBROLAN SEBELUMNYA'. Jika pengguna menanyakan kembali hal yang BARU SAJA dibahas, jawablah secara alami (misal: 'Seperti yang saya jelaskan di atas...'). Jangan mengulang jawaban formal panjang secara kaku.
        3. Gunakan 'KONTEKS DARI BUKU REFERENSI' sebagai dasar utama fakta ilmiah untuk menjawab pertanyaan pengetahuan atau sejarah. 
        4. Jika informasi tidak ada di 'KONTEKS DARI BUKU REFERENSI' maupun di 'RIWAYAT OBROLAN SEBELUMNYA', katakan dengan sopan bahwa informasi tidak ditemukan. Jangan mengarang jawaban.

        RIWAYAT OBROLAN SEBELUMNYA:
        {riwayat_teks}

        KONTEKS DARI BUKU REFERENSI:
        {context}

        PERTANYAAN TERBARU PENGGUNA:
        {user_query}

        JAWABAN ANDA:
        """

        bot_response = ""
        model_digunakan = ""

        with st.spinner("✍️ AI Sedang berpikir..."):
            # -----------------------------------------------------------------
            # ✨ JALUR PENYELAMAT INDEPENDEN (GEMINI FLASH -> GROQ -> GEMINI PRO)
            # -----------------------------------------------------------------
            jalur_sukses = False

            # Langkah A: Coba jalankan Gemini Flash Utama
            try:
                bot_response, model_digunakan = panggil_gemini(prompt)
                jalur_sukses = True
            except Exception:
                jalur_sukses = False

            # Langkah B: Jika Gemini Flash limit, LANGSUNG lempar ke Groq API yang kilat dan gratis
            if not jalur_sukses:
                try:
                    bot_response, model_digunakan = panggil_groq(prompt)
                    jalur_sukses = True
                except Exception:
                    jalur_sukses = False

            # Langkah C: Jika Groq juga limit, gunakan benteng terakhir Gemini 2.5 Pro
            if not jalur_sukses:
                try:
                    bot_response, model_digunakan = panggil_gemini_cadangan(prompt)
                except Exception as e_final:
                    st.error(f"❌ Seluruh layanan AI (Gemini Flash, Groq, Gemini Pro) sedang penuh! Eror: {e_final}")
                    st.stop()
            # -----------------------------------------------------------------

        # Tampilkan jawaban bersih di layar browser
        st.markdown(bot_response)
        st.caption(f"🤖 Dibalas oleh: {model_digunakan}")
        st.session_state.messages.append({"role": "assistant", "content": bot_response})


