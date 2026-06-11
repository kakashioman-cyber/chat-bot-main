import os
import streamlit as st
import hashlib
from dotenv import load_dotenv
from langchain_community.vectorstores import UpstashVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # ✨ Menggunakan model lokal stabil
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
        # Inisialisasi model Gemini terlebih dahulu
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # ✨ LANGKAH 1: AI ROUTER (Gemini mendeteksi jenis pertanyaan secara instan)
        jenis_pertanyaan = "umum"
        try:
            prompt_router = f"""
            Klasifikasikan pertanyaan pengguna ini ke dalam salah satu kategori: 'sejarah' atau 'umum'.
            - Kategori 'sejarah': Jika menanyakan tentang fakta sejarah Indonesia, tanggal, tokoh pahlawan, atau isi buku referensi.
            - Kategori 'umum': Jika berupa sapaan, ucapan terima kasih, obrolan santai, atau menanyakan ingatan/informasi personal yang baru saja dibahas (seperti 'siapa nama saya', 'siapa aku', 'tadi kita bahas apa').

            PERTANYAAN: {user_query}
            KATEGORI (Jawab HANYA dengan 1 kata saja, 'sejarah' atau 'umum'):
            """
            respon_router = model.generate_content(prompt_router)
            jenis_pertanyaan = respon_router.text.lower().strip()
        except Exception:
            jenis_pertanyaan = "umum" # Jaga-jaga jika API error, default ke umum

        # ✨ LANGKAH 2: PROSES BERDASARKAN HASIL DETEKSI AI
        context = ""
        
        # Jika AI mendeteksi ini pertanyaan sejarah, kita tampilkan spinner khusus pencarian buku
        if "sejarah" in jenis_pertanyaan:
            with st.spinner("🔍 Sedang mencari di buku sejarah..."):
                try:
                    docs = db.similarity_search(user_query, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    st.error(f"⚠️ Gagal mengakses Upstash: {e}")
        
        # Jika "umum", kita langsung lompat ke tahap pembuatan jawaban (Bypass Upstash)
        # Kita beri spinner halus saat AI merangkai kata agar user tahu bot sedang bekerja
        with st.spinner("✍️ Sedang mengetik..."):
            try:
                # Susun Riwayat Obrolan Sebelumnya (Maksimal 6 pesan terakhir)
                riwayat_teks = ""
                for msg in st.session_state.messages[:-1][-6:]: 
                    peran = "Pengguna" if msg["role"] == "user" else "Asisten/Anda"
                    riwayat_teks += f"{peran}: {msg['content']}\n"
                
                if not riwayat_teks:
                    riwayat_teks = "(Belum ada obrolan sebelumnya)"
                
                # Prompt Akhir untuk Gemini menghasilkan jawaban edukatif
                prompt_akhir = f"""
                Anda adalah seorang pakar Sejarah Nasional Indonesia yang ramah dan edukatif.
                
                TUGAS ANDA:
                1. Jika KONTEKS SEJARAH DARI BUKU terisi, jawablah pertanyaan pengguna HANYA berdasarkan informasi tersebut. Jika tidak ditemukan di buku, katakan dengan sopan bahwa informasi tersebut tidak ada di buku referensi.
                2. Jika KONTEKS SEJARAH DARI BUKU KOSONG, berarti ini adalah obrolan umum atau pertanyaan tentang ingatan/riwayat. Jawablah secara alami berdasarkan informasi yang tertulis di RIWAYAT OBROLAN SEBELUMNYA (seperti sapaan, nama pengguna, atau panggilan 'siapa aku').

                RIWAYAT OBROLAN SEBELUMNYA:
                {riwayat_teks}

                KONTEKS SEJARAH DARI BUKU:
                {context}

                PERTANYAAN TERBARU PENGGUNA:
                {user_query}

                JAWABAN ANDA:
                """

                response = model.generate_content(prompt_akhir)
                bot_response = response.text
                
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat menyusun jawaban: {e}")



