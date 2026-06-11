# File skrip untuk mengindeks data
# Untuk memproses dokumen agar masuk ke dalam database pencarian.


import os
import time
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
# ✨ IMPOR LANGSUNG DRIVER RESMI UPSTASH
from upstash_vector import Index

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
UPSTASH_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

if not api_key or not UPSTASH_URL or not UPSTASH_TOKEN:
    raise ValueError("❌ Periksa kembali file .env Anda! API Key atau Kredensial Upstash kosong.")

genai.configure(api_key=api_key)

# Kelas Pembantu Gemini Embeddings 768 Dimensi
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = genai.embed_content(
                    model="models/gemini-embedding-001", 
                    content=batch_texts, 
                    task_type="retrieval_document",
                    output_dimensionality=768
                )
                embeddings.extend(response['embedding'])
                print(f"📦 Berhasil memproses vektor Gemini ke {i} sampai {i + len(batch_texts)}")
                time.sleep(6) 
            except Exception as e:
                print(f"⚠️ Kuota habis, menunggu 30 detik untuk pemulihan...")
                time.sleep(30)
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
            output_dimensionality=768
        )
        return response['embedding']

def generate_unique_id(chunk):
    source = chunk.metadata.get("source", "unknown")
    content = chunk.page_content
    return hashlib.md5(f"{source}_{content}".encode('utf-8')).hexdigest()

def main():
    print("📂 Membaca file Buku Sejarah di folder './data'...")
    all_docs = PyPDFDirectoryLoader("./data").load() + DirectoryLoader("./data", glob="*.txt", loader_cls=TextLoader).load()
    if not all_docs:
        print("❌ Folder './data' kosong!")
        return
    
    print(f"📄 Memotong teks dokumen menjadi bagian kecil...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    # ✨ SOLUSI UTAMA: Inisialisasi Klien Driver Native Upstash
    upstash_client = Index(url=UPSTASH_URL, token=UPSTASH_TOKEN)

    # DAFTAR KATA KUNCI YANG INGIN DIABAIKAN
    kata_kunci_sampah = [
        "daftar gambar", "daftar tabel", "kata pengantar", 
        "prakata", "halaman sengaja dikosongkan", "glosarium", "indeks",
        "halaman judul", "sambutan", "puji syukur"
    ]
    
    chunks_bersih = []
    for chunk in chunks:
        teks_lowercased = chunk.page_content.lower().strip()
        if len(teks_lowercased) < 15: 
            continue
        if any(kata in teks_lowercased for kata in kata_kunci_sampah):
            continue
        if teks_lowercased.isdigit():
            continue
        chunks_bersih.append(chunk)
        
    print(f"🧹 Filter Selesai: Dari {len(chunks)} dipangkas menjadi {len(chunks_bersih)} bagian bersih.")
    
    if not chunks_bersih:
        print("😎 Tidak ada dokumen baru atau bersih yang perlu diunggah.")
        return
    
    print(f"🚀 Memulai konversi {len(chunks_bersih)} bagian dokumen bersih dengan Gemini...")
    list_teks = [chunk.page_content for chunk in chunks_bersih]
    
    mesin_embedding = GeminiEmbeddings()
    vektor_hasil_gemini = mesin_embedding.embed_documents(list_teks)
    
    # Format data berupa objek Tuple/Kamus resmi untuk upstash_vector SDK
    vektor_siap_kirim = []
    for i, chunk in enumerate(chunks_bersih):
        chunk_id = generate_unique_id(chunk)
        
        metadata_gabungan = chunk.metadata.copy()
        metadata_gabungan["text"] = chunk.page_content 
        
        # Format pengiriman Driver Native: (id, vector, metadata)
        vektor_siap_kirim.append((
            chunk_id,
            vektor_hasil_gemini[i],
            metadata_gabungan
        ))
        
    print("🔌 Mengunggah kumpulan vektor kustom langsung ke Cloud Upstash...")
    
    # ✨ TEMBAK LANGSUNG LEWAT DRIVER NATIVE (100% AMAN & ANTI-BUG)
    upstash_client.upsert(vectors=vektor_siap_kirim)
    
    print("✅ SELESAI! Seluruh database RAG berhasil disimpan di Upstash Cloud!")

if __name__ == "__main__":
    main()
