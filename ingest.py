# File skrip untuk mengindeks data
# Untuk memproses dokumen agar masuk ke dalam database pencarian.


import os
import time
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # ✨ Menggunakan Hugging Face gratis
from upstash_vector import Index

load_dotenv()
UPSTASH_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise ValueError("❌ Kredensial Upstash Vector tidak ditemukan di file .env!")

# 1. Inisialisasi Model Embedding Gratis dari Hugging Face (100% Bebas Kuota)
print("🤗 Memuat model embedding Hugging Face (all-MiniLM-L6-v2)...")
mesin_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def generate_unique_id(chunk):
    source = chunk.metadata.get("source", "unknown")
    # Bersihkan path folder agar menyisakan nama file aslinya saja
    nama_file_saja = os.path.basename(source)
    content = chunk.page_content
    return hashlib.md5(f"{nama_file_saja}_{content}".encode('utf-8')).hexdigest()

def main():
    print("📂 Membaca file Buku Sejarah di folder './data'...")
    all_docs = PyPDFDirectoryLoader("./data").load() + DirectoryLoader("./data", glob="*.txt", loader_cls=TextLoader).load()
    if not all_docs:
        print("❌ Folder './data' kosong!")
        return
    
    print(f"📄 Memotong teks dokumen menjadi bagian kecil...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    # Inisialisasi Klien Driver Native Upstash Cloud
    upstash_client = Index(url=UPSTASH_URL, token=UPSTASH_TOKEN)

    # DAFTAR KATA KUNCI YANG INGIN DIABAIKAN
    kata_kunci_sampah = [
        "daftar gambar", "daftar tabel", "kata pengantar", 
        "prakata", "halaman sengaja dikosongkan", "glosarium", "indeks",
        "halaman judul", "sambutan", "puji syukur"
    ]
    
    chunks_bersih = []
    print("🔍 Menyaring data sampah dan mendeteksi file duplikat di cloud...")
    
    for chunk in chunks:
        teks_lowercased = chunk.page_content.lower().strip()
        if len(teks_lowercased) < 15: 
            continue
        if any(kata in teks_lowercased for kata in kata_kunci_sampah):
            continue
        if teks_lowercased.isdigit():
            continue
            
        # ✨ SISTEM ANTIDUPLIKAT PINTAR LOKAL:
        chunk_id = generate_unique_id(chunk)
        
        # Tanya cepat ke Upstash apakah ID bagian ini sudah pernah diunggah
        hasil_cek = upstash_client.fetch(ids=[chunk_id], include_vectors=False)
        if hasil_cek and hasil_cek is not None and len(hasil_cek) > 0:
            continue # Jika sudah ada di cloud, langsung lewati tanpa proses ulang
            
        chunks_bersih.append(chunk)
        
    print(f"🧹 Filter Selesai: Dari {len(chunks)} bagian, ditemukan {len(chunks_bersih)} bagian BARU.")
    
    if not chunks_bersih:
        print("😎 Semua dokumen di folder './data' sudah tersimpan di Upstash Cloud. Tidak ada data baru.")
        return
    
    print(f"🚀 Memulai konversi {len(chunks_bersih)} teks baru ke vektor 384 Dimensi (Gratis via Hugging Face)...")
    list_teks = [chunk.page_content for chunk in chunks_bersih]
    
    # ✨ PROSES EMBEDDING SEKARANG GRATIS LEWAT LAPTOP ANDA TANPA PERLU KUOTA GOOGLE GEMINI
    vektor_hasil_huggingface = mesin_embedding.embed_documents(list_teks)
    
    # Susun data sesuai objek tuple Upstash
    vektor_siap_kirim = []
    for i, chunk in enumerate(chunks_bersih):
        chunk_id = generate_unique_id(chunk)
        
        metadata_gabungan = chunk.metadata.copy()
        # Ambil nama file bersihnya saja untuk metadata
        metadata_gabungan["source"] = os.path.basename(chunk.metadata.get("source", "unknown"))
        metadata_gabungan["text"] = chunk.page_content 
        
        vektor_siap_kirim.append((
            chunk_id,
            vektor_hasil_huggingface[i], # Array berisi 384 angka koordinat
            metadata_gabungan
        ))
        
    print("🔌 Mengunggah kumpulan vektor kustom langsung ke Cloud Upstash...")
    # Kirim data menggunakan format native (aman dari bug parameter LangChain)
    upstash_client.upsert(vectors=vektor_siap_kirim)
    print("✅ SELESAI! Seluruh database RAG berhasil diperbarui ke Cloud Upstash menggunakan 384 Dimensi!")

if __name__ == "__main__":
    main()