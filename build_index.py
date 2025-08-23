import shutil
from pathlib import Path

# Atur model embedding sekali di awal biar konsisten dan nggak fallback ke provider lain
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

# Import modul inti buat baca file markdown, bikin potongan teks (nodes), lalu bangun index vektor
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# Lokasi data mentah (markdown) dan folder penyimpanan index
DATA_DIR = Path("data")
PERSIST_ROOT = Path("storage")

# Daftar kategori harus sama persis dengan struktur folder kamu
CATEGORIES = {
    "character": DATA_DIR / "character",
    "factions":  DATA_DIR / "factions",
    "items":     DATA_DIR / "items",
    "maps":      DATA_DIR / "maps",
    "npc":       DATA_DIR / "npc",
    "timeline":  DATA_DIR / "timeline",
}

# Kita pecah dokumen jadi potongan ~500an karakter dengan overlap biar konteks tetap nyambung
parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

def build_one(category: str, src_dir: Path, persist_root: Path):
    # Ambil semua file .md di folder kategori ini
    md_files = sorted(str(p) for p in src_dir.glob("*.md"))
    if not md_files:
        print(f"⚠️  Lewati '{category}': gak ketemu file .md di {src_dir}")
        return

    # Bersihkan index lama biar nggak ke-mix sama versi sebelumnya
    persist_dir = persist_root / category
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    print(f"📥 [{category}] Baca {len(md_files)} file markdown...")
    docs = SimpleDirectoryReader(input_files=md_files, filename_as_id=True).load_data()

    # Ubah dokumen menjadi nodes (unit kecil yang bisa di-embed) lalu bikin index vektornya
    print(f"🔨 [{category}] Susun nodes & bangun index...")
    nodes = parser.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)

    # Simpan index per kategori ke folder storage/<kategori>
    print(f"💾 [{category}] Simpan index ke {persist_dir} ...")
    index.storage_context.persist(persist_dir=str(persist_dir))

def main():
    # Pastikan folder storage ada
    PERSIST_ROOT.mkdir(parents=True, exist_ok=True)

    # Bangun index per kategori sesuai daftar di atas
    for cat, path in CATEGORIES.items():
        build_one(cat, path, PERSIST_ROOT)

    print("✅ Beres. Semua index tersimpan di folder 'storage/' per kategori.")

if __name__ == "__main__":
    main()
