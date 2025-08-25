import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ===== Embedding via Jina AI (multilingual, gratis) =====
from llama_index.core import Settings
from llama_index.embeddings.jinaai import JinaEmbedding

Settings.embed_model = JinaEmbedding(
    api_key=os.getenv("JINA_API_KEY"),
    model="jina-embeddings-v3",  
    task="text-matching",
)

# ===== Inti indexing =====
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

DATA_DIR = Path("data")
PERSIST_ROOT = Path("storage")

# Sesuaikan dengan struktur folder data kamu
CATEGORIES = {
    "character": DATA_DIR / "character",
    "factions":  DATA_DIR / "factions",
    "items":     DATA_DIR / "items",
    "maps":      DATA_DIR / "maps",
    "npc":       DATA_DIR / "npc",
    "timeline":  DATA_DIR / "timeline",
}

# Pecah dokumen agar retrieval akurat
parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

def build_one(category: str, src_dir: Path, persist_root: Path):
    md_files = sorted(str(p) for p in src_dir.glob("*.md"))
    if not md_files:
        print(f"⚠️  Lewati '{category}': tidak ada .md di {src_dir}")
        return

    persist_dir = persist_root / category
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    print(f"📥 [{category}] Baca {len(md_files)} file markdown...")
    docs = SimpleDirectoryReader(input_files=md_files, filename_as_id=True).load_data()

    print(f"🔨 [{category}] Susun nodes & bangun index...")
    nodes = parser.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)

    print(f"💾 [{category}] Simpan index ke {persist_dir} ...")
    index.storage_context.persist(persist_dir=str(persist_dir))

def main():
    PERSIST_ROOT.mkdir(parents=True, exist_ok=True)
    for cat, path in CATEGORIES.items():
        build_one(cat, path, PERSIST_ROOT)
    print("✅ Beres. Semua index tersimpan di folder 'storage/' per kategori.")

if __name__ == "__main__":
    main()
