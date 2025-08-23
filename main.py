import os
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, MessageHandler, CallbackQueryHandler,
    CommandHandler, filters, ContextTypes
)

# --- LlamaIndex setup (embedding lokal gratis) ---
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

# --- ENV ---
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert BOT_TOKEN, "TELEGRAM_BOT_TOKEN belum diset"
assert GROQ_API_KEY, "GROQ_API_KEY belum diset"

# --- LLM Groq (tanpa .env, default kuat) ---
PREFERRED_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
MAX_TOKENS = 1024

def init_llm():
    model_to_use = PREFERRED_MODELS[0]
    try:
        from llama_index.llms.groq import Groq
        return Groq(api_key=GROQ_API_KEY, model=model_to_use,
                    temperature=0.2, max_tokens=MAX_TOKENS, request_timeout=60.0)
    except Exception:
        from llama_index.llms.openai_like import OpenAILike
        return OpenAILike(api_base="https://api.groq.com/openai/v1",
                          api_key=GROQ_API_KEY, model=model_to_use,
                          temperature=0.2, max_tokens=MAX_TOKENS, timeout=60.0)

Settings.llm = init_llm()

# --- Optional: reranker untuk ningkatin relevansi retrieval ---
reranker = None
try:
    from llama_index.postprocessor.sentence_transformer_rerank import SentenceTransformerRerank
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=8)
except Exception:
    pass

# --- Index loader ---
PERSIST_ROOT = Path("storage")
CATEGORIES = ["character", "factions", "items", "maps", "npc", "timeline"]
retrievers = {}

def load_retrievers():
    if not PERSIST_ROOT.exists():
        raise RuntimeError("Folder 'storage/' tidak ada. Jalankan build_index.py dulu.")
    for cat in CATEGORIES:
        cat_dir = PERSIST_ROOT / cat
        if not cat_dir.exists():
            print(f"⚠️  Index '{cat}' tidak ada, dilewati.")
            continue
        sc = StorageContext.from_defaults(persist_dir=str(cat_dir))
        index = load_index_from_storage(sc)
        retrievers[cat] = index.as_retriever(similarity_top_k=8)
    print("✅ Retrievers loaded:", sorted(retrievers.keys()))

load_retrievers()

# --- Aliases + helper ---
ALIASES = {
    "character": ["character", "karakter"],
    "factions":  ["faction", "factions", "tim", "team", "faksi"],
    "items":     ["item", "items", "artefak", "artifact", "barang"],
    "maps":      ["maps", "map", "peta", "lokasi", "area"],
    "npc":       ["npc"],
    "timeline":  ["timeline", "linimasa", "alur", "sejarah"],
}

def detect_category_and_query(text: str):
    """Dukung format 'maps: apa ...'. Kalau tidak ada prefix, return (None, full_text)."""
    raw = (text or "").strip()
    if ":" in raw:
        prefix, rest = raw.split(":", 1)
        prefix = prefix.lower().strip()
        rest = rest.strip()
        for cat, keys in ALIASES.items():
            if prefix in keys:
                return cat, rest
    return None, raw

def collect_sources(nodes, max_files=5):
    files = []
    for n in nodes:
        meta = getattr(n, "node", n).metadata if hasattr(n, "node") else {}
        fname = (meta.get("file_name") or meta.get("filename") or meta.get("id"))
        if fname and fname not in files:
            files.append(fname)
        if len(files) >= max_files:
            break
    return ", ".join(files) if files else "dataset"

# === UI: /start, /reset, pilih kategori ===
def category_keyboard():
    rows = [
        [
            InlineKeyboardButton("Character", callback_data="CAT|character"),
            InlineKeyboardButton("Factions", callback_data="CAT|factions"),
        ],
        [
            InlineKeyboardButton("Items", callback_data="CAT|items"),
            InlineKeyboardButton("Maps", callback_data="CAT|maps"),
        ],
        [
            InlineKeyboardButton("NPC", callback_data="CAT|npc"),
            InlineKeyboardButton("Timeline", callback_data="CAT|timeline"),
        ],
        [InlineKeyboardButton("🔎 Semua Kategori", callback_data="CAT|all")],
    ]
    return InlineKeyboardMarkup(rows)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("selected_cat", None)
    await update.message.reply_text(
        "Pilih kategori yang ingin kamu tanyakan:",
        reply_markup=category_keyboard()
    )

async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("selected_cat", None)
    await update.message.reply_text("Kategori direset. Ketik /start untuk pilih lagi.")

async def pick_category_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    try:
        _, cat = q.data.split("|", 1)
    except Exception:
        await q.edit_message_text("Pilihan tidak dikenal. /start lagi ya.")
        return

    if cat == "all":
        context.user_data["selected_cat"] = None
        await q.edit_message_text("Mode: 🔎 semua kategori.\nTulis pertanyaanmu:")
        return

    if cat not in retrievers:
        await q.edit_message_text(
            f"Kategori '{cat}' belum ter-index. Jalankan build_index.py untuk folder itu dulu."
        )
        return

    context.user_data["selected_cat"] = cat
    pretty = cat.capitalize()
    await q.edit_message_text(f"Mode: 🗂 {pretty}\nSilakan ketik pertanyaanmu…")

# === Handler tanya-jawab ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q_raw = update.message.text or ""
    loading = await update.message.reply_text("⏳ Lagi nyari jawabannya...")

    try:
        # 1) Prefix override (items:/maps:/dst)
        cat_from_prefix, q = detect_category_and_query(q_raw)
        # 2) Default ke kategori yang dipilih via /start
        selected_cat = context.user_data.get("selected_cat")
        cat = cat_from_prefix if cat_from_prefix is not None else selected_cat

        if cat and cat not in retrievers:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=loading.message_id,
                text=f"Kategori '{cat}' belum ter-index. Jalankan build_index.py dulu."
            )
            return

        # 3) Retrieval
        retrieved_nodes = []
        if cat:
            retrieved_nodes.extend(retrievers[cat].retrieve(q))
        else:
            for r in retrievers.values():
                retrieved_nodes.extend(r.retrieve(q))

        # 4) Rerank (jika tersedia)
        if reranker and retrieved_nodes:
            try:
                retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query=q)
            except Exception:
                pass

        contexts = [n.node.text for n in retrieved_nodes]
        context_text = "\n\n---\n\n".join(contexts[:12])

        if not context_text.strip():
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=loading.message_id,
                text="🙇‍♂️ Maaf, belum ketemu di basis data. Coba spesifikkan pertanyaan atau cek index."
            )
            return

        source_note = f"(Sumber: {collect_sources(retrieved_nodes)})"
        mode_note = f"[Mode: {('Semua' if not cat else cat)}]"

        prompt = (
            "Jawablah pertanyaan hanya dengan memanfaatkan KONTEN berikut. "
            "Jika jawaban persis tidak ada, gunakan informasi paling relevan dari KONTEN untuk memberi penjelasan ringkas. "
            "Jika sama sekali tidak relevan, tulis: 'Informasi tidak ditemukan di basis data.'\n\n"
            f"--- KONTEN ---\n{context_text}\n\n"
            f"--- PERTANYAAN ---\n{q}\n\n"
            "--- PETUNJUK ---\n"
            "- Utamakan fakta dari KONTEN.\n"
            "- Jika tidak ada fakta langsung, boleh simpulkan singkat dari potongan yang terkait.\n"
            "- Jika benar-benar kosong, tulis: 'Informasi tidak ditemukan di basis data.'\n"
        )

        resp = await Settings.llm.acomplete(prompt)
        answer = (getattr(resp, 'text', None) or str(resp)).strip()

        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading.message_id,
            text=f"🧠 {answer}\n\n{source_note}\n{mode_note}"
        )

        # 5) Feedback 1–5
        keyboard_fb = InlineKeyboardMarkup([[
            InlineKeyboardButton("1", callback_data=f"fb|1|{q_raw}"),
            InlineKeyboardButton("2", callback_data=f"fb|2|{q_raw}"),
            InlineKeyboardButton("3", callback_data=f"fb|3|{q_raw}"),
            InlineKeyboardButton("4", callback_data=f"fb|4|{q_raw}"),
            InlineKeyboardButton("5", callback_data=f"fb|5|{q_raw}")
        ]])
        await update.message.reply_text("Nilai jawaban ini (1=buruk, 5=bagus):", reply_markup=keyboard_fb)

        # 6) Tawarkan interaksi berikutnya
        keyboard_next = InlineKeyboardMarkup([
            [InlineKeyboardButton("Tanya lagi 🔄", callback_data="NEXT|again")],
            [InlineKeyboardButton("Selesai ✅", callback_data="NEXT|done")]
        ])
        await update.message.reply_text("Apakah masih ingin bertanya?", reply_markup=keyboard_next)

    except Exception as e:
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading.message_id,
            text=f"❌ Error saat memproses: {e}"
        )

# === Callback setelah jawaban: tanya lagi / selesai ===
async def next_step_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    try:
        _, action = q.data.split("|", 1)
    except Exception:
        await q.edit_message_text("Pilihan tidak dikenal.")
        return

    if action == "again":
        # buka menu kategori lagi
        await q.edit_message_text("Pilih kategori baru:", reply_markup=category_keyboard())
    else:
        # saran langkah berikutnya biar jelas
        tips = (
            "Terima kasih sudah menggunakan bot ini 👋\n\n"
            "Saran berikutnya:\n"
            "• Ketik /start untuk ganti kategori dan tanya lagi\n"
            "• Ketik /reset kalau mau hapus pilihan kategori yang aktif\n"
            "• Pastikan data kamu up-to-date lalu rebuild index (jalankan build_index.py) kalau ada perubahan"
        )
        await q.edit_message_text(tips)

# === Simpan feedback ke CSV ===
async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    try:
        _, score, question = q.data.split("|", 2)
    except Exception:
        await q.edit_message_text("⚠️ Format feedback tidak valid.")
        return

    user = q.from_user
    row = {"user_id": user.id, "username": user.username or "",
           "score": score, "question": question, "timestamp": datetime.now().isoformat()}

    with open("feedback_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

    await q.edit_message_text(f"✅ Terima kasih! Feedback {score}/5 terekam.")

# === Run bot ===
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CallbackQueryHandler(pick_category_callback, pattern=r"^CAT\|"))
    app.add_handler(CallbackQueryHandler(next_step_callback, pattern=r"^NEXT\|"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(feedback_callback, pattern=r"^fb\|"))
    print("🤖 Bot aktif. Gunakan /start untuk memilih kategori.")
    app.run_polling()

if __name__ == "__main__":
    main()
