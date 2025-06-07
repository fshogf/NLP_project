import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# ========== تغییر برای تنظیم درست project_root (راه‌حل ۱) ==========
# فرض می‌کنیم این اسکریپت در فولدر project/test/ قرار دارد.
# os.path.dirname(__file__) -> ".../project/test"
# os.path.dirname(os.path.dirname(__file__)) -> ".../project"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ================================================================

# مسیر فایل‌ها و مدل‌ها
book_path       = os.path.join(project_root, "book.txt")
questions_path  = os.path.join(project_root, "test", "test.txt")
output_path     = os.path.join(project_root, "test", "result.txt")

mpnet_model_path = os.path.join(project_root, "models", "mpnet_local")
phi2_model_path  = os.path.join(project_root, "models", "phi-2")

# ====== مرحله 1: بارگذاری مدل‌ها ======
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"==> Using device: {device}")

# مدل امبدینگ را روی همان دستگاه (CPU/GPU) تنظیم می‌کنیم
embedder = SentenceTransformer(mpnet_model_path, device=device)

# مدل زبانی (phi-2) و توکنایزر مربوطه
tokenizer = AutoTokenizer.from_pretrained(phi2_model_path)
model     = AutoModelForCausalLM.from_pretrained(phi2_model_path).to(device)

# ====== مرحله 2: خواندن کتاب و سؤالات ======
with open(book_path, "r", encoding="utf-8") as f:
    book_text = f.read()

with open(questions_path, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

# ====== مرحله 3: تقسیم‌بندی متن به چانک ======
def chunk_text(text, max_words=250, overlap=0.4):
    """
    متن بلند را به تکه‌هایی با حداکثر max_words کلمه تقسیم می‌کند.
    overlap درصد همپوشانی بین چانک‌هاست (مثلاً اگر overlap=0.4،
    هر چانک 40٪ از چانک قبلی را تکرار می‌کند).
    """
    words = text.split()
    step = int(max_words * (1 - overlap))
    if step < 1:
        step = max_words
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i : i + max_words]
        chunks.append(" ".join(chunk))
    return chunks

chunks = chunk_text(book_text, max_words=250, overlap=0.4)

print("==> Computing embeddings for all chunks …")
chunk_embeddings = embedder.encode(
    chunks,
    convert_to_tensor=True,
    show_progress_bar=True
)
print("    ✔ Chunk embeddings computed.")

# ====== مرحله 4: استخراج چانک‌های مرتبط و تولید پاسخ ======
def answer_question(question, chunks, chunk_embeddings, top_n=5):
    """
    1) embedding سؤال را می‌سازد.
    2) شبیهت کسینوسی بین embedding سؤال و embedding همهٔ چانک‌ها را محاسبه می‌کند.
    3) top_n چانکِ مرتبط را استخراج می‌کند.
    4) قبل از کنار هم‌چیدن، مطمئن می‌شود مجموعهٔ نهاییِ توکن‌ها (prompt + context)
       از حدِ 1024 توکن فراتر نرود.
    5) پرامپت را می‌سازد و به مدل زبان می‌دهد تا پاسخ را شبیه‌سازی کند.
    """
    # 4.1 – embedding سؤال
    q_embedding = embedder.encode(question, convert_to_tensor=True)

    # 4.2 – محاسبهٔ شبیهت کسینوسی و انتخاب top_n اندیس
    cos_scores = util.cos_sim(q_embedding, chunk_embeddings)[0]
    top_indices = torch.topk(cos_scores, k=top_n).indices.tolist()

    # 4.3 – محدود کردن تعداد و حجم توکنی برای مدل زبانی
    #   هدف این است که مجموع توکن‌های "best_chunks + سؤال" زیر 1024 بماند.
    q_tokens = tokenizer.encode(f"Question: {question}\nAnswer:", add_special_tokens=False)
    q_token_count = len(q_tokens)
    max_input_tokens = 1024 - q_token_count - 20  # ۲۰ توکن حاشیه برای پاسخ

    selected_chunks = []
    total_tokens = 0
    for idx in top_indices:
        chunk = chunks[idx]
        chunk_tok = tokenizer.encode(chunk, add_special_tokens=False)
        if total_tokens + len(chunk_tok) <= max_input_tokens:
            selected_chunks.append(chunk)
            total_tokens += len(chunk_tok)
        if total_tokens >= max_input_tokens:
            break

    if not selected_chunks:
        selected_chunks = [chunks[top_indices[0]]]

    best_chunks = "\n---\n".join(selected_chunks)

    # 4.4 – ساخت پرامپت نهایی
    prompt = (
    "As a specialist in philosophy of mind, analyze the following question using the relevant information below. "
    "Write a self-contained, precise, and academic answer. Do not repeat the question or instructions.\n\n"
    "Context:\n"
    f"{best_chunks}\n\n"
    f"Question:\n{question}\n\n"
    "Answer:"
)


    # 4.5 – توکنیزه و انتقال به device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        add_special_tokens=True
    ).to(device)

    # 4.6 – تولید پاسخ با try/except تا در صورت بروز خطای CUDA، کل حلقه قطع نشود
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.6,
                top_p=0.85,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
    except Exception as e:
        print(f"    [!] خطا در تولید پاسخ برای سؤال: {e}")
        return "(Error: could not generate answer)"

    # 4.7 – دیکد خروجی و استخراج بخش Answer:
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in output_text:
        answer = output_text.split("Answer:")[-1].strip()
    else:
        answer = output_text.strip()

    if len(answer) < 20:
        return "(No clear answer generated — consider review.)"
    return answer

# ====== مرحله 5: تابع اسکوردهی پاسخ ======
def score_answer(question, answer):
    """
    معیار ساده: شبیهت کسینوسیِ embedding سؤال و embedding پاسخ را
    به مقیاس 0 تا 100 تبدیل می‌کند.
    """
    if answer.startswith("(Error") or answer.startswith("(No clear"):
        return 0
    q_emb = embedder.encode(question, convert_to_tensor=True)
    a_emb = embedder.encode(answer,   convert_to_tensor=True)
    sim = util.cos_sim(q_emb, a_emb).item()
    score = round(((sim + 1) / 2) * 100)
    return max(0, min(score, 100))

# ====== مرحله 6: اجرای کلی و نوشتن خروجی ======
with open(output_path, "w", encoding="utf-8") as f_out:
    f_out.write("=== Semantic QA Results ===\n\n")
    for i, question in enumerate(questions, start=1):
        print(f"({i}/{len(questions)}) 🔍 Question: {question}")
        start_time = time.time()

        answer = answer_question(question, chunks, chunk_embeddings, top_n=5)
        elapsed_time = time.time() - start_time
        score = score_answer(question, answer)

        print(f"    ✅ Score: {score}/100   ⏱️ Time: {elapsed_time:.2f}s")
        print(f"    ▶️ Answer preview: {answer[:80].replace(chr(10), ' ')}...\n")

        f_out.write(f"Question {i}: {question}\n")
        f_out.write(f"Answer: {answer}\n")
        f_out.write(f"Score: {score}/100\n")
        f_out.write(f"Time: {elapsed_time:.2f} seconds\n")
        f_out.write("-" * 40 + "\n\n")

print("✅ All questions answered, scored, and saved to:", output_path)
