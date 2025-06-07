# NLP_project

# 🧠 Philosophy of Mind - Semantic QA System

This project implements a semantic question-answering (QA) system tailored for philosophical questions, particularly in the field of **20th and 21st Century Philosophy of Mind**.

The system uses transformer-based models to retrieve semantically relevant content from a source book and generate accurate, self-contained, and academically toned answers to conceptual questions.

---

## 📁 Folder Structure

```
project/
├── book.txt                          ← Main textbook (Volume 6)
├── models/
│   ├── phi-2/                        ← Phi-2 language model (local)
│   └── mpnet_local/                 ← MPNet sentence embedding model (local)
├── test/
│   ├── test.txt                      ← List of conceptual questions
│   ├── main.py                       ← QA pipeline script (Colab-ready)
│   ├── result.txt                    ← Generated answers and similarity scores
```

---

## 🚀 How It Works

1. Reads full book text from `book.txt`
2. Splits it into overlapping chunks (250 words, 40% overlap)
3. Embeds all chunks using MPNet
4. Embeds each question and selects top 5 related chunks
5. Constructs a prompt and generates answers using phi-2
6. Scores the answers via cosine similarity to the question
7. Saves result to `result.txt`

---

## 📚 Book Used

- **Title:** *The History of the Philosophy of Mind – Volume 6*
- **Format:** Plain UTF-8 text (`book.txt`)
- **Topics Covered:**  
  Consciousness, intentionality, introspection, the extended mind, representationalism, functionalism, etc.

  📥 Download the book (text format):
https://drive.google.com/file/d/1k4bFE3mKHrrfPrSYaBMX-y5-bYU9R_m2/view

---

## ❓ Suitable Questions

This system is optimized for **semantic, conceptual, and philosophical questions**, such as:

- What are the limitations of introspection in modern philosophy of mind?
- How did the extended mind thesis challenge traditional cognitive boundaries?
- How does functionalism differ from identity theory in explaining mental states?
- Why is representational content crucial to theories of perception?

It is **not optimized** for simple fact lookup or year-based factual queries.

---

## 🧠 Models Used

| Model           | Use                           | Source (Local Folder)   |
|-----------------|--------------------------------|--------------------------|
| MPNet           | Embedding & semantic search    | `/models/mpnet_local/`   |
| Phi-2 (Causal)  | Language generation            | `/models/phi-2/`         |

You may also replace them with Hugging Face versions:

```python

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

```

---

## 💻 How to Run in Google Colab

To run this project in Google Colab:

### 🔹 Step 1 – Upload required files

In the **left sidebar → Files tab** of Colab, upload these from your `project/` folder:

```
📁 /content/
├── book.txt
├── models/
│   ├── phi-2/
│   └── mpnet_local/
├── test/
│   ├── test.txt
│   ├── main.py
```
---

### 🔹 Step 2 – Install dependencies

Run the following in a new code cell:

```python
!pip install transformers sentence-transformers
```

---

### 🔹 Step 3 – Adjust paths (if needed)

Make sure inside `main.py` the paths match:

```python
book_path        = "/content/book.txt"
questions_path   = "/content/test/test.txt"
output_path      = "/content/test/result.txt"
phi2_model_path  = "/content/models/phi-2"
mpnet_model_path = "/content/models/mpnet_local"
```

---

### 🔹 Step 4 – Run main script

You can either:

- Run the full contents of `main.py` in cells, or  
- Use the `%run` command in Colab:

```python
%run /content/test/main.py
```

---

### 🟢 Output

A file named `result.txt` will be created in `/content/test/`, containing:

- Each question
- The generated answer
- A semantic similarity score (0–100)
- Time taken to generate

---

## 🔗 External Links

- 📄 [Google Doc Report]()*
- 📥 [Phi-2 Model (HF)](https://huggingface.co/microsoft/phi-2)
- 📥 [MPNet Model (HF)](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

---
