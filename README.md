# NLP_project
# 🧠 Semantic-QA System for Philosophy of Mind

This project implements a semantic question-answering (QA) system tailored for philosophical questions, particularly in the field of **20th and 21st Century Philosophy of Mind**.

It uses transformer-based models to retrieve semantically relevant content from a source book and generate accurate, self-contained, and academically toned answers to conceptual questions.

---

## ✅ Project Description

The system performs the following key steps:

1. **Reads a philosophical text file** (`book.txt`)
2. **Splits it into overlapping chunks** of ~250 words
3. **Computes embeddings** of all chunks using MPNet
4. **Embeds each question** and finds top 5 semantically similar chunks
5. **Builds a structured prompt** with context and question
6. **Generates an answer** using a causal language model (phi-2)
7. **Scores the answer** based on embedding similarity
8. **Writes outputs** (answer, score, time) to `result.txt`

---

## 📚 Source Book

- **Title:** *20th and 21st Century Philosophy of Mind*
- **Format:** UTF-8 plain text file
- **Content:** Covers key topics like consciousness, intentionality, the mind-body problem, extended cognition, introspection, etc.

📥 **Download the book (text format):**  
https://drive.google.com/file/d/1k4bFE3mKHrrfPrSYaBMX-y5-bYU9R_m2/view

---

## ❓ Sample Questions It Can Answer Well

This system is optimized for **high-level conceptual questions**, such as:

- *What is the extended mind hypothesis, and under what conditions can external tools be part of cognition?*  
- *How did behaviorism and functionalism approach the mind-body problem differently?*  
- *What role does introspection play in modern theories of consciousness?*  
- *How is intentionality defined and debated in 20th-century thought?*

Avoid using purely factual questions (e.g., "Who said X in year Y?") as the system is built for **semantic understanding**, not lookup.

---

## 🔍 Models Used

1. **MPNet (local version)**  
   For semantic similarity and chunk retrieval.  
   🔗 [https://huggingface.co/sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

2. **Phi-2 (local version)**  
   Lightweight causal language model for answer generation.  
   🔗 [https://huggingface.co/microsoft/phi-2](https://huggingface.co/microsoft/phi-2)

---

