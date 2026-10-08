# HR Policy Assistant (RAG) - Meher Sambalpuri Fashion

**A hallucination-free HR chatbot built with RAG (Retrieval-Augmented Generation).** Ask HR questions in plain English; answers come **only** from the HR policy PDF, with clause numbers and source text shown.

## Quick start (about 2 minutes)
```bash
cd hr-policy-rag
python -m venv venv
venv\Scripts\activate          # Windows      |  macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open **http://127.0.0.1:5050**. That's it, no API key needed.

## Two modes
| Mode | When | Answer style |
|---|---|---|
| Offline (default) | no key | Quotes the best matching clauses from the PDF |
| AI (Claude) | `ANTHROPIC_API_KEY` set | Claude writes a short friendly answer using ONLY the retrieved clauses |

To enable AI mode: copy `.env.example` to `.env`, paste your key, restart. If the AI call ever fails, the app silently falls back to offline mode.

## How the RAG pipeline works
1. **Ingest** (`rag.py`): `pypdf` extracts text, removes headers/footers/table-of-contents.
2. **Chunk**: split by numbered clauses (4.2, 6.1...) so each chunk is one complete rule; tables become their own chunk. Unnumbered PDFs fall back to 900-character overlapping windows.
3. **Index**: TF-IDF (unigrams + bigrams, light stemming) plus a small HR synonym map (e.g. "wfh" = work from home).
4. **Retrieve**: cosine similarity, top 4 clauses. Below a score threshold the bot says "not found, contact HR" instead of guessing.
5. **Answer**: Claude (optional) or the extractive answerer, citing clause numbers.

## Using your own PDF
Click **Upload PDF** in the sidebar (text-based PDFs, up to 15 MB). **Use sample** restores the Meher handbook.

## Files
```
app.py              Flask server + API (/api/ask, /api/upload, /api/status, /api/reset, /api/pdf)
rag.py              RAG engine (chunking, retrieval, answering)
static/index.html   Web UI (single file, no build step)
data/               Policy PDF(s)
tests/test_app.py   Run: python tests/test_app.py
requirements.txt  .env.example  README.md
```

## Ideas to level up (great for your resume)
- Swap TF-IDF for embeddings (`sentence-transformers` + FAISS/Chroma) and compare retrieval accuracy.
- Add a small evaluation set (question -> expected clause) and report top-1 / top-3 accuracy.
- Add login + per-employee questions log for HR analytics; deploy on Render / Railway.

## Hallucination guards
- Answers are quoted from retrieved clauses (or, in AI mode, written from them only), always with clause numbers.
- Questions the policy does not cover (images, prices, CEO, stock options, jokes...) are refused with "contact HR" instead of a guess.
- Typos and Hinglish ("casul leave kitne milte") are still understood.

## Browser-only version (no server needed)
`index.html` (root, same as `static/standalone.html`) is a single-file version that runs entirely in the browser (retrieval in JavaScript), including PDF upload,
PDF viewer and a product illustration gallery. Open it directly, or host it free with **GitHub Pages**
(repo Settings > Pages > deploy from branch `main`, folder `/ (root)`; the root `index.html` is this same page, so your site opens directly at `https://<username>.github.io/hr-policy-rag/`).
PDF upload and viewing need internet once (the PDF reader library loads from a CDN).

## Tech stack
Python, Flask, pypdf, scikit-learn (TF-IDF), vanilla JavaScript/HTML/CSS, optional Anthropic Claude API.

## Push to GitHub
```bash
git init
git add .
git commit -m "HR Policy Assistant: RAG chatbot for Meher Sambalpuri Fashion"
git branch -M main
git remote add origin https://github.com/<your-username>/hr-policy-rag.git
git push -u origin main
```
Sample company and policies are fictional, created for learning.
