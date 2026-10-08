"""Flask server for the HR Policy Assistant.  Run:  python app.py"""
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

import rag

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DEFAULT_PDF = DATA / "Meher_Sambalpuri_Fashion_HR_Policy_Handbook.pdf"
UPLOADED_PDF = DATA / "uploaded_policy.pdf"
UPLOADED_NAME = DATA / "uploaded_name.txt"


def load_env():
    f = BASE / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
DATA.mkdir(exist_ok=True)
app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
lock = threading.Lock()
engine = rag.Engine()
load_error = ""


def active_pdf():
    return UPLOADED_PDF if UPLOADED_PDF.exists() else DEFAULT_PDF


def load_active():
    global engine, load_error
    path = active_pdf()
    is_default = path == DEFAULT_PDF
    name = path.name
    if not is_default and UPLOADED_NAME.exists():
        name = UPLOADED_NAME.read_text(encoding="utf-8").strip() or name
    try:
        if not path.exists():
            raise FileNotFoundError("No policy PDF found in the data/ folder.")
        engine = rag.Engine().build(path, name, is_default)
        load_error = ""
    except Exception as e:
        engine, load_error = rag.Engine(), str(e)


load_active()


def err(msg, code=400):
    return jsonify({"error": msg}), code


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    info = engine.info()
    info["error"] = load_error
    info["suggestions"] = (
        ["How many casual leaves do I get?", "When is salary credited?", "What is the notice period?",
         "Can I work from home?", "What is the maternity leave policy?", "How do I report harassment?",
         "Which festivals are paid holidays?", "What discount do employees get on products?"]
        if info["is_default"] else
        ["What is the leave policy?", "What are the working hours?", "How is salary paid?", "What is the notice period?"])
    return jsonify(info)


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    q = str(data.get("question", "")).strip()
    if not q:
        return err("Please type a question.")
    if len(q) > 500:
        return err("Question is too long (max 500 characters).")
    history = data.get("history") if isinstance(data.get("history"), list) else []
    history = [h for h in history if isinstance(h, dict)]
    try:
        with lock:
            eng = engine
        return jsonify(eng.answer(q, history))
    except Exception as e:
        print("[app] ask failed:", e)
        return err("Something went wrong while answering. Please try again.", 500)


@app.post("/api/upload")
def upload():
    global engine, load_error
    f = request.files.get("file")
    if not f or not f.filename:
        return err("Please choose a PDF file.")
    if not f.filename.lower().endswith(".pdf"):
        return err("Only PDF files are supported.")
    tmp = DATA / "_incoming.pdf"
    f.save(tmp)
    try:
        if not tmp.read_bytes()[:5].startswith(b"%PDF"):
            raise ValueError("That file doesn't look like a valid PDF.")
        new = rag.Engine().build(tmp, f.filename, False)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return err(f"Could not read this PDF: {e}")
    with lock:
        os.replace(tmp, UPLOADED_PDF)
        UPLOADED_NAME.write_text(f.filename, encoding="utf-8")
        engine, load_error = new, ""
    return jsonify({"ok": True, "name": f.filename})


@app.post("/api/reset")
def reset():
    with lock:
        UPLOADED_PDF.unlink(missing_ok=True)
        UPLOADED_NAME.unlink(missing_ok=True)
        load_active()
    return jsonify({"ok": True})


@app.get("/api/pdf")
def pdf():
    p = active_pdf()
    return send_file(p, mimetype="application/pdf") if p.exists() else err("No PDF loaded.", 404)


@app.errorhandler(413)
def too_big(_):
    return err("File is too large (max 15 MB).", 413)


@app.errorhandler(404)
def nf(_):
    return err("Not found.", 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    mode = "AI mode (Claude)" if os.environ.get("ANTHROPIC_API_KEY") else "Offline mode (no API key needed)"
    print(f"\n  HR Policy Assistant running in {mode}\n  Open  http://127.0.0.1:{port}  in your browser\n")
    app.run(host="127.0.0.1", port=port, debug=False)
