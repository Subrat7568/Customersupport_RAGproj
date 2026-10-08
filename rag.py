"""Core RAG engine: PDF -> clause-aware chunks -> TF-IDF retrieval -> grounded answer.

Works fully offline. If ANTHROPIC_API_KEY is set, Claude writes the final answer
using ONLY the retrieved policy text; otherwise a built-in extractive answerer is used.
"""
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

TOP_K = 4
MIN_SCORE = 0.07          # below this, the question is treated as "not in the policy"
MAX_HISTORY = 6

STOP = set(ENGLISH_STOP_WORDS) - {"leave", "off"}
SYNONYMS = {
    "wfh": "work from home remote", "remote": "work from home", "home": "work from home remote",
    "salary": "salary payroll payslip paid", "payday": "salary paid date", "paid": "salary payroll",
    "pay": "salary payroll", "credited": "salary paid", "ctc": "salary structure",
    "sick": "sick leave medical", "ill": "sick leave medical", "fever": "sick leave medical",
    "pregnant": "maternity leave", "pregnancy": "maternity leave", "baby": "maternity paternity childbirth",
    "wedding": "marriage leave", "married": "marriage leave", "death": "bereavement leave",
    "vacation": "earned leave privilege", "holiday": "holidays festivals", "festival": "holidays festivals",
    "off": "leave holiday", "resign": "resignation notice period exit", "quit": "resignation notice period",
    "resigning": "resignation notice period", "fired": "termination disciplinary", "terminate": "termination",
    "harass": "harassment POSH complaint", "harassment": "POSH internal committee complaint",
    "insurance": "medical insurance cover", "mediclaim": "medical insurance", "health": "medical insurance health",
    "pf": "provident fund", "hike": "increment appraisal", "raise": "increment appraisal",
    "appraisal": "performance rating increment", "promotion": "promotions career growth",
    "timing": "working hours timings shift", "timings": "working hours shift", "late": "late grace attendance",
    "laptop": "IT assets devices", "chatgpt": "generative AI tools", "ai": "generative AI tools",
    "discount": "employee discount products", "travel": "travel expenses allowance hotel",
    "reimbursement": "expense claims reimbursement", "complaint": "grievance redressal",
    "dress": "dress code uniform", "friday": "handloom day dress", "course": "training certification reimbursement",
    "learning": "training certification", "referral": "referral bonus", "overtime": "overtime double pay comp-off",
    "weaver": "artisan weavers", "weavers": "artisan artisans", "safety": "safety fire PPE accident",
    "canteen": "canteen meals welfare", "bus": "transport shuttle", "retirement": "retirement age",
    "probation": "probation confirmation notice period", "bonus": "bonus festival gift",
    "gift": "festival gift bonus", "confidential": "confidentiality", "gratuity": "gratuity years of service",
}

GENERIC = set("show tell explain give list please need want know detail details information info policy policies company employee employees staff rule rules handbook hr say ask kitne kitni milte milta milti hai hain kya ka ki ke mujhe mera meri kab kaise kyun kyu nahi nhi karna kar sakta sakte sakti chahiye batao bata mein me se ko aur ya par tak wala wali hota hoti hote lagta lagte happen happens happened apply applies applicable yaar bhai".split())
IMG_RE = [re.compile(r"\b(show|display|see|view|give|send|share|get|find|generate|create|make|draw)\b[^.?!]*\b(images?|pictures?|photos?|pics?|videos?)\b"),
          re.compile(r"\b(images?|pictures?|photos?|pics?)\s+of\b")]
OFF_RE = re.compile(r"\b(joke|poem|song|story|python|javascript|recipe|weather|news|cricket|movie|bitcoin|stock market|horoscope|price|prices|mrp|catalogue|catalog)\b")
HI_RE = re.compile(r"^\s*(hi+|hello|hey|namaste|namaskar|good\s(morning|afternoon|evening)|thanks?|thank you|ok(ay)?|bye)\b[\s!.?]*$")
CLAUSE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\s+(\S.*)$")
SECTION_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][^\n]{3,90})$")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def stem(w):
    for suf in ("ing", "ed", "s"):
        if len(w) > 4 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def tokens(text):
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


def analyzer(text):
    t = tokens(text)
    return t + [a + "_" + b for a, b in zip(t, t[1:])]


def near1(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    i = j = d = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1; continue
        d += 1
        if d > 1:
            return False
        if len(a) > len(b): i += 1
        elif len(a) < len(b): j += 1
        else: i += 1; j += 1
    return d + (len(a) - i) + (len(b) - j) <= 1


def expand(question):
    extra = [SYNONYMS[w] for w in re.findall(r"[a-z0-9]+", question.lower()) if w in SYNONYMS]
    return question + " " + " ".join(extra)


# ----------------------------------------------------------------- PDF -> chunks
def read_pages(path):
    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password-protected.")
    pages = []
    for p in reader.pages:
        try:
            txt = p.extract_text(extraction_mode="layout") or ""
        except Exception:
            txt = p.extract_text() or ""
        pages.append(txt)
    return pages


def _norm(line):
    return re.sub(r"\d+", "#", line.strip())


def clean_pages(pages):
    """Drop repeated headers/footers and table-of-contents pages."""
    n = len(pages)
    cnt = Counter()
    for p in pages:
        for k in {_norm(re.sub(r"\s{2,}", " ", l)) for l in p.splitlines() if l.strip()}:
            cnt[k] += 1
    thr = max(2, int(n * 0.6))
    out = []
    for i, p in enumerate(pages, 1):
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        if n >= 3:
            lines = [l for l in lines if cnt[_norm(re.sub(r"\s{2,}", " ", l))] < thr]
        toc_like = sum(1 for l in lines if re.search(r"(\s{2,}|\.{3,}\s*)\d{1,3}$", l))
        if len(lines) >= 5 and toc_like >= 0.5 * len(lines):
            continue
        lines = [l for l in lines if not re.fullmatch(r"(page\s*)?\d+(\s*(/|of)\s*\d+)?", l, re.I)]
        out.append((i, lines))
    return out


def _finish(entries):
    return "\n".join(e.strip() for e in entries if e.strip())


def chunk_by_clauses(cleaned):
    chunks, cur, sec_no, sec_title, last = [], None, None, "", 0

    def flush():
        nonlocal cur
        if cur:
            text = _finish(cur["entries"])
            if len(text) >= (20 if cur["clause"] else 40):
                chunks.append({"section": cur["section"], "title": cur["title"], "clause": cur["clause"] or f"{cur['section']} (table)",
                               "page": cur["page"], "text": text})
        cur = None

    for pno, lines in cleaned:
        for line in lines:
            m = SECTION_RE.match(line)
            if m and int(m.group(1)) > last:
                flush()
                sec_no, sec_title, last = m.group(1), m.group(2).strip(), int(m.group(1))
                cur = {"section": sec_no, "title": sec_title, "clause": None, "page": pno, "entries": []}
                continue
            m = CLAUSE_RE.match(line)
            if m and sec_no and m.group(1) == sec_no:
                flush()
                cur = {"section": sec_no, "title": sec_title, "clause": f"{m.group(1)}.{m.group(2)}", "page": pno,
                       "entries": [m.group(3)]}
                continue
            if not cur:
                continue
            if re.search(r"\S\s{2,}\S", line):              # table row -> own line, cells joined by |
                cur["entries"].append(re.sub(r"\s{2,}", " | ", line))
            elif cur["entries"]:
                cur["entries"][-1] += " " + line
            else:
                cur["entries"].append(line)
    flush()
    return chunks


def chunk_by_window(cleaned, size=900, overlap=150):
    text, spans = "", []
    for pno, lines in cleaned:
        spans.append((len(text), pno))
        text += " ".join(re.sub(r"\s{2,}", " ", l) for l in lines) + " "
    chunks, i, k = [], 0, 1
    while i < len(text):
        piece = text[i:i + size]
        if i + size < len(text):                              # end on a sentence boundary when possible
            cut = max(piece.rfind(". "), piece.rfind("? "), piece.rfind("! "))
            if cut > size * 0.5:
                piece = piece[:cut + 1]
        piece = piece.strip()
        if len(piece) > 30:
            page = [p for s, p in spans if s <= i][-1]
            chunks.append({"section": str(k), "title": f"Part {k}", "clause": f"p{page}-{k}", "page": page, "text": piece})
            k += 1
        if i + size >= len(text):
            break
        i += max(len(piece) - overlap, 1)
    return chunks


def build_chunks(path):
    raw = read_pages(path)
    cleaned = clean_pages(raw)
    if not any(l for _, ls in cleaned for l in ls):
        raise ValueError("No readable text found. This looks like a scanned PDF (images only) - please upload a text-based PDF.")
    chunks = chunk_by_clauses(cleaned)
    if len({c["section"] for c in chunks}) < 3 or len(chunks) < 5:
        chunks = chunk_by_window(cleaned)
    if not chunks:
        raise ValueError("Could not extract any content from this PDF.")
    return chunks, len(raw)


# ----------------------------------------------------------------- Engine
class Engine:
    def __init__(self):
        self.chunks, self.vec, self.mat = [], None, None
        self.name, self.pages, self.is_default, self.hr_email = "", 0, True, ""

    @property
    def ready(self):
        return self.mat is not None

    def build(self, path, name=None, is_default=True):
        chunks, pages = build_chunks(path)
        vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        mat = vec.fit_transform([f"{c['title']} {c['title']} {c['text']}" for c in chunks])
        self.chunks, self.vec, self.mat = chunks, vec, mat
        self.name, self.pages, self.is_default = name or Path(path).name, pages, is_default
        m = EMAIL_RE.search(" ".join(c["text"] for c in chunks))
        self.hr_email = m.group(0) if m else ""
        return self

    def sections(self):
        seen = {}
        for c in self.chunks:
            seen.setdefault(c["section"], c["title"])
        return [{"no": k, "title": v} for k, v in seen.items()]

    def info(self):
        return {"ready": self.ready, "name": self.name, "pages": self.pages, "chunks": len(self.chunks),
                "is_default": self.is_default, "sections": self.sections(), "ai": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())}

    def search(self, query, k=TOP_K):
        qv = self.vec.transform([expand(query)])
        sims = linear_kernel(qv, self.mat).ravel()
        order = sims.argsort()[::-1][:k]
        return [dict(self.chunks[i], score=round(float(sims[i]), 3)) for i in order if sims[i] > 0]

    def unknown_ratio(self, q):
        vocab = self.vec.vocabulary_
        words = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in STOP and len(w) > 1 and w not in GENERIC and not w.isdigit()]
        u = 0
        for w in words:
            s = stem(w)
            if s in vocab or any(t in vocab for t in tokens(SYNONYMS.get(w, ""))):
                continue
            if len(s) >= 4 and any("_" not in k and near1(s, k) for k in vocab):
                continue
            u += 1
        return u, len(words)

    def _nf(self, text, kind):
        return {"answer": text, "found": False, "mode": "none", "kind": kind, "sources": []}

    # -------- answering
    def answer(self, question, history=None):
        history = history or []
        ql = question.lower()
        if HI_RE.match(ql):
            return self._nf("Namaskar! I'm the HR Policy Assistant. Ask me about leave, salary, holidays, notice period, safety and more, and I'll answer from the handbook.", "chat")
        if any(rx.search(ql) for rx in IMG_RE):
            return self._nf("I can only answer from the HR policy text, and this document contains no images, so I can't show photos.", "images")
        if OFF_RE.search(ql):
            return self._nf("That's outside the HR policy, so I won't guess. I can help with leave, salary, holidays, notice period, safety, benefits and other HR topics.", "off")
        if self.ready:
            u, n = self.unknown_ratio(question)
            if u and u * 3 >= n:
                contact = f" at {self.hr_email}" if self.hr_email else ""
                return self._nf(f"I couldn't find this in the HR policy document, so I don't want to guess. Please contact the HR department{contact} for help.", "nf")
        if not self.ready:
            return {"answer": "No policy document is loaded yet. Please upload an HR policy PDF.", "found": False, "mode": "none", "sources": []}
        prev = next((h["content"] for h in reversed(history) if h.get("role") == "user"), "")
        query = f"{prev} {question}" if prev and len(question.split()) <= 4 else question
        hits = self.search(query)
        good = [h for h in hits if h["score"] >= MIN_SCORE]
        if not good:
            contact = f" at {self.hr_email}" if self.hr_email else ""
            return {"answer": "I couldn't find this in the HR policy document, so I don't want to guess. "
                              f"Please contact the HR department{contact} for help.", "found": False, "mode": "none", "sources": []}
        good = [h for h in good if h["score"] >= 0.4 * good[0]["score"]]
        text, note = call_claude(question, good, history, self.hr_email)
        mode = "ai" if text else "offline"
        if not text:
            text = extractive_answer(question, good)
        return {"answer": text, "found": True, "mode": mode, "note": note,
                "sources": [{"clause": h["clause"], "section": h["title"], "page": h["page"], "text": h["text"], "score": h["score"]} for h in good]}


def extractive_answer(question, hits):
    qt = set(tokens(expand(question)))
    parts = []
    for h in hits[:2]:
        lines = h["text"].split("\n")
        if len(lines) > 1:       # table chunk: show the rows that best match, plus the header row
            ranked = sorted(range(len(lines)), key=lambda i: (-len(qt & set(tokens(lines[i]))), i))
            keep = sorted({0, *[i for i in ranked[:2] if qt & set(tokens(lines[i]))]})
            body = "\n".join("- " + lines[i] for i in keep)
        else:
            sents = re.split(r"(?<=[.!?])\s+", h["text"])
            scored = [(len(qt & set(tokens(s))), i, s) for i, s in enumerate(sents)]
            best = sorted(sorted(scored, key=lambda x: (-x[0], x[1]))[:2], key=lambda x: x[1])
            body = " ".join(s for sc, i, s in best if sc > 0) or " ".join(sents[:2])
        parts.append(f"**[{h['clause']}] {h['title']}**\n{body}")
    return "Here is what the HR policy says:\n\n" + "\n\n".join(parts)


SYSTEM = ("You are the friendly HR Policy Assistant for the company whose handbook is provided in CONTEXT. "
          "Answer ONLY from CONTEXT. If the answer is not in CONTEXT, say you could not find it in the policy and suggest contacting HR. "
          "Never invent numbers, dates or rules. Be concise and clear (max ~120 words), use short bullet points when listing, "
          "and cite the clause numbers you used in square brackets like [4.2]. Reply in the language of the question.")


def call_claude(question, hits, history, hr_email=""):
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None, ""
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    ctx = "\n\n".join(f"[{h['clause']}] (Section: {h['title']})\n{h['text']}" for h in hits)
    msgs, last = [], None
    for h in history[-MAX_HISTORY:]:
        r, c = h.get("role"), str(h.get("content", ""))[:1500]
        if r in ("user", "assistant") and c and r != last and (msgs or r == "user"):
            msgs.append({"role": r, "content": c}); last = r
    if msgs and msgs[-1]["role"] == "user":
        msgs.pop()
    msgs.append({"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"})
    body = {"model": model, "max_tokens": 600, "system": SYSTEM, "messages": msgs, "temperature": 0.1}
    try:
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.load(r)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        return (text or None), ("" if text else "Empty AI reply; showing policy text instead.")
    except Exception as e:                                   # never crash the app because of the LLM
        print(f"[rag] Claude call failed: {e}", file=sys.stderr)
        return None, "AI service unavailable; showing policy text instead."
