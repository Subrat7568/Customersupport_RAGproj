"""Run:  python tests/test_app.py   (all checks should print OK)"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as A

c = A.app.test_client()
s = c.get("/api/status").get_json()
assert s["ready"] and s["chunks"] > 50 and len(s["sections"]) >= 20, s
print("OK status:", s["chunks"], "chunks,", len(s["sections"]), "sections")

def top(q):
    r = c.post("/api/ask", json={"question": q}).get_json()
    return r, [x["clause"] for x in r["sources"]]

for q, want in [("When is salary credited?", "6.1"), ("Can I work from home?", "19.1"), ("How do I report sexual harassment?", "10.2"),
                ("Can I use ChatGPT for work?", "17.6"), ("What discount do I get on products?", "6.7"),
                ("How many casual leaves do I get?", "4 (table)"), ("What is the gratuity rule?", "6.5")]:
    r, cl = top(q)
    assert r["found"] and want in cl[:2], (q, cl)
    print("OK", q, "->", cl[:3])

for q in ["show some images of sambalpuri saree", "what is the price of sambalpuri saree", "Is there a gym in the office?", "Do you offer stock options?", "tell me a joke"]:
    assert not top(q)[0]["found"], q
print("OK hallucination guards (images/price/gym/stock/joke refused)")
for q in ["What happens if I am late?", "casul leave kitne milte", "Does the company provide medical insurance?"]:
    assert top(q)[0]["found"], q
print("OK typos / Hinglish still answered")
r, _ = top("Who won the cricket world cup?")
assert not r["found"]; print("OK out-of-scope question handled")
assert c.post("/api/ask", json={"question": ""}).status_code == 400
assert c.post("/api/ask", data="garbage").status_code == 400
assert c.post("/api/upload", data={"file": (io.BytesIO(b"not a pdf"), "x.pdf")}, content_type="multipart/form-data").status_code == 400
assert c.post("/api/upload", data={"file": (io.BytesIO(b"hi"), "x.txt")}, content_type="multipart/form-data").status_code == 400
print("OK bad input handled"); 
pdf = A.DEFAULT_PDF.read_bytes()
u = c.post("/api/upload", data={"file": (io.BytesIO(pdf), "My_Policy.pdf")}, content_type="multipart/form-data")
assert u.status_code == 200 and c.get("/api/status").get_json()["name"] == "My_Policy.pdf"
assert c.post("/api/reset").status_code == 200 and c.get("/api/status").get_json()["is_default"]
print("OK upload + reset"); assert c.get("/").status_code == 200 and c.get("/api/pdf").status_code == 200
print("\nALL TESTS PASSED")
