import streamlit as st
import zipfile
from pathlib import Path
import re
import os
import numpy as np
from typing import Optional, List, Dict

# ============================================================
# Page config + custom theme (navy blue)
# ============================================================

st.set_page_config(
page_title="CodeDocMate Lite",
page_icon="🧠",
layout="wide",
)

# Custom CSS for navy blue style
st.markdown(
"""
   <style>
   .stApp {
       background: radial-gradient(circle at top left, #1d2a3f 0, #020617 40%, #020617 100%);
       color: #e5e7eb;
   }
   .codemate-hero {
       padding: 1.5rem 1.75rem;
       border-radius: 1.25rem;
       background: linear-gradient(120deg, rgba(56,189,248,0.2), rgba(59,130,246,0.12));
       border: 1px solid rgba(148,163,184,0.4);
       backdrop-filter: blur(12px);
   }
   .codemate-pill {
       display: inline-flex;
       align-items: center;
       gap: 0.4rem;
       padding: 0.15rem 0.7rem;
       border-radius: 999px;
       font-size: 0.75rem;
       background: rgba(15,23,42,0.8);
       border: 1px solid rgba(148,163,184,0.6);
       color: #e5e7eb;
   }
   .codemate-section {
       padding: 1.25rem 1.5rem;
       border-radius: 1.25rem;
       background: rgba(15,23,42,0.9);
       border: 1px solid rgba(30,64,175,0.8);
       box-shadow: 0 18px 40px rgba(15,23,42,0.5);
   }
   .codemate-section-soft {
       padding: 1.0rem 1.25rem;
       border-radius: 1.1rem;
       background: rgba(15,23,42,0.8);
       border: 1px solid rgba(30,64,175,0.4);
   }
   .codemate-metric {
       padding: 0.75rem 0.9rem;
       border-radius: 0.9rem;
       background: rgba(15,23,42,0.9);
       border: 1px solid rgba(148,163,184,0.4);
       font-size: 0.8rem;
   }
   .codemate-metric h3 {
       font-size: 0.85rem;
       margin-bottom: 0.2rem;
       color: #9ca3af;
   }
   .codemate-metric span {
       font-size: 1.1rem;
       font-weight: 600;
       color: #e5e7eb;
   }
   .stButton>button {
       border-radius: 999px;
       padding: 0.5rem 1.3rem;
       border: 1px solid rgba(56,189,248,0.7);
       background: linear-gradient(120deg, #0284c7, #0ea5e9);
       color: white;
       font-weight: 600;
   }
   .stButton>button:hover {
       border-color: #e5e7eb;
       background: linear-gradient(120deg, #0369a1, #0284c7);
   }
   .codemate-badge-muted {
       display:inline-flex;
       align-items:center;
       gap:0.3rem;
       padding:0.1rem 0.5rem;
       border-radius:999px;
       border:1px solid rgba(148,163,184,0.5);
       font-size:0.7rem;
       color:#9ca3af;
   }
   </style>
   """,
unsafe_allow_html=True,
)

# ============================================================
# Settings
# ============================================================

SUPPORTED_EXT = [".py"]  # şimdilik sadece Python dosyaları


# ============================================================
# Helper functions: files & chunking
# ============================================================

def save_and_extract_zip(uploaded_file, extract_root: Path) -> Path:
"""
   ZIP dosyasını diske kaydedip açar.
   Proje kök klasörünün Path'ini döndürür.
   """
extract_root.mkdir(parents=True, exist_ok=True)

zip_path = extract_root / "project.zip"
with open(zip_path, "wb") as f:
f.write(uploaded_file.read())

with zipfile.ZipFile(zip_path, "r") as zf:
zf.extractall(extract_root)

return extract_root


def find_code_files(root: Path):
"""
   Kök klasör altında desteklenen tüm kod dosyalarını bulur.
   """
return [p for p in root.rglob("*") if p.suffix in SUPPORTED_EXT]


def read_file(path: Path) -> str:
"""
   UTF-8 ile güvenli okuma.
   """
return path.read_text(encoding="utf-8", errors="ignore")


def simple_chunk(code: str, max_lines: int = 40):
"""
   Kodu satır sayısına göre basit parçalara böler.
   (Gelişmiş chunking için placeholder.)
   """
lines = code.splitlines()
for i in range(0, len(lines), max_lines):
chunk_lines = lines[i:i + max_lines]
yield "\n".join(chunk_lines)


def extract_function_name_from_query(query: str) -> Optional[str]:
"""
   Soru içinden fonksiyon ismini yakalamaya çalışır.
   Örnek: "What does calculate_loss() do?" -> "calculate_loss"
   """
match = re.search(r"([a-zA-Z_]\w*)\s*\(", query)
if match:
return match.group(1)
return None


# ============================================================
# Simple local retrieval (no embeddings, no LLM)
# ============================================================

def score_chunk(chunk_code: str, query: str, func_name: Optional[str] = None) -> int:
"""
   Çok basit text-based skorlayıcı:
   - Soru içindeki kelimeleri sayar
   - Fonksiyon adı geçiyorsa ekstra puan verir
   """
text = chunk_code.lower()
q = query.lower()
tokens = [t for t in re.findall(r"\w+", q) if len(t) >= 3]

score = 0
for tok in tokens:
score += text.count(tok)

if func_name and func_name.lower() in text:
score += 5

return score


def retrieve_top_k(
chunks: List[Dict[str, str]],
query: str,
func_name: Optional[str] = None,
k: int = 3,
) -> List[Dict[str, str]]:
"""
   Basit local retrieval:
   - Her chunk için text skor hesaplar
   - En yüksek skorlu k chunk'ı döndürür
   """
scored = []
for ch in chunks:
s = score_chunk(ch["code"], query, func_name=func_name)
if s > 0:
scored.append((s, ch))

# Hiç skor çıkmazsa, fallback olarak ilk k chunk
if not scored:
return chunks[:k]

scored.sort(key=lambda x: x[0], reverse=True)
top_chunks = [c for _, c in scored[:k]]
return top_chunks


# ============================================================
# Explanation (stub / rule-based)
# ============================================================

def local_explanation(chunks: List[Dict[str, str]], query: str) -> str:
"""
   LLM kullanmadan, geri getirilen kod parçalarına göre
   açıklama üreten stub / kural tabanlı açıklama.
   """
bullet_snippets = []
for ch in chunks:
code_preview = "\n".join(ch["code"].splitlines()[:10])
bullet_snippets.append(
f"- **File:** `{ch['path']}`\n\n"
f"  ```python\n{code_preview}\n  ```\n"
)

joined = "\n".join(bullet_snippets)

explanation = f"""
### 🔍 Prototype explanation (local only)

You asked:

> {query}

CodeDocMate Lite scanned your project and highlighted these code regions as the most relevant:

{joined}

This **Lite** version does **not** call any LLM or external API.
It uses simple keyword and function-name matching to surface potentially
relevant parts of your codebase.

In a full RAG + LLM version, these snippets would be passed into a language model
to generate higher-level documentation and deeper explanations.
"""
return explanation


# ============================================================
# Hero section (top)
# ============================================================

with st.container():
st.markdown(
"""
       <div class="codemate-hero">
         <div class="codemate-pill">
           <span>🧠 CodeDocMate Lite</span>
           <span>Local-only · No API key</span>
         </div>
         <h1 style="margin-top:0.6rem; margin-bottom:0.3rem; font-size:2.0rem;">
            Navy Blue RAG Playground for Codebases
        RAG Playground for Codebases
         </h1>
         <p style="margin:0; font-size:0.95rem; color:#cbd5f5;">
           Upload a zipped Python project, explore its structure, and see which parts of the
           code are most related to your natural-language questions — all without any LLM calls.
         </p>
       </div>
       """,
unsafe_allow_html=True,
)

# Metrics row
col_m1, col_m2, col_m3 = st.columns(3)

num_files = st.session_state.get("num_files", 0)
num_chunks = st.session_state.get("num_chunks", 0)
current_project = st.session_state.get("project_name", "—")

with col_m1:
st.markdown(
f"""
           <div class="codemate-metric">
             <h3>Indexed Python files</h3>
             <span>{num_files}</span>
           </div>
           """,
unsafe_allow_html=True,
)
with col_m2:
st.markdown(
f"""
           <div class="codemate-metric">
             <h3>Total code chunks</h3>
             <span>{num_chunks}</span>
           </div>
           """,
unsafe_allow_html=True,
)
with col_m3:
st.markdown(
f"""
           <div class="codemate-metric">
             <h3>Active project</h3>
             <span>{current_project}</span>
           </div>
           """,
unsafe_allow_html=True,
)

st.markdown("")  # küçük boşluk


# ============================================================
# Main layout – two big sections
# ============================================================

left_col, right_col = st.columns([1.15, 1.1])

# ------------------ LEFT: Upload & index ---------------------

with left_col:
st.markdown('<div class="codemate-section">', unsafe_allow_html=True)

st.subheader("1. Upload & index project", anchor=False)

uploaded_zip = st.file_uploader(
"Upload your project (.zip)",
type=["zip"],
help="For the Lite version, only .py files inside the ZIP are used.",
)

default_project_id = "demo-project"
if uploaded_zip is not None:
default_project_id = uploaded_zip.name
if default_project_id.lower().endswith(".zip"):
default_project_id = default_project_id[:-4]

project_id = st.text_input("Project ID", value=default_project_id)

if uploaded_zip is not None and project_id:
st.info(
"ZIP file received. When you click the button below, "
"CodeDocMate Lite will extract your project, scan for `.py` files, "
"and create line-based chunks for local retrieval."
)

if st.button("Process & Index Project", key="index_btn"):
project_root = Path("projects") / project_id
root = save_and_extract_zip(uploaded_zip, project_root)

code_files = find_code_files(root)

st.write(f"Found **{len(code_files)}** Python files:")
for p in code_files:
st.write(f"- `{p.relative_to(root)}`")

all_chunks: List[Dict[str, str]] = []
for p in code_files:
code = read_file(p)
for ch in simple_chunk(code, max_lines=40):
all_chunks.append(
{
"path": str(p.relative_to(root)),
"code": ch,
}
)

st.write(f"Created **{len(all_chunks)}** code chunks.")
st.success("Project is indexed in memory (local-only, no embeddings).")

# Update session state for metrics & retrieval
st.session_state["chunks"] = all_chunks
st.session_state["num_files"] = len(code_files)
st.session_state["num_chunks"] = len(all_chunks)
st.session_state["project_name"] = project_id

elif uploaded_zip is None:
st.caption("Tip: start with a small sample project to quickly test the flow.")

st.markdown("</div>", unsafe_allow_html=True)


# ------------------ RIGHT: Ask question ----------------------

with right_col:
st.markdown('<div class="codemate-section">', unsafe_allow_html=True)

st.subheader("2. Ask a question about the code", anchor=False)

if "chunks" not in st.session_state or not st.session_state["chunks"]:
st.warning(
"No project is indexed yet. Upload and process a project on the left first."
)
else:
col_q1, col_q2 = st.columns([1.8, 1.2])

with col_q1:
default_q = "What does the function calculate_loss() do?"
query = st.text_area(
"Your question",
value=default_q,
height=120,
help="Ask about functions, data flow, preprocessing, training logic, etc.",
)

with col_q2:
st.markdown(
"""
               <div class="codemate-section-soft">
                 <div class="codemate-badge-muted">
                   <span>⚙️ Mode</span><span>Local stub</span>
                 </div>
                 <p style="margin:0.4rem 0 0.3rem; font-size:0.8rem; color:#e5e7eb;">
                   This Lite build does not call any LLM.
                   It uses keyword and function-name matching only.
                 </p>
                 <ul style="font-size:0.78rem; padding-left:1.1rem; margin-top:0.3rem;">
                   <li>Mention target functions like <code>calculate_loss()</code>.</li>
                   <li>Ask about model inputs, outputs, or preprocessing.</li>
                   <li>Compare two functions or files.</li>
                 </ul>
               </div>
               """,
unsafe_allow_html=True,
)

if st.button("Retrieve & explain", key="explain_btn"):
chunks = st.session_state["chunks"]

func_name = extract_function_name_from_query(query)
if func_name:
st.info(f"Detected function name in question: `{func_name}`")
else:
st.info("No explicit function name detected in the question.")

top_chunks = retrieve_top_k(
chunks,
query,
func_name=func_name,
k=3,
)

explanation_md = local_explanation(top_chunks, query)
st.markdown(explanation_md)

with st.expander("Show retrieved chunks in full", expanded=False):
for i, ch in enumerate(top_chunks, start=1):
st.markdown(f"#### Chunk {i} – `{ch['path']}`")
st.code(ch["code"], language="python")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Bottom: System overview section
# ============================================================

st.markdown("")
st.markdown('<div class="codemate-section-soft">', unsafe_allow_html=True)

st.markdown("### System overview (Lite build)")
st.markdown(
"""
**CodeDocMate Lite** demonstrates a RAG-like flow with a fully local stack:

1. You upload a zipped project.
2. The app extracts it and scans for supported source files (currently `.py`).
3. Each file is split into line-based chunks to allow more focused retrieval.
4. When you ask a natural-language question, the system:
  - Tries to detect a function name like `calculate_loss`.
  - Performs keyword + function-name matching over all chunks.
5. The highest-scoring chunks are shown alongside a rule-based explanation.

This navy blue Lite version is ideal for:
- Presentations where no API key is available.
- Explaining the *idea* of RAG on top of a codebase.
- Quickly testing how questions map to different regions of your project.
"""
)

st.markdown("</div>", unsafe_allow_html=True)
