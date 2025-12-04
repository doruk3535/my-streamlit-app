import streamlit as st
import zipfile
from pathlib import Path
import re
import os
import numpy as np

# ============================================================
# 0. Page config
# ============================================================

st.set_page_config(
    page_title="CodeDocMate Lite",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# 1. Basic settings
# ============================================================

SUPPORTED_EXT = [".py"]  # şimdilik sadece Python dosyaları


# ============================================================
# 2. Helper functions: files & chunking
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


def extract_function_name_from_query(query: str):
    """
    Soru içinden fonksiyon ismini yakalamaya çalışır.
    Örnek: "What does calculate_loss() do?" -> "calculate_loss"
    """
    match = re.search(r"([a-zA-Z_]\w*)\s*\(", query)
    if match:
        return match.group(1)
    return None


# ============================================================
# 3. Simple local retrieval (no embeddings, no LLM)
# ============================================================

def score_chunk(chunk_code: str, query: str, func_name: str | None = None) -> int:
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


def retrieve_top_k(chunks, query: str, func_name: str | None = None, k: int = 3):
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
# 4. Explanation (stub / rule-based)
# ============================================================

def local_explanation(chunks, query: str) -> str:
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
### Prototype explanation (no LLM)

You asked:

> {query}

CodeDocMate Lite searched your project and found these relevant code regions:

{joined}

This **local-only prototype** does not use any LLM or external API.
It relies on simple keyword and function-name matching across your codebase.
In a full RAG + LLM version, this step would call an LLM with these snippets
to generate a more natural, high-level explanation.
"""
    return explanation


# ============================================================
# 5. UI Layout
# ============================================================

# --- Sidebar ---
st.sidebar.title("🧠 CodeDocMate Lite")
st.sidebar.caption("Local-only, no-LLM prototype")

st.sidebar.markdown(
    """
**Status**

- 🌐 No OpenAI / LLM
- 🗂 In-memory chunk store
- 🔍 Simple keyword-based retrieval
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Made for experimenting with RAG-like flows **without** any API key."
)

# --- Main title ---
st.title("CodeDocMate Lite – Local-Only RAG Prototype")

st.markdown(
    """
This version of **CodeDocMate** works **without** any LLM or OpenAI API.

- Upload a zipped Python project.
- The system scans and chunks your `.py` files.
- When you ask a question, it uses simple keyword search
  and function-name matching to find relevant code chunks.
- It then builds a **prototype explanation** around those chunks.

Use this to **demo the pipeline** without any external costs.
"""
)

# Tabs for a cleaner, more modern layout
tab_upload, tab_ask, tab_overview = st.tabs(
    ["1️⃣ Upload & Index", "2️⃣ Ask About Code", "ℹ️ System Overview"]
)

# ============================================================
# 6. Tab 1 – Upload & index
# ============================================================

with tab_upload:
    st.header("1. Upload and index project")

    uploaded_zip = st.file_uploader("Upload your project (.zip)", type=["zip"])

    default_project_id = "demo-project"
    if uploaded_zip is not None:
        default_project_id = uploaded_zip.name
        if default_project_id.lower().endswith(".zip"):
            default_project_id = default_project_id[:-4]

    project_id = st.text_input("Project ID", value=default_project_id)

    if uploaded_zip is not None and project_id:
        st.info("Zip file received. Click the button below to process and index the project.")

        if st.button("Process & Index Project", type="primary"):
            project_root = Path("projects") / project_id
            root = save_and_extract_zip(uploaded_zip, project_root)

            code_files = find_code_files(root)
            st.write(f"Found **{len(code_files)}** Python files:")

            for p in code_files:
                st.write(f"- `{p.relative_to(root)}`")

            all_chunks = []
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
            st.session_state["chunks"] = all_chunks
            st.success("Project is indexed in memory (local-only).")


# ============================================================
# 7. Tab 2 – Ask question
# ============================================================

with tab_ask:
    st.header("2. Ask a question about the code")

    if "chunks" not in st.session_state or not st.session_state["chunks"]:
        st.warning("No project is indexed yet. Please go to **Upload & Index** first.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            default_q = "What does the function calculate_loss() do?"
            query = st.text_area("Your question:", value=default_q, height=100)

        with col2:
            st.markdown("**Explanation mode**")
            st.radio(
                "",
                ["Stub (no LLM, local-only)"],
                index=0,
                key="mode_radio",
            )
            st.caption(
                "In this Lite version, only a local stub explanation is available.\n"
                "RAG + LLM mode is disabled because there is no API usage."
            )

        if st.button("Retrieve & Explain", type="primary"):
            chunks = st.session_state["chunks"]

            func_name = extract_function_name_from_query(query)
            if func_name:
                st.info(f"Detected function name in query: `{func_name}`")
            else:
                st.info("No explicit function name detected in the question.")

            top_chunks = retrieve_top_k(chunks, query, func_name=func_name, k=3)

            explanation_md = local_explanation(top_chunks, query)
            st.markdown(explanation_md)

            with st.expander("Show retrieved chunks in full"):
                for i, ch in enumerate(top_chunks, start=1):
                    st.markdown(f"#### Chunk {i} – `{ch['path']}`")
                    st.code(ch["code"], language="python")


# ============================================================
# 8. Tab 3 – System overview
# ============================================================

with tab_overview:
    st.header("System Overview (Lite Version)")

    st.markdown(
        """
**CodeDocMate Lite** illustrates the overall RAG-style workflow **without** any LLM:

1. The user uploads a zipped project.
2. The system extracts the archive and scans for supported source files (currently `.py`).
3. Each file is split into smaller, line-based chunks for more fine-grained retrieval.
4. When the user asks a natural-language question, the system:
   - Tries to detect a function name (e.g., `calculate_loss`)
   - Performs simple keyword matching across all chunks
5. The most relevant chunks are returned and shown to the user together with
   a prototype, rule-based explanation.

In a **full RAG + LLM** version, these chunks would be passed to an LLM
(e.g., GPT-4.x) to generate high-level, human-readable documentation.
Here, we keep everything **local and API-free** to make the pipeline
easy to demo and safe to run anywhere.
"""
    )
