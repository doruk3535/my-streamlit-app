from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ============================================================
# Page config + basit navy blue tema
# ============================================================

st.set_page_config(
    page_title="CodeDocMate Lite",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f2937 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
    }
    .codemate-hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1.3rem;
        background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(56,189,248,0.12));
        border: 1px solid rgba(148,163,184,0.6);
        backdrop-filter: blur(12px);
    }
    .codemate-pill {
        display:inline-flex;
        align-items:center;
        gap:0.35rem;
        padding:0.12rem 0.7rem;
        border-radius:999px;
        font-size:0.75rem;
        background:rgba(15,23,42,0.9);
        border:1px solid rgba(148,163,184,0.7);
        color:#e5e7eb;
    }
    .codemate-section {
        padding: 1.1rem 1.3rem;
        border-radius: 1.1rem;
        background: rgba(15,23,42,0.95);
        border: 1px solid rgba(30,64,175,0.9);
        box-shadow: 0 20px 40px rgba(15,23,42,0.6);
    }
    .codemate-metric {
        padding: 0.7rem 0.9rem;
        border-radius: 0.9rem;
        background: rgba(15,23,42,0.96);
        border: 1px solid rgba(148,163,184,0.55);
        font-size: 0.8rem;
    }
    .codemate-metric h3 {
        font-size: 0.8rem;
        margin-bottom: 0.15rem;
        color: #9ca3af;
    }
    .codemate-metric span {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.3rem;
        border: 1px solid rgba(56,189,248,0.7);
        background: linear-gradient(120deg, #0369a1, #0ea5e9);
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stButton>button:hover {
        border-color: #e5e7eb;
        background: linear-gradient(120deg, #075985, #0284c7);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
        max-width: 1200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Veri yapıları
# ============================================================

SUPPORTED_EXT = [".py"]  # istersek ileride .js, .java ekleriz


@dataclass
class Chunk:
    id: int
    file_path: str
    start_line: int
    end_line: int
    text: str


@dataclass
class ProjectIndex:
    project_id: str
    root_dir: Path
    files: List[Dict[str, int]]      # {path, n_lines, n_chunks}
    chunks: List[Chunk]
    vectorizer: TfidfVectorizer
    tfidf_matrix: np.ndarray         # (n_chunks, vocab)


# ============================================================
# Yardımcı fonksiyonlar – dosya & chunk
# ============================================================

def save_and_extract_zip(uploaded_file, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    zip_path = extract_root / "project.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.read())
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    return extract_root


def find_code_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix in SUPPORTED_EXT]


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def simple_chunk(code: str, max_lines: int = 45) -> List[tuple[int, int, str]]:
    lines = code.splitlines()
    out: List[tuple[int, int, str]] = []
    for i in range(0, len(lines), max_lines):
        chunk_lines = lines[i:i + max_lines]
        start = i + 1
        end = i + len(chunk_lines)
        out.append((start, end, "\n".join(chunk_lines)))
    return out


def extract_function_name_from_query(query: str) -> Optional[str]:
    import re
    m = re.search(r"([a-zA-Z_]\w*)\s*\(", query)
    if m:
        return m.group(1)
    return None


# ============================================================
# Indexing & retrieval (TF-IDF)
# ============================================================

def build_project_index(uploaded_zip, project_id: str) -> ProjectIndex:
    root_dir = save_and_extract_zip(uploaded_zip, Path("projects") / project_id)
    code_files = find_code_files(root_dir)

    files: List[Dict[str, int]] = []
    chunks: List[Chunk] = []
    texts: List[str] = []
    cid = 0

    for path in code_files:
        code = read_file(path)
        file_chunks = simple_chunk(code, max_lines=45)

        for start, end, text in file_chunks:
            chunks.append(
                Chunk(
                    id=cid,
                    file_path=str(path.relative_to(root_dir)),
                    start_line=start,
                    end_line=end,
                    text=text,
                )
            )
            texts.append(text)
            cid += 1

        files.append(
            {
                "path": str(path.relative_to(root_dir)),
                "n_lines": len(code.splitlines()),
                "n_chunks": len(file_chunks),
            }
        )

    if not texts:
        raise RuntimeError("No supported source files found in project.")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        max_features=20000,
    )
    tfidf_sparse = vectorizer.fit_transform(texts)
    tfidf_matrix = normalize(tfidf_sparse, norm="l2", axis=1).toarray()

    return ProjectIndex(
        project_id=project_id,
        root_dir=root_dir,
        files=files,
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )


def retrieve_top_k(
    index: ProjectIndex,
    query: str,
    func_name: Optional[str],
    k: int = 5,
) -> List[tuple[Chunk, float]]:
    query_text = query
    if func_name:
        query_text += f" function {func_name} {func_name}"

    q_vec = index.vectorizer.transform([query_text])
    q_vec = normalize(q_vec, norm="l2", axis=1).toarray()[0]

    sims = np.dot(index.tfidf_matrix, q_vec)  # (n_chunks,)
    top_idx = np.argsort(sims)[::-1][:k]

    results: List[tuple[Chunk, float]] = []
    for idx in top_idx:
        score = float(sims[idx])
        results.append((index.chunks[idx], score))

    return results


def build_explanation(chunks_with_scores: List[tuple[Chunk, float]], query: str) -> str:
    lines = [
        "### 🔍 Local explanation (no LLM)\n",
        "You asked:\n",
        f"> {query}\n",
        "CodeDocMate Lite highlighted these chunks as the most relevant:\n",
    ]
    for ch, score in chunks_with_scores:
        score_pct = round(score * 100, 1)
        preview = "\n".join(ch.text.splitlines()[:10])
        lines.append(
            f"- **File:** `{ch.file_path}` · lines **{ch.start_line}–{ch.end_line}** "
            f"· relevance ~ **{score_pct}%**\n\n"
            f"  ```python\n{preview}\n  ```\n"
        )
    lines.append(
        "\nThis build does **not** call any LLM. It only uses TF-IDF over your code "
        "to find relevant regions and wraps them in a simple explanation template."
    )
    return "\n".join(lines)


# ============================================================
# Session state
# ============================================================

if "project_index" not in st.session_state:
    st.session_state.project_index = None  # type: ignore


# ============================================================
# Hero + metrics
# ============================================================

pi: Optional[ProjectIndex] = st.session_state.project_index

n_files = len(pi.files) if pi else 0
n_chunks = len(pi.chunks) if pi else 0
pname = pi.project_id if pi else "—"

st.markdown(
    f"""
    <div class="codemate-hero">
      <div class="codemate-pill">
        <span>🧠 CodeDocMate Lite</span>
        <span>Local · No API key</span>
      </div>
      <h1 style="margin-top:0.6rem; margin-bottom:0.3rem; font-size:1.9rem;">
        Minimal navy-blue code understanding playground
      </h1>
      <p style="margin:0; font-size:0.95rem; color:#e5e7eb;">
        Upload a zipped Python project, index it into chunks and ask natural-language
        questions to see which parts of the code are most relevant — fully offline, no LLM.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""
        <div class="codemate-metric">
          <h3>Indexed Python files</h3>
          <span>{n_files}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="codemate-metric">
          <h3>Total code chunks</h3>
          <span>{n_chunks}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="codemate-metric">
          <h3>Active project</h3>
          <span>{pname}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# ============================================================
# 3 sekmeli ana layout
# ============================================================

tab_overview, tab_qna, tab_explorer = st.tabs(
    ["Overview", "Project & Q&A", "Explorer"]
)

# ---------------------- Overview -----------------------------

with tab_overview:
    st.markdown('<div class="codemate-section">', unsafe_allow_html=True)
    st.subheader("Overview", anchor=False)
    st.markdown(
        """
        **CodeDocMate Lite** = küçük ama akıllı bir kod keşif aracı.

        - Projeyi `.zip` olarak yüklüyorsun  
        - `.py` dosyalarını bulup ~45 satırlık chunk’lere bölüyoruz  
        - Her chunk için TF-IDF vektörü çıkarıyoruz  
        - Soru sorduğunda, en alakalı chunk’leri bulup gösteriyoruz  

        Bu sürüm:
        - 🌐 Hiçbir LLM / OpenAI API kullanmıyor  
        - 🧠 Sadece TF-IDF + cosine similarity  
        - 🎯 Demo, ders, sunum ve denemeler için ideal  
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Project & Q&A ------------------------

with tab_qna:
    st.markdown('<div class="codemate-section">', unsafe_allow_html=True)
    st.subheader("Upload project & ask questions", anchor=False)

    col_left, col_right = st.columns([1.1, 1.2])

    with col_left:
        st.markdown("#### 1) Upload & index")

        uploaded_zip = st.file_uploader(
            "Upload your project (.zip)",
            type=["zip"],
            help="Lite sürüm şu anda sadece .py dosyalarını indexliyor.",
        )

        default_project_id = "demo-project"
        if uploaded_zip is not None:
            default_project_id = uploaded_zip.name
            if default_project_id.lower().endswith(".zip"):
                default_project_id = default_project_id[:-4]

        project_id = st.text_input("Project ID", value=default_project_id)

        if uploaded_zip is not None and project_id:
            if st.button("Process & Index Project"):
                try:
                    index = build_project_index(uploaded_zip, project_id)
                    st.session_state.project_index = index
                    st.success(
                        f"Indexed `{project_id}` – {len(index.files)} files, "
                        f"{len(index.chunks)} chunks."
                    )
                except Exception as e:
                    st.error(f"Failed to index project: {e}")
        else:
            st.caption("Önce küçük bir örnek proje ile test edebilirsin.")

    with col_right:
        st.markdown("#### 2) Ask a question")

        if st.session_state.project_index is None:
            st.info("Önce soldan bir proje yükleyip indexle.")
        else:
            index: ProjectIndex = st.session_state.project_index

            default_q = "What does the function calculate_loss() do?"
            query = st.text_area(
                "Your question",
                value=default_q,
                height=120,
            )

            k = st.slider(
                "How many chunks should we show?",
                min_value=3,
                max_value=10,
                value=5,
            )

            if st.button("Retrieve & explain"):
                func_name = extract_function_name_from_query(query)
                if func_name:
                    st.info(f"Detected function name: `{func_name}`")

                results = retrieve_top_k(index, query, func_name, k=k)
                explanation_md = build_explanation(results, query)
                st.markdown(explanation_md)

                with st.expander("Show retrieved chunks in full", expanded=False):
                    for i, (ch, score) in enumerate(results, start=1):
                        st.markdown(
                            f"#### #{i} · `{ch.file_path}` · "
                            f"lines {ch.start_line}–{ch.end_line} · score={score:.3f}"
                        )
                        st.code(ch.text, language="python")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Explorer -----------------------------

with tab_explorer:
    st.markdown('<div class="codemate-section">', unsafe_allow_html=True)
    st.subheader("Explorer", anchor=False)

    if st.session_state.project_index is None:
        st.info("Önce bir proje indexle, sonra dosyaları burada gezebilirsin.")
    else:
        index: ProjectIndex = st.session_state.project_index
        st.caption(
            f"{len(index.files)} file · {len(index.chunks)} chunks in project `{index.project_id}`"
        )

        paths = [f["path"] for f in index.files]
        selected = st.selectbox("Select a file", paths)

        if selected:
            code_path = index.root_dir / selected
            try:
                code_text = read_file(code_path)
                st.code(code_text, language="python")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
