import streamlit as st  # type: ignore
import zipfile
from pathlib import Path
import re
import os
import numpy as np
from openai import OpenAI

# ----------------------------
# Global configuration
# ----------------------------

# Read API key from environment; if missing, client will be None
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"

SUPPORTED_EXT = [".py"]  # Currently we only support Python files


# ----------------------------
# Helper functions: file handling and chunking
# ----------------------------

def save_and_extract_zip(uploaded_file, extract_root: Path) -> Path:
    """
    Save the uploaded zip file to disk and extract it.
    Returns the path to the project root directory.
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
    Recursively find all supported code files under the given root.
    """
    return [p for p in root.rglob("*") if p.suffix in SUPPORTED_EXT]


def read_file(path: Path) -> str:
    """
    Read a text file safely using UTF-8 encoding.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def simple_chunk(code: str, max_lines: int = 40):
    """
    Split code into simple line-based chunks.
    This is a placeholder for more advanced, structure-aware chunking.
    """
    lines = code.splitlines()
    for i in range(0, len(lines), max_lines):
        chunk_lines = lines[i:i + max_lines]
        yield "\n".join(chunk_lines)


def extract_function_name_from_query(query: str):
    """
    Try to extract a function name from a natural language question.
    Example: "What does calculate_loss() do?" -> "calculate_loss"
    """
    match = re.search(r"([a-zA-Z_]\w*)\s*\(", query)
    if match:
        return match.group(1)
    return None


# ----------------------------
# Helper functions: embeddings and retrieval
# ----------------------------

def embed_text(text: str) -> list[float]:
    """
    Create an embedding vector for the given text using the configured model.
    Raises RuntimeError if no OpenAI client is available.
    """
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot create embeddings.")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def cosine_sim(a, b) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def retrieve_top_k(chunks, query: str, k: int = 3):
    """
    Embedding-based retrieval:
    1. Create an embedding for the query.
    2. Compute cosine similarity with each chunk embedding.
    3. Return the top-k most similar chunks.
    """
    query_emb = embed_text(query)
    scored = []

    for ch in chunks:
        score = cosine_sim(query_emb, ch["embedding"])
        scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:k]]
    return top_chunks


# ----------------------------
# Helper functions: explanations
# ----------------------------

def fake_explanation(code_snippet: str, query: str) -> str:
    """
    Placeholder explanation for comparison and fallback.
    """
    return (
        "### Prototype explanation (stub)\n\n"
        "This is a placeholder explanation for your question:\n\n"
        f"> {query}\n\n"
        "The system searched for a code region related to your query "
        "and found the following snippet:\n\n"
        "python\n"
        + code_snippet
        + "\n\n\n"
        "In the full RAG version, this part calls an LLM with retrieved "
        "code context to generate a detailed and grounded explanation."
    )


def llm_explanation(chunks, query: str) -> str:
    """
    Generate a real explanation using an LLM, given retrieved chunks and the query.
    """
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot call LLM.")

    context_parts = []
    for ch in chunks:
        context_parts.append(f"# File: {ch['path']}\n{ch['code']}")

    context = "\n\n".join(context_parts)

    prompt = (
        "You are a software engineering assistant. "
        "Explain what the following code does in clear, concise natural language. "
        "Focus on the intent of the functions, important parameters, and any "
        "non-trivial logic.\n\n"
        f"Developer question:\n{query}\n\n"
        "Relevant code snippets:\n"
        f"{context}\n\n"
        "Now provide a helpful explanation for the developer."
    )

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt,
    )

    text = response.output[0].content[0].text
    return text


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("CodeDocMate – Early RAG Prototype")

st.markdown(
    "This prototype demonstrates the first working step of CodeDocMate.\n\n"
    "- You can upload a zipped Python, Java, C, C++ etc. project.\n"
    "- The system scans and chunks the source files.\n"
    "- Each chunk is embedded into a vector space using an OpenAI embedding model.\n"
    "- Given a natural language question, the system retrieves relevant chunks\n"
    "  and either shows a stub explanation or calls an LLM (RAG mode).\n\n"
    "In the final version, this will be extended with more advanced chunking,\n"
    "better retrieval, and richer documentation generation."
)

if client is None:
    st.warning(
        "OPENAI_API_KEY is not set. Embeddings and LLM calls will fail. "
        "You can still run the stub mode, but RAG + LLM will require a valid API key."
    )

# ----------------------------
# Section 1: Project upload and indexing
# ----------------------------

st.header("1. Upload and index project")

# First let the user upload a zip file
uploaded_zip = st.file_uploader("Upload your project (.zip)", type=["zip"])

# Derive a default project ID from the zip file name (without .zip extension)
default_project_id = "demo-project"
if uploaded_zip is not None:
    default_project_id = uploaded_zip.name
    if default_project_id.lower().endswith(".zip"):
        default_project_id = default_project_id[:-4]

# Pre-fill the Project ID field with the derived default
project_id = st.text_input("Project ID", value=default_project_id)

if uploaded_zip is not None and project_id:
    st.info("Zip file received. Click the button below to process and index the project.")

    if st.button("Process & Index Project"):
        project_root = Path("projects") / project_id
        root = save_and_extract_zip(uploaded_zip, project_root)

        code_files = find_code_files(root)
        st.write(f"Found *{len(code_files)}* Python files:")

        for p in code_files:
            st.write(f"- {p.relative_to(root)}")

        all_chunks = []
        try:
            for p in code_files:
                code = read_file(p)
                for ch in simple_chunk(code, max_lines=40):
                    emb = embed_text(ch)
                    all_chunks.append(
                        {
                            "path": str(p.relative_to(root)),
                            "code": ch,
                            "embedding": emb,
                        }
                    )

            st.write(f"Created *{len(all_chunks)}* code chunks with embeddings.")
            st.session_state["chunks"] = all_chunks
            st.success("Project is indexed in memory (simple in-memory vector store).")
        except Exception as e:
            st.error(
                "Failed to create embeddings. Check your OPENAI_API_KEY and billing/quota.\n\n"
                f"Details: {e}"
            )

# ----------------------------
# Section 2: Question and explanation
# ----------------------------

st.header("2. Ask a question about the code")

default_q = "What does the function calculate_loss() do?"
mode = st.radio(
    "Explanation mode",
    ["Stub (no LLM)", "RAG + LLM"],
    index=0,
)
query = st.text_area("Your question:", value=default_q, height=80)

if st.button("Retrieve & Explain"):
    if "chunks" not in st.session_state or not st.session_state["chunks"]:
        st.error("No project is indexed yet. Please upload and process a project first.")
    else:
        chunks = st.session_state["chunks"]

        func_name = extract_function_name_from_query(query)
        if func_name:
            st.write(f"Detected function name in query: {func_name}")
            query_for_embedding = query + f" (function name: {func_name})"
        else:
            st.write("No function name detected in the query.")
            query_for_embedding = query

        try:
            if mode == "Stub (no LLM)":
                # Use only the first chunk as a simple baseline
                best = chunks[0]
                explanation_md = fake_explanation(best["code"], query)
            else:
                top_chunks = retrieve_top_k(chunks, query_for_embedding, k=3)

                if not top_chunks:
                    st.error("No code snippets are available.")
                    st.stop()

                explanation_text = llm_explanation(top_chunks, query)
                explanation_md = "### LLM explanation\n\n" + explanation_text

            st.markdown(explanation_md)

        except Exception as e:
            st.error(
                "Failed to run RAG + LLM explanation. "
                "This is usually caused by a missing/invalid API key or insufficient quota.\n\n"
                f"Details: {e}"
            )
