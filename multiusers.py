"""멀티유저/멀티세션 RAG 챗봇 — DB user 테이블 로그인, user_id 기준 세션·벡터 저장."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

CONFIG_KEYS = ("SUPABASE_URL", "SUPABASE_ANON_KEY", "OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"
VECTOR_BATCH_SIZE = 10
RAG_MATCH_COUNT = 10
CHATBOT_TITLE = "기획예산처 RAG 챗봇"

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""

PAGE_STYLE = """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
"""


def _resolve_log_path() -> Path | None:
    """Pick a writable log directory (repo logs locally, temp on Streamlit Cloud)."""
    log_name = f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
    for base in (LOG_DIR, Path(tempfile.gettempdir()) / "multiusers_logs"):
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base / log_name
        except OSError:
            continue
    return None


def _setup_logging() -> logging.Logger:
    log_path = _resolve_log_path()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if log_path is not None:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.WARNING)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers")


logger = _setup_logging()


def _load_environment() -> None:
    """Load .env then override with st.secrets when present (Streamlit Cloud)."""
    load_dotenv(dotenv_path=ENV_PATH)
    try:
        secrets = st.secrets
        for key in CONFIG_KEYS:
            if key in secrets and str(secrets[key]).strip():
                os.environ[key] = str(secrets[key]).strip()
    except Exception:  # noqa: BLE001
        pass


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _missing_keys() -> list[str]:
    labels = {
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "SUPABASE_URL": "SUPABASE_URL",
        "SUPABASE_ANON_KEY": "SUPABASE_ANON_KEY",
    }
    return [labels[k] for k in CONFIG_KEYS if not os.getenv(k, "").strip()]


def get_supabase_client() -> Client | None:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        return None
    return create_client(url, key)


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return ChatOpenAI(model=MODEL_NAME, temperature=0.7, api_key=api_key)


def get_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAIEmbeddings(api_key=api_key)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def _current_user_id() -> str | None:
    return st.session_state.get("current_user_id")


def _require_user_id() -> str:
    uid = _current_user_id()
    if not uid:
        raise ValueError("로그인이 필요합니다.")
    return uid


def register_user(client: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 입력해 주세요."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."

    existing = (
        client.table("user")
        .select("id")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False, "이미 사용 중인 아이디입니다."

    pw_hash = hash_password(password)
    client.table("user").insert(
        {"login_id": login_id, "password_hash": pw_hash}
    ).execute()
    return True, "회원가입이 완료되었습니다. 로그인해 주세요."


def login_user(client: Client, login_id: str, password: str) -> tuple[bool, str, str | None]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 입력해 주세요.", None

    resp = (
        client.table("user")
        .select("id, password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None

    row = rows[0]
    if not verify_password(password, row["password_hash"]):
        return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None

    return True, "로그인되었습니다.", str(row["id"])


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str,
    context: str,
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", str(out)) or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""

    raw = remove_separators(str(raw))
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def generate_session_title(client: Client, messages: list[dict[str, str]]) -> str:
    first_user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    first_asst = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
    if not first_user:
        return f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    try:
        llm = get_llm()
        prompt = (
            "다음 첫 질문과 답변을 한 줄(30자 이내) 한국어 제목으로 요약하세요. "
            "따옴표나 설명 없이 제목만 출력하세요.\n\n"
            f"질문: {first_user[:500]}\n답변: {(first_asst or '')[:500]}"
        )
        out = llm.invoke([HumanMessage(content=prompt)])
        title = remove_separators(str(getattr(out, "content", "") or "")).strip()
        if title:
            return title[:80]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Title generation failed: %s", exc)

    return first_user[:40] + ("..." if len(first_user) > 40 else "")


def _session_owned(client: Client, session_id: str, user_id: str) -> bool:
    resp = (
        client.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return bool(resp.data)


def fetch_session_list(client: Client, user_id: str) -> list[dict[str, Any]]:
    resp = (
        client.table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return resp.data or []


def fetch_session_messages(
    client: Client, session_id: str, user_id: str
) -> list[dict[str, str]]:
    if not _session_owned(client, session_id, user_id):
        return []
    resp = (
        client.table("chat_messages")
        .select("role, content, sort_order")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("sort_order")
        .execute()
    )
    rows = resp.data or []
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def fetch_vector_file_names(client: Client, session_id: str, user_id: str) -> list[str]:
    if not _session_owned(client, session_id, user_id):
        return []
    resp = (
        client.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = sorted({r["file_name"] for r in (resp.data or []) if r.get("file_name")})
    return names


def upsert_messages(
    client: Client, session_id: str, user_id: str, messages: list[dict[str, str]]
) -> None:
    if not _session_owned(client, session_id, user_id):
        raise ValueError("접근 권한이 없는 세션입니다.")
    client.table("chat_messages").delete().eq("session_id", session_id).eq(
        "user_id", user_id
    ).execute()
    if not messages:
        return
    rows = [
        {
            "session_id": session_id,
            "user_id": user_id,
            "role": m["role"],
            "content": m["content"],
            "sort_order": idx,
        }
        for idx, m in enumerate(messages)
    ]
    for i in range(0, len(rows), 50):
        client.table("chat_messages").insert(rows[i : i + 50]).execute()


def auto_save_session(client: Client, *, title: str | None = None) -> str:
    user_id = _require_user_id()
    session_id = st.session_state.work_session_id
    messages = st.session_state.chat_history

    if title is None:
        if messages:
            title = generate_session_title(client, messages)
        else:
            title = st.session_state.get("session_title") or "새 세션"

    existing = (
        client.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        client.table("chat_sessions").update({"title": title}).eq("id", session_id).eq(
            "user_id", user_id
        ).execute()
    else:
        client.table("chat_sessions").insert(
            {"id": session_id, "user_id": user_id, "title": title}
        ).execute()

    upsert_messages(client, session_id, user_id, messages)
    st.session_state.session_title = title
    return session_id


def insert_new_session_save(client: Client) -> str | None:
    user_id = _require_user_id()
    messages = st.session_state.chat_history
    if not messages:
        st.sidebar.warning("저장할 대화가 없습니다.")
        return None

    new_id = str(uuid.uuid4())
    title = generate_session_title(client, messages)
    client.table("chat_sessions").insert(
        {"id": new_id, "user_id": user_id, "title": title}
    ).execute()
    upsert_messages(client, new_id, user_id, messages)

    old_id = st.session_state.work_session_id
    if old_id and old_id != new_id and _session_owned(client, old_id, user_id):
        _copy_vectors_to_session(client, old_id, new_id, user_id)

    st.session_state.work_session_id = new_id
    st.session_state.session_title = title
    st.session_state.selected_session_id = new_id
    return new_id


def _copy_vectors_to_session(
    client: Client, source_id: str, target_id: str, user_id: str
) -> None:
    if not _session_owned(client, source_id, user_id) or not _session_owned(
        client, target_id, user_id
    ):
        return
    resp = (
        client.table("vector_documents")
        .select("file_name, content, metadata, embedding")
        .eq("session_id", source_id)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return
    payload = [
        {
            "session_id": target_id,
            "file_name": r["file_name"],
            "content": r["content"],
            "metadata": r.get("metadata") or {},
            "embedding": r["embedding"],
        }
        for r in rows
    ]
    for i in range(0, len(payload), VECTOR_BATCH_SIZE):
        client.table("vector_documents").insert(payload[i : i + VECTOR_BATCH_SIZE]).execute()


def delete_session_from_db(client: Client, session_id: str, user_id: str) -> None:
    if not _session_owned(client, session_id, user_id):
        raise ValueError("삭제할 수 없는 세션입니다.")
    client.table("vector_documents").delete().eq("session_id", session_id).execute()
    client.table("chat_messages").delete().eq("session_id", session_id).eq(
        "user_id", user_id
    ).execute()
    client.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()


def load_session_into_state(client: Client, session_id: str, user_id: str) -> None:
    if not _session_owned(client, session_id, user_id):
        raise ValueError("로드할 수 없는 세션입니다.")

    messages = fetch_session_messages(client, session_id, user_id)
    files = fetch_vector_file_names(client, session_id, user_id)

    st.session_state.work_session_id = session_id
    st.session_state.selected_session_id = session_id
    st.session_state.chat_history = messages
    st.session_state.conversation_memory = messages[-50:]
    st.session_state.processed_names = files

    sess = (
        client.table("chat_sessions")
        .select("title")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if sess.data:
        st.session_state.session_title = sess.data[0].get("title", "")


def clear_screen_state() -> None:
    st.session_state.work_session_id = str(uuid.uuid4())
    st.session_state.selected_session_id = None
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_names = []
    st.session_state.session_title = "새 세션"


def store_vectors_direct(
    client: Client,
    session_id: str,
    user_id: str,
    documents: list[Document],
    embeddings_model: OpenAIEmbeddings,
) -> None:
    if not documents:
        return
    if not _session_owned(client, session_id, user_id):
        client.table("chat_sessions").insert(
            {"id": session_id, "user_id": user_id, "title": st.session_state.session_title}
        ).execute()

    texts = [d.page_content for d in documents]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), VECTOR_BATCH_SIZE):
        batch = texts[i : i + VECTOR_BATCH_SIZE]
        vectors.extend(embeddings_model.embed_documents(batch))

    rows: list[dict[str, Any]] = []
    for doc, emb in zip(documents, vectors, strict=True):
        file_name = doc.metadata.get("file_name") or "unknown.pdf"
        rows.append(
            {
                "session_id": session_id,
                "file_name": file_name,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": emb,
            }
        )

    for i in range(0, len(rows), VECTOR_BATCH_SIZE):
        client.table("vector_documents").insert(rows[i : i + VECTOR_BATCH_SIZE]).execute()


def search_vectors_rpc(
    client: Client,
    session_id: str,
    user_id: str,
    query: str,
    embeddings_model: OpenAIEmbeddings,
    k: int = RAG_MATCH_COUNT,
) -> list[Document]:
    if not _session_owned(client, session_id, user_id):
        return []

    query_emb = embeddings_model.embed_query(query)
    try:
        resp = client.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_emb,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        data = resp.data or []
        return [
            Document(
                page_content=row["content"],
                metadata={
                    "file_name": row.get("file_name"),
                    "session_id": row.get("session_id"),
                    "similarity": row.get("similarity"),
                },
            )
            for row in data
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC match_vector_documents failed: %s", exc)
        return _search_vectors_fallback(client, session_id, user_id, query, embeddings_model, k)


def _search_vectors_fallback(
    client: Client,
    session_id: str,
    user_id: str,
    query: str,
    embeddings_model: OpenAIEmbeddings,
    k: int,
) -> list[Document]:
    if not _session_owned(client, session_id, user_id):
        return []
    resp = (
        client.table("vector_documents")
        .select("file_name, content, metadata, embedding")
        .eq("session_id", session_id)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return []

    query_emb = embeddings_model.embed_query(query)

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        emb = row.get("embedding")
        if isinstance(emb, str) or not emb:
            continue
        scored.append((cosine(query_emb, emb), row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]
    return [
        Document(
            page_content=row["content"],
            metadata={"file_name": row.get("file_name"), "session_id": session_id},
        )
        for _, row in top
    ]


def _process_pdf_uploads(
    client: Client,
    uploaded_files: list[Any],
    session_id: str,
    user_id: str,
) -> list[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("PDF 임베딩에 OPENAI_API_KEY가 필요합니다.")

    embeddings_model = OpenAIEmbeddings(api_key=api_key)
    processed_names: list[str] = []

    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs:
                d.metadata["file_name"] = uf.name
                d.metadata["session_id"] = session_id

            if not docs:
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            splits = splitter.split_documents(docs)
            store_vectors_direct(client, session_id, user_id, splits, embeddings_model)
            processed_names.append(uf.name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return processed_names


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "logged_in": False,
        "current_user_id": None,
        "current_login_id": "",
        "chat_history": [],
        "conversation_memory": [],
        "processed_names": [],
        "work_session_id": str(uuid.uuid4()),
        "selected_session_id": None,
        "session_title": "새 세션",
        "session_list": [],
        "sessions_loaded": False,
        "dropdown_key": 0,
        "auth_mode": "로그인",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_header() -> None:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            f"""
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">기획예산처</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def _render_missing_keys() -> bool:
    missing = _missing_keys()
    if missing:
        src = "Streamlit secrets 또는 `.env`"
        st.error(f"{src} ({ENV_PATH})에 다음 설정이 필요합니다: " + ", ".join(missing))
        return True
    return False


def _logout() -> None:
    st.session_state.logged_in = False
    st.session_state.current_user_id = None
    st.session_state.current_login_id = ""
    clear_screen_state()
    st.session_state.session_list = []
    st.session_state.sessions_loaded = False


def _render_auth_screen(client: Client) -> None:
    _render_header()
    st.markdown("### 로그인 / 회원가입")
    st.caption("Supabase Auth 없이 DB `user` 테이블로 계정을 관리합니다.")

    mode = st.radio("모드", ("로그인", "회원가입"), horizontal=True, key="auth_mode")
    login_id = st.text_input("아이디 (login_id)")
    password = st.text_input("비밀번호", type="password")

    if st.button("확인", type="primary"):
        if mode == "회원가입":
            ok, msg = register_user(client, login_id, password)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        else:
            ok, msg, uid = login_user(client, login_id, password)
            if ok and uid:
                st.session_state.logged_in = True
                st.session_state.current_user_id = uid
                st.session_state.current_login_id = login_id.strip()
                clear_screen_state()
                st.session_state.sessions_loaded = False
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


def _render_sidebar(client: Client, user_id: str) -> None:
    with st.sidebar:
        st.markdown(f"**로그인:** `{st.session_state.current_login_id}`")
        if st.button("로그아웃"):
            _logout()
            st.rerun()

        st.markdown(f"**모델:** `{MODEL_NAME}`")

        if not st.session_state.sessions_loaded:
            try:
                st.session_state.session_list = fetch_session_list(client, user_id)
                st.session_state.sessions_loaded = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Session list load failed: %s", exc)
                st.sidebar.error(f"세션 목록 로드 실패: {exc}")

        sessions = st.session_state.session_list
        options: list[str] = []
        id_by_label: dict[str, str] = {}
        for s in sessions:
            label = f"{s.get('title', '제목 없음')} ({str(s.get('id', ''))[:8]}…)"
            options.append(label)
            id_by_label[label] = str(s["id"])

        selected_label = st.selectbox(
            "세션 선택",
            options=["— 세션 선택 —"] + options,
            key=f"session_select_{st.session_state.dropdown_key}",
        )

        if selected_label != "— 세션 선택 —" and selected_label in id_by_label:
            picked_id = id_by_label[selected_label]
            if st.session_state.selected_session_id != picked_id:
                try:
                    load_session_into_state(client, picked_id, user_id)
                    st.session_state.selected_session_id = picked_id
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"세션 자동 로드 실패: {exc}")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("세션저장"):
                try:
                    new_id = insert_new_session_save(client)
                    if new_id:
                        st.session_state.session_list = fetch_session_list(client, user_id)
                        st.success("새 세션이 저장되었습니다.")
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"세션 저장 실패: {exc}")

            if st.button("세션삭제"):
                sid = st.session_state.selected_session_id or st.session_state.work_session_id
                if not sid:
                    st.warning("삭제할 세션이 없습니다.")
                else:
                    try:
                        delete_session_from_db(client, sid, user_id)
                        st.session_state.session_list = fetch_session_list(client, user_id)
                        clear_screen_state()
                        st.session_state.sessions_loaded = True
                        st.success("세션이 삭제되었습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 삭제 실패: {exc}")

        with col_b:
            if st.button("세션로드"):
                sid = st.session_state.selected_session_id
                if not sid and selected_label in id_by_label:
                    sid = id_by_label[selected_label]
                if not sid:
                    st.warning("로드할 세션을 선택하세요.")
                else:
                    try:
                        load_session_into_state(client, sid, user_id)
                        st.success("세션을 불러왔습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 로드 실패: {exc}")

            if st.button("화면초기화"):
                clear_screen_state()
                st.rerun()

        if st.button("vectordb"):
            sid = st.session_state.work_session_id
            try:
                names = fetch_vector_file_names(client, sid, user_id)
                if names:
                    st.markdown("**벡터 DB 파일 목록**")
                    for n in names:
                        st.text(f"- {n}")
                else:
                    st.info("현재 세션에 저장된 벡터 파일이 없습니다.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"벡터 DB 조회 실패: {exc}")

        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기"):
            if not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    sid = st.session_state.work_session_id
                    names = _process_pdf_uploads(client, list(uploads), sid, user_id)
                    st.session_state.processed_names = list(
                        dict.fromkeys(st.session_state.processed_names + names)
                    )
                    auto_save_session(client)
                    st.session_state.session_list = fetch_session_list(client, user_id)
                    st.success("PDF 처리 및 자동 저장이 완료되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        st.text(
            f"작업 세션 ID: {st.session_state.work_session_id[:8]}…\n"
            f"제목: {st.session_state.session_title}\n"
            f"대화 수: {len(st.session_state.chat_history)}\n"
            f"저장된 세션 수: {len(st.session_state.session_list)}"
        )


def _render_chat(client: Client, user_id: str) -> None:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    sid = st.session_state.work_session_id
    has_vectors = bool(st.session_state.processed_names)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            if not has_vectors:
                full_answer = (
                    "# 안내\n\n"
                    "RAG를 사용하려면 PDF를 업로드한 뒤 **파일 처리하기**를 눌러 주세요."
                )
                placeholder.markdown(remove_separators(full_answer))
            else:
                llm = get_llm()
                emb = get_embeddings()
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                docs = search_vectors_rpc(client, sid, user_id, user_input, emb)
                context = "\n\n".join(d.page_content for d in docs) if docs else "(관련 문서 없음)"
                messages = _build_rag_messages(user_input, context, mem_txt)

                acc = ""
                for chunk in llm.stream(messages):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        acc += piece
                        placeholder.markdown(remove_separators(acc) + "▌")
                full_answer = remove_separators(acc)
                placeholder.markdown(full_answer)

                if full_answer and not full_answer.lstrip().startswith("# 오류"):
                    follow = _generate_followup_section(llm, user_input, full_answer)
                    if follow:
                        full_answer += follow
                        placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청 처리 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
        st.session_state.conversation_memory.append(
            {"role": "assistant", "content": full_answer}
        )
        if len(st.session_state.conversation_memory) > 50:
            st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

        try:
            auto_save_session(client)
            st.session_state.session_list = fetch_session_list(client, user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("자동 저장 실패: %s", exc)


def main() -> None:
    st.set_page_config(page_title=CHATBOT_TITLE, page_icon="📚", layout="wide")
    _load_environment()
    _init_session()
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    if _render_missing_keys():
        _render_header()
        return

    client = get_supabase_client()
    if client is None:
        _render_header()
        st.error("Supabase 클라이언트를 생성할 수 없습니다. URL/KEY를 확인하세요.")
        return

    if not st.session_state.logged_in:
        _render_auth_screen(client)
        return

    _render_header()
    user_id = _require_user_id()
    _render_sidebar(client, user_id)
    _render_chat(client, user_id)


if __name__ == "__main__":
    main()
