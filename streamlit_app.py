"""
streamlit_app.py  —  Multi-Modal RAG Knowledge Engine
───────────────────────────────────────────────────────
Performance optimisations over v1:
  • Persistent httpx.Client via st.cache_resource  (no new TCP per call)
  • Health check cached 15 s                       (was: every render)
  • Document list cached 30 s + manual invalidation (was: every render)
  • All citation HTML batched into ONE st.markdown  (was: loop of calls)
  • Navigation buttons skip rerun when already on page
  • st.fragment for job-poll loop                   (partial rerenders only)
  • CSS injected once via st.cache_data             (was: every render)

Run:
  streamlit run streamlit_app.py
"""

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

# ─── Page config (must be first st call) ──────────────────────────────────────

st.set_page_config(
    page_title="RAG · Knowledge Engine",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Persistent HTTP client ────────────────────────────────────────────────────

@st.cache_resource
def get_client() -> httpx.Client:
    return httpx.Client(
        base_url=API_BASE,
        timeout=httpx.Timeout(connect=3, read=90, write=30, pool=5),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )


def api(method: str, path: str, **kwargs):
    try:
        r = get_client().request(method, path, **kwargs)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), None
    except httpx.ConnectError:
        return None, "Cannot connect to API at `localhost:8000`."
    except Exception as e:
        return None, str(e)


# ─── Cached API calls ──────────────────────────────────────────────────────────

@st.cache_data(ttl=15, show_spinner=False)
def fetch_health() -> bool:
    data, err = api("GET", "/health")
    return err is None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_documents() -> list:
    data, _ = api("GET", "/documents")
    return data or []


def invalidate_documents():
    fetch_documents.clear()


# ─── CSS injected once ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Mono',monospace!important;background:#080b10;color:#e8edf3}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem 2.5rem 4rem;max-width:1200px}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1e2a38}
[data-testid="stSidebar"] *{font-family:'IBM Plex Mono',monospace!important}
.pt{font-family:'Syne',sans-serif!important;font-size:2.4rem;font-weight:800;letter-spacing:-.02em;line-height:1.1;color:#e8edf3;margin-bottom:.2rem}
.pt span{color:#00e5ff}
.ps{font-size:.7rem;color:#5a6a7a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:1.8rem}
.sl{font-size:.6rem;letter-spacing:.22em;text-transform:uppercase;color:#5a6a7a;margin-bottom:.7rem;display:flex;align-items:center;gap:8px}
.sl::after{content:'';flex:1;height:1px;background:#1e2a38}
.tag{display:inline-block;font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;padding:3px 9px;border-radius:2px;margin:2px}
.tc{background:rgba(0,229,255,.08);color:#00e5ff;border:1px solid rgba(0,229,255,.25)}
.to{background:rgba(255,107,53,.08);color:#ff6b35;border:1px solid rgba(255,107,53,.25)}
.tg{background:rgba(127,255,107,.08);color:#7fff6b;border:1px solid rgba(127,255,107,.25)}
.td{background:rgba(255,255,255,.04);color:#5a6a7a;border:1px solid #1e2a38}
.card{background:#0f1620;border:1px solid #1e2a38;border-radius:3px;padding:1.25rem;margin-bottom:.75rem}
.card-c{border-top:2px solid #00e5ff}
.ans{background:#0a1018;border:1px solid #1e2a38;border-top:2px solid #00e5ff;padding:1.4rem;border-radius:3px;font-size:.85rem;line-height:1.85;color:#e8edf3;white-space:pre-wrap}
.uv{background:rgba(255,107,53,.12);color:#ff6b35;border-radius:2px;padding:1px 5px}
.cit{background:#0a1018;border-left:3px solid #00e5ff;padding:.65rem 1rem;margin:.4rem 0;border-radius:0 3px 3px 0;font-size:.75rem}
.cit-m{font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:#5a6a7a;margin-bottom:.35rem}
.cit-t{color:#9ab0c5;line-height:1.6}
.warn{background:rgba(255,107,53,.06);border:1px solid rgba(255,107,53,.3);border-radius:3px;padding:.7rem 1rem;font-size:.72rem;color:#ff6b35;margin-top:.5rem}
.sg{display:flex;gap:.75rem;margin-bottom:1.4rem;flex-wrap:wrap}
.sc{flex:1;min-width:110px;background:#0f1620;border:1px solid #1e2a38;border-radius:3px;padding:.9rem;text-align:center}
.sv{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:700;color:#00e5ff;line-height:1}
.sk{font-size:.6rem;letter-spacing:.15em;text-transform:uppercase;color:#5a6a7a;margin-top:3px}
.badge{display:inline-block;padding:3px 10px;border-radius:2px;font-size:.6rem;letter-spacing:.12em;text-transform:uppercase}
.b-ok{background:rgba(127,255,107,.1);color:#7fff6b;border:1px solid rgba(127,255,107,.3)}
.b-pend{background:rgba(0,229,255,.1);color:#00e5ff;border:1px solid rgba(0,229,255,.3)}
.b-err{background:rgba(255,107,53,.1);color:#ff6b35;border:1px solid rgba(255,107,53,.3)}
.tr{display:flex;gap:1rem;font-size:.65rem;color:#5a6a7a;letter-spacing:.08em;text-transform:uppercase;margin-top:.5rem;flex-wrap:wrap}
.tv{color:#9ab0c5}
.dr{display:flex;align-items:center;justify-content:space-between;background:#0a1018;border:1px solid #1e2a38;border-radius:3px;padding:.65rem 1.1rem;margin-bottom:.4rem;font-size:.75rem}
.dn{color:#e8edf3;font-weight:600}
.dm{color:#5a6a7a;font-size:.62rem;margin-top:2px}
div[data-testid="stTextInput"] input,div[data-testid="stTextArea"] textarea{background:#0a1018!important;border:1px solid #1e2a38!important;color:#e8edf3!important;font-family:'IBM Plex Mono',monospace!important;border-radius:3px!important}
div[data-testid="stTextInput"] input:focus,div[data-testid="stTextArea"] textarea:focus{border-color:#00e5ff!important;box-shadow:0 0 0 1px #00e5ff33!important}
div[data-testid="stFileUploader"]{background:#0a1018;border:1px dashed #1e2a38!important;border-radius:3px}
.stButton>button{background:transparent!important;border:1px solid #00e5ff!important;color:#00e5ff!important;font-family:'IBM Plex Mono',monospace!important;font-size:.72rem!important;letter-spacing:.08em;border-radius:2px!important;padding:.45rem 1.1rem!important;transition:background .15s,color .15s}
.stButton>button:hover{background:#00e5ff!important;color:#080b10!important}
div[data-testid="stMultiSelect"] *{font-family:'IBM Plex Mono',monospace!important}
.stCheckbox label{font-size:.75rem!important;color:#9ab0c5!important}
hr{border:none;border-top:1px solid #1e2a38;margin:1.2rem 0}
</style>"""

st.markdown(_css(), unsafe_allow_html=True)

# ─── Session state defaults ────────────────────────────────────────────────────

for k, v in {"page": "query", "last_result": None, "ingest_jobs": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 .8rem">
      <div style="font-size:.58rem;letter-spacing:.22em;text-transform:uppercase;color:#5a6a7a;margin-bottom:7px">⬡ &nbsp;LLM / GenAI Engineer</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:800;color:#e8edf3;line-height:1.1">
        Multi-Modal<br/><span style="color:#00e5ff">RAG Engine</span>
      </div>
    </div><hr/>
    """, unsafe_allow_html=True)

    is_online = fetch_health()
    st.markdown(
        '<span class="badge b-ok">● API Online</span>' if is_online else
        '<span class="badge b-err">● API Offline</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    for key, label in [("query","◈  Query"), ("ingest","↑  Ingest"), ("documents","≡  Documents")]:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            if st.session_state.page != key:
                st.session_state.page = key
                st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="sl">Settings</div>', unsafe_allow_html=True)
    apply_guard = st.checkbox("NLI Hallucination Guard", value=True)
    use_stream  = st.checkbox("Streaming Response", value=False)
    st.markdown("""<br/><div style="font-size:.6rem;color:#5a6a7a;line-height:1.7">
      RAG · Hybrid BM25+Dense<br>CLIP · text-embedding-3<br>Qdrant · Cohere Rerank<br>GPT-4o · NLI Guard
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: QUERY
# ══════════════════════════════════════════════════════════════════════════════

def page_query():
    st.markdown(
        '<div class="pt">Ask the<br/><span>Knowledge Base</span></div>'
        '<div class="ps">Multi-modal retrieval · Grounded citations · NLI verified</div>',
        unsafe_allow_html=True,
    )

    docs = fetch_documents()   # from cache — zero network cost

    col_q, col_f = st.columns([3, 1])
    with col_q:
        query = st.text_area("q", placeholder="e.g. What accuracy did the model achieve on S&P500?",
                             height=110, label_visibility="collapsed", key="query_input")
    with col_f:
        st.markdown('<div class="sl">Filter Docs</div>', unsafe_allow_html=True)
        doc_opts = {f"{d['doc_name']} ({d['chunk_count']}ch)": d["doc_id"] for d in docs}
        selected = st.multiselect("Docs", options=list(doc_opts.keys()), label_visibility="collapsed")
        filter_ids = [doc_opts[l] for l in selected] or None

    if st.button("◈  Run Query", disabled=not (query or "").strip(), key="run_query"):
        if use_stream:
            _run_streaming(query, filter_ids)
        else:
            with st.spinner("Retrieving & generating…"):
                data, err = api("POST", "/query", json={
                    "query": query, "apply_guard": apply_guard, "filter_doc_ids": filter_ids,
                })
            if err:
                st.markdown(f'<div class="warn">⚠ {err}</div>', unsafe_allow_html=True)
            else:
                st.session_state.last_result = data
                _render_result(data)

    elif st.session_state.last_result:
        _render_result(st.session_state.last_result)


def _run_streaming(query: str, filter_ids):
    st.markdown('<div class="sl">Answer</div>', unsafe_allow_html=True)
    placeholder = st.empty()
    full = ""
    params = {"q": query}
    if filter_ids:
        params["doc_ids"] = ",".join(filter_ids)
    try:
        with httpx.stream("GET", f"{API_BASE}/query/stream", params=params, timeout=120) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    tok = line[6:]
                    if tok == "[DONE]":
                        break
                    full += tok
                    placeholder.markdown(f'<div class="ans">{full}▌</div>', unsafe_allow_html=True)
        placeholder.markdown(f'<div class="ans">{full}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Streaming error: {e}")


def _render_result(data: dict):
    answer    = data.get("answer", "")
    citations = data.get("citations", [])
    warnings  = data.get("warnings", [])

    st.markdown('<div class="sl">Answer</div>', unsafe_allow_html=True)
    answer_html = answer.replace("[⚠ unverified]", '<span class="uv">[⚠ unverified]</span>')
    st.markdown(
        f'<div class="ans">{answer_html}</div>'
        f'<div class="tr">'
        f'<span>Model: <span class="tv">{data.get("model","")}</span></span>'
        f'<span>Prompt: <span class="tv">{data.get("prompt_tokens",0)}</span> tok</span>'
        f'<span>Completion: <span class="tv">{data.get("completion_tokens",0)}</span> tok</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if warnings:
        st.markdown(
            f'<div class="warn">⚠ NLI Guard flagged {len(warnings)} sentence(s):<br>'
            + "<br>".join(f"• {w}" for w in warnings) + "</div>",
            unsafe_allow_html=True,
        )

    if citations:
        type_tag = {"text":'<span class="tag td">text</span>',"image":'<span class="tag tc">image</span>',"table":'<span class="tag to">table</span>'}
        parts = ['<div style="margin-top:1rem"><div class="sl">Sources & Citations</div>']
        for c in citations:
            img = '<span class="tag tg">📷 img</span>' if c.get("has_image") else ""
            parts.append(
                f'<div class="cit">'
                f'<div class="cit-m">[{c["index"]}] &nbsp; {c["doc_name"]} &nbsp;·&nbsp; p.{c["page"]} &nbsp;'
                f'{type_tag.get(c.get("chunk_type","text"),"")} {img}</div>'
                f'<div class="cit-t">{c["text_snippet"]}</div></div>'
            )
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)   # ONE call for all citations


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INGEST
# ══════════════════════════════════════════════════════════════════════════════

def page_ingest():
    st.markdown(
        '<div class="pt">Ingest<br/><span>Documents</span></div>'
        '<div class="ps">PDF · PNG · JPG · DOCX · Async Celery pipeline</div>',
        unsafe_allow_html=True,
    )
    col_up, col_jobs = st.columns(2, gap="large")

    with col_up:
        st.markdown('<div class="sl">Upload File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop file", type=["pdf","png","jpg","jpeg","webp","docx"],
                                    label_visibility="collapsed")
        if uploaded:
            st.markdown(
                f'<div class="card card-c" style="padding:.9rem 1.1rem;margin-top:.5rem">'
                f'<div style="font-size:.62rem;color:#5a6a7a;text-transform:uppercase">Ready to ingest</div>'
                f'<div style="font-size:.88rem;font-weight:600;color:#e8edf3;margin-top:2px">{uploaded.name}</div>'
                f'<div style="font-size:.62rem;color:#5a6a7a">{uploaded.size/1024:.1f} KB · {uploaded.type}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("↑  Start Ingestion", key="do_ingest"):
                with st.spinner("Uploading…"):
                    data, err = api("POST", "/ingest",
                                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
                if err:
                    st.markdown(f'<div class="warn">⚠ {err}</div>', unsafe_allow_html=True)
                else:
                    st.session_state.ingest_jobs.insert(0, {
                        "task_id": data["task_id"], "doc_id": data["doc_id"],
                        "filename": data["filename"], "status": "PENDING",
                    })
                    invalidate_documents()
                    st.toast(f"✓ Queued: {data['filename']}", icon="📄")
                    st.rerun()

        st.markdown("""<hr/>
        <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:.8rem">
          <span class="tag tc">PDF</span><span class="tag tc">PNG</span>
          <span class="tag tc">JPG</span><span class="tag tc">WEBP</span><span class="tag to">DOCX</span>
        </div>
        <div style="font-size:.68rem;color:#5a6a7a;line-height:1.85">
          → Two-pass PDF (text + images + tables)<br>
          → CLIP ViT-L/14 image embeddings<br>
          → GPT-4o Vision captions for figures<br>
          → Hybrid BM25 + dense indexing
        </div>""", unsafe_allow_html=True)

    with col_jobs:
        _job_panel()


@st.fragment(run_every=5)
def _job_panel():
    """Polls pending jobs every 5 s. Only this fragment rerenders — not the whole page."""
    st.markdown('<div class="sl">Ingestion Jobs</div>', unsafe_allow_html=True)
    jobs = st.session_state.ingest_jobs
    if not jobs:
        st.markdown('<div style="font-size:.75rem;color:#5a6a7a;padding:.75rem 0">No jobs yet.</div>',
                    unsafe_allow_html=True)
        return

    badge_map = {
        "SUCCESS": '<span class="badge b-ok">✓ done</span>',
        "FAILURE": '<span class="badge b-err">✗ failed</span>',
        "STARTED": '<span class="badge b-pend">⟳ processing</span>',
        "PROGRESS":'<span class="badge b-pend">⟳ processing</span>',
        "PENDING": '<span class="badge b-pend">… queued</span>',
    }

    for job in jobs:
        if job["status"] not in ("SUCCESS", "FAILURE"):
            data, _ = api("GET", f"/jobs/{job['task_id']}")
            if data and data["status"] != job["status"]:
                job["status"] = data["status"]
                if data.get("result"):
                    job["chunks"] = data["result"].get("chunk_count", "?")
                if data["status"] == "SUCCESS":
                    invalidate_documents()

    parts = []
    for job in jobs:
        badge = badge_map.get(job["status"], badge_map["PENDING"])
        chunks = f"· {job.get('chunks','?')} chunks" if job["status"] == "SUCCESS" else ""
        parts.append(
            f'<div class="card" style="padding:.8rem 1rem;margin-bottom:.4rem">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-size:.8rem;font-weight:600;color:#e8edf3">{job["filename"]}</span>'
            f'{badge}</div>'
            f'<div style="font-size:.6rem;color:#5a6a7a;margin-top:3px">{job["task_id"][:22]}… {chunks}</div>'
            f'</div>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def page_documents():
    st.markdown(
        '<div class="pt">Indexed<br/><span>Documents</span></div>'
        '<div class="ps">Manage the knowledge base · Qdrant collection</div>',
        unsafe_allow_html=True,
    )

    col_r, _ = st.columns([1, 5])
    with col_r:
        if st.button("↻  Refresh", key="refresh_docs"):
            invalidate_documents()
            st.rerun()

    docs = fetch_documents()
    total = sum(d.get("chunk_count", 0) for d in docs)

    st.markdown(f"""
    <div class="sg">
      <div class="sc"><div class="sv">{len(docs)}</div><div class="sk">Documents</div></div>
      <div class="sc"><div class="sv">{total:,}</div><div class="sk">Total Chunks</div></div>
      <div class="sc"><div class="sv">{round(total/max(len(docs),1))}</div><div class="sk">Avg / Doc</div></div>
    </div>
    """, unsafe_allow_html=True)

    if not docs:
        st.markdown('<div style="font-size:.8rem;color:#5a6a7a;padding:.75rem 0">No documents indexed yet.</div>',
                    unsafe_allow_html=True)
        return

    st.markdown('<div class="sl">Documents</div>', unsafe_allow_html=True)
    for doc in docs:
        ext = doc["doc_name"].rsplit(".", 1)[-1].lower() if "." in doc["doc_name"] else "?"
        tag_cls = "tc" if ext in ("pdf","png","jpg","jpeg","webp") else "to"
        col_info, col_del = st.columns([6, 1])
        with col_info:
            st.markdown(
                f'<div class="dr"><div>'
                f'<div class="dn">{doc["doc_name"]}</div>'
                f'<div class="dm">{doc["doc_id"][:28]}… &nbsp;·&nbsp; {doc["chunk_count"]} chunks</div>'
                f'</div><span class="tag {tag_cls}">{ext.upper()}</span></div>',
                unsafe_allow_html=True,
            )
        with col_del:
            if st.button("✕", key=f"del_{doc['doc_id']}", help="Delete"):
                _, err = api("DELETE", f"/documents/{doc['doc_id']}")
                if err:
                    st.error(err)
                else:
                    invalidate_documents()
                    st.toast(f"Deleted {doc['doc_name']}", icon="🗑")
                    st.rerun()


# ─── Router ───────────────────────────────────────────────────────────────────

{"query": page_query, "ingest": page_ingest, "documents": page_documents}[st.session_state.page]()