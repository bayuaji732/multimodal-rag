"""
streamlit_app.py  —  Multi-Modal RAG Knowledge Engine  v2
──────────────────────────────────────────────────────────
New in v2:
  • CSV / XLSX / XLS upload + live table preview before ingestion
  • Table citations rendered with dimension badges + header chips
  • Answer renderer handles markdown tables (st.markdown native)
  • Format legend panel in sidebar
  • st.fragment job poller unchanged (still partial-rerender only)

Performance:
  • Persistent httpx.Client via st.cache_resource
  • Health / document list cached with TTL + manual bust
  • All citation HTML batched into one st.markdown call
  • CSS injected once via st.cache_data
"""
from __future__ import annotations

import io
from pathlib import Path

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG · Knowledge Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── HTTP client ──────────────────────────────────────────────────────────────

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


# ─── Cached API helpers ────────────────────────────────────────────────────────

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


# ─── Format catalogue (single source of truth for UI) ────────────────────────

FORMAT_INFO: dict[str, dict] = {
    ".pdf":  {"label": "PDF",  "color": "blue",   "table": True,  "image": True,  "text": True},
    ".csv":  {"label": "CSV",  "color": "green",  "table": True,  "image": False, "text": False},
    ".xlsx": {"label": "XLSX", "color": "green",  "table": True,  "image": False, "text": False},
    ".xls":  {"label": "XLS",  "color": "green",  "table": True,  "image": False, "text": False},
    ".docx": {"label": "DOCX", "color": "orange", "table": True,  "image": False, "text": True},
    ".png":  {"label": "PNG",  "color": "blue",   "table": False, "image": True,  "text": False},
    ".jpg":  {"label": "JPG",  "color": "blue",   "table": False, "image": True,  "text": False},
    ".jpeg": {"label": "JPEG", "color": "blue",   "table": False, "image": True,  "text": False},
    ".webp": {"label": "WEBP", "color": "blue",   "table": False, "image": True,  "text": False},
}

UPLOAD_TYPES = [ext.lstrip(".") for ext in FORMAT_INFO]


# ─── CSS (injected once) ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

/* ── base ── */
html,body,[class*="css"]{
  font-family:'IBM Plex Mono',monospace!important;
  background:#080b10;color:#e8edf3;
}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem 2.5rem 4rem;max-width:1260px}

/* ── sidebar ── */
[data-testid="stSidebar"]{background:#0a0f18;border-right:1px solid #1a2535}
[data-testid="stSidebar"] *{font-family:'IBM Plex Mono',monospace!important}

/* ── typography ── */
.pt{font-family:'Syne',sans-serif!important;font-size:2.4rem;font-weight:800;
    letter-spacing:-.02em;line-height:1.1;color:#e8edf3;margin-bottom:.2rem}
.pt span{color:#00e5ff}
.ps{font-size:.68rem;color:#5a6a7a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:1.8rem}

/* ── section labels ── */
.sl{font-size:.58rem;letter-spacing:.22em;text-transform:uppercase;color:#5a6a7a;
    margin-bottom:.7rem;display:flex;align-items:center;gap:8px}
.sl::after{content:'';flex:1;height:1px;background:#1a2535}

/* ── tags ── */
.tag{display:inline-block;font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;
     padding:3px 9px;border-radius:2px;margin:2px;white-space:nowrap}
.tc {background:rgba(0,229,255,.08); color:#00e5ff; border:1px solid rgba(0,229,255,.25)}
.to {background:rgba(255,107,53,.08);color:#ff6b35; border:1px solid rgba(255,107,53,.25)}
.tg {background:rgba(127,255,107,.08);color:#7fff6b;border:1px solid rgba(127,255,107,.25)}
.td {background:rgba(255,255,255,.04);color:#5a6a7a;border:1px solid #1a2535}
.tp {background:rgba(168,85,247,.08); color:#c084fc;border:1px solid rgba(168,85,247,.25)}

/* ── cards ── */
.card{background:#0d1520;border:1px solid #1a2535;border-radius:3px;padding:1.25rem;margin-bottom:.75rem}
.card-c{border-top:2px solid #00e5ff}
.card-t{border-top:2px solid #7fff6b}
.card-o{border-top:2px solid #ff6b35}

/* ── answer block ── */
.ans{background:#090e18;border:1px solid #1a2535;border-top:2px solid #00e5ff;
     padding:1.4rem;border-radius:3px;font-size:.85rem;line-height:1.85;
     color:#e8edf3;white-space:pre-wrap}
.uv{background:rgba(255,107,53,.12);color:#ff6b35;border-radius:2px;padding:1px 5px}

/* ── markdown table inside answer ── */
.ans-md table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.78rem}
.ans-md th{background:rgba(0,229,255,.06);color:#00e5ff;padding:.45rem .7rem;
           border:1px solid #1a2535;text-align:left;font-weight:500;letter-spacing:.06em}
.ans-md td{padding:.4rem .7rem;border:1px solid #1a2535;color:#9ab0c5;vertical-align:top}
.ans-md tr:nth-child(even) td{background:rgba(255,255,255,.015)}

/* ── citations ── */
.cit{background:#090e18;border-left:3px solid #00e5ff;padding:.7rem 1rem;
     margin:.35rem 0;border-radius:0 3px 3px 0;font-size:.74rem}
.cit-table{border-left-color:#7fff6b}
.cit-image{border-left-color:#ff6b35}
.cit-m{font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;color:#5a6a7a;margin-bottom:.35rem}
.cit-t{color:#9ab0c5;line-height:1.6;margin-top:.3rem}
.cit-headers{display:flex;flex-wrap:wrap;gap:4px;margin-top:.45rem}
.ch{font-size:.55rem;padding:2px 7px;background:rgba(127,255,107,.06);
    color:#7fff6b;border:1px solid rgba(127,255,107,.2);border-radius:2px;
    letter-spacing:.06em}

/* ── warnings ── */
.warn{background:rgba(255,107,53,.06);border:1px solid rgba(255,107,53,.3);
      border-radius:3px;padding:.7rem 1rem;font-size:.72rem;color:#ff6b35;margin-top:.5rem}

/* ── stats grid ── */
.sg{display:flex;gap:.75rem;margin-bottom:1.4rem;flex-wrap:wrap}
.sc{flex:1;min-width:110px;background:#0d1520;border:1px solid #1a2535;border-radius:3px;
    padding:.9rem;text-align:center}
.sv{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:700;color:#00e5ff;line-height:1}
.sk{font-size:.58rem;letter-spacing:.15em;text-transform:uppercase;color:#5a6a7a;margin-top:3px}

/* ── badges ── */
.badge{display:inline-block;padding:3px 10px;border-radius:2px;font-size:.58rem;
       letter-spacing:.12em;text-transform:uppercase}
.b-ok  {background:rgba(127,255,107,.1);color:#7fff6b;border:1px solid rgba(127,255,107,.3)}
.b-pend{background:rgba(0,229,255,.1);  color:#00e5ff;border:1px solid rgba(0,229,255,.3)}
.b-err {background:rgba(255,107,53,.1); color:#ff6b35;border:1px solid rgba(255,107,53,.3)}

/* ── token row ── */
.tr{display:flex;gap:1rem;font-size:.62rem;color:#5a6a7a;
    letter-spacing:.08em;text-transform:uppercase;margin-top:.5rem;flex-wrap:wrap}
.tv{color:#9ab0c5}

/* ── document row ── */
.dr{display:flex;align-items:center;justify-content:space-between;
    background:#090e18;border:1px solid #1a2535;border-radius:3px;
    padding:.65rem 1.1rem;margin-bottom:.4rem;font-size:.75rem}
.dn{color:#e8edf3;font-weight:600}
.dm{color:#5a6a7a;font-size:.6rem;margin-top:2px}

/* ── table preview ── */
.tbl-preview{overflow-x:auto;border:1px solid #1a2535;border-radius:3px;
             background:#090e18;padding:.6rem}
.tbl-preview table{border-collapse:collapse;font-size:.7rem;width:100%;min-width:400px}
.tbl-preview th{background:rgba(127,255,107,.06);color:#7fff6b;padding:.35rem .6rem;
                border:1px solid #1a2535;font-weight:500;letter-spacing:.04em;white-space:nowrap}
.tbl-preview td{padding:.3rem .6rem;border:1px solid #1a2535;color:#9ab0c5}
.tbl-preview tr:nth-child(even) td{background:rgba(255,255,255,.015)}
.tbl-preview .more{font-size:.62rem;color:#5a6a7a;padding:.5rem .6rem;
                   text-align:center;border-top:1px solid #1a2535;font-style:italic}

/* ── format legend ── */
.fmt-row{display:flex;align-items:center;gap:10px;padding:.45rem 0;
         border-bottom:1px solid #0f1825;font-size:.68rem}
.fmt-row:last-child{border-bottom:none}
.fmt-cap{width:50px;flex-shrink:0}
.fmt-desc{color:#5a6a7a;font-size:.62rem;line-height:1.5}
.fmt-caps{display:flex;gap:4px;flex-wrap:wrap;margin-top:2px}

/* ── inputs ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea{
  background:#090e18!important;border:1px solid #1a2535!important;
  color:#e8edf3!important;font-family:'IBM Plex Mono',monospace!important;border-radius:3px!important}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus{
  border-color:#00e5ff!important;box-shadow:0 0 0 1px rgba(0,229,255,.2)!important}
div[data-testid="stFileUploader"]{
  background:#090e18;border:1px dashed #1a2535!important;border-radius:3px}

/* ── buttons ── */
.stButton>button{
  background:transparent!important;border:1px solid #00e5ff!important;
  color:#00e5ff!important;font-family:'IBM Plex Mono',monospace!important;
  font-size:.7rem!important;letter-spacing:.08em;border-radius:2px!important;
  padding:.45rem 1.1rem!important;transition:background .15s,color .15s}
.stButton>button:hover{background:#00e5ff!important;color:#080b10!important}

/* ── misc ── */
div[data-testid="stMultiSelect"] *{font-family:'IBM Plex Mono',monospace!important}
.stCheckbox label{font-size:.72rem!important;color:#9ab0c5!important}
hr{border:none;border-top:1px solid #1a2535;margin:1.2rem 0}
</style>"""


st.markdown(_css(), unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────

for k, v in {
    "page": "query",
    "last_result": None,
    "ingest_jobs": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 .8rem">
      <div style="font-size:.56rem;letter-spacing:.22em;text-transform:uppercase;
                  color:#5a6a7a;margin-bottom:7px">⬡ &nbsp;LLM / GenAI Engineer</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:800;
                  color:#e8edf3;line-height:1.1">
        Multi-Modal<br/><span style="color:#00e5ff">RAG Engine</span>
      </div>
    </div><hr/>
    """, unsafe_allow_html=True)

    is_online = fetch_health()
    st.markdown(
        '<span class="badge b-ok">● API Online</span>'  if is_online else
        '<span class="badge b-err">● API Offline</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    for key, label in [
        ("query",     "◈  Query"),
        ("ingest",    "↑  Ingest"),
        ("documents", "≡  Documents"),
    ]:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            if st.session_state.page != key:
                st.session_state.page = key
                st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="sl">Settings</div>', unsafe_allow_html=True)
    apply_guard = st.checkbox("NLI Hallucination Guard", value=True)
    use_stream  = st.checkbox("Streaming Response",      value=False)

    # ── Format legend ──────────────────────────────────────────────────────────
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="sl">Supported Formats</div>', unsafe_allow_html=True)

    legend_rows = [
        (".pdf",  "tc", "Text · Tables · Images"),
        (".csv",  "tg", "Table (full file)"),
        (".xlsx", "tg", "Table (per sheet)"),
        (".xls",  "tg", "Table (per sheet)"),
        (".docx", "to", "Text · Tables"),
        (".png",  "tc", "Image (CLIP embedded)"),
        (".jpg",  "tc", "Image (CLIP embedded)"),
        (".webp", "tc", "Image (CLIP embedded)"),
    ]
    rows_html = "".join(
        f'<div class="fmt-row">'
        f'<span class="tag {cls} fmt-cap">{ext}</span>'
        f'<span class="fmt-desc">{desc}</span>'
        f'</div>'
        for ext, cls, desc in legend_rows
    )
    st.markdown(f'<div style="margin-top:.3rem">{rows_html}</div>', unsafe_allow_html=True)
    st.markdown("""<br/><div style="font-size:.58rem;color:#5a6a7a;line-height:1.8">
      RAG · Hybrid BM25+Dense<br>CLIP · text-embedding-3<br>
      Qdrant · Cohere Rerank<br>GPT-4o · NLI Guard
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_type_tag(ct: str) -> str:
    m = {"text": '<span class="tag td">text</span>',
         "image":'<span class="tag tc">image</span>',
         "table":'<span class="tag tg">table</span>'}
    return m.get(ct, f'<span class="tag td">{ct}</span>')


def _format_tag(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    info = FORMAT_INFO.get(ext, {})
    cls  = {"blue": "tc", "green": "tg", "orange": "to"}.get(info.get("color", ""), "td")
    label = info.get("label", ext.upper())
    return f'<span class="tag {cls}">{label}</span>'


# ─── Table preview (CSV / XLSX before ingestion) ──────────────────────────────

def _render_table_preview(uploaded) -> None:
    """Read the first 10 rows of a CSV or XLSX and render an HTML preview."""
    ext = Path(uploaded.name).suffix.lower()
    try:
        if ext == ".csv":
            import csv, io as _io
            raw = uploaded.getvalue()
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    text    = raw.decode(enc)
                    sample  = text[:4096]
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    except csv.Error:
                        dialect = csv.excel
                    reader  = csv.reader(_io.StringIO(text), dialect)
                    rows    = [r for r in reader if any(c.strip() for c in r)]
                    break
                except UnicodeDecodeError:
                    continue

        elif ext in (".xlsx", ".xls"):
            import openpyxl
            wb   = openpyxl.load_workbook(io.BytesIO(uploaded.getvalue()),
                                          read_only=True, data_only=True)
            ws   = wb[wb.sheetnames[0]]
            rows = [
                [str(c) if c is not None else "" for c in row]
                for row in ws.iter_rows(values_only=True)
            ]
            rows = [r for r in rows if any(c.strip() for c in r)]
            wb.close()
        else:
            return

        if not rows:
            return

        preview    = rows[:11]   # header + 10 data rows
        total_rows = len(rows) - 1
        header     = preview[0]
        data       = preview[1:]

        ths = "".join(f"<th>{h}</th>" for h in header)
        trs = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in data
        )
        more = ""
        if total_rows > 10:
            more = f'<div class="more">… {total_rows - 10:,} more rows not shown</div>'

        st.markdown(
            f'<div class="tbl-preview">'
            f'<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>'
            f'{more}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:.6rem;color:#5a6a7a;margin-top:.35rem">'
            f'{len(header)} columns · {total_rows:,} data rows</div>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.markdown(
            f'<div class="warn" style="font-size:.68rem">Preview unavailable: {exc}</div>',
            unsafe_allow_html=True,
        )


# ── Answer renderer ───────────────────────────────────────────────────────────

def _render_result(data: dict) -> None:
    answer    = data.get("answer", "")
    citations = data.get("citations", [])
    warnings  = data.get("warnings", [])

    st.markdown('<div class="sl">Answer</div>', unsafe_allow_html=True)

    # Replace [⚠ unverified] markers with styled span
    answer_display = answer.replace(
        "[⚠ unverified]", '<span class="uv">[⚠ unverified]</span>'
    )

    # If answer contains a markdown table, render with st.markdown (handles
    # tables natively) inside a styled wrapper; otherwise use raw HTML div.
    if "|" in answer and "---" in answer:
        st.markdown(
            f'<div class="ans-md" style="background:#090e18;border:1px solid #1a2535;'
            f'border-top:2px solid #00e5ff;padding:1.4rem;border-radius:3px;'
            f'font-size:.85rem;line-height:1.85">',
            unsafe_allow_html=True,
        )
        st.markdown(answer_display, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="ans">{answer_display}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="tr">'
        f'<span>Model: <span class="tv">{data.get("model","")}</span></span>'
        f'<span>Prompt: <span class="tv">{data.get("prompt_tokens",0):,}</span> tok</span>'
        f'<span>Completion: <span class="tv">{data.get("completion_tokens",0):,}</span> tok</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if warnings:
        st.markdown(
            f'<div class="warn">⚠ NLI Guard flagged {len(warnings)} sentence(s):<br>'
            + "<br>".join(f"• {w}" for w in warnings) + "</div>",
            unsafe_allow_html=True,
        )

    if not citations:
        return

    st.markdown('<div style="margin-top:1.1rem"><div class="sl">Sources & Citations</div>', unsafe_allow_html=True)
    parts: list[str] = []

    for c in citations:
        ct        = c.get("chunk_type", "text")
        cit_cls   = f"cit cit-{ct}"
        type_tag  = _chunk_type_tag(ct)
        img_badge = '<span class="tag tc">📷 img</span>' if c.get("has_image") else ""
        snippet   = c.get("text_snippet", "")

        # ── table-specific extras ──────────────────────────────────────────────
        table_meta = ""
        if ct == "table":
            rows  = c.get("table_rows")
            cols  = c.get("table_cols")
            title = c.get("table_title") or ""
            hdrs  = c.get("table_headers") or []

            dims = ""
            if rows is not None and cols is not None:
                dims = (
                    f'<span class="tag tg" style="font-size:.55rem">'
                    f'{rows}r × {cols}c</span>'
                )
            title_line = (
                f'<div style="font-size:.65rem;color:#7fff6b;margin:.3rem 0 .1rem">'
                f'↳ {title}</div>'
                if title else ""
            )
            headers_html = ""
            if hdrs:
                chips = "".join(f'<span class="ch">{h}</span>' for h in hdrs[:8])
                more  = f'<span class="ch" style="opacity:.5">+{len(hdrs)-8}</span>' if len(hdrs) > 8 else ""
                headers_html = f'<div class="cit-headers">{chips}{more}</div>'

            table_meta = title_line + (
                f'<div style="display:flex;align-items:center;gap:6px;margin-top:.2rem">'
                f'{dims}{headers_html}</div>'
                if (dims or headers_html) else ""
            )

        parts.append(
            f'<div class="{cit_cls}">'
            f'<div class="cit-m">[{c["index"]}] &nbsp; {c["doc_name"]} &nbsp;·&nbsp; '
            f'p.{c["page"]} &nbsp;{type_tag} {img_badge}</div>'
            f'{table_meta}'
            f'<div class="cit-t">{snippet}</div>'
            f'</div>'
        )

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: QUERY
# ══════════════════════════════════════════════════════════════════════════════

def page_query():
    st.markdown(
        '<div class="pt">Ask the<br/><span>Knowledge Base</span></div>'
        '<div class="ps">Multi-modal retrieval · Grounded citations · NLI verified</div>',
        unsafe_allow_html=True,
    )

    docs = fetch_documents()

    col_q, col_f = st.columns([3, 1])
    with col_q:
        query = st.text_area(
            "q",
            placeholder=(
                "e.g. Which region had the highest revenue in Q3?\n"
                "     What accuracy did the model achieve?\n"
                "     Summarise the key findings from the report."
            ),
            height=120,
            label_visibility="collapsed",
            key="query_input",
        )
    with col_f:
        st.markdown('<div class="sl">Filter Docs</div>', unsafe_allow_html=True)
        doc_opts = {
            f"{d['doc_name']} ({d['chunk_count']}ch)": d["doc_id"]
            for d in docs
        }
        selected   = st.multiselect("Docs", options=list(doc_opts.keys()),
                                    label_visibility="collapsed")
        filter_ids = [doc_opts[l] for l in selected] or None

    if st.button("◈  Run Query", disabled=not (query or "").strip(), key="run_query"):
        if use_stream:
            _run_streaming(query, filter_ids)
        else:
            with st.spinner("Retrieving & generating…"):
                data, err = api("POST", "/query", json={
                    "query":          query,
                    "apply_guard":    apply_guard,
                    "filter_doc_ids": filter_ids,
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
    full        = ""
    params      = {"q": query}
    if filter_ids:
        params["doc_ids"] = ",".join(filter_ids)
    try:
        with httpx.stream("GET", f"{API_BASE}/query/stream",
                          params=params, timeout=120) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    tok = line[6:]
                    if tok == "[DONE]":
                        break
                    full += tok
                    placeholder.markdown(
                        f'<div class="ans">{full}▌</div>',
                        unsafe_allow_html=True,
                    )
        placeholder.markdown(f'<div class="ans">{full}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Streaming error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INGEST
# ══════════════════════════════════════════════════════════════════════════════

def page_ingest():
    st.markdown(
        '<div class="pt">Ingest<br/><span>Documents</span></div>'
        '<div class="ps">PDF · CSV · XLSX · XLS · DOCX · PNG · JPG · Async Celery pipeline</div>',
        unsafe_allow_html=True,
    )

    col_up, col_jobs = st.columns(2, gap="large")

    with col_up:
        st.markdown('<div class="sl">Upload File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop file",
            type=UPLOAD_TYPES,
            label_visibility="collapsed",
        )

        if uploaded:
            ext  = Path(uploaded.name).suffix.lower()
            info = FORMAT_INFO.get(ext, {})
            caps = []
            if info.get("table"):  caps.append('<span class="tag tg">table</span>')
            if info.get("image"):  caps.append('<span class="tag tc">image</span>')
            if info.get("text"):   caps.append('<span class="tag td">text</span>')
            caps_html = " ".join(caps)

            st.markdown(
                f'<div class="card card-t" style="padding:.9rem 1.1rem;margin-top:.5rem">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                f'<div>'
                f'<div style="font-size:.6rem;color:#5a6a7a;text-transform:uppercase">Ready to ingest</div>'
                f'<div style="font-size:.88rem;font-weight:600;color:#e8edf3;margin-top:2px">{uploaded.name}</div>'
                f'<div style="font-size:.6rem;color:#5a6a7a">{uploaded.size/1024:.1f} KB · {uploaded.type}</div>'
                f'</div>'
                f'<div style="text-align:right">{_format_tag(uploaded.name)}<br/>'
                f'<div style="margin-top:4px">{caps_html}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # ── Live preview for tabular files ────────────────────────────────
            if ext in (".csv", ".xlsx", ".xls"):
                st.markdown('<div class="sl" style="margin-top:.8rem">Table Preview</div>',
                            unsafe_allow_html=True)
                _render_table_preview(uploaded)

            st.markdown("<br/>", unsafe_allow_html=True)

            if st.button("↑  Start Ingestion", key="do_ingest"):
                with st.spinner("Uploading…"):
                    data, err = api(
                        "POST", "/ingest",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    )
                if err:
                    st.markdown(f'<div class="warn">⚠ {err}</div>', unsafe_allow_html=True)
                else:
                    st.session_state.ingest_jobs.insert(0, {
                        "task_id":  data["task_id"],
                        "doc_id":   data["doc_id"],
                        "filename": data["filename"],
                        "status":   "PENDING",
                    })
                    invalidate_documents()
                    st.toast(f"Queued: {data['filename']}", icon="📄")
                    st.rerun()

        # ── Format capability matrix ───────────────────────────────────────────
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="sl">Format Capabilities</div>', unsafe_allow_html=True)
        matrix_rows = [
            (".pdf",  "tc", "✓", "✓", "✓"),
            (".csv",  "tg", "✓", "—", "—"),
            (".xlsx", "tg", "✓", "—", "—"),
            (".xls",  "tg", "✓", "—", "—"),
            (".docx", "to", "✓", "—", "✓"),
            (".png",  "tc", "—", "✓", "—"),
            (".jpg",  "tc", "—", "✓", "—"),
            (".webp", "tc", "—", "✓", "—"),
        ]
        header_row = (
            '<div style="display:grid;grid-template-columns:70px 1fr 1fr 1fr;'
            'gap:4px;font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;'
            'color:#5a6a7a;padding:.3rem 0;border-bottom:1px solid #1a2535">'
            '<span>Format</span><span>Table</span><span>Image</span><span>Text</span>'
            '</div>'
        )
        data_rows = "".join(
            f'<div style="display:grid;grid-template-columns:70px 1fr 1fr 1fr;'
            f'gap:4px;font-size:.68rem;padding:.3rem 0;border-bottom:1px solid #0f1825;'
            f'color:#9ab0c5;align-items:center">'
            f'<span class="tag {cls}" style="width:fit-content">{ext}</span>'
            f'<span style="color:{"#7fff6b" if t=="✓" else "#2a3a4a"}">{t}</span>'
            f'<span style="color:{"#00e5ff" if i=="✓" else "#2a3a4a"}">{i}</span>'
            f'<span style="color:{"#9ab0c5" if tx=="✓" else "#2a3a4a"}">{tx}</span>'
            f'</div>'
            for ext, cls, t, i, tx in matrix_rows
        )
        st.markdown(
            f'<div style="background:#090e18;border:1px solid #1a2535;border-radius:3px;'
            f'padding:.6rem 1rem">{header_row}{data_rows}</div>',
            unsafe_allow_html=True,
        )

    with col_jobs:
        _job_panel()


@st.fragment(run_every=5)
def _job_panel():
    st.markdown('<div class="sl">Ingestion Jobs</div>', unsafe_allow_html=True)
    jobs = st.session_state.ingest_jobs

    if not jobs:
        st.markdown(
            '<div style="font-size:.75rem;color:#5a6a7a;padding:.75rem 0">No jobs yet.</div>',
            unsafe_allow_html=True,
        )
        return

    badge_map = {
        "SUCCESS":  '<span class="badge b-ok">✓ done</span>',
        "FAILURE":  '<span class="badge b-err">✗ failed</span>',
        "STARTED":  '<span class="badge b-pend">⟳ processing</span>',
        "PROGRESS": '<span class="badge b-pend">⟳ processing</span>',
        "PENDING":  '<span class="badge b-pend">… queued</span>',
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

    parts: list[str] = []
    for job in jobs:
        badge  = badge_map.get(job["status"], badge_map["PENDING"])
        chunks = f'· {job.get("chunks","?")} chunks' if job["status"] == "SUCCESS" else ""
        fmt_tag = _format_tag(job["filename"])
        parts.append(
            f'<div class="card" style="padding:.8rem 1rem;margin-bottom:.4rem">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-size:.78rem;font-weight:600;color:#e8edf3">{job["filename"]}</span>'
            f'{badge}</div>'
            f'<div style="display:flex;align-items:center;gap:8px;margin-top:.35rem">'
            f'{fmt_tag}'
            f'<span style="font-size:.58rem;color:#5a6a7a">'
            f'{job["task_id"][:22]}… {chunks}</span>'
            f'</div></div>'
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

    docs  = fetch_documents()
    total = sum(d.get("chunk_count", 0) for d in docs)

    st.markdown(
        f'<div class="sg">'
        f'<div class="sc"><div class="sv">{len(docs)}</div><div class="sk">Documents</div></div>'
        f'<div class="sc"><div class="sv">{total:,}</div><div class="sk">Total Chunks</div></div>'
        f'<div class="sc"><div class="sv">{round(total/max(len(docs),1))}</div>'
        f'<div class="sk">Avg / Doc</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not docs:
        st.markdown(
            '<div style="font-size:.8rem;color:#5a6a7a;padding:.75rem 0">'
            'No documents indexed yet.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="sl">Documents</div>', unsafe_allow_html=True)

    for doc in docs:
        col_info, col_del = st.columns([6, 1])
        with col_info:
            fmt_tag = _format_tag(doc["doc_name"])
            st.markdown(
                f'<div class="dr">'
                f'<div>'
                f'<div class="dn">{doc["doc_name"]}</div>'
                f'<div class="dm">{doc["doc_id"][:28]}… &nbsp;·&nbsp; {doc["chunk_count"]} chunks</div>'
                f'</div>{fmt_tag}</div>',
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

{
    "query":     page_query,
    "ingest":    page_ingest,
    "documents": page_documents,
}[st.session_state.page]()