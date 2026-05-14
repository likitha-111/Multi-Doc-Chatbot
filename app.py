import streamlit as st
import requests
import uuid
import time
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(
    page_title="RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8001")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

  :root {
    --bg:        #0a0d12;
    --bg2:       #111520;
    --bg3:       #181d2a;
    --border:    #1e2536;
    --accent:    #4f8eff;
    --accent2:   #a78bfa;
    --success:   #34d399;
    --warn:      #fbbf24;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --user-bg:   #1a2340;
    --bot-bg:    #111520;
    --card:      #12172080;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

  /* Main area */
  .main .block-container { padding: 2rem 2.5rem 4rem; max-width: 900px; }

  /* Typography */
  h1, h2, h3, .syne { font-family: 'Syne', sans-serif !important; }
  code, .mono { font-family: 'DM Mono', monospace !important; font-size: 0.85em; }

  /* Header strip */
  .header-strip {
    display: flex; align-items: center; gap: 14px;
    padding: 1.2rem 0 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
  }
  .header-strip .logo {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; box-shadow: 0 0 20px #4f8eff40;
  }
  .header-strip h1 {
    margin: 0; font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .header-strip .sub { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }

  /* Chat bubbles */
  .chat-wrap { display: flex; flex-direction: column; gap: 1.2rem; padding: 0.5rem 0; }

  .msg-row { display: flex; gap: 12px; align-items: flex-start; }
  .msg-row.user { flex-direction: row-reverse; }

  .avatar {
    width: 34px; height: 34px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 700;
  }
  .avatar.user { background: linear-gradient(135deg,#4f8eff,#a78bfa); }
  .avatar.bot  { background: var(--bg3); border: 1px solid var(--border); }

  .bubble {
    max-width: 78%; padding: 0.9rem 1.1rem; border-radius: 14px;
    line-height: 1.65; font-size: 0.95rem;
  }
  .bubble.user {
    background: var(--user-bg);
    border: 1px solid #1e3a6a;
    border-bottom-right-radius: 4px;
  }
  .bubble.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }

  /* Source chips */
  .sources-wrap { margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 6px; }
  .src-chip {
    background: #0f1a2e; border: 1px solid #1e3a6a; border-radius: 20px;
    padding: 3px 10px; font-size: 0.73rem; color: var(--accent);
    font-family: 'DM Mono', monospace; cursor: default;
    transition: background 0.2s;
  }
  .src-chip:hover { background: #1a2e4a; }
  .src-chip .pg { color: var(--muted); }

  /* Source expand snippet */
  .src-snippet {
    background: #0a0f1a; border-left: 3px solid var(--accent);
    padding: 8px 12px; border-radius: 0 8px 8px 0;
    font-size: 0.78rem; color: var(--muted); margin-top: 4px;
    font-family: 'DM Mono', monospace; line-height: 1.5;
  }

  /* Stats row */
  .stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
  }
  .stat-card .val {
    font-family: 'Syne', sans-serif; font-size: 1.8rem;
    font-weight: 800; color: var(--accent);
  }
  .stat-card .lbl { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

  /* Doc pill in sidebar */
  .doc-pill {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 12px; margin-bottom: 6px;
    font-size: 0.82rem;
  }
  .doc-pill .fname { color: var(--text); font-weight: 500; word-break: break-all; }
  .doc-pill .chnk  { color: var(--muted); font-size: 0.73rem; margin-top: 2px; }

  /* Streamlit overrides */
  .stTextInput > div > div > input,
  .stTextArea textarea {
    background: var(--bg2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px #4f8eff30 !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
  }
  .stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
  .stButton > button:active { transform: translateY(0) !important; }

  [data-testid="stFileUploader"] {
    background: var(--bg2) !important; border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
  }
  [data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

  div[data-baseweb="notification"] { display: none !important; }

  .stSpinner > div { border-top-color: var(--accent) !important; }

  hr { border-color: var(--border) !important; }

  /* Thinking dots */
  .thinking { display: flex; gap: 5px; padding: 6px 2px; align-items: center; }
  .thinking span {
    width: 8px; height: 8px; background: var(--accent);
    border-radius: 50%; animation: bounce 1.2s infinite;
  }
  .thinking span:nth-child(2) { animation-delay: 0.2s; background: var(--accent2); }
  .thinking span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%,60%,100% { transform: translateY(0); opacity:.6; }
    30% { transform: translateY(-6px); opacity:1; }
  }

  /* welcome card */
  .welcome {
    background: linear-gradient(135deg,#0d1929,#0f1522);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2.5rem; text-align: center; margin: 2rem 0;
  }
  .welcome h2 { font-family:'Syne',sans-serif; font-size:1.5rem; color:var(--text); }
  .welcome p  { color: var(--muted); font-size: 0.9rem; line-height: 1.7; }
  .welcome .steps { display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin-top:1.5rem; }
  .step {
    background:var(--bg3); border:1px solid var(--border); border-radius:12px;
    padding:.8rem 1.2rem; font-size:0.82rem; color:var(--muted);
  }
  .step .num {
    font-family:'Syne',sans-serif; font-size:1.1rem;
    font-weight:800; color:var(--accent); display:block; margin-bottom:4px;
  }
</style>
""", unsafe_allow_html=True)


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_info" not in st.session_state:
    st.session_state.docs_info = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

def fetch_documents():
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=10)
        if r.status_code == 200:
            data = r.json()
            st.session_state.docs_info = data["documents"]
            st.session_state.total_chunks = data["total_chunks"]
    except Exception:
        pass

def render_sources(sources):
    if not sources:
        return ""
    chips = ""
    snippets = ""
    for s in sources:
        chips += f'<span class="src-chip">📄 {s["filename"]} <span class="pg">p.{s["page"]}</span></span>'
        snippets += f'<div class="src-snippet">"{s["snippet"]}…"</div>'
    return f'<div class="sources-wrap">{chips}</div>{snippets}'

with st.sidebar:
    st.markdown('<h2 style="font-family:Syne,sans-serif;font-size:1.1rem;margin-bottom:1rem;">⚙️ Configuration</h2>', unsafe_allow_html=True)

    groq_key = GROQ_API_KEY
    if groq_key:
        st.markdown('<div style="background:#0d2218;border:1px solid #166534;border-radius:8px;padding:8px 12px;font-size:0.8rem;color:#34d399;">✅ Groq API key loaded from environment</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:#2d1212;border:1px solid #7f1d1d;border-radius:8px;padding:8px 12px;font-size:0.8rem;color:#f87171;">⚠️ GROQ_API_KEY not set in environment</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h2 style="font-family:Syne,sans-serif;font-size:1.1rem;margin-bottom:1rem;">📤 Upload Documents</h2>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⬆ Process Documents", use_container_width=True) and uploaded:
        if not groq_key:
            st.error("⚠️ GROQ_API_KEY environment variable is not set.")
        else:
            with st.spinner("Embedding documents…"):
                files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
                try:
                    r = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
                    if r.status_code == 200:
                        data = r.json()
                        st.success(f"✅ {data['message']} ({data['total_chunks']} chunks)")
                        fetch_documents()
                    else:
                        st.error(f"Upload failed: {r.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.markdown("---")

    # Document list
    fetch_documents()
    docs = st.session_state.docs_info
    total = st.session_state.total_chunks

    if docs:
        st.markdown(f'<h2 style="font-family:Syne,sans-serif;font-size:1rem;margin-bottom:.8rem;">📚 Knowledge Base <span style="color:#4f8eff;font-size:.85rem;">({len(docs)} docs)</span></h2>', unsafe_allow_html=True)
        for d in docs:
            st.markdown(f'''
            <div class="doc-pill">
              <div class="fname">📄 {d["filename"]}</div>
              <div class="chnk">{d["chunks"]} chunks indexed</div>
            </div>''', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#64748b;font-size:.82rem;">No documents uploaded yet.</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear DB", use_container_width=True):
            try:
                r = requests.delete(f"{API_BASE}/clear", timeout=10)
                if r.status_code == 200:
                    st.session_state.docs_info = []
                    st.session_state.total_chunks = 0
                    st.session_state.messages = []
                    st.rerun()
            except Exception as e:
                st.error(str(e))
    with col2:
        if st.button("💬 New Chat", use_container_width=True):
            try:
                requests.delete(f"{API_BASE}/session/{st.session_state.session_id}", timeout=5)
            except Exception:
                pass
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.rerun()

    st.markdown(f'<p style="color:#334155;font-size:.72rem;margin-top:1rem;">Session: <span style="font-family:\'DM Mono\',monospace;">{st.session_state.session_id}</span></p>', unsafe_allow_html=True)


st.markdown("""
<div class="header-strip">
  <div class="logo">🧠</div>
  <div>
    <h1>RAG Assistant</h1>
    <div class="sub">Multi-Document Knowledge Assistant · Powered by Groq + LangChain</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Stats bar
docs = st.session_state.docs_info
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="stat-card"><div class="val">{len(docs)}</div><div class="lbl">Documents Loaded</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="val">{st.session_state.total_chunks}</div><div class="lbl">Chunks Indexed</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="val">{len(st.session_state.messages)}</div><div class="lbl">Messages in Chat</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# Welcome screen
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
      <h2>Start Your Document Conversation</h2>
      <p>Upload PDFs into the knowledge base, then ask anything.<br>
         DocMind finds the right chunks, cites sources, and remembers your conversation.</p>
      <div class="steps">
        <div class="step"><span class="num">01</span>Set GROQ_API_KEY env var</div>
        <div class="step"><span class="num">02</span>Upload PDF files</div>
        <div class="step"><span class="num">03</span>Process &amp; embed</div>
        <div class="step"><span class="num">04</span>Ask questions below</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Chat history
chat_html = '<div class="chat-wrap">'
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        chat_html += f'''
        <div class="msg-row user">
          <div class="avatar user">U</div>
          <div class="bubble user">{content}</div>
        </div>'''
    else:
        src_html = render_sources(sources)
        chat_html += f'''
        <div class="msg-row bot">
          <div class="avatar bot">🧠</div>
          <div class="bubble bot">
            {content}
            {src_html}
          </div>
        </div>'''

chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)


thinking_placeholder = st.empty()

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([5, 1])
    with cols[0]:
        question = st.text_input(
            "Your question",
            placeholder="Ask anything about your documents…",
            label_visibility="collapsed",
        )
    with cols[1]:
        submitted = st.form_submit_button("Send →", use_container_width=True)

if submitted and question.strip():
    if not groq_key:
        st.error("⚠️ GROQ_API_KEY environment variable is not set. Run: export GROQ_API_KEY=gsk_...")
    elif not docs:
        st.error("⚠️ No documents in the knowledge base. Upload PDFs first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question.strip()})

        # Show thinking
        thinking_placeholder.markdown('''
        <div class="msg-row bot" style="margin-top:1rem;">
          <div class="avatar bot">🧠</div>
          <div class="bubble bot">
            <div class="thinking">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>''', unsafe_allow_html=True)

        try:
            r = requests.post(
                f"{API_BASE}/chat",
                json={
                    "question": question.strip(),
                    "session_id": st.session_state.session_id,
                    "groq_api_key": groq_key,
                },
                timeout=60,
            )
            thinking_placeholder.empty()

            if r.status_code == 200:
                data = r.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"],
                })
                st.rerun()
            else:
                thinking_placeholder.empty()
                err = r.json().get("detail", r.text)
                st.error(f"API Error: {err}")
                st.session_state.messages.pop()

        except requests.exceptions.ConnectionError:
            thinking_placeholder.empty()
            st.error("❌ Cannot connect to backend. Make sure the FastAPI server is running on port 8001.")
            st.session_state.messages.pop()
        except Exception as e:
            thinking_placeholder.empty()
            st.error(f"Error: {e}")
            st.session_state.messages.pop()