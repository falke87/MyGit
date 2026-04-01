import streamlit as st
from groq import Groq
import time
import re
import os
from datetime import datetime
from pathlib import Path

# ── Load API key from .env file ───────────────────────────────────────────
def load_api_key() -> str:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("GROQ_API_KEY", "")

DEFAULT_API_KEY = load_api_key()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Humanizer", page_icon="✍️", layout="wide")

st.markdown("""
<style>
    .stTextArea textarea { font-size: 14px; line-height: 1.6; font-family: 'Consolas', 'Courier New', monospace; }
    .log-box {
        background: #1e1e1e; color: #d4d4d4; font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px; padding: 12px; border-radius: 8px; max-height: 350px;
        overflow-y: auto; white-space: pre-wrap; word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ("result_text", ""),
    ("logs", []),
    ("stats", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def extract_filename_from_html(html: str) -> str:
    """Extract H1 title from HTML and convert to slug for filename."""
    match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
    if match:
        title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
        slug = title.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s]+', '-', slug).strip('-')
        slug = re.sub(r'-+', '-', slug)
        if slug:
            return f"{slug}.html"
    return "humanized_article.html"


def add_log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] [{level}] {message}")


def count_words(text: str) -> int:
    """Count words in text, stripping HTML tags for accurate count."""
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'&\w+;', ' ', clean)
    return len(clean.split())


def split_html_into_sections(html: str) -> list[str]:
    """Split HTML into sections at <h2> boundaries for chunk processing."""
    h2_pattern = re.compile(r'(?=<h2[\s>])', re.IGNORECASE)
    positions = [m.start() for m in h2_pattern.finditer(html)]

    if not positions:
        return [html]

    sections = []
    if positions[0] > 0:
        sections.append(html[:positions[0]])

    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(html)
        sections.append(html[pos:end])

    return sections


def build_chunks(html: str, max_words: int = 600) -> list[str]:
    """Build chunks from HTML sections, merging small sections together."""
    sections = split_html_into_sections(html)
    chunks = []
    current_chunk = ""
    current_words = 0

    for section in sections:
        section_words = count_words(section)
        if current_words + section_words > max_words and current_chunk:
            chunks.append(current_chunk)
            current_chunk = section
            current_words = section_words
        else:
            current_chunk += section
            current_words += section_words

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


SYSTEM_PROMPT = """Rewrite AI-generated HTML to sound human. Beat AI detectors (GPTZero, Turnitin, Originality.ai).

REWRITE: <p>, <h2>, <h3> text, <li> items, text in <strong>/<em>/<a> tags.

NEVER TOUCH: <h1> title (keep exact), HTML comments (<!-- -->), all HTML tags/attributes/styles, photo credits, <img> tags, URLs, numbers/prices/percentages/dates, proper nouns (names, brands, places), tables, footer, code blocks, disclaimers, quotes.

STYLE: Vary sentence length (mix short+long). Use simple words. Never use: delve, crucial, comprehensive, leverage, utilize, moreover, furthermore, robust, streamline, pivotal, foster, elevate, harness, navigate. Use casual connectors (but, so, though, honestly, look, the thing is). Add human touches: rhetorical questions, asides, start with And/But. Be direct, opinionated, like a friend. Vary paragraph length. Same language as input.

OUTPUT: Only rewritten HTML. No explanations, no markdown fences."""


def humanize_chunk(client: Groq, chunk: str, chunk_num: int, total_chunks: int, context_summary: str = "", log_placeholder=None) -> str:
    """Rewrite a single HTML chunk using Groq Llama, with auto-retry on rate limit."""
    add_log(f"Processing chunk {chunk_num}/{total_chunks} ({count_words(chunk)} words)")

    user_message = ""
    if context_summary:
        user_message += f"[CONTEXT — the article section right before this one discussed: {context_summary}. Use this for tone/flow continuity only. Do NOT repeat or rewrite it.]\n\n"
    user_message += chunk

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=4096,
                temperature=0.9,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            result = response.choices[0].message.content
            usage = response.usage
            add_log(f"Chunk {chunk_num} done — {usage.prompt_tokens} in / {usage.completion_tokens} out tokens")
            return result
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = "rate" in error_msg.lower() or "429" in error_msg or "413" in error_msg
            if is_rate_limit and attempt < max_retries - 1:
                wait = [65, 65, 70, 75][attempt]
                add_log(f"Rate limited — waiting {wait}s before retry ({attempt+1}/{max_retries})", "WARN")
                if log_placeholder:
                    log_placeholder.markdown(format_logs(), unsafe_allow_html=True)
                time.sleep(wait)
            else:
                add_log(f"Error on chunk {chunk_num}: {e}", "ERROR")
                raise


def extract_summary(html_chunk: str) -> str:
    """Extract a brief text summary from an HTML chunk for context passing."""
    clean = re.sub(r'<[^>]+>', ' ', html_chunk)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if len(clean) > 200:
        clean = clean[-200:]
    return clean


def humanize_text(api_key: str, text: str, progress_bar, status_text, log_placeholder) -> str:
    """Main function to humanize the full HTML text."""
    st.session_state.logs = []
    client = Groq(api_key=api_key)

    word_count = count_words(text)
    add_log(f"Starting humanization: {word_count} words")
    add_log("Model: Llama 3.3 70B (Groq)")
    add_log(f"Input format: {'HTML detected' if '<' in text else 'plain text'}")
    log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

    chunks = build_chunks(text, max_words=600)
    if len(chunks) == 1:
        add_log("Text fits in single pass")
    else:
        add_log(f"Split into {len(chunks)} chunks by H2 sections")
        for i, c in enumerate(chunks):
            add_log(f"  Chunk {i+1}: {count_words(c)} words")
    log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

    results = []
    total = len(chunks)
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        chunk_num = i + 1

        # Wait 65s between chunks to reset Groq TPM counter
        if i > 0:
            add_log(f"Waiting 65s for rate limit reset...")
            log_placeholder.markdown(format_logs(), unsafe_allow_html=True)
            status_text.markdown(f"Waiting 65s before chunk **{chunk_num}/{total}**...")
            time.sleep(65)

        status_text.markdown(f"Rewriting chunk **{chunk_num}/{total}**...")
        progress_bar.progress(i / total)
        log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

        context = extract_summary(results[-1]) if results else ""
        result = humanize_chunk(client, chunk, chunk_num, total, context, log_placeholder)
        results.append(result)

        log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

    elapsed = time.time() - start_time
    final_text = "\n".join(results)
    final_words = count_words(final_text)

    add_log(f"Completed in {elapsed:.1f}s")
    add_log(f"Input: {word_count} words -> Output: {final_words} words")

    st.session_state.stats = {
        "input_words": word_count,
        "output_words": final_words,
        "chunks": total,
        "time": round(elapsed, 1),
    }

    progress_bar.progress(1.0)
    status_text.markdown("**Done!**")
    log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

    return final_text


def format_logs() -> str:
    if not st.session_state.logs:
        return '<div class="log-box">Waiting...</div>'
    lines = "\n".join(st.session_state.logs)
    return f'<div class="log-box">{lines}</div>'


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input(
        "Groq API Key",
        value=DEFAULT_API_KEY,
        type="password",
        help="Pre-configured. Change if needed.",
    )
    st.caption("**Free** — Groq + Llama 3.3 70B")
    st.divider()

    st.markdown("**How it works**")
    st.markdown(
        "1. Paste HTML article (AI-generated)\n"
        "2. HTML is split into chunks by H2 sections\n"
        "3. Llama 3.3 70B rewrites text content only\n"
        "4. HTML structure, metadata, images preserved"
    )
    st.divider()
    st.markdown("**What gets rewritten**")
    st.markdown("Paragraphs, headings, list items")
    st.markdown("**What stays untouched**")
    st.markdown(
        "HTML tags/styles, comments, photo credits, "
        "metadata, numbers, proper nouns, tables, "
        "images, quotes, code blocks, navigation"
    )
    st.divider()
    st.markdown("**Limits**")
    st.markdown("Up to **10,000 words** per request")

    if st.session_state.stats:
        st.divider()
        st.markdown("**Last run stats**")
        s = st.session_state.stats
        st.markdown(
            f"- Input: {s['input_words']} words\n"
            f"- Output: {s['output_words']} words\n"
            f"- Chunks: {s['chunks']}\n"
            f"- Time: {s['time']}s"
        )

# ── Main UI ────────────────────────────────────────────────────────────────
st.title("AI Humanizer")
st.markdown("Paste AI-generated HTML articles. Only text content gets rewritten — HTML structure, images, metadata stay intact.")

col_input, col_output = st.columns(2, gap="medium")

with col_input:
    st.markdown("### Input (HTML)")
    input_text = st.text_area(
        "Paste AI-generated HTML here",
        height=500,
        placeholder="Paste your AI-generated HTML article here (up to 10,000 words)...",
        label_visibility="collapsed",
    )

    word_count = count_words(input_text) if input_text.strip() else 0
    wcol1, wcol2 = st.columns([1, 1])
    with wcol1:
        if word_count > 10000:
            st.error(f"{word_count:,} words (max 10,000)")
        else:
            st.caption(f"{word_count:,} words")

    with wcol2:
        can_run = bool(input_text.strip() and api_key.strip() and word_count <= 10000)
        run_btn = st.button("Humanize", type="primary", disabled=not can_run, use_container_width=True)
        if not api_key.strip():
            st.caption("API key missing — check sidebar.")

with col_output:
    st.markdown("### Output (HTML)")
    if st.session_state.result_text:
        st.text_area(
            "Humanized HTML",
            value=st.session_state.result_text,
            height=500,
            label_visibility="collapsed",
        )
        out_words = count_words(st.session_state.result_text)
        st.caption(f"{out_words:,} words")

        dl_filename = extract_filename_from_html(st.session_state.result_text)
        st.download_button(
            label=f"Download: {dl_filename}",
            data=st.session_state.result_text,
            file_name=dl_filename,
            mime="text/html",
            use_container_width=True,
        )
    else:
        st.text_area(
            "Humanized HTML",
            value="",
            height=500,
            placeholder="Humanized HTML will appear here...",
            disabled=True,
            label_visibility="collapsed",
        )

# ── Preview ────────────────────────────────────────────────────────────────
if st.session_state.result_text:
    with st.expander("Preview rendered HTML"):
        st.components.v1.html(st.session_state.result_text, height=800, scrolling=True)

# ── Logs ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Logs")
log_placeholder = st.empty()
log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

if run_btn:
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    try:
        result = humanize_text(api_key, input_text, progress_bar, status_text, log_placeholder)
        st.session_state.result_text = result
        st.rerun()
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower() or "401" in error_msg:
            st.error("Invalid API key. Check your Groq API key at console.groq.com")
        elif "rate" in error_msg.lower() or "429" in error_msg:
            st.error("Rate limited. Wait a moment and try again.")
        else:
            st.error(f"Error: {e}")
        add_log(f"Error: {e}", "ERROR")
        log_placeholder.markdown(format_logs(), unsafe_allow_html=True)
