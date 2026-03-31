import streamlit as st
import anthropic
import time
import logging
import re
from datetime import datetime
from io import StringIO

# ── Logging setup ──────────────────────────────────────────────────────────
log_stream = StringIO()
log_handler = logging.StreamHandler(log_stream)
log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger = logging.getLogger("humanizer")
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Humanizer", page_icon="✍️", layout="wide")

st.markdown("""
<style>
    .stTextArea textarea { font-size: 15px; line-height: 1.6; }
    .log-box {
        background: #1e1e1e; color: #d4d4d4; font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px; padding: 12px; border-radius: 8px; max-height: 300px;
        overflow-y: auto; white-space: pre-wrap; word-wrap: break-word;
    }
    .stat-card {
        background: #f8f9fa; border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #4CAF50; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ("result_text", ""),
    ("logs", []),
    ("processing", False),
    ("stats", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def add_log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] [{level}] {message}")


def count_words(text: str) -> int:
    return len(text.split())


def split_into_chunks(text: str, max_words: int = 1200) -> list[str]:
    """Split text into chunks at paragraph boundaries, respecting max word count."""
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_count = 0

    for para in paragraphs:
        para_words = count_words(para)
        if current_count + para_words > max_words and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [para]
            current_count = para_words
        else:
            current_chunk.append(para)
            current_count += para_words

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


SYSTEM_PROMPT = """You are an expert human writer who rewrites AI-generated text to sound completely natural and human-written. Your goal is to make the text undetectable by AI detection tools (GPTZero, Originality.ai, Turnitin, etc.) while preserving the original meaning and information.

REWRITING RULES:
1. SENTENCE STRUCTURE: Vary sentence lengths dramatically. Mix very short sentences (3-5 words) with longer, complex ones. Real humans don't write uniformly.
2. WORD CHOICE: Replace formal/academic words with simpler everyday alternatives. Avoid AI-typical words: "delve", "crucial", "comprehensive", "facilitate", "leverage", "utilize", "moreover", "furthermore", "it's important to note", "in conclusion", "landscape", "paradigm", "multifaceted", "nuanced", "robust".
3. TRANSITIONS: Use natural, casual connectors: "but", "and", "so", "though", "anyway", "honestly", "the thing is", "look", "here's the deal". Avoid mechanical transitions.
4. IMPERFECTIONS: Add natural human touches — occasional parenthetical asides, rhetorical questions, colloquial expressions. Humans sometimes start sentences with "And" or "But".
5. FLOW: Don't follow a rigid structure. Humans digress slightly, circle back, emphasize unexpectedly. Break the predictable pattern.
6. TONE: Write like a knowledgeable person explaining to a friend, not like a textbook. Be direct and opinionated where appropriate.
7. PARAGRAPHS: Vary paragraph length. Some can be just one sentence. Others can be longer blocks.
8. AVOID: Don't use bullet points or numbered lists unless the original specifically has them. Don't add headers that weren't there.
9. PRESERVE: Keep all factual information, key arguments, and the overall message intact. The meaning must stay the same.
10. LANGUAGE: Respond in the SAME language as the input text. If the input is in Vietnamese, rewrite in Vietnamese. If English, rewrite in English. Etc.

OUTPUT: Return ONLY the rewritten text. No explanations, no meta-comments, no "Here's the rewritten version:" prefix. Just the humanized text."""


def humanize_chunk(client: anthropic.Anthropic, chunk: str, chunk_num: int, total_chunks: int, context_before: str = "") -> str:
    """Rewrite a single chunk using Claude Sonnet."""
    add_log(f"Processing chunk {chunk_num}/{total_chunks} ({count_words(chunk)} words)")

    user_message = ""
    if context_before:
        user_message += f"[CONTEXT - this is what came right before this section, for continuity. Do NOT rewrite this part, only use it for context:]\n{context_before[-500:]}\n\n[TEXT TO REWRITE:]\n"
    user_message += chunk

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=1.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        result = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        add_log(f"Chunk {chunk_num} done: {input_tokens} input tokens, {output_tokens} output tokens")
        return result
    except anthropic.APIError as e:
        add_log(f"API error on chunk {chunk_num}: {e}", "ERROR")
        raise
    except Exception as e:
        add_log(f"Unexpected error on chunk {chunk_num}: {e}", "ERROR")
        raise


def humanize_text(api_key: str, text: str, progress_bar, status_text, log_placeholder) -> str:
    """Main function to humanize the full text."""
    st.session_state.logs = []
    client = anthropic.Anthropic(api_key=api_key)

    word_count = count_words(text)
    add_log(f"Starting humanization: {word_count} words")
    log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

    if word_count <= 1500:
        chunks = [text]
        add_log("Text is short enough for single-pass processing")
    else:
        chunks = split_into_chunks(text, max_words=1200)
        add_log(f"Text split into {len(chunks)} chunks")
    log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

    results = []
    total = len(chunks)
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        status_text.markdown(f"Processing chunk **{chunk_num}/{total}**...")
        progress_bar.progress(i / total)
        log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

        context = results[-1] if results else ""
        result = humanize_chunk(client, chunk, chunk_num, total, context)
        results.append(result)

        log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

    elapsed = time.time() - start_time
    final_text = "\n\n".join(results)
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
    api_key = st.text_input("Anthropic API Key", type="password", help="Get your key at console.anthropic.com")
    st.caption("Uses **Claude Sonnet** for rewriting.")
    st.divider()

    st.markdown("**How it works**")
    st.markdown(
        "1. Paste AI-generated text\n"
        "2. Text is split into chunks if needed\n"
        "3. Each chunk is rewritten by Claude Sonnet\n"
        "4. Results are merged into final output"
    )
    st.divider()
    st.markdown("**Limits**")
    st.markdown("- Up to **10,000 words** per request\n- Supports any language\n- Preserves original meaning")

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
st.markdown("Paste AI-generated text, get human-sounding text that bypasses AI detectors.")

col_input, col_output = st.columns(2, gap="medium")

with col_input:
    st.markdown("### Input")
    input_text = st.text_area(
        "Paste AI-generated text here",
        height=450,
        placeholder="Paste your AI-generated text here (up to 10,000 words)...",
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
            st.caption("Enter API key in sidebar first.")

with col_output:
    st.markdown("### Output")
    if st.session_state.result_text:
        st.text_area(
            "Humanized text",
            value=st.session_state.result_text,
            height=450,
            label_visibility="collapsed",
        )
        out_words = count_words(st.session_state.result_text)
        st.caption(f"{out_words:,} words")
    else:
        st.text_area(
            "Humanized text",
            value="",
            height=450,
            placeholder="Humanized text will appear here...",
            disabled=True,
            label_visibility="collapsed",
        )

# ── Processing ─────────────────────────────────────────────────────────────
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
    except anthropic.AuthenticationError:
        st.error("Invalid API key. Check your Anthropic API key.")
        add_log("Authentication failed - invalid API key", "ERROR")
        log_placeholder.markdown(format_logs(), unsafe_allow_html=True)
    except anthropic.RateLimitError:
        st.error("Rate limited. Wait a moment and try again.")
        add_log("Rate limited by API", "ERROR")
        log_placeholder.markdown(format_logs(), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")
        add_log(f"Fatal error: {e}", "ERROR")
        log_placeholder.markdown(format_logs(), unsafe_allow_html=True)
