import streamlit as st
import anthropic
import time
import re
from datetime import datetime

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


def add_log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] [{level}] {message}")


def count_words(text: str) -> int:
    """Count words in text, stripping HTML tags for accurate count."""
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'&\w+;', ' ', clean)
    return len(clean.split())


def split_html_into_sections(html: str) -> list[str]:
    """Split HTML into sections at <h2> boundaries for chunk processing.
    Each section includes content from one <h2> to the next (or end).
    Leading content before the first <h2> is its own section.
    """
    # Find all h2 positions
    h2_pattern = re.compile(r'(?=<h2[\s>])', re.IGNORECASE)
    positions = [m.start() for m in h2_pattern.finditer(html)]

    if not positions:
        return [html]

    sections = []
    # Content before first h2
    if positions[0] > 0:
        sections.append(html[:positions[0]])

    # Each h2 section
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(html)
        sections.append(html[pos:end])

    return sections


def build_chunks(html: str, max_words: int = 1500) -> list[str]:
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


SYSTEM_PROMPT = """You are an expert human writer who rewrites AI-generated HTML articles to sound completely natural and human-written. Your goal is to make the text undetectable by AI detection tools (GPTZero, Originality.ai, Turnitin, etc.) while preserving the original meaning.

═══════════════════════════════════════════
CRITICAL: WHAT TO REWRITE vs WHAT TO PRESERVE
═══════════════════════════════════════════

✅ REWRITE THESE (make them sound human):
- <p> paragraph text content
- <h1>, <h2>, <h3> heading text content
- <li> list item text content
- Text inside <strong>, <em>, <a> tags (rewrite the text, keep the tags)

🚫 DO NOT TOUCH — PRESERVE 100% EXACTLY:
- ALL HTML tags, attributes, inline styles, classes (keep every tag identical)
- HTML comments (<!-- anything -->)
- Photo credits: "Photo by X on Pexels", "Image by X on Unsplash", etc.
- <img> tags and their alt text, src URLs
- Metadata in comments (URL slug, meta title, meta description, dates, author)
- Numbers: prices ($500, $1,500), percentages (25%, 30-40%), statistics, dates
- Proper nouns: person names, brand names (Aldi, Costco, Wharton, USDA), product names, place names
- Navigation elements, buttons, disclaimers, copyright notices
- Quotes and testimonials
- Code blocks
- <table> content (keep all table data exactly as-is)
- Tip/callout boxes (<div> with border-left styling) — rewrite the advice text naturally but keep the exact formatting, icons, and <strong> label
- Footer/author sections
- Internal link placeholders/comments
- Any URLs or href values

═══════════════════════════════════════════
REWRITING STYLE RULES
═══════════════════════════════════════════

1. SENTENCE STRUCTURE: Vary lengths dramatically. Mix punchy short sentences (3-6 words) with longer flowing ones. Real humans are unpredictable.
2. WORD CHOICE: Use everyday words. NEVER use these AI-typical words/phrases: "delve", "crucial", "comprehensive", "facilitate", "leverage", "utilize", "moreover", "furthermore", "it's important to note", "in conclusion", "landscape", "paradigm", "multifaceted", "nuanced", "robust", "streamline", "harness", "navigate", "realm", "foster", "elevate", "pivotal".
3. TRANSITIONS: Natural connectors only: "but", "and", "so", "though", "anyway", "honestly", "the thing is", "look", "here's the deal", "thing is". Kill mechanical transitions.
4. HUMAN TOUCHES: Parenthetical asides, rhetorical questions, mild opinions, starting sentences with "And" or "But". Humans aren't perfectly polished.
5. FLOW: Break predictable patterns. Digress slightly, circle back, emphasize unexpectedly. Don't follow a rigid template.
6. TONE: Knowledgeable friend, not textbook. Direct. Sometimes opinionated.
7. PARAGRAPHS: Vary length. One-sentence paragraphs are fine. Long ones too.
8. LANGUAGE: Write in the SAME language as the input. Vietnamese stays Vietnamese, English stays English.

═══════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════

Return ONLY the rewritten HTML. No explanations, no markdown fences, no "Here's the rewritten version:" prefix. Just the HTML with humanized text content and all structure preserved exactly."""


def humanize_chunk(client: anthropic.Anthropic, chunk: str, chunk_num: int, total_chunks: int, context_summary: str = "") -> str:
    """Rewrite a single HTML chunk using Claude Sonnet."""
    add_log(f"Processing chunk {chunk_num}/{total_chunks} ({count_words(chunk)} words)")

    user_message = ""
    if context_summary:
        user_message += f"[CONTEXT — the article section right before this one discussed: {context_summary}. Use this for tone/flow continuity only. Do NOT repeat or rewrite it.]\n\n"
    user_message += chunk

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16384,
            temperature=1.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        result = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        add_log(f"Chunk {chunk_num} done — {input_tokens} in / {output_tokens} out tokens")
        return result
    except anthropic.APIError as e:
        add_log(f"API error on chunk {chunk_num}: {e}", "ERROR")
        raise
    except Exception as e:
        add_log(f"Unexpected error on chunk {chunk_num}: {e}", "ERROR")
        raise


def extract_summary(html_chunk: str) -> str:
    """Extract a brief text summary from an HTML chunk for context passing."""
    clean = re.sub(r'<[^>]+>', ' ', html_chunk)
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Take last ~200 chars as context hint
    if len(clean) > 200:
        clean = clean[-200:]
    return clean


def humanize_text(api_key: str, text: str, progress_bar, status_text, log_placeholder) -> str:
    """Main function to humanize the full HTML text."""
    st.session_state.logs = []
    client = anthropic.Anthropic(api_key=api_key)

    word_count = count_words(text)
    add_log(f"Starting humanization: {word_count} words")
    add_log(f"Input format: HTML detected" if "<" in text else "Input format: plain text")
    log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

    if word_count <= 2000:
        chunks = [text]
        add_log("Text fits in single pass")
    else:
        chunks = build_chunks(text, max_words=1500)
        add_log(f"Split into {len(chunks)} chunks by H2 sections")
        for i, c in enumerate(chunks):
            add_log(f"  Chunk {i+1}: {count_words(c)} words")
    log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

    results = []
    total = len(chunks)
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        status_text.markdown(f"Rewriting chunk **{chunk_num}/{total}**...")
        progress_bar.progress(i / total)
        log_placeholder.markdown(format_logs(), unsafe_allow_html=True)

        context = extract_summary(results[-1]) if results else ""
        result = humanize_chunk(client, chunk, chunk_num, total, context)
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
    log_lines = st.session_state.logs
    lines = "\n".join(log_lines)
    return f'<div class="log-box">{lines}</div>'


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Anthropic API Key", type="password", help="Get your key at console.anthropic.com")
    st.caption("Uses **Claude Sonnet** for rewriting.")
    st.divider()

    st.markdown("**How it works**")
    st.markdown(
        "1. Paste HTML article (AI-generated)\n"
        "2. HTML is split into chunks by H2 sections\n"
        "3. Claude Sonnet rewrites text content only\n"
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
            st.caption("Enter API key in sidebar first.")

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

        st.download_button(
            label="Download HTML",
            data=st.session_state.result_text,
            file_name="humanized_article.html",
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

# ── Preview tab ────────────────────────────────────────────────────────────
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
