import streamlit as st
import anthropic
import requests
import json
import zipfile
from io import BytesIO
import urllib.parse
import time
import random

st.set_page_config(
    page_title="Story → Images",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .scene-header { font-weight: 700; font-size: 15px; margin-bottom: 2px; }
    .scene-caption { color: #888; font-size: 13px; margin-bottom: 8px; }
    .placeholder-box {
        background: #f5f5f5;
        border: 2px dashed #ccc;
        border-radius: 10px;
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #aaa;
        font-size: 14px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ──────────────────────────────────────────────────────
for key, default in [
    ("scenes", []),
    ("characters", []),
    ("char_prompt", ""),
    ("genre", ""),
    ("images", {}),
    ("custom_prompts", {}),
    ("analyzed", False),
    ("img_width", 1024),
    ("img_height", 576),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ─────────────────────────────────────────────────────────────────
def analyze_story(story: str, num_images: int, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    system = (
        "You are an expert story analyst and visual director. "
        "Respond ONLY with valid JSON, no markdown, no explanations."
    )
    user = f"""Analyze this story and create exactly {num_images} sequential scenes for image generation.

STORY:
{story}

Return ONLY a JSON object matching this exact structure (no extra keys):
{{
  "main_characters": [
    {{
      "name": "character name",
      "visual_description": "Very detailed physical description: age range, hair color/length/style, eye color, skin tone, face shape, build/height, distinctive features, signature outfit with specific colors"
    }}
  ],
  "character_consistency_prompt": "Compact combined visual description of ALL main characters to embed in every prompt. Example: 'John (30s, short black hair, brown eyes, athletic build, navy jacket), Maria (20s, long red hair, green eyes, floral dress)'",
  "story_genre_style": "e.g. modern urban drama, fantasy adventure, romantic comedy",
  "scenes": [
    {{
      "scene_number": 1,
      "title": "short scene title",
      "description": "1–2 sentences describing what happens",
      "image_prompt": "COMPLETE prompt for image generation. Must include: character consistency details + specific action/emotion + environment details. ALWAYS append exactly: photorealistic DSLR photography, ultra-sharp focus, vibrant saturated colors, bright cheerful natural daylight, golden sunlight fill even indoors, high resolution 8K, professional color grading, cinematic composition"
    }}
  ]
}}

STRICT RULES:
1. Exactly {num_images} scenes, in chronological story order.
2. Every image_prompt MUST embed the character visual descriptions for consistency.
3. Lighting must ALWAYS be bright, vivid, daylight — never dark, dim, moody, or shadowy — even for night scenes, basements, caves, or windowless rooms. Treat it as an always-on bright studio light + golden daylight.
4. All images must look like real photographs shot with a DSLR camera.
5. Return ONLY valid JSON."""

    msg = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = msg.content[0].text.strip()
    # Strip markdown code fences if present
    if "```" in raw:
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
    return json.loads(raw)


def generate_image(prompt: str, width: int, height: int, seed: int | None = None) -> tuple[bytes | None, int]:
    if seed is None:
        seed = random.randint(1, 2_000_000)
    encoded = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?model=flux&width={width}&height={height}&seed={seed}&nologo=true&enhance=true"
    )
    try:
        resp = requests.get(url, timeout=150)
        if resp.status_code == 200 and resp.content:
            return resp.content, seed
    except Exception:
        pass
    return None, seed


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Cài đặt")
    api_key = st.text_input(
        "🔑 Claude API Key",
        type="password",
        help="Lấy tại console.anthropic.com → API Keys",
    )
    st.caption("Dùng cho bước phân tích kịch bản.")

    st.divider()
    st.markdown("**Tạo ảnh**")
    st.success("✅ Pollinations.ai (miễn phí, không cần API key)\nModel: Flux — photorealistic")

    st.divider()
    st.markdown("**Kích thước ảnh**")
    ratio = st.selectbox(
        "Tỉ lệ",
        ["16:9 — Landscape (1024×576)", "4:3 (1024×768)", "1:1 Square (768×768)", "9:16 — Portrait (576×1024)"],
    )
    dim_map = {
        "16:9 — Landscape (1024×576)": (1024, 576),
        "4:3 (1024×768)": (1024, 768),
        "1:1 Square (768×768)": (768, 768),
        "9:16 — Portrait (576×1024)": (576, 1024),
    }
    st.session_state.img_width, st.session_state.img_height = dim_map[ratio]

    st.divider()
    if st.session_state.analyzed:
        st.markdown("**Nhân vật chính**")
        for c in st.session_state.characters:
            with st.expander(c["name"]):
                st.caption(c["visual_description"])


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🎬 Story → Image Generator")
st.markdown("Đưa kịch bản vào, Claude phân tích tự động, rồi tạo loạt ảnh nhất quán theo từng phân cảnh.")

tab_input, tab_scenes, tab_download = st.tabs(["📖 Kịch bản", "🖼️ Phân cảnh & Ảnh", "⬇️ Tải về"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Story input & analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_input:
    story = st.text_area(
        "Dán kịch bản / câu chuyện vào đây",
        height=420,
        placeholder=(
            "Dán toàn bộ nội dung câu chuyện hoặc kịch bản vào đây.\n"
            "Claude sẽ tự động nhận diện nhân vật, chia phân cảnh và tạo prompt ảnh."
        ),
    )

    col_n, col_btn = st.columns([1, 3])
    with col_n:
        num_images = st.number_input("Số ảnh cần tạo", min_value=1, max_value=60, value=10, step=1)

    with col_btn:
        analyze_disabled = not (story.strip() and api_key.strip())
        analyze_btn = st.button(
            "🔍 Phân tích kịch bản bằng Claude",
            type="primary",
            disabled=analyze_disabled,
            use_container_width=True,
        )
        if analyze_disabled and not api_key.strip():
            st.caption("⚠️ Nhập Claude API Key trong sidebar trước.")

    if analyze_btn:
        with st.spinner("Claude đang đọc và phân tích kịch bản…"):
            try:
                data = analyze_story(story.strip(), num_images, api_key.strip())
                st.session_state.scenes = data["scenes"]
                st.session_state.characters = data["main_characters"]
                st.session_state.char_prompt = data.get("character_consistency_prompt", "")
                st.session_state.genre = data.get("story_genre_style", "")
                st.session_state.images = {}
                st.session_state.custom_prompts = {}
                st.session_state.analyzed = True

                st.success(
                    f"✅ Phân tích xong! "
                    f"Tìm thấy **{len(st.session_state.characters)} nhân vật**, "
                    f"**{len(st.session_state.scenes)} phân cảnh** sẵn sàng tạo ảnh."
                )
                if st.session_state.genre:
                    st.info(f"Thể loại / phong cách: **{st.session_state.genre}**")

                with st.expander("👥 Nhân vật nhận diện được"):
                    for c in st.session_state.characters:
                        st.markdown(f"**{c['name']}**: {c['visual_description']}")

                st.info("Chuyển sang tab **🖼️ Phân cảnh & Ảnh** để tạo ảnh.")

            except json.JSONDecodeError:
                st.error("Claude trả về dữ liệu không đúng JSON. Thử lại lần nữa.")
            except anthropic.AuthenticationError:
                st.error("API Key không hợp lệ. Kiểm tra lại trong sidebar.")
            except Exception as e:
                st.error(f"Lỗi: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Scenes & image generation
# ══════════════════════════════════════════════════════════════════════════════
with tab_scenes:
    if not st.session_state.analyzed:
        st.info("👆 Vào tab **📖 Kịch bản** để phân tích kịch bản trước.")
        st.stop()

    n_scenes = len(st.session_state.scenes)
    n_done = len(st.session_state.images)
    w = st.session_state.img_width
    h = st.session_state.img_height

    # Header bar
    hcol1, hcol2, hcol3 = st.columns([3, 1, 1])
    with hcol1:
        st.markdown(f"**{n_scenes} phân cảnh** · Thể loại: *{st.session_state.genre}* · Kích thước: {w}×{h}")
    with hcol2:
        st.metric("Đã tạo", f"{n_done}/{n_scenes}")
    with hcol3:
        gen_all_btn = st.button("🎨 Tạo tất cả ảnh", type="primary", use_container_width=True)

    if gen_all_btn:
        progress = st.progress(0.0)
        status = st.empty()
        for i, scene in enumerate(st.session_state.scenes):
            status.markdown(f"⏳ Đang tạo ảnh **{i + 1}/{n_scenes}**: *{scene['title']}*")
            prompt = st.session_state.custom_prompts.get(i, scene["image_prompt"])
            img_bytes, seed = generate_image(prompt, w, h)
            if img_bytes:
                st.session_state.images[i] = {"data": img_bytes, "seed": seed, "prompt": prompt}
            else:
                st.warning(f"Phân cảnh {i + 1}: tạo ảnh thất bại, thử lại sau.")
            progress.progress((i + 1) / n_scenes)
            time.sleep(0.3)
        status.markdown("✅ Tạo xong!")
        st.rerun()

    st.divider()

    # Grid: 2 columns
    for row_start in range(0, n_scenes, 2):
        cols = st.columns(2, gap="medium")
        for col_idx in range(2):
            si = row_start + col_idx
            if si >= n_scenes:
                break
            scene = st.session_state.scenes[si]
            current_prompt = st.session_state.custom_prompts.get(si, scene["image_prompt"])

            with cols[col_idx]:
                st.markdown(
                    f'<div class="scene-header">Cảnh {scene["scene_number"]}: {scene["title"]}</div>'
                    f'<div class="scene-caption">{scene["description"]}</div>',
                    unsafe_allow_html=True,
                )

                if si in st.session_state.images:
                    img_info = st.session_state.images[si]
                    st.image(img_info["data"], use_container_width=True)

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("🔄 Tạo lại", key=f"regen_{si}", use_container_width=True):
                            with st.spinner("Đang tạo lại…"):
                                img_bytes, seed = generate_image(current_prompt, w, h)
                                if img_bytes:
                                    st.session_state.images[si] = {
                                        "data": img_bytes,
                                        "seed": seed,
                                        "prompt": current_prompt,
                                    }
                                    st.rerun()
                                else:
                                    st.error("Tạo ảnh thất bại, thử lại.")
                    with btn_col2:
                        if st.button("✏️ Sửa prompt", key=f"edit_toggle_{si}", use_container_width=True):
                            key_open = f"prompt_open_{si}"
                            st.session_state[key_open] = not st.session_state.get(key_open, False)
                            st.rerun()

                else:
                    st.markdown(
                        '<div class="placeholder-box">Chưa có ảnh</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"▶️ Tạo ảnh này", key=f"gen_single_{si}", use_container_width=True):
                        with st.spinner("Đang tạo…"):
                            img_bytes, seed = generate_image(current_prompt, w, h)
                            if img_bytes:
                                st.session_state.images[si] = {
                                    "data": img_bytes,
                                    "seed": seed,
                                    "prompt": current_prompt,
                                }
                                st.rerun()
                            else:
                                st.error("Tạo ảnh thất bại, thử lại.")

                # Prompt editor (shown when toggled)
                if st.session_state.get(f"prompt_open_{si}", False):
                    new_prompt = st.text_area(
                        "Chỉnh sửa prompt rồi tạo lại:",
                        value=current_prompt,
                        height=120,
                        key=f"prompt_editor_{si}",
                    )
                    save_col, apply_col = st.columns(2)
                    with save_col:
                        if st.button("💾 Lưu prompt", key=f"save_prompt_{si}", use_container_width=True):
                            st.session_state.custom_prompts[si] = new_prompt
                            st.session_state[f"prompt_open_{si}"] = False
                            st.rerun()
                    with apply_col:
                        if st.button("💾+🔄 Lưu & Tạo lại", key=f"apply_prompt_{si}", use_container_width=True):
                            st.session_state.custom_prompts[si] = new_prompt
                            st.session_state[f"prompt_open_{si}"] = False
                            with st.spinner("Đang tạo lại…"):
                                img_bytes, seed = generate_image(new_prompt, w, h)
                                if img_bytes:
                                    st.session_state.images[si] = {
                                        "data": img_bytes,
                                        "seed": seed,
                                        "prompt": new_prompt,
                                    }
                                    st.rerun()

                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Download
# ══════════════════════════════════════════════════════════════════════════════
with tab_download:
    if not st.session_state.images:
        st.info("Chưa có ảnh nào. Hãy tạo ảnh ở tab **🖼️ Phân cảnh & Ảnh** trước.")
    else:
        n_ready = len(st.session_state.images)
        n_total = len(st.session_state.scenes)
        st.markdown(f"**{n_ready}/{n_total} ảnh** sẵn sàng để tải.")

        # Build ZIP in memory
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, img_info in sorted(st.session_state.images.items()):
                scene = st.session_state.scenes[idx]
                safe = "".join(c if c.isalnum() or c in " _-" else "" for c in scene["title"])[:40].strip()
                fname = f"scene_{scene['scene_number']:03d}_{safe}.jpg"
                zf.writestr(fname, img_info["data"])
        zip_buf.seek(0)

        st.download_button(
            label=f"⬇️ Tải tất cả {n_ready} ảnh (ZIP)",
            data=zip_buf,
            file_name="story_images.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

        st.divider()
        st.markdown("### Xem trước tất cả ảnh")
        for idx, img_info in sorted(st.session_state.images.items()):
            scene = st.session_state.scenes[idx]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(img_info["data"], caption=f"Cảnh {scene['scene_number']}: {scene['title']}", use_container_width=True)
            with c2:
                st.markdown(f"**Cảnh {scene['scene_number']}: {scene['title']}**")
                st.markdown(scene["description"])
