import streamlit as st
import requests
import re
import base64
import time
from streamlit.components.v1 import html

# --- Session State Initialization (MUST be at the top) ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'last_transcript' not in st.session_state:
    st.session_state['last_transcript'] = None
if 'last_summary' not in st.session_state:
    st.session_state['last_summary'] = None
if 'notes' not in st.session_state:
    st.session_state['notes'] = {}
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'dark'
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = ''

# --- Helper Functions ---
def extract_video_id(url_or_id):
    patterns = [
        r'(?:v=|youtu.be/|/v/|/embed/|/shorts/)([\w-]{11})',
        r'^([\w-]{11})$'
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    return url_or_id.strip()

def get_youtube_embed_url(video_id):
    return f"https://www.youtube.com/embed/{video_id}"

def get_text_download_link(text, filename="summary.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Summary</a>'

def get_markdown_download_link(text, filename="summary.md"):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/markdown;base64,{b64}" download="{filename}">📥 Download Markdown</a>'

def get_html_download_link(text, filename="summary.html"):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}">📥 Download HTML</a>'

# --- Custom CSS for enhanced UI ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    html, body, .stApp {
        font-family: 'Montserrat', sans-serif !important;
        background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
        color: #f5f6fa;
    }
    .stApp {
        background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
    }
    .main-card {
        background: rgba(30, 30, 40, 0.85);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        padding: 2.5em 2em 2em 2em;
        margin-bottom: 2em;
        border: 1.5px solid rgba(255,255,255,0.08);
    }
    .section-title {
        font-size: 1.5em;
        font-weight: 700;
        color: #FFD700;
        margin-bottom: 0.5em;
        display: flex;
        align-items: center;
        gap: 0.5em;
    }
    .icon {
        font-size: 1.2em;
        margin-right: 0.3em;
    }
    .summary-section {
        background: rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 1.5em;
        margin-bottom: 1.5em;
        box-shadow: 0 2px 8px 0 rgba(31, 38, 135, 0.10);
    }
    .chunk-summary {
        background: rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        color: #f5f6fa;
    }
    .video-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #FFD700;
        margin-bottom: 0.5em;
    }
    .divider {
        border: none;
        border-top: 2px solid #FFD70033;
        margin: 2em 0 1.5em 0;
    }
    .floating-feedback {
        position: fixed;
        bottom: 32px;
        right: 32px;
        z-index: 9999;
    }
    .floating-feedback button {
        background: linear-gradient(90deg, #ff4b2b 0%, #ff416c 100%);
        color: #fff;
        border: none;
        border-radius: 50px;
        padding: 0.8em 2em;
        font-size: 1.1em;
        font-weight: bold;
        box-shadow: 0 4px 16px 0 rgba(255,75,43,0.15);
        cursor: pointer;
        transition: background 0.2s;
    }
    .floating-feedback button:hover {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
    }
    .stSpinner > div > div {
        border-top-color: #FFD700 !important;
        border-right-color: #FFD700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Modern Header with Logo and Description ---
st.markdown("""
<div style='display: flex; align-items: center; gap: 1em;'>
    <img src='https://img.icons8.com/color/96/youtube-play.png' width='60'/>
    <div>
        <h1 style='margin-bottom:0;'>YouTube Video Summarizer <span style='font-size:0.7em;'>(Gemini 2.5 Flash)</span></h1>
        <div style='font-size:1.1em; color:#FFD700;'>AI-powered, fast, and beautiful summaries for any YouTube video.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("Enter a YouTube video link to get an advanced summary. No API key needed in the UI.")

# --- Sidebar with Collapsible Advanced Settings ---
with st.sidebar.expander('⚙️ Advanced Settings', expanded=True):
    summary_style = st.selectbox(
        "Summary Style",
        ["bullet_points", "paragraph", "detailed"],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    temperature = st.slider("AI Temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Max Tokens per Chunk", 500, 2000, 1000, 100)
    summary_format = st.radio("Summary Format", ["Plain Text", "Markdown", "HTML"])

# --- Theme Toggle with Visual Feedback ---
theme_icon = "🌙" if st.session_state['theme'] == 'dark' else "☀️"
if st.sidebar.button(f"{theme_icon} Toggle Theme"):
    st.session_state['theme'] = 'light' if st.session_state['theme'] == 'dark' else 'dark'
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("Made with Streamlit & Gemini API")

# --- Sidebar History in Expander ---
if st.session_state['history']:
    with st.sidebar.expander('🕑 History', expanded=False):
        for idx, item in enumerate(st.session_state['history'][:5]):
            st.markdown(f"<b>{item.get('video_title', 'Video')}</b>", unsafe_allow_html=True)
            st.markdown(f"<a href='{item.get('video_url', '#')}' target='_blank'>🔗 Open Video</a>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:0.9em;'>Summary: {item.get('summary', '')[:60]}...</span>", unsafe_allow_html=True)
            st.markdown("---", unsafe_allow_html=True)

# Backend health check (auto-refresh)
def check_api_health():
    try:
        health = requests.get("http://localhost:8000/health/", timeout=5)
        return health.status_code == 200
    except Exception:
        return False

api_online = check_api_health()
if api_online:
    st.sidebar.success("API: Online")
else:
    st.sidebar.error("API: Unavailable")

# Clear history button
if st.sidebar.button("Clear History"):
    st.session_state['history'] = []
    st.success("History cleared.")

url = st.text_input("YouTube Video URL or ID", placeholder="Paste YouTube link here...", help="Supports full URL or just the video ID.")

summarize_btn = st.button("🚀 Summarize", disabled=not api_online)

# Always use 'detailed' as the summary style for best results
selected_summary_style = 'detailed'

if summarize_btn:
    summary_error = False
    if not url:
        st.warning("Please provide the YouTube URL or ID.")
    else:
        with st.spinner("Summarizing..."):
            progress = st.progress(0)
            try:
                response = requests.get(
                    "http://localhost:8000/summarize/",
                    params={
                        "url": url,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "summary_style": selected_summary_style
                    },
                    timeout=180
                )
                progress.progress(50)
                if response.status_code == 200:
                    data = response.json()
                    # Check for OpenRouter 401 or error in summary
                    if (
                        '[OpenRouter Error]' in data.get('summary', '') or
                        '[OpenRouter Error]' in data.get('chunk_summaries', [''])[0] or
                        '401 Unauthorized' in data.get('summary', '')
                    ):
                        st.error("OpenRouter API error: Unauthorized. Please check your OpenRouter API key, model permissions, or try a different model/key. See https://openrouter.ai/ for help.")
                        progress.empty()
                        summary_error = True
                    if not summary_error:
                        # Save to history
                        st.session_state['history'].insert(0, data)
                        st.session_state['last_summary'] = data
                        # Save transcript if available
                        st.session_state['last_transcript'] = data.get('transcript', None)
                        progress.progress(100)
                else:
                    try:
                        error = response.json().get('detail') or response.json().get('error', 'Unknown error')
                    except Exception:
                        error = response.text
                    st.error(f"Error: {error}")
                    progress.empty()
            except Exception as e:
                st.error(f"Request failed: {e}")
                progress.empty()

# --- Main Content Tabs ---
if st.session_state['last_summary'] is not None:
    data = st.session_state['last_summary']
    video_id = data.get("video_id", "")
    tabs = st.tabs(["Summary", "Word Cloud", "Notes", "Transcript"])
    with tabs[0]:
        st.markdown(f'<div class="video-title">{data.get("video_title", "Video")}</div>', unsafe_allow_html=True)
        # Video preview
        if video_id:
            st.video(f"https://www.youtube.com/watch?v={video_id}")
        st.markdown(f'<div class="summary-section"><b>Final Summary:</b><br>{data.get("summary", "No summary returned.")}</div>', unsafe_allow_html=True)
        # Actions: Download, Copy
        col1, col2, col3 = st.columns([1,1,1])
        summary_text = data.get("summary", "")
        with col1:
            if summary_format == "Plain Text":
                st.markdown(get_text_download_link(summary_text), unsafe_allow_html=True)
            elif summary_format == "Markdown":
                st.markdown(get_markdown_download_link(summary_text), unsafe_allow_html=True)
            else:
                st.markdown(get_html_download_link(summary_text), unsafe_allow_html=True)
        with col2:
            st.button("📋 Copy Summary", on_click=lambda: st.session_state.update({'copy_summary': summary_text}))
            if st.session_state.get('copy_summary'):
                st.code(st.session_state['copy_summary'], language=None)
                st.session_state['copy_summary'] = None
        with col3:
            st.markdown(f'{data.get("clickable_link", "")}', unsafe_allow_html=True)
        st.markdown(f'<b>Transcript Length:</b> {data.get("transcript_length", 0)} characters')
        st.markdown(f'<b>Processing Time:</b> {data.get("processing_time", 0)} seconds')
        # Transcript viewer (simulate, as backend doesn't return transcript)
        if st.session_state['last_transcript']:
            with st.expander("Show Full Transcript"):
                st.text_area("Transcript", st.session_state['last_transcript'], height=200, key=f"expander_transcript_{video_id}")
                st.download_button("📥 Download Transcript", st.session_state['last_transcript'], file_name="transcript.txt")
                st.button("📋 Copy Transcript", on_click=lambda: st.session_state.update({'copy_transcript': st.session_state['last_transcript']}))
                if st.session_state.get('copy_transcript'):
                    st.code(st.session_state['copy_transcript'], language=None)
                    st.session_state['copy_transcript'] = None
        st.markdown("---")
        st.subheader("Chunk Summaries")
        for i, chunk in enumerate(data.get("chunk_summaries", [])):
            with st.expander(f"Chunk {i+1}"):
                st.markdown(f'<div class="chunk-summary">{chunk}</div>', unsafe_allow_html=True)
    # --- Word Cloud Tab ---
    with tabs[1]:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(data.get("summary", ""))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    # --- Notes Tab ---
    with tabs[2]:
        if 'notes' not in st.session_state:
            st.session_state['notes'] = {}
        note = st.text_area("Notes", st.session_state['notes'].get(video_id, ""), key=f"note_{video_id}", height=100)
        if st.button("💾 Save Note", key=f"save_note_{video_id}"):
            st.session_state['notes'][video_id] = note
            st.success("Note saved!")
        if st.session_state['notes'].get(video_id):
            st.info(f"Your note for this video: {st.session_state['notes'][video_id]}")
    # --- Transcript Tab ---
    with tabs[3]:
        transcript = st.session_state['last_transcript']
        if transcript:
            st.text_area("Transcript", transcript, height=200, key=f"transcript_{video_id}")
        else:
            st.text_area(
                "Transcript",
                "Transcript not available for this video.\n\nPossible reasons:\n- The video may not have captions enabled.\n- The video is too new or restricted.\n- Try another video or check your backend logs.",
                height=200,
                disabled=True,
                key=f"transcript_{video_id}_missing"
            )

# --- Sidebar: About & Help ---
with st.sidebar.expander('ℹ️ About & Help', expanded=False):
    st.markdown('''
    **YouTube Video Summarizer** uses Google Gemini AI to generate detailed summaries and transcripts for any YouTube video. 
    - Paste a YouTube link or ID and click **Summarize**.
    - Use the tabs to view the summary, word cloud, notes, or transcript.
    - Adjust settings in the sidebar for custom results.
    - [GitHub Repo](https://github.com/sohomghosh47)
    ''')

# --- Sidebar: Theme Preview ---
st.sidebar.markdown('---')
st.sidebar.markdown(f"**Theme Preview:** {'🌙 Dark' if st.session_state['theme']=='dark' else '☀️ Light'}")

# --- Sidebar: Clear All Button ---
if st.sidebar.button("🧹 Clear All"):
    st.session_state['history'] = []
    st.session_state['last_summary'] = None
    st.session_state['last_transcript'] = None
    st.session_state['notes'] = {}
    st.success("All app state cleared.")

# --- Footer ---
st.markdown("""
---
<div style='text-align:center; color: #888; font-size: 0.95em;'>
    Made with ❤️ by Sohom & Gemini API | <a href='https://github.com/Sagexd08' target='_blank'>GitHub</a> | <a href='https://streamlit.io/' target='_blank'>Streamlit</a>
</div>
""", unsafe_allow_html=True)

# --- Feedback Form ---
st.markdown("""
---
### 💬 Feedback
If you have suggestions or found a bug, please [open an issue](https://github.com/Sagexd08/youtube-summarizer/issues) or leave a comment below:
""")
st.text_area("Your feedback", "", key="feedback", height=80)
st.button("Submit Feedback", on_click=lambda: st.success("Thank you for your feedback!"))

# --- Floating Feedback Button ---
st.markdown('''
<div class="floating-feedback">
    <button onclick="window.open('https://github.com/Sagexd08/youtube-summarizer/issues', '_blank')">
        💬 Feedback
    </button>
</div>
''', unsafe_allow_html=True)
