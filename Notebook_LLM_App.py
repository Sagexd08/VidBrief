# 📺 YouTube Video Summarizer with Gemini LLM
#
# This script guides you through building an interactive YouTube video summarization tool using Google APIs and LLMs.
#
# ---
#
# 📦 1. Setup & Install
#
# We'll install the required libraries and provide instructions for enabling the necessary APIs.

# Install required packages (uncomment if running interactively)
# !pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2 google-generativeai ipywidgets youtube-transcript-api nltk spacy tqdm

#
# ### API Setup Instructions
#
# 1. YouTube Data API
#    - Go to https://console.cloud.google.com/apis/library/youtube.googleapis.com
#    - Enable the YouTube Data API v3 for your project.
#    - Create an API key and save it securely.
#
# 2. Gemini API (Google Generative AI)
#    - Go to https://aistudio.google.com/app/apikey
#    - Generate an API key for Gemini and save it securely.
#
# ---

# ## 🔑 2. Authentication
#
# We'll securely load or prompt for your API keys and build authenticated service clients.

import os
import getpass
import googleapiclient.discovery
import google.generativeai as genai
from IPython.display import display
import ipywidgets as widgets

# Prompt for YouTube Data API key
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    YOUTUBE_API_KEY = getpass.getpass('Enter your YouTube Data API key: ')

# Prompt for Gemini API key
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    GEMINI_API_KEY = getpass.getpass('Enter your Gemini API key: ')

# Build YouTube API client
youtube = googleapiclient.discovery.build(
    'youtube', 'v3', developerKey=YOUTUBE_API_KEY
)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# ## ▶️ 3. Fetch Video Transcript
#
# Enter a YouTube video URL or ID below and click 'Fetch Transcript'. If captions are unavailable, you'll see an error message.

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re

def extract_video_id(url_or_id):
    # Extracts video ID from URL or returns as is
    patterns = [
        r'(?:v=|youtu.be/|/v/|/embed/|/shorts/)([\w-]{11})',
        r'^([\w-]{11})$'
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    return url_or_id.strip()

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        print(f'Error: {e}')
        return None

video_input = widgets.Text(
    value='',
    placeholder='Paste YouTube URL or ID',
    description='Video:',
    style={'description_width': 'initial'},
)
fetch_button = widgets.Button(description='Fetch Transcript', button_style='primary')
transcript_output = widgets.Output()

def on_fetch_clicked(b):
    with transcript_output:
        transcript_output.clear_output()
        vid = extract_video_id(video_input.value)
        transcript = fetch_transcript(vid)
        if transcript:
            print(f'Transcript fetched for video ID: {vid} ({len(transcript)} segments)')
        else:
            print('No transcript available for this video.')
        global last_transcript, last_video_id
        last_transcript = transcript
        last_video_id = vid

fetch_button.on_click(on_fetch_clicked)
display(widgets.HBox([video_input, fetch_button]))
display(transcript_output)

# Initialize globals
last_transcript = None
last_video_id = None

# ## ✂️ 4. Text Preprocessing & Chunking
#
# We'll clean the transcript and split it into manageable chunks for the LLM.

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def clean_transcript(transcript):
    # Remove timestamps, speaker tags, join text
    if not transcript:
        return ''
    text = ' '.join([seg['text'] for seg in transcript])
    text = re.sub(r'\[.*?\]', '', text)  # Remove [tags]
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_tokens=1000):
    # Approximate 1 token ~ 4 chars
    sentences = sent_tokenize(text)
    chunks = []
    current = ''
    for sent in sentences:
        if len(current) + len(sent) < max_tokens * 4:
            current += ' ' + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks

# Example usage after fetching transcript:
# cleaned = clean_transcript(last_transcript)
# chunks = chunk_text(cleaned)

# ## 🤖 5. Summarization via Gemini API
#
# We'll summarize each chunk using Gemini.

from tqdm.notebook import tqdm

def summarize_chunk(chunk):
    prompt = (
        'Summarize the following transcript segment in 3 concise bullet points.\n', chunk
    )
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        '\n'.join(prompt),
        generation_config={
            'temperature': 0.3,
            'max_output_tokens': 256
        }
    )
    return response.text.strip()

def summarize_all_chunks(chunks):
    summaries = []
    for chunk in tqdm(chunks, desc='Summarizing chunks'):
        summary = summarize_chunk(chunk)
        summaries.append(summary)
    return summaries

# ## 📝 6. Final Synthesis
#
# We'll merge the chunk summaries into a final, cohesive summary.

def synthesize_final_summary(chunk_summaries):
    prompt = (
        'You are an assistant. Merge these bullet summaries into a 5-sentence overview of the video.\n', '\n'.join(chunk_summaries)
    )
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        '\n'.join(prompt),
        generation_config={
            'temperature': 0.3,
            'max_output_tokens': 256
        }
    )
    return response.text.strip()

# ## 🎨 7. Interactive Output
#
# Preview the transcript, per-chunk summaries, and the final summary. Optionally, export the summary.

def display_results(transcript, chunks, chunk_summaries, final_summary):
    transcript_preview = widgets.Output()
    with transcript_preview:
        print(clean_transcript(transcript)[:2000] + ('...' if len(clean_transcript(transcript)) > 2000 else ''))
    transcript_accordion = widgets.Accordion(children=[transcript_preview])
    transcript_accordion.set_title(0, 'Transcript Preview')

    chunk_boxes = []
    for i, (chunk, summary) in enumerate(zip(chunks, chunk_summaries)):
        box = widgets.Accordion(children=[widgets.Output()])
        with box.children[0]:
            print('Chunk:', chunk[:300] + ('...' if len(chunk) > 300 else ''))
            print('\nSummary:\n', summary)
        box.set_title(0, f'Chunk {i+1}')
        chunk_boxes.append(box)

    final_box = widgets.Output()
    with final_box:
        print(final_summary)
    final_accordion = widgets.Accordion(children=[final_box])
    final_accordion.set_title(0, 'Final Summary')

    export_button = widgets.Button(description='Export Summary as .txt')
    export_output = widgets.Output()
    def on_export_clicked(b):
        with export_output:
            export_output.clear_output()
            fname = f'summary_{last_video_id}.txt'
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(final_summary)
            print(f'Summary saved as {fname}')
    export_button.on_click(on_export_clicked)

    display(transcript_accordion)
    for box in chunk_boxes:
        display(box)
    display(final_accordion)
    display(widgets.HBox([export_button, export_output]))

# ## 🚀 8. Example Run
#
# Let's demonstrate the summarizer on a popular YouTube video. (Example: Never Gonna Give You Up: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
#
# **To run the full pipeline:**
# 1. Enter the video URL or ID above and fetch the transcript.
# 2. Run the following cell to process, summarize, and display results.

# Example pipeline run
if last_transcript:
    cleaned = clean_transcript(last_transcript)
    chunks = chunk_text(cleaned)
    chunk_summaries = summarize_all_chunks(chunks)
    final_summary = synthesize_final_summary(chunk_summaries)
    display_results(last_transcript, chunks, chunk_summaries, final_summary)
else:
    print('Please fetch a transcript first.')
