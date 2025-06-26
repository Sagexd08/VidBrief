#!/usr/bin/env python3
"""
Complete YouTube Summarizer Demo
This demonstrates the full functionality before running the notebook
"""

import os
import re
import sys
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Configure APIs
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def extract_video_id(url_or_id):
    """Extract video ID from URL or return as is"""
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
    """Fetch transcript for a video"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        print(f'Error: {e}')
        return None

def clean_transcript(transcript):
    """Clean transcript text"""
    if not transcript:
        return ''
    text = ' '.join([seg['text'] for seg in transcript])
    text = re.sub(r'\[.*?\]', '', text)  # Remove [tags]
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_tokens=1000):
    """Split text into chunks"""
    sentences = sent_tokenize(text)
    chunks = []
    current = ''
    for sent in sentences:
        if len(current) + len(sent) < max_tokens * 4:  # Approximate 1 token ~ 4 chars
            current += ' ' + sent
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks

def summarize_chunk(chunk):
    """Summarize a single chunk"""
    prompt = (
        'Summarize the following transcript segment in 3 concise bullet points.\\n\\n' +
        chunk
    )
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        prompt,
        generation_config={
            'temperature': 0.3,
            'max_output_tokens': 256
        }
    )
    return response.text.strip()

def summarize_all_chunks(chunks):
    """Summarize all chunks"""
    summaries = []
    for chunk in tqdm(chunks, desc='Summarizing chunks'):
        summary = summarize_chunk(chunk)
        summaries.append(summary)
    return summaries

def synthesize_final_summary(chunk_summaries):
    """Create final summary from chunk summaries"""
    prompt = (
        'You are an assistant. Merge these bullet summaries into a 5-sentence overview of the video.\\n\\n' +
        '\\n'.join(chunk_summaries)
    )
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        prompt,
        generation_config={
            'temperature': 0.3,
            'max_output_tokens': 256
        }
    )
    return response.text.strip()

def main():
    """Main demo function"""
    print("🎬 YouTube Video Summarizer Demo")
    print("=" * 50)
    
    # Setup NLTK
    nltk.download('punkt', quiet=True)
    
    # Demo with Rick Roll video (has captions)
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    video_id = extract_video_id(video_url)
    
    print(f"📺 Processing video: {video_id}")
    
    # Fetch transcript
    print("📝 Fetching transcript...")
    transcript = fetch_transcript(video_id)
    
    if not transcript:
        print("❌ No transcript available for this video")
        return
    
    print(f"✅ Transcript fetched: {len(transcript)} segments")
    
    # Clean and chunk
    print("🧹 Cleaning and chunking text...")
    cleaned = clean_transcript(transcript)
    chunks = chunk_text(cleaned)
    
    print(f"📄 Text cleaned: {len(cleaned)} characters")
    print(f"📦 Created {len(chunks)} chunks")
    
    # Show preview
    print("\\n📖 Transcript Preview:")
    print("-" * 30)
    print(cleaned[:500] + "..." if len(cleaned) > 500 else cleaned)
    
    # Summarize chunks
    print("\\n🤖 Summarizing chunks...")
    chunk_summaries = summarize_all_chunks(chunks)
    
    print("\\n📋 Chunk Summaries:")
    print("-" * 30)
    for i, summary in enumerate(chunk_summaries, 1):
        print(f"Chunk {i}:")
        print(summary)
        print()
    
    # Final summary
    print("🎯 Creating final summary...")
    final_summary = synthesize_final_summary(chunk_summaries)
    
    print("\\n🎬 Final Video Summary:")
    print("=" * 50)
    print(final_summary)
    print("=" * 50)
    
    # Save to file
    filename = f"summary_{video_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"YouTube Video Summary\\n")
        f.write(f"Video ID: {video_id}\\n")
        f.write(f"URL: {video_url}\\n\\n")
        f.write(f"Final Summary:\\n{final_summary}\\n\\n")
        f.write("Chunk Summaries:\\n")
        for i, summary in enumerate(chunk_summaries, 1):
            f.write(f"\\nChunk {i}:\\n{summary}\\n")
    
    print(f"\\n💾 Summary saved to: {filename}")
    print("\\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()