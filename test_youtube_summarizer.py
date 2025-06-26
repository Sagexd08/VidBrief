#!/usr/bin/env python3
"""
Test script for YouTube Summarizer functionality
This will help identify and fix issues before running the full notebook
"""

import os
import re
import sys
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import nltk

def test_imports():
    """Test if all required imports work"""
    try:
        import googleapiclient.discovery
        import google.generativeai as genai
        from youtube_transcript_api import YouTubeTranscriptApi
        import nltk
        from nltk.tokenize import sent_tokenize
        import ipywidgets as widgets
        from tqdm import tqdm
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

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

def test_video_id_extraction():
    """Test video ID extraction"""
    test_cases = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ")
    ]
    
    for url, expected in test_cases:
        result = extract_video_id(url)
        if result == expected:
            print(f"✓ Video ID extraction: {url} -> {result}")
        else:
            print(f"✗ Video ID extraction failed: {url} -> {result} (expected {expected})")

def test_transcript_fetch():
    """Test transcript fetching with a known video"""
    test_video_id = "dQw4w9WgXcQ"  # Rick Roll - should have captions
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(test_video_id)
        if transcript:
            print(f"✓ Transcript fetched successfully: {len(transcript)} segments")
            print(f"  Sample: {transcript[0]['text'][:50]}...")
            return transcript
        else:
            print("✗ No transcript returned")
            return None
    except (TranscriptsDisabled, NoTranscriptFound):
        print("✗ No transcript available for test video")
        return None
    except Exception as e:
        print(f"✗ Transcript fetch error: {e}")
        return None

def clean_transcript(transcript):
    """Clean transcript text"""
    if not transcript:
        return ''
    text = ' '.join([seg['text'] for seg in transcript])
    text = re.sub(r'\[.*?\]', '', text)  # Remove [tags]
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def test_nltk_setup():
    """Test NLTK setup and download required data"""
    try:
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        test_text = "This is a test. This is another sentence."
        sentences = sent_tokenize(test_text)
        if len(sentences) == 2:
            print("✓ NLTK sentence tokenization working")
            return True
        else:
            print(f"✗ NLTK tokenization issue: got {len(sentences)} sentences")
            return False
    except Exception as e:
        print(f"✗ NLTK error: {e}")
        return False

def chunk_text(text, max_tokens=1000):
    """Split text into chunks"""
    try:
        from nltk.tokenize import sent_tokenize
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
    except Exception as e:
        print(f"✗ Text chunking error: {e}")
        return []

def test_gemini_api():
    """Test Gemini API connection"""
    # Use the provided API key
    api_key = "AIzaSyBD1MCAu02mn3qv18D3eEhlbRmK2LG3rPU"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Test with a simple prompt
        response = model.generate_content(
            "Summarize this text in one sentence: The quick brown fox jumps over the lazy dog.",
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 50
            }
        )
        
        if response and response.text:
            print(f"✓ Gemini API working: {response.text.strip()}")
            return True
        else:
            print("✗ Gemini API returned empty response")
            return False
            
    except Exception as e:
        print(f"✗ Gemini API error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing YouTube Summarizer Components...")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("Stopping tests due to import failures")
        return
    
    # Test video ID extraction
    test_video_id_extraction()
    
    # Test NLTK setup
    test_nltk_setup()
    
    # Test transcript fetching
    transcript = test_transcript_fetch()
    
    if transcript:
        # Test text processing
        cleaned = clean_transcript(transcript)
        print(f"✓ Text cleaning: {len(cleaned)} characters")
        
        chunks = chunk_text(cleaned)
        print(f"✓ Text chunking: {len(chunks)} chunks")
        
        if chunks:
            print(f"  First chunk preview: {chunks[0][:100]}...")
    
    # Test Gemini API
    test_gemini_api()
    
    print("=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main()