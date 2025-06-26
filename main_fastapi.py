from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import nltk
import re
import time
from datetime import datetime
from typing import Optional, List
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app with metadata
app = FastAPI(
    title="YouTube Video Summarizer API",
    description="Advanced API for summarizing YouTube videos using AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download punkt for NLTK if not already present
nltk.download('punkt', quiet=True)

# Pydantic models
class SummaryRequest(BaseModel):
    url: str
    api_key: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.3
    summary_style: Optional[str] = "bullet_points"  # bullet_points, paragraph, detailed

class SummaryResponse(BaseModel):
    video_id: str
    video_title: str
    video_url: str
    clickable_link: str
    summary: str
    chunk_summaries: List[str]
    transcript_length: int
    transcript: str  # <-- Add transcript field
    processing_time: float
    timestamp: str

class TestResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    endpoints: List[str]

def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL or return as is if already an ID"""
    patterns = [
        r'(?:v=|youtu.be/|/v/|/embed/|/shorts/)([\w-]{11})',
        r'^([\w-]{11})$'
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    return url_or_id.strip()

def get_video_title(video_id: str) -> str:
    """Get video title (simplified version - in production you'd use YouTube API)"""
    try:
        # This is a placeholder - in production, use YouTube Data API
        return f"Video {video_id}"
    except:
        return f"Video {video_id}"

def clean_transcript(transcript: list) -> str:
    """Clean and format transcript text (handles both dict and object types)"""
    if not transcript:
        return ''
    texts = []
    for seg in transcript:
        if isinstance(seg, dict):
            texts.append(seg.get('text', ''))
        else:
            texts.append(getattr(seg, 'text', ''))
    text = ' '.join(texts)
    text = re.sub(r'\[.*?\]', '', text)  # Remove [tags]
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, max_tokens: int = 1000) -> List[str]:
    """Split text into manageable chunks"""
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

def get_summary_prompt(style: str) -> str:
    """Get appropriate prompt based on summary style"""
    prompts = {
        "bullet_points": "Summarize the following transcript segment in 3-5 concise bullet points:",
        "paragraph": "Summarize the following transcript segment in a clear, coherent paragraph:",
        "detailed": "Provide a detailed summary of the following transcript segment, including key points and context:"
    }
    return prompts.get(style, prompts["bullet_points"])

def summarize_chunk(chunk: str, api_key: str, temperature: float = 0.3, style: str = "bullet_points") -> str:
    """Summarize a single text chunk using Gemini 2.5 Flash (Google Generative AI)"""
    try:
        prompt = f"{get_summary_prompt(style)}\n\n{chunk}"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt, generation_config={
            "temperature": temperature,
            "max_output_tokens": 300
        })
        if hasattr(response, 'text'):
            return response.text.strip()
        return str(response).strip()
    except Exception as e:
        return f"[Gemini Error] {str(e)}"

def create_final_summary(chunk_summaries: List[str], api_key: str, temperature: float = 0.3) -> str:
    """Create final cohesive summary from chunk summaries using Gemini 2.5 Flash (Google Generative AI)"""
    try:
        combined_summaries = '\n'.join(chunk_summaries)
        prompt = f"Create a comprehensive 5-7 sentence summary of this video based on these section summaries:\n\n{combined_summaries}"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt, generation_config={
            "temperature": temperature,
            "max_output_tokens": 400
        })
        if hasattr(response, 'text'):
            return response.text.strip()
        return str(response).strip()
    except Exception as e:
        return f"[Gemini Error] {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <title>YouTube Summarizer API</title>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <link rel='preconnect' href='https://fonts.googleapis.com'>
        <link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>
        <link href='https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Fira+Mono&display=swap' rel='stylesheet'>
        <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css'>
        <style>
            body {
                font-family: 'Roboto', Arial, sans-serif;
                margin: 0;
                min-height: 100vh;
                background: linear-gradient(120deg, #f8fafc 0%, #e3e6ed 100%);
                color: #23272f;
            }
            .llm-header {
                background: linear-gradient(90deg, #6a82fb 0%, #fc5c7d 100%);
                padding: 36px 0 24px 0;
                border-bottom-left-radius: 32px;
                border-bottom-right-radius: 32px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
                text-align: center;
            }
            .llm-header h1 {
                font-size: 2.8rem;
                font-weight: 700;
                color: #fff;
                margin: 0 0 0.2em 0;
                letter-spacing: 1px;
            }
            .llm-header .subtitle {
                color: #f8fafc;
                font-size: 1.25rem;
                margin-bottom: 0.5em;
            }
            .llm-header .llm-badge {
                display: inline-block;
                background: #fff;
                color: #6a82fb;
                font-weight: 700;
                border-radius: 20px;
                padding: 6px 18px;
                font-size: 1em;
                margin-top: 10px;
                box-shadow: 0 2px 8px rgba(106,130,251,0.08);
            }
            .llm-main {
                max-width: 900px;
                margin: -40px auto 0 auto;
                background: #fff;
                border-radius: 24px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
                padding: 40px 36px 32px 36px;
                position: relative;
                z-index: 2;
            }
            .llm-cards {
                display: flex;
                flex-wrap: wrap;
                gap: 28px;
                margin-bottom: 32px;
                justify-content: space-between;
            }
            .llm-card {
                flex: 1 1 260px;
                min-width: 260px;
                background: linear-gradient(120deg, #f8fafc 0%, #e3e6ed 100%);
                border-radius: 18px;
                box-shadow: 0 2px 8px rgba(31, 38, 135, 0.06);
                padding: 28px 22px 22px 22px;
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
                border-left: 6px solid #6a82fb;
            }
            .llm-card .icon {
                font-size: 1.7em;
                color: #fc5c7d;
                margin-bottom: 8px;
            }
            .llm-card .title {
                font-weight: 700;
                font-size: 1.1em;
                color: #23272f;
            }
            .llm-card .desc {
                color: #6c757d;
                font-size: 1em;
            }
            .llm-links {
                text-align: center;
                margin-top: 2em;
            }
            .llm-links a {
                color: #6a82fb;
                font-weight: 700;
                text-decoration: none;
                margin: 0 18px;
                font-size: 1.1em;
                transition: color 0.2s;
            }
            .llm-links a:hover {
                color: #fc5c7d;
                text-decoration: underline;
            }
            .llm-footer {
                text-align: center;
                color: #6c757d;
                font-size: 1em;
                margin-top: 3em;
                padding-bottom: 1em;
            }
            .floating-feedback {
                position: fixed;
                bottom: 32px;
                right: 32px;
                z-index: 999;
            }
            .feedback-btn {
                background: linear-gradient(90deg, #fc5c7d 0%, #6a82fb 100%);
                color: #fff;
                font-weight: bold;
                border: none;
                border-radius: 50px;
                padding: 16px 32px;
                font-size: 1.1em;
                box-shadow: 0 4px 16px rgba(106,130,251, 0.18);
                cursor: pointer;
                transition: background 0.2s, box-shadow 0.2s;
            }
            .feedback-btn:hover {
                background: linear-gradient(90deg, #6a82fb 0%, #fc5c7d 100%);
                box-shadow: 0 6px 24px rgba(252,92,125, 0.18);
            }
        </style>
    </head>
    <body>
        <div class="llm-header">
            <h1><i class="fa-brands fa-youtube"></i> YouTube Video Summarizer API</h1>
            <div class="subtitle">Notebook LLM+ inspired UI for summarizing YouTube videos with <span style='color:#ffd700;'>Gemini 2.5 Flash</span>.</div>
            <div class="llm-badge"><i class="fa-solid fa-bolt"></i> FastAPI & Gemini 2.5 Flash</div>
        </div>
        <div class="llm-main">
            <div class="llm-cards">
                <div class="llm-card">
                    <span class="icon"><i class="fa-solid fa-vial"></i></span>
                    <span class="title">GET /test/</span>
                    <span class="desc">Test endpoint to verify API status</span>
                </div>
                <div class="llm-card">
                    <span class="icon"><i class="fa-solid fa-wand-magic-sparkles"></i></span>
                    <span class="title">GET /summarize/</span>
                    <span class="desc">Summarize a YouTube video<br><b>Parameters:</b> url, api_key, max_tokens, temperature, summary_style</span>
                </div>
                <div class="llm-card">
                    <span class="icon"><i class="fa-solid fa-wand-magic-sparkles"></i></span>
                    <span class="title">POST /summarize/</span>
                    <span class="desc">Summarize a YouTube video (JSON body)</span>
                </div>
                <div class="llm-card">
                    <span class="icon"><i class="fa-solid fa-book"></i></span>
                    <span class="title">GET /docs</span>
                    <span class="desc">Interactive API Documentation</span>
                </div>
                <div class="llm-card">
                    <span class="icon"><i class="fa-solid fa-heart-pulse"></i></span>
                    <span class="title">GET /health/</span>
                    <span class="desc">Health check endpoint</span>
                </div>
            </div>
            <div class="llm-links">
                <a href="/docs"><i class="fa-solid fa-book"></i> API Docs</a>
                <a href="/test/"><i class="fa-solid fa-vial"></i> Test API</a>
                <a href="/health/"><i class="fa-solid fa-heart-pulse"></i> Health</a>
            </div>
        </div>
        <div class="floating-feedback">
            <button class="feedback-btn" onclick="window.open('https://forms.gle/2v8Qn6k8Qw6Qw6Qw6', '_blank')">
                <i class="fa-solid fa-comment-dots"></i> Feedback
            </button>
        </div>
        <div class="llm-footer">
            <span>Made with <i class="fa-solid fa-heart" style="color:#fc5c7d;"></i> by <b>Your Team</b> &mdash; Powered by <b>Gemini 2.5 Flash</b></span>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/test/", response_model=TestResponse)
async def test_endpoint():
    """Enhanced test endpoint with comprehensive information"""
    return TestResponse(
        status="success",
        message="YouTube Summarizer API is running perfectly! 🚀",
        timestamp=datetime.now().isoformat(),
        endpoints=[
            "GET /",
            "GET /test/",
            "GET /summarize/",
            "POST /summarize/",
            "GET /docs",
            "GET /redoc"
        ]
    )

@app.get("/summarize/", response_model=SummaryResponse)
async def summarize_youtube_get(
    url: str = Query(..., description="YouTube video URL or ID"),
    api_key: Optional[str] = Query(None, description="Gemini API Key (optional)"),
    max_tokens: int = Query(1000, description="Maximum tokens per chunk"),
    temperature: float = Query(0.3, description="AI temperature (0.0-1.0)"),
    summary_style: str = Query("bullet_points", description="Summary style: bullet_points, paragraph, detailed")
):
    """Summarize YouTube video via GET request"""
    return await process_video_summary(url, api_key, max_tokens, temperature, summary_style)

@app.post("/summarize/", response_model=SummaryResponse)
async def summarize_youtube_post(request: SummaryRequest):
    """Summarize YouTube video via POST request"""
    return await process_video_summary(
        request.url, 
        request.api_key, 
        request.max_tokens, 
        request.temperature, 
        request.summary_style
    )

async def process_video_summary(
    url: str, 
    api_key: Optional[str] = None, 
    max_tokens: int = 1000, 
    temperature: float = 0.3, 
    summary_style: str = "bullet_points"
) -> SummaryResponse:
    """Process video summary with enhanced features"""
    start_time = time.time()
    # Use Gemini API key from environment if not provided
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyAUqtiM6OQJ8cDvH6SJ2nSS5w1ZKcKO8PA"
        if not api_key:
            raise HTTPException(status_code=500, detail="Gemini API key not found in environment or fallback.")

    try:
        # Extract video ID
        video_id = extract_video_id(url)
        if not video_id or len(video_id) != 11:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL or video ID")
        
        # Fetch transcript (prefer Hindi, fallback to default)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
            transcript = None
            try:
                transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                # Try to get Hindi transcript
                try:
                    transcript = transcripts.find_transcript(['hi']).fetch()
                except Exception:
                    # Fallback to manually search for Hindi
                    for t in transcripts:
                        if t.language_code == 'hi':
                            transcript = t.fetch()
                            break
                # Fallback to English or first available
                if not transcript:
                    try:
                        transcript = transcripts.find_transcript(['en']).fetch()
                    except Exception:
                        transcript = transcripts.find_generated_transcript(['en']).fetch() if transcripts._generated_transcripts else None
                # Fallback to first transcript
                if not transcript:
                    transcript = transcripts._manually_created_transcripts[0].fetch() if transcripts._manually_created_transcripts else None
            except Exception:
                # Fallback to old method
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                raise HTTPException(status_code=404, detail="Transcript not available for this video (no Hindi or English transcript found)")
        except (TranscriptsDisabled, NoTranscriptFound):
            raise HTTPException(status_code=404, detail="Transcript not available for this video")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching transcript: {str(e)}")
        
        # Process transcript
        cleaned_text = clean_transcript(transcript)
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="No valid transcript content found")
        
        # Create chunks
        chunks = chunk_text(cleaned_text, max_tokens)
        
        # Generate summaries for each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = summarize_chunk(chunk, api_key, temperature, summary_style)
            chunk_summaries.append(summary)
        # Create final summary
        final_summary = create_final_summary(chunk_summaries, api_key, temperature)
        
        # Generate URLs
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        clickable_link = f'<a href="{video_url}" target="_blank" style="color: #1976d2; text-decoration: none;">🎬 Watch Video</a>'
        
        # Get video title
        video_title = get_video_title(video_id)
        
        processing_time = time.time() - start_time
        
        return SummaryResponse(
            video_id=video_id,
            video_title=video_title,
            video_url=video_url,
            clickable_link=clickable_link,
            summary=final_summary,
            chunk_summaries=chunk_summaries,
            transcript_length=len(cleaned_text),
            transcript=cleaned_text,  # <-- Return the cleaned transcript
            processing_time=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "YouTube Summarizer API",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)