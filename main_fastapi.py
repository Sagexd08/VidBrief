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
    """Summarize a single text chunk using Gemini AI"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"{get_summary_prompt(style)}\n\n{chunk}"
        
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': 300
            }
        )
        return response.text.strip() if response and response.text else "Summary not available"
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def create_final_summary(chunk_summaries: List[str], api_key: str, temperature: float = 0.3) -> str:
    """Create final cohesive summary from chunk summaries"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        combined_summaries = '\n'.join(chunk_summaries)
        prompt = f"Create a comprehensive 5-7 sentence summary of this video based on these section summaries:\n\n{combined_summaries}"
        
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': 400
            }
        )
        return response.text.strip() if response and response.text else "Final summary not available"
    except Exception as e:
        return f"Error generating final summary: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Summarizer API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; }
            h1 { color: #fff; text-align: center; }
            .endpoint { background: rgba(255,255,255,0.2); padding: 15px; margin: 10px 0; border-radius: 8px; }
            .method { color: #4CAF50; font-weight: bold; }
            a { color: #FFD700; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 YouTube Video Summarizer API</h1>
            <p>Advanced API for summarizing YouTube videos using AI technology.</p>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /test/</h3>
                <p>Test endpoint to verify API status</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /summarize/</h3>
                <p>Summarize a YouTube video</p>
                <p><strong>Parameters:</strong> url, api_key, max_tokens, temperature, summary_style</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /summarize/</h3>
                <p>Summarize a YouTube video (JSON body)</p>
            </div>
            
            <p><a href="/docs">📚 Interactive API Documentation</a></p>
            <p><a href="/test/">🧪 Test API Status</a></p>
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
    
    # Use API key from environment if not provided
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Gemini API key not found in environment.")
    
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