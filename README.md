# VidBrief: Advanced YouTube Video Summarizer

VidBrief is an advanced, user-friendly web application that summarizes YouTube videos using Google Gemini AI. It features a modern Streamlit frontend and a robust FastAPI backend, providing concise, customizable summaries, video previews, word clouds, sentiment analysis, and more—all without exposing your API key to the frontend.

## Features

- 🎬 **YouTube Video Summarization** using Gemini AI
- 📝 **Customizable Summary Style**: bullet points, paragraph, or detailed
- 🎛️ **Adjustable AI Parameters**: temperature and max tokens
- 📺 **Embedded Video Preview**
- 📊 **Summary Sentiment Analysis**
- ☁️ **Word Cloud Visualization**
- 📋 **Copy & Download Summary** (plain text, markdown, HTML)
- 🗒️ **Personal Notes** for each summary
- 🕒 **Session-based History** with timestamps
- 🔗 **Shareable Links** (local demo)
- 🌗 **Theme Toggle** (dark/light)
- 🩺 **Backend Health Check**
- 🚫 **No API Key in Frontend** (secure, .env-based backend)

## Demo

![VidBrief Demo Screenshot](demo_screenshot.png)

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/Sagexd08/VidBrief.git
cd VidBrief
```

### 2. Install Dependencies

It's recommended to use a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a requirements.txt, install manually:

```sh
pip install streamlit fastapi uvicorn google-generativeai youtube-transcript-api nltk requests python-dotenv wordcloud matplotlib
```

### 3. Set Up Your Gemini API Key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the FastAPI Backend

```sh
uvicorn main_fastapi_enhanced:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run the Streamlit Frontend

```sh
streamlit run app_streamlit.py
```

Open the provided URL (usually http://localhost:8501) in your browser.

## Usage

1. Enter a YouTube video URL or ID.
2. Adjust summary style and AI parameters in the sidebar.
3. Click **Summarize**.
4. View the summary, sentiment, word cloud, and video preview.
5. Download, copy, or add notes to your summary.
6. Browse your session history in the sidebar.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Credits

- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Gemini AI](https://ai.google.dev/)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- [WordCloud](https://github.com/amueller/word_cloud)

---

Made with ❤️ by [Sagexd08](https://github.com/Sagexd08) and contributors.
