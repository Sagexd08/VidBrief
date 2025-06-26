# PowerShell script to run FastAPI and Streamlit together
Start-Process powershell -ArgumentList 'uvicorn main_fastapi:app --reload' -WindowStyle Minimized
Start-Process powershell -ArgumentList 'streamlit run app_streamlit.py' -WindowStyle Minimized
Write-Host "Both FastAPI (http://localhost:8000) and Streamlit (http://localhost:8501) are starting..."
