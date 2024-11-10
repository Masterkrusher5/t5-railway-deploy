from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
import pdfplumber
import uvicorn
from pathlib import Path

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://summarizer-text.w3spaces.com/", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
summarizer = pipeline("summarization", model="MK-5/t5-small-Abstractive-Summarizer")
class TextInput(BaseModel):
    text: str
    max_length: int = 50

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize")
async def summarize(input_data: TextInput):
    text = input_data.text
    max_length = input_data.max_length

    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty or consist only of symbols.")

    try:
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]["summary_text"]
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    try:
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)