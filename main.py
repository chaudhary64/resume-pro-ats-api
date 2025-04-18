import os
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import PyPDF2 as pdf
import google.generativeai as genai

import pytesseract
from pdf2image import convert_from_bytes

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(
    title="Resume ATS & Writing Analyzer API",
    description="API to analyze resume PDFs against job descriptions using Gemini",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_gemini_response(resume_text: str, job_description: str) -> str:
    input_prompt = f"""
    Analyze this resume against the job description with strict ATS scoring and detailed writing improvements. 
    Follow this exact JSON structure:

    {{
      "ATS_Analysis": {{
        "Total_Score": "X%",
        "Breakdown": {{
          "Keyword_Match": "X%",
          "Experience_Match": "X%",
          "Skill_Alignment": "X%",
          "Grammar_Score": "X%"
        }},
        "Missing_Keywords": {{
          "Hard_Skills": ["list"],
          "Soft_Skills": ["list"],
          "Critical_Missing": ["top 5"]
        }},
        "Experience_Gaps": {{
          "Years_Short": X,
          "Missing_Roles": ["list"],
          "Industry_Gaps": ["list"]
        }}
      }},
      "Writing_Improvements": {{
        "Total_Errors": X,
        "Errors": [
          {{
            "Original_Text": "exact phrase",
            "Section": "specific section",
            "Line_Number": X,
            "Error_Type": "Grammar|Style|Formatting|Word_Choice",
            "Correction": "exact replacement",
            "Explanation": "technical reason",
            "Severity": "Critical|High|Medium"
          }}
        ],
        "Style_Recommendations": [
          {{
            "Issue": "specific problem",
            "Example": "original text",
            "Improved_Version": "rewritten text",
            "Section": "where to apply"
          }}
        ]
      }},
      "Optimization_Tips": ["prioritized list"]
    }}

    Analysis Requirements:
    1. ATS Scoring (60% weight):
       - Compare skills/experience with JD
       - Calculate keyword match percentage
       - Identify critical missing requirements
    
    2. Writing Analysis (40% weight):
       - Find ALL grammatical errors with exact locations
       - Require exact replacement text
       - Classify error types technically
       - Highlight style inconsistencies
       - Suggest measurable improvements
    
    3. Formatting Checks:
       - Bullet point consistency
       - Tense uniformity
       - Date formats
       - Section ordering

    Job Description: {job_description}
    Resume Text: {resume_text}
    """

    model = genai.GenerativeModel('gemini-2.5-flash', generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.1,
        "max_output_tokens": 4096
    })
    result = model.generate_content(input_prompt)
    return result.text

def extract_text_from_pdf(file: UploadFile) -> str:
    # Try PyPDF2 first (for text-based PDFs)
    file.file.seek(0)
    reader = pdf.PdfReader(file.file)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    if text.strip():
        return text

    # If no text found, try OCR (for image-based PDFs)
    file.file.seek(0)
    images = convert_from_bytes(file.file.read())
    ocr_text = ""
    for image in images:
        ocr_text += pytesseract.image_to_string(image)
    return ocr_text

@app.post("/analyze_resume")
async def analyze_resume(
    file: UploadFile = File(..., description="PDF Resume"),
    job_description: str = Form(..., description="Job Description")
):
    """
    Analyze a resume PDF against a job description.
    Returns ATS and writing analysis as JSON.
    """
    try:
        resume_text = extract_text_from_pdf(file)
        if not resume_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text from PDF or image. Make sure your resume contains readable text."}
            )

        gemini_response = get_gemini_response(resume_text, job_description)
        try:
            response_json = json.loads(gemini_response)
        except json.JSONDecodeError:
            # Return raw response for debugging if JSON parsing fails
            return JSONResponse(
                status_code=500,
                content={"error": "Gemini output is not valid JSON.", "raw_response": gemini_response}
            )

        return JSONResponse(content=response_json)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "Resume Analyzer API is running!"}
