
import tempfile
import zipfile
import tarfile
import os
import json
import sys
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import base64
import io
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

app = FastAPI(title="Data Analyst Agent", description="AI-powered data analysis API using Gemini")

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

client = genai.Client(api_key=GEMINI_API_KEY)

def get_content_type_for_image(filename):
    """Determine content type based on file extension"""
    ext = filename.lower().split('.')[-1]
    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext == "gif":
        return "image/gif"
    elif ext == "webp":
        return "image/webp"
    else:
        return "application/octet-stream"

class FileData:
    """Class to represent uploaded file data"""
    def __init__(self, name, content, content_type, is_image=False, is_text=False):
        self.name = name
        self.content = content
        self.content_type = content_type
        self.is_image = is_image
        self.is_text = is_text

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Data Analyst Agent API is running", "status": "healthy"}

@app.post("/api/")
async def analyze_data(request: Request):
    """Main endpoint for data analysis requests"""
    try:
        form = await request.form()
        
        if "questions.txt" not in form:
            return JSONResponse(
                status_code=400, 
                content={"error": "questions.txt is required"}
            )
        
        questions_content = ""
        processed_files = []
        files_info_response = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process all uploaded files
            for name, file_or_field in form.items():
                if not hasattr(file_or_field, 'filename'):
                    files_info_response[name] = file_or_field
                    continue
                
                file = file_or_field
                content = await file.read()
                
                if name == "questions.txt":
                    questions_content = content.decode('utf-8')
                    files_info_response[name] = {
                        "filename": file.filename, 
                        "content_preview": questions_content[:200]
                    }
                    continue
                
                # Handle compressed files
                is_zip = file.filename.lower().endswith('.zip')
                is_tar = file.filename.lower().endswith(('.tar', '.tar.gz', '.tgz'))
                
                if is_zip or is_tar:
                    files_info_response[name] = {
                        "filename": file.filename, 
                        "extracted_files": []
                    }
                    archive_path = io.BytesIO(content)
                    
                    try:
                        if is_zip:
                            with zipfile.ZipFile(archive_path) as zf:
                                zf.extractall(temp_dir)
                        elif is_tar:
                            with tarfile.open(fileobj=archive_path) as tf:
                                tf.extractall(temp_dir)
                        
                        # Process extracted files
                        for extracted_filename in os.listdir(temp_dir):
                            extracted_filepath = os.path.join(temp_dir, extracted_filename)
                            if os.path.isfile(extracted_filepath):
                                with open(extracted_filepath, 'rb') as f:
                                    extracted_content = f.read()
                                
                                is_image = extracted_filename.lower().endswith(
                                    ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')
                                )
                                content_type = (get_content_type_for_image(extracted_filename) 
                                              if is_image else 'text/plain')
                                
                                processed_files.append(FileData(
                                    name=extracted_filename,
                                    content=extracted_content,
                                    content_type=content_type,
                                    is_image=is_image,
                                    is_text=not is_image
                                ))
                                files_info_response[name]["extracted_files"].append({
                                    "filename": extracted_filename
                                })
                    
                    except (zipfile.BadZipFile, tarfile.ReadError) as e:
                        files_info_response[name]["error"] = f"Could not decompress file: {e}"
                
                else:
                    # Handle regular files
                    is_image = (file.content_type and file.content_type.startswith('image/') or
                              file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')))
                    
                    processed_files.append(FileData(
                        name=file.filename,
                        content=content,
                        content_type=file.content_type or ('image/png' if is_image else 'text/plain'),
                        is_image=is_image,
                        is_text=not is_image
                    ))
                    files_info_response[name] = {"filename": file.filename}
        
        # Generate response using Gemini
        if questions_content:
            response_text = await generate_gemini_response(questions_content, processed_files)
            return JSONResponse(content={
                "files_processed": files_info_response,
                "response": response_text
            })
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "No questions provided"}
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

async def generate_gemini_response(questions_content: str, processed_files: list) -> str:
    """Generate response using Google Gemini API"""
    try:
        # Prepare the content for Gemini
        content_parts = []
        
        # Add the questions
        content_parts.append(f"Here are the questions I need you to answer:\n---\n{questions_content}\n---\n")
        
        # Add file information
        if processed_files:
            content_parts.append("To answer the questions, you have access to the following files:\n")
            
            for p_file in processed_files:
                if p_file.is_image:
                    # For images, we'll describe them
                    content_parts.append(f"- An image file named `{p_file.name}`\n")
                elif p_file.is_text:
                    try:
                        decoded_content = p_file.content.decode('utf-8')
                        # Limit content preview to avoid token limits
                        preview = "\n".join(decoded_content.splitlines()[:50])
                        if len(preview) > 4000:  # Limit to ~4000 characters
                            preview = preview[:4000] + "...\n[Content truncated]"
                        content_parts.append(f"- A text/data file named `{p_file.name}`. Content preview:\n```\n{preview}\n```\n")
                    except UnicodeDecodeError:
                        content_parts.append(f"- A binary file named `{p_file.name}` (content not displayable as text)\n")
        
        # Enhanced system prompt for data analysis
        system_instruction = """You are a world-class data analyst AI with expertise in Python, data science, statistics, and visualization. 

Your task is to analyze the provided data and answer questions with high-quality, executable Python code.

CRITICAL INSTRUCTIONS:
1. Analyze the questions and data files carefully
2. Write robust, production-quality Python code that:
   - Handles data cleaning and preprocessing
   - Performs the requested analysis
   - Creates visualizations when requested
   - Returns results in the exact format specified
3. For plots/images: Save to file and return as base64 data URI
4. For numerical answers: Return exact values as specified
5. Handle errors gracefully
6. Use appropriate libraries (pandas, matplotlib, seaborn, numpy, scipy, etc.)
7. Your response should contain ONLY executable Python code, no explanations

The code should be complete and runnable, handling all aspects of data loading, processing, analysis, and output formatting."""
        
        # Combine all content
        full_content = "".join(content_parts)
        
        # Generate response using Gemini
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": full_content}]
                }
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=8192,
                temperature=0.1,
                top_p=0.8,
                top_k=40
            )
        )
        
        return response.text
    
    except Exception as e:
        error_code = f"""
# Error occurred while processing with Gemini API: {str(e)}
# Returning basic error response
print("Error: Could not process the request with Gemini API")
"""
        return error_code

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )