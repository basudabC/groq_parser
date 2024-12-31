import pdfplumber
import json
from typing import Dict, List, Any, Optional
import re
import zipfile
import os
import random
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from time import sleep

load_dotenv()

class GroqAPIError(Exception):
    """Custom exception for Groq API errors"""
    pass

class MultiGroqClient:
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("No API keys provided")
        
        # Filter out empty or invalid format keys
        self.api_keys = [key.strip() for key in api_keys if self._is_valid_key_format(key)]
        if not self.api_keys:
            raise ValueError("No valid API keys found")
            
        self.current_client = None
        self.used_keys = set()
        self.invalid_keys = set()  # Track permanently invalid keys
        self._initialize_client()
        
    def _is_valid_key_format(self, key: str) -> bool:
        """Basic validation of API key format"""
        if not key or not isinstance(key, str):
            return False
        key = key.strip()
        # Basic format check for Groq API keys
        return bool(re.match(r'^gsk_[A-Za-z0-9]{32,}$', key))

    def _validate_api_key(self, key: str) -> bool:
        """Test if an API key is valid by making a minimal API call"""
        try:
            test_client = Groq(api_key=key)
            # Make a minimal API call to verify the key
            test_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            if "invalid_api_key" in str(e).lower() or "401" in str(e):
                return False
            # If it's a different error (e.g., rate limit), assume key might be valid
            return True

    def _initialize_client(self) -> Optional[str]:
        """Initialize or reinitialize the Groq client with a random unused and valid API key"""
        available_keys = [key for key in self.api_keys 
                         if key not in self.used_keys and key not in self.invalid_keys]
        
        if not available_keys:
            if len(self.invalid_keys) == len(self.api_keys):
                raise GroqAPIError("All API keys are invalid")
            # Reset used keys if all valid keys have been tried
            self.used_keys.clear()
            available_keys = [key for key in self.api_keys if key not in self.invalid_keys]

        selected_key = random.choice(available_keys)
        
        # Validate the selected key
        if not self._validate_api_key(selected_key):
            print(f"API key validation failed for key ending in ...{selected_key[-4:]}")
            self.invalid_keys.add(selected_key)
            return self._initialize_client()  # Recursively try another key
            
        self.current_client = Groq(api_key=selected_key)
        self.used_keys.add(selected_key)
        print(f"Successfully initialized client with key ending in ...{selected_key[-4:]}")
        return selected_key

    def create_chat_completion(self, *args, **kwargs) -> Any:
        """Wrapper for chat completion with automatic failover and retry logic"""
        max_retries = len(self.api_keys)
        retries = 0
        
        while retries < max_retries:
            try:
                return self.current_client.chat.completions.create(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                retries += 1
                
                if "invalid_api_key" in error_str or "401" in error_str:
                    # Mark the current key as invalid
                    current_key = self.current_client.api_key
                    self.invalid_keys.add(current_key)
                    print(f"Invalid API key detected (ending in ...{current_key[-4:]})")
                
                if retries < max_retries:
                    print(f"API call failed: {str(e)}. Attempt {retries} of {max_retries}")
                    try:
                        self._initialize_client()
                        # Add a small delay before retry
                        sleep(1)
                    except GroqAPIError as ge:
                        raise ge
                else:
                    raise GroqAPIError("All API keys failed or are invalid") from e

class ResumeProcessor:
    def __init__(self, zip_path: str, output_folder: str, groq_api_keys: List[str]):
        self.zip_path = zip_path
        self.output_folder = output_folder
        try:
            self.client = MultiGroqClient(groq_api_keys)
        except ValueError as e:
            raise ValueError(f"Failed to initialize MultiGroqClient: {str(e)}")

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

    def process_all_resumes(self):
        """Process all PDF files in the zip folder"""
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Extract all PDFs to a temporary folder
            temp_folder = "temp_pdfs"
            os.makedirs(temp_folder, exist_ok=True)

            for file in zip_ref.namelist():
                if file.lower().endswith('.pdf'):
                    zip_ref.extract(file, temp_folder)
                    pdf_path = os.path.join(temp_folder, file)

                    # Process each PDF
                    print(f"Processing: {file}")
                    try:
                        resume_data = self.extract_from_pdf(pdf_path)
                        llm_processed_data = self.process_with_llm(resume_data)

                        # Save to JSON with original filename
                        output_filename = os.path.splitext(file)[0] + '.json'
                        output_path = os.path.join(self.output_folder, output_filename)

                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(llm_processed_data, f, indent=2, ensure_ascii=False)

                        print(f"Saved: {output_filename}")

                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")

                    # Clean up the temporary PDF file
                    os.remove(pdf_path)

            # Remove temporary folder
            os.rmdir(temp_folder)

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and structure from a single PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                sections = []
                current_section = {"heading": "", "content": []}

                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        lines = text.split('\n')

                        for line in lines:
                            clean_line = line.strip()
                            if clean_line:
                                if self._is_likely_header(clean_line):
                                    if current_section["heading"] or current_section["content"]:
                                        sections.append(current_section.copy())
                                    current_section = {
                                        "heading": clean_line,
                                        "content": []
                                    }
                                else:
                                    current_section["content"].append(clean_line)
                                full_text += clean_line + "\n"

                if current_section["heading"] or current_section["content"]:
                    sections.append(current_section)

                contact_info = self._extract_contact_info(full_text)

                return {
                    "raw_text": full_text.strip(),
                    "contact_info": contact_info,
                    "sections": sections
                }

        except Exception as e:
            raise Exception(f"Error extracting text: {str(e)}")

    def _is_likely_header(self, line: str) -> bool:
        """Determine if a line is likely a section header"""
        header_patterns = [
            r'^[A-Z\s&-]+$',
            r'^(EDUCATION|EXPERIENCE|SKILLS|PROJECTS|SUMMARY|WORK|CONTACT|ACHIEVEMENTS|CERTIFICATIONS|LANGUAGES)',
            r'^[A-Z][a-zA-Z\s]+:',
            r'^[A-Z][a-zA-Z\s]{2,30}$'
        ]
        return any(re.match(pattern, line.strip()) for pattern in header_patterns)

    def _extract_contact_info(self, text: str) -> Dict[str, List[str]]:
        """Extract contact information from text"""
        contact_info = {
            "emails": [],
            "phones": [],
            "links": [],
            "location": []
        }

        first_lines = text.split('\n')[:10]
        for line in first_lines:
            emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', line)
            contact_info["emails"].extend(emails)

            phones = re.findall(r'[\+]?[\d\s-]{10,}', line)
            contact_info["phones"].extend([p.strip() for p in phones])

            links = re.findall(r'(?:https?://)?(?:www\.)?[\w\.-]+\.\w+/[\w\.-]+', line)
            contact_info["links"].extend(links)

            locations = re.findall(r'[\w\s]+,\s*[\w\s]+', line)
            contact_info["location"].extend([loc.strip() for loc in locations])

        return contact_info

    def process_with_llm(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process extracted resume data with Groq LLM using multiple API keys"""
        system_prompt = {
            "role": "system",
            "content": """
            You are an expert HR assistant skilled in creating structured reports. Your task is to process the input data and generate a JSON output with essential details. 
            Include the number of years served in each company and role in brackets. If any required column value is missing, leave it blank. 
            Here's an example of the output:
            [
              {
                "Name": "john doe",
                "Emails": "john@gmail.com",
                "Mobile": "1886378566",
                "Present Salary": "Tk.",
                "Expected Salary": "Tk.",
                "Date of Birth": "",
                "Permanent Address": "",
                "Company with Duration": "Bandor Steel Industries Ltd(2yrs)",
                "Job Title with Duration": "Company Legal Adviser(2yrs)",
                "Institution": "",
                "Graduation": "",
                "Total Years of experience": ""
              }
            ]
            """
        }

        resume_text = f"""
        Raw Text:
        {resume_data['raw_text']}

        Contact Information:
        {json.dumps(resume_data['contact_info'], indent=2)}

        Sections:
        {json.dumps(resume_data['sections'], indent=2)}
        """

        try:
            response = self.client.create_chat_completion(
                model="llama3-8b-8192",
                messages=[
                    system_prompt,
                    {"role": "user", "content": resume_text}
                ],
                max_tokens=1024,
                temperature=0.9
            )
            
            try:
                llm_output = json.loads(response.choices[0].message.content)
                return llm_output
            except json.JSONDecodeError:
                return {"raw_llm_output": response.choices[0].message.content}
                
        except GroqAPIError as e:
            return {"error": f"LLM processing failed: {str(e)}"}

def main():
    # Configuration
    zip_path = "/content/all.zip"
    output_folder = "processed_resumes"
    
    # Get multiple API keys from environment variables
    groq_api_keys = []
    
    # Try different environment variable patterns
    # Pattern 1: GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
    for i in range(1, 6):
        key = os.environ.get(f"GROQ_API_KEY_{i}")
        if key:
            groq_api_keys.append(key)
    
    # Pattern 2: GROQ_API_KEY
    default_key = os.environ.get("GROQ_API_KEY")
    if default_key:
        groq_api_keys.append(default_key)
    
    # Pattern 3: Comma-separated keys in GROQ_API_KEYS
    multiple_keys = os.environ.get("GROQ_API_KEYS")
    if multiple_keys:
        groq_api_keys.extend([k.strip() for k in multiple_keys.split(',')])

    if not groq_api_keys:
        raise ValueError("No GROQ API keys found in environment variables")

    try:
        # Process all resumes
        processor = ResumeProcessor(zip_path, output_folder, groq_api_keys)
        processor.process_all_resumes()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()