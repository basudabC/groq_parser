import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import shutil
import os
from typing import Dict, Any, List
import zipfile
import json
from f_script_api_v2 import ResumeProcessor, GroqAPIError
from sscript import process_resumes

def get_groq_api_keys() -> List[str]:
    """Collect all available GROQ API keys from environment variables"""
    api_keys = []
    
    # Pattern 1: GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
    for i in range(1, 6):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            api_keys.append(key)
    
    # Pattern 2: GROQ_API_KEY
    default_key = os.getenv("GROQ_API_KEY")
    if default_key:
        api_keys.append(default_key)
    
    # Pattern 3: Comma-separated keys in GROQ_API_KEYS
    multiple_keys = os.getenv("GROQ_API_KEYS")
    if multiple_keys:
        api_keys.extend([k.strip() for k in multiple_keys.split(',')])
    
    return list(set(api_keys))  # Remove duplicates

def normalize_data(value):
    """Normalize data types for DataFrame"""
    if isinstance(value, list):
        return ', '.join(map(str, value))
    elif pd.isna(value) or value is None:
        return ''
    return str(value)

def init_db():
    """Initialize SQLite database with resume table"""
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            emails TEXT UNIQUE,
            mobile TEXT UNIQUE,
            present_salary TEXT,
            expected_salary TEXT,
            date_of_birth TEXT,
            permanent_address TEXT,
            company_with_duration TEXT,
            job_title_with_duration TEXT,
            institution TEXT,
            graduation TEXT,
            total_years_of_experience TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def process_and_normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process and normalize DataFrame before database insertion"""
    # Convert all columns to string type and handle lists
    for column in df.columns:
        df[column] = df[column].apply(normalize_data)
    
    # Ensure all required columns exist
    required_columns = [
        'Name', 'Emails', 'Mobile', 'Present Salary', 'Expected Salary',
        'Date of Birth', 'Permanent Address', 'Company with Duration',
        'Job Title with Duration', 'Institution', 'Graduation',
        'Total Years of experience'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    return df[required_columns]

def insert_to_db(df: pd.DataFrame) -> tuple[int, int]:
    """Insert DataFrame to SQLite database, preventing duplicates"""
    conn = sqlite3.connect('resumes.db')
    cursor = conn.cursor()
    
    df = process_and_normalize_df(df)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    total_records = len(df)
    inserted_records = 0
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    insert_sql = '''
    INSERT INTO resumes (
        name, emails, mobile, present_salary, expected_salary,
        date_of_birth, permanent_address, company_with_duration,
        job_title_with_duration, institution, graduation,
        total_years_of_experience, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    
    for _, row in df.iterrows():
        try:
            values = [
                row['name'], row['emails'], row['mobile'],
                row['present_salary'], row['expected_salary'],
                row['date_of_birth'], row['permanent_address'],
                row['company_with_duration'], row['job_title_with_duration'],
                row['institution'], row['graduation'],
                row['total_years_of_experience'], current_time
            ]
            cursor.execute(insert_sql, values)
            inserted_records += 1
        except sqlite3.IntegrityError:
            continue
    
    conn.commit()
    conn.close()
    return inserted_records, total_records

def search_db(filters: Dict[str, Any]) -> pd.DataFrame:
    """Search database with given filters"""
    conn = sqlite3.connect('resumes.db')
    
    query = "SELECT * FROM resumes WHERE 1=1"
    params = []
    
    if filters.get('created_at'):
        query += " AND DATE(created_at) = ?"
        params.append(filters['created_at'])
    
    if filters.get('graduation'):
        query += " AND graduation LIKE ?"
        params.append(f"%{filters['graduation']}%")
    
    if filters.get('experience'):
        query += " AND total_years_of_experience LIKE ?"
        params.append(f"%{filters['experience']}%")
    
    if filters.get('mobile'):
        query += " AND mobile LIKE ?"
        params.append(f"%{filters['mobile']}%")
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def clean_processed_folder():
    """Clean the processed_resumes folder"""
    if os.path.exists("processed_resumes"):
        shutil.rmtree("processed_resumes")
        os.makedirs("processed_resumes")

def main():
    st.title("Resume Processor")
    
    # Initialize session state for error tracking
    if 'processing_errors' not in st.session_state:
        st.session_state.processing_errors = []
    
    clean_processed_folder()
    init_db()
    
    # Get API keys
    api_keys = get_groq_api_keys()
    if not api_keys:
        st.error("No GROQ API keys found in environment variables. Please add at least one API key.")
        return
    
    # Display number of available API keys (optional)
    st.sidebar.info(f"Number of API keys available: {len(api_keys)}")
    
    # File upload
    uploaded_file = st.file_uploader("Upload ZIP file containing resumes", type=['zip'])
    
    if uploaded_file:
        # Save the uploaded file
        with open("uploaded.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Initialize processor with multiple API keys
            processor = ResumeProcessor(
                "uploaded.zip",
                "processed_resumes",
                api_keys
            )
            
            with st.spinner("Processing resumes..."):
                try:
                    processor.process_all_resumes()
                    result_df = process_resumes("processed_resumes")
                    
                    if not result_df.empty:
                        st.subheader("Preview Processed Data")
                        st.dataframe(result_df)
                        
                        if st.button("Insert to Database"):
                            inserted, total = insert_to_db(result_df)
                            st.success(f"Inserted {inserted} out of {total} records. {total - inserted} duplicates skipped.")
                            clean_processed_folder()
                            
                            if os.path.exists("uploaded.zip"):
                                os.remove("uploaded.zip")
                    else:
                        st.warning("No valid data found in the processed resumes.")
                
                except GroqAPIError as e:
                    st.error(f"API Error: {str(e)}")
                    st.session_state.processing_errors.append(str(e))
                
                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")
                    st.session_state.processing_errors.append(str(e))
        
        except Exception as e:
            st.error(f"Initialization Error: {str(e)}")
            st.session_state.processing_errors.append(str(e))
        
        finally:
            if os.path.exists("uploaded.zip"):
                os.remove("uploaded.zip")
    
    # Display error log in sidebar if there are errors
    # if st.session_state.processing_errors:
    #     with st.sidebar.expander("Error Log"):
    #         for error in st.session_state.processing_errors:
    #             st.write(error)
    #         if st.button("Clear Error Log"):
    #             st.session_state.processing_errors = []
    
    # Search section
    st.subheader("Search Database")
    col1, col2 = st.columns(2)
    
    with col1:
        search_date = st.date_input("Created At")
        search_graduation = st.text_input("Graduation")
    
    with col2:
        search_experience = st.text_input("Experience")
        search_mobile = st.text_input("Mobile")
    
    if st.button("Search"):
        filters = {
            'created_at': search_date.strftime('%Y-%m-%d') if search_date else None,
            'graduation': search_graduation if search_graduation else None,
            'experience': search_experience if search_experience else None,
            'mobile': search_mobile if search_mobile else None
        }
        
        filters = {k: v for k, v in filters.items() if v is not None}
        
        if filters:
            results = search_db(filters)
            st.subheader("Search Results")
            st.dataframe(results)
        else:
            st.warning("Please enter at least one search criterion")

if __name__ == "__main__":
    main()