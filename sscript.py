# sscript.py
import re
import json
import os
import pandas as pd
from io import StringIO

def normalize_json_data(data):
    """Normalize JSON data before converting to DataFrame"""
    if isinstance(data, list):
        # Process each record in the list
        normalized_records = []
        for record in data:
            normalized_record = {}
            for key, value in record.items():
                # Convert lists to comma-separated strings
                if isinstance(value, list):
                    normalized_record[key] = ', '.join(map(str, value))
                # Handle None values
                elif value is None:
                    normalized_record[key] = ''
                # Convert everything else to string
                else:
                    normalized_record[key] = str(value)
            normalized_records.append(normalized_record)
        return normalized_records
    return []

def process_resumes(json_folder):
    """
    Process resume JSON files and convert them to a structured DataFrame
    """
    all_data = []
    
    # Define the required columns
    required_columns = [
        'Name', 
        'Emails', 
        'Mobile', 
        'Present Salary',
        'Expected Salary',
        'Date of Birth',
        'Permanent Address',
        'Company with Duration',
        'Job Title with Duration',
        'Institution',
        'Graduation',
        'Total Years of experience'
    ]
    
    # Iterate through JSON files in the folder
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder, filename)
            
            try:
                # Read JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                # Handle different possible JSON structures
                if 'raw_llm_output' in content:
                    raw_output = content['raw_llm_output']
                    
                    # Try extracting JSON from the raw output
                    json_str = None
                    
                    # Try backticks first
                    extracted = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_output, re.DOTALL)
                    if extracted:
                        json_str = extracted.group(1)
                    else:
                        # Try square brackets or curly braces
                        extracted = re.search(r'(\[.*\]|\{.*\})', raw_output, re.DOTALL)
                        if extracted:
                            json_str = extracted.group(1)
                    
                    if json_str:
                        try:
                            # Parse the extracted JSON
                            data = json.loads(json_str)
                            # Normalize the data
                            normalized_data = normalize_json_data(data if isinstance(data, list) else [data])
                            if normalized_data:
                                df = pd.DataFrame(normalized_data)
                                all_data.append(df)
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON in file {filename}")
                else:
                    # Direct JSON structure
                    normalized_data = normalize_json_data([content] if not isinstance(content, list) else content)
                    if normalized_data:
                        df = pd.DataFrame(normalized_data)
                        all_data.append(df)
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
    
    # Combine all DataFrames
    if all_data:
        try:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in final_df.columns:
                    final_df[col] = ''
            
            # Select only the required columns in the specified order
            final_df = final_df[required_columns]
            
            # Final normalization of data types
            for col in final_df.columns:
                final_df[col] = final_df[col].astype(str).replace('nan', '').replace('None', '')
            
            return final_df
        
        except Exception as e:
            print(f"Error creating final DataFrame: {str(e)}")
            return pd.DataFrame(columns=required_columns)
    
    return pd.DataFrame(columns=required_columns)