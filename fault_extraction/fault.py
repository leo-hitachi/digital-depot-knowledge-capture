import io
import base64
import os
from openai import OpenAI
from openai import AzureOpenAI
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os


os.system('clear')
print(os.getcwd())
load_dotenv(dotenv_path=f"{os.getcwd()}/.env")
# api_key = os.getenv("OPENAI_API_KEY")
# print("api key",api_key)
# exit(0)
transcript_file = '../data/IMG_0381.tsv'


# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
#     api_version="2025-01-01-preview",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )

client = OpenAI()

DEFAULT_MODEL = "gpt-4o"



json_format = {          
          "detected_anomalies":[{"timestamp":"1450", "description":"description of defect"},
                                {"timestamp":"1620", "description":"description of defect"}]
        }


system_prompt = f"""You are a rolling stock maintenance technician. Your job is to report any detected anomalies."""

def openai_check(model,text):
    print(f"Model: {model}")
    print("======================")    

    prompt = f"""Analyze this transcript. 
    {text}
    Report on any issues. Take note as there are many keywords which sound like an issue, however they could be negated within the sentence. Respond using the format below. Be accurate with the timestamps. The timestamp should be the start timestamp of the line that reports the defect. Do not return any other commentary.
    {json.dumps(json_format)}
    """

    openai_messages = [          
                    {'role': 'system', 
                        'content':[
                            {
                                'type':'text',
                                'text':system_prompt
                            }
                        ]                    
                    },                                               
                    {'role': 'user', 
                        'content':[
                            {
                                'type':'text',
                                'text':prompt
                            }
                        ]                    
                    }                
                ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=0.0,
            max_tokens=2000,
            frequency_penalty=0.0,
            stream=True
        )

        return_string = ""
        for chunk in response:
            try:
                return_string += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end='', flush=True)
            except Exception as e:
                print("")

        # Clean up the response to remove markdown and extract pure JSON
        # Remove ```json and ``` markers and any extra whitespace
        cleaned_response = return_string.replace('```json', '').replace('```', '').strip()
        
        # Parse the JSON response
        try:
            result = json.loads(cleaned_response)                                    
            
            return result
        except json.JSONDecodeError as e:
            print(f"\nFailed to parse JSON response: {e}")
            print(f"Raw response: {cleaned_response}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None

    print("\n\n")

def generate_variations(description: str, n: int = 20) -> list:
    """
    Generate variations of an anomaly description using GPT-4o.
    
    Args:
        description (str): The anomaly description to vary.
        n (int): Number of variations to generate.

    Returns:
        list: A list of generated description variations.
    """
    prompt = (
        f"Generate {n} variations of the following anomaly description. "
        f"Use words a british technician might use. Some are more verbose than others. Also consider English might not be their first language. They might not reference the component by full name, e.g. FASG Mesh might be referenced as just mesh. Mesh might be referenced as grill or grid, etc. Do not replace words if you are not sure whether they represent the same object. Do not include quotes or markdown.:\n\n"
        f"'{description}'"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,  # higher temp = more creative variations
    )

    # Extract text from the response
    variations = response.choices[0].message.content.strip().split("\n")

    # Clean up (remove numbering/bullets if present)
    cleaned = [v.lstrip("0123456789. -") for v in variations if v.strip()]

    return cleaned[:n]

def find_index_for_timestamp(df, timestamp):
    """
    Given a DataFrame with 'start' and 'end' columns,
    return the index of the row where the timestamp falls into the interval.
    """
    result = df[(df['start'] <= timestamp) & (df['end'] > timestamp)]
    if not result.empty:
        return result.index[0]
    else:
        return None  # Or raise an error if preferred

def extract_stem(tsv_path):
    """
    Loads a TSV file into a pandas DataFrame and extracts the filename stem.
    
    Args:
        tsv_path (str or Path): Path to the .tsv file
    
    Returns:
        tuple: (DataFrame, stem) where `stem` is the filename without extension
    """
    path = Path(tsv_path)    
    stem = path.stem
    return stem

def find_files_starting_with(number, folder, digits=4, extension=".jpg"):
    """
    Find all files in the folder that start with the given number (zero-padded).
    
    Args:
        number (int): The number to match at the start of the filename.
        folder (str or Path): The directory to search in.
        digits (int): Total number of digits in the filename (default is 4 for '0001', '0023', etc.)
        extension (str): File extension to filter by (default: '.tsv')
    
    Returns:
        List[Path]: Matching file paths
    """
    folder = Path(folder)
    prefix = str(number).zfill(digits)  # Pad number with zeros
    matching_files = sorted([
        f for f in folder.glob(f"{prefix}*{extension}")
        if f.is_file()
    ])
    return matching_files

folder_name = extract_stem(transcript_file)
print(f"Filename stem: {folder_name}")

# Load the TSV file into a DataFrame
df = pd.read_csv(transcript_file, sep='\t')

# Display the DataFrame
print(df)

transcript = '\n'.join(f"{row['start']} {row['text']}" for _, row in df.iterrows())
print(transcript)
# transcript = ""
# with open(transcript_file, "r") as file:
#     transcript = file.read()
#     print(transcript)


defects = openai_check(DEFAULT_MODEL,transcript)

print(defects)
detected_anomalies = defects["detected_anomalies"]
print(detected_anomalies)
print("\n\n\n")
count = 1
print(f"Anomalies extracted from transcript {transcript_file} ")
for anomaly in detected_anomalies:
    timestamp = anomaly["timestamp"]
    description = anomaly["description"]
    index = find_index_for_timestamp(df, int(timestamp))
    # print(index)
    filenames = find_files_starting_with(index, f'../data/{folder_name}')
    # print(filenames)    

    variations = generate_variations(description)
    print(f"Anomaly {count}")
    print("====================")
    print(f"Description: {description}")
    print(f"Variations:")
    for variation in variations:
        print(variation)
    print(f"Line: {index}")
    print(f"Timestamp: {timestamp}")
    print(f"Images: ")
    for filename in filenames:
        print(filename)
    print("\n\n")
    count +=1