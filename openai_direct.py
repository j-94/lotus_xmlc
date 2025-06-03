"""
Direct OpenAI integration for XMLC.
This script uses OpenAI's API directly without LOTUS.
"""

import os
import sys
import pandas as pd
import yaml
from bs4 import BeautifulSoup
import html
import re
import warnings
from tqdm import tqdm
import time
import json
from openai import OpenAI

# Suppress warnings
warnings.filterwarnings('ignore')

# Check for OpenAI API key in environment
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    print("❌ Error: OpenAI API key not found in environment variables.")
    print("Please set your OPENAI_API_KEY environment variable before running this script.")
    print("Example: export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)
else:
    print("✅ OpenAI API key found in environment variables.")
    
# Initialize OpenAI client
client = OpenAI(api_key=api_key)
print("✅ OpenAI client initialized.")

# Function to clean text
def clean_text(text):
    """Clean text by removing HTML tags, decoding HTML entities, and normalizing whitespace."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Normalize whitespace (multiple spaces/newlines -> single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to assign labels using OpenAI
def assign_labels_openai(text, label_names, label_descriptions, top_k=5, max_retries=3):
    """
    Assign ontology labels to a text using OpenAI.
    
    Args:
        text: The text to classify
        label_names: List of label names
        label_descriptions: List of label descriptions
        top_k: Number of top labels to return
        max_retries: Maximum number of retries on API failure
        
    Returns:
        List of assigned label names
    """
    try:
        # Create a prompt for the language model
        prompt = f"""
        You are an expert classifier. Your task is to assign the most relevant labels to the following text.
        
        TEXT:
        {text}
        
        AVAILABLE LABELS:
        """
        
        # Add label descriptions to the prompt
        for i, (name, desc) in enumerate(zip(label_names, label_descriptions)):
            prompt += f"{i+1}. {name}: {desc}\n"
            
        prompt += f"""
        INSTRUCTIONS:
        1. Analyze the text carefully.
        2. Select up to {top_k} labels that best match the content of the text.
        3. Return ONLY the label names in a comma-separated list.
        4. If no labels match, return "None".
        
        SELECTED LABELS:
        """
        
        # Use OpenAI to get the response
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that classifies text into categories."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )
                
                # Extract the response text
                response_text = response.choices[0].message.content.strip()
                
                # Parse the response to extract label names
                if response_text and response_text.lower() != "none":
                    # Split by commas and clean up each label
                    predicted_labels = [label.strip() for label in response_text.split(',')]
                    
                    # Filter to only include valid labels
                    valid_labels = [label for label in predicted_labels if label in label_names]
                    
                    # Limit to top_k
                    return valid_labels[:top_k]
                else:
                    # Fallback to text matching if OpenAI returns "None"
                    return text_matching_fallback(text, label_names, label_descriptions, top_k)
                    
            except Exception as api_error:
                print(f"OpenAI API error (attempt {attempt+1}/{max_retries}): {api_error}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                    continue
                else:
                    # Fallback to text matching after all retries fail
                    print("Falling back to text matching...")
                    return text_matching_fallback(text, label_names, label_descriptions, top_k)
                    
    except Exception as e:
        print(f"Error assigning labels: {e}")
        return []

# Fallback method using text matching
def text_matching_fallback(text, label_names, label_descriptions, top_k=5):
    """
    Fallback method using text matching when OpenAI classification fails.
    """
    try:
        matched_labels = []
        for i, (name, desc) in enumerate(zip(label_names, label_descriptions)):
            # Make sure the description is a string
            if not isinstance(desc, str):
                desc = str(desc)
                
            # More sophisticated text matching
            # 1. Check if any significant term from the description is in the text
            desc_terms = [term.lower() for term in desc.lower().split() if len(term) > 3]
            
            # 2. Check if any significant term from the label name is in the text
            name_terms = []
            # Split the label name by underscores and camel case
            for part in name.replace('_', ' ').split():
                # Split camel case (e.g., "CamelCase" -> "Camel Case")
                camel_case_parts = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', part)
                if camel_case_parts:
                    name_terms.extend([p.lower() for p in camel_case_parts])
                else:
                    name_terms.append(part.lower())
            
            # 3. Combine all terms
            all_terms = desc_terms + name_terms
            
            # 4. Check if any term is in the text
            text_lower = text.lower()
            matching_terms = [term for term in all_terms if term in text_lower]
            
            # 5. If we have matching terms, add the label
            if matching_terms:
                matched_labels.append(name)
                if len(matched_labels) >= top_k:
                    break
                    
        # Return the matched labels
        return matched_labels[:top_k]
            
    except Exception as e:
        print(f"Error in text matching fallback: {e}")
        return []

def main():
    # Load Data
    data_path = "Dataset.jsonl"
    if not os.path.exists(data_path):
        print(f"❌ Error: File not found at {data_path}")
        sys.exit(1)
        
    print(f"Loading data from {data_path}...")
    df = pd.read_json(data_path, lines=True)
    print(f"✅ Loaded {len(df)} items.")
    
    # Clean Data
    print("Cleaning text fields...")
    df['cleaned_title'] = df['title'].apply(clean_text)
    df['cleaned_content'] = df['content'].apply(clean_text)
    
    # Handle Missing Essential Text
    rows_before = len(df)
    df = df[(df['cleaned_title'] != "") | (df['cleaned_content'] != "")]
    rows_dropped = rows_before - len(df)
    print(f"Dropped {rows_dropped} rows with empty title AND content.")
    
    # Fill remaining NaNs in cleaned text columns with empty string
    df['cleaned_title'] = df['cleaned_title'].fillna("")
    df['cleaned_content'] = df['cleaned_content'].fillna("")
    
    # Parse Dates
    df['created_at_dt'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Handle Duplicates
    duplicates_before = df.duplicated(subset=['url']).sum()
    if duplicates_before > 0:
        df = df.sort_values(['url', 'created_at_dt'], ascending=[True, False])
        df = df.drop_duplicates(subset=['url'], keep='first')
        print(f"Removed {duplicates_before} duplicate URLs.")
    
    # Create Unified Text Input
    df['combined_text'] = df.apply(
        lambda row: (row['cleaned_title'] + " " + row['cleaned_content']).strip(),
        axis=1
    )
    
    # Extract Existing Tags
    df['existing_tags'] = df['metadata'].apply(
        lambda x: x.get('raindrop_tags', []) if isinstance(x, dict) else []
    )
    
    # Load Ontology
    ontology_path = "Ontology.yaml"
    if not os.path.exists(ontology_path):
        print(f"❌ Error: Ontology file not found at {ontology_path}")
        sys.exit(1)
        
    print(f"Loading ontology from {ontology_path}...")
    with open(ontology_path, 'r') as file:
        ontology = yaml.safe_load(file)
    
    # Extract labels from ontology
    if isinstance(ontology, dict):
        # Get top-level categories (excluding special categories like ContentType)
        labels = {}
        for key, value in ontology.items():
            if isinstance(value, dict) and 'description' in value:
                labels[key] = value['description']
        
        print(f"✅ Extracted {len(labels)} top-level labels from the ontology.")
    else:
        print("❌ Error: Ontology is not a dictionary")
        sys.exit(1)
    
    # Prepare labels for processing
    label_names = list(labels.keys())
    label_descriptions = [labels[name] for name in label_names]
    
    # Process a sample first
    sample_size = 10  # Process just 10 items as a test
    df_sample = df.head(sample_size)
    
    print(f"\nProcessing a sample of {sample_size} items...")
    df_sample['assigned_labels'] = df_sample['combined_text'].apply(
        lambda text: assign_labels_openai(text, label_names, label_descriptions)
    )
    
    # Count the number of items with assigned labels
    items_with_labels = sum(1 for labels in df_sample['assigned_labels'] if len(labels) > 0)
    print(f"\nAssigned labels to {items_with_labels} out of {len(df_sample)} items ({items_with_labels/len(df_sample)*100:.1f}% of sample).")
    
    # Count the frequency of each label
    label_counts = {}
    for labels in df_sample['assigned_labels']:
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    # Display the label distribution
    print("\nSample Label Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {label}: {count} items ({count/len(df_sample)*100:.1f}%)")
    
    # Save the sample results to a CSV file
    output_path = "enriched_sample_openai_direct.csv"
    df_sample.to_csv(output_path, index=False)
    print(f"\n✅ Saved enriched sample dataset to {output_path}")
    
    # Ask if the user wants to process the full dataset
    process_full = input("\nDo you want to process the full dataset? (y/n): ")
    if process_full.lower() == 'y':
        print("\nProcessing the full dataset...")
        # Process in batches to avoid memory issues
        batch_size = 50  # Smaller batch size for API rate limits
        num_batches = (len(df) + batch_size - 1) // batch_size
        
        # Create a new DataFrame to store the results
        df_enriched = pd.DataFrame()
        
        for i in tqdm(range(num_batches), desc="Processing batches"):
            # Get the current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            df_batch = df.iloc[start_idx:end_idx].copy()
            
            # Assign labels
            df_batch['assigned_labels'] = df_batch['combined_text'].apply(
                lambda text: assign_labels_openai(text, label_names, label_descriptions)
            )
            
            # Append to the result DataFrame
            df_enriched = pd.concat([df_enriched, df_batch])
            
            # Sleep to avoid rate limits
            time.sleep(1)
        
        # Count the number of items with assigned labels
        items_with_labels = sum(1 for labels in df_enriched['assigned_labels'] if len(labels) > 0)
        print(f"\nAssigned labels to {items_with_labels} out of {len(df_enriched)} items ({items_with_labels/len(df_enriched)*100:.1f}% of dataset).")
        
        # Count the frequency of each label
        label_counts = {}
        for labels in df_enriched['assigned_labels']:
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        # Display the label distribution
        print("\nLabel Distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {label}: {count} items ({count/len(df_enriched)*100:.1f}%)")
        
        # Save the results to a CSV file
        output_path = "enriched_dataset_openai_direct.csv"
        df_enriched.to_csv(output_path, index=False)
        print(f"\n✅ Saved enriched dataset to {output_path}")
    else:
        print("\nSkipping full dataset processing.")
    
    print("\nProcessing completed successfully!")

if __name__ == "__main__":
    main()