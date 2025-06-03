"""
XMLC-LOTUS: Bookmark Data Enrichment with Ontology Labels

This script implements a pipeline for enriching bookmark data (JSONL) using a custom ontology (YAML)
and the LOTUS framework. The goal is to prepare data for eXtreme Multi-Label Classification (XMLC)
and potential Knowledge Graph construction.
"""

import os
import warnings
import sys
import pandas as pd
import numpy as np
import yaml
from bs4 import BeautifulSoup
import html
import json
from datetime import datetime
import re
from tqdm import tqdm

# Check for OpenAI API key in environment
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    # Prompt for API key if not found in environment
    api_key = input("Please enter your OpenAI API key: ")
    os.environ['OPENAI_API_KEY'] = api_key
    print("✅ OpenAI API key set from user input.")
else:
    print("✅ OpenAI API key found in environment variables.")

# Monkey patch numpy.lib.stride_tricks to add broadcast_to if needed
import numpy as np
if not hasattr(np.lib.stride_tricks, 'broadcast_to'):
    print("Adding broadcast_to to numpy.lib.stride_tricks")
    from numpy.lib.stride_tricks import as_strided
    
    def broadcast_to(array, shape, subok=False):
        """Broadcast an array to a new shape."""
        array = np.array(array, copy=False, subok=subok)
        shape = tuple(shape)
        
        if array.shape == shape:
            return array
            
        # Get the broadcast shape
        strides = list(array.strides)
        for i, (old_dim, new_dim) in enumerate(zip(array.shape, shape)):
            if old_dim == 1 and new_dim != 1:
                strides[i] = 0
                
        # Add new dimensions
        for _ in range(len(shape) - len(array.shape)):
            strides.append(0)
            
        return as_strided(array, shape=shape, strides=strides)
        
    np.lib.stride_tricks.broadcast_to = broadcast_to

# Suppress common warnings for cleaner execution logs
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress BeautifulSoup warnings about URLs and filenames
from bs4 import MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Import LOTUS framework
try:
    import lotus
    print("✅ LOTUS framework imported successfully.")
except ImportError as e:
    print(f"❌ Error importing LOTUS: {e}")
    print("Please install LOTUS: pip install lotus-ai")
    sys.exit(1)

# Configure LOTUS Framework
try:
    # Initialize Language Model (LM)
    lm = lotus.models.LM("gpt-4o-mini")  # Balance capability/cost
    print("✅ Language Model initialized.")

    # Initialize Retrieval Model (RM)
    rm = lotus.models.SentenceTransformersRM("intfloat/e5-base-v2")  # Strong sentence embeddings
    print("✅ Retrieval Model initialized.")

    # Initialize Reranker Model (optional)
    reranker = lotus.models.CrossEncoderReranker("mixedbread-ai/mxbai-rerank-large-v1")
    print("✅ Reranker Model initialized.")

    # Try to configure LOTUS with the models
    if hasattr(lotus, 'settings'):
        # Try to set models through settings
        if hasattr(lotus.settings, 'set_lm'):
            lotus.settings.set_lm(lm)
        if hasattr(lotus.settings, 'set_rm'):
            lotus.settings.set_rm(rm)
        if hasattr(lotus.settings, 'set_reranker'):
            lotus.settings.set_reranker(reranker)
        print("✅ LOTUS configured through settings.")
    else:
        # Try to set global variables directly
        if hasattr(lotus, 'lm'):
            lotus.lm = lm
        if hasattr(lotus, 'rm'):
            lotus.rm = rm
        if hasattr(lotus, 'reranker'):
            lotus.reranker = reranker
        print("✅ LOTUS configured through direct assignment.")

except Exception as e:
    print(f"❌ Error configuring LOTUS: {e}")
    print("Please check your API key and ensure LOTUS is properly installed.")
    sys.exit(1)

# Load Bookmark Data
try:
    # Define the path to the JSONL file
    data_path = "Dataset.jsonl"  # Path to the dataset
    
    # Check if file exists
    if os.path.exists(data_path):
        # Read JSONL file into a Pandas DataFrame
        df = pd.read_json(data_path, lines=True)
        
        # Verify successful loading
        print(f"✅ Data loaded successfully from {data_path}")
        print(f"   - Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for expected columns
        expected_columns = ['id', 'url', 'source', 'title', 'content', 'created_at', 'domain', 'metadata']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Warning: Missing expected columns: {missing_columns}")
        else:
            print("✅ All expected columns are present.")
            
    else:
        print(f"❌ Error: File not found at {data_path}")
        print("Please check the file path and ensure the file exists.")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# Clean Text Fields
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

print("Cleaning text fields...")
# Apply cleaning to title and content
df['cleaned_title'] = df['title'].apply(clean_text)
df['cleaned_content'] = df['content'].apply(clean_text)

# Handle Missing Essential Text
# Drop rows if both cleaned_title AND cleaned_content are empty
rows_before = len(df)
df = df[(df['cleaned_title'] != "") | (df['cleaned_content'] != "")]
rows_dropped = rows_before - len(df)
print(f"Dropped {rows_dropped} rows with empty title AND content.")

# Fill remaining NaNs in cleaned text columns with empty string
df['cleaned_title'] = df['cleaned_title'].fillna("")
df['cleaned_content'] = df['cleaned_content'].fillna("")

# Parse Dates
# Convert string dates to datetime objects
df['created_at_dt'] = pd.to_datetime(df['created_at'], errors='coerce')

# Handle Duplicates
# Count duplicates before removal
duplicates_before = df.duplicated(subset=['url']).sum()
print(f"Found {duplicates_before} duplicate URLs.")

# Sort by url then created_at_dt (desc), then drop duplicates keeping the first (most recent)
if duplicates_before > 0:
    df = df.sort_values(['url', 'created_at_dt'], ascending=[True, False])
    df = df.drop_duplicates(subset=['url'], keep='first')
    print(f"Kept the most recent entry for each duplicate URL.")

print("✅ Data cleaning completed successfully.")

# Create Unified Text Input
# Concatenate cleaned_title and cleaned_content into a single field
df['combined_text'] = df.apply(
    lambda row: (row['cleaned_title'] + " " + row['cleaned_content']).strip(),
    axis=1
)

print("Created 'combined_text' column by concatenating title and content.")

# Extract Existing Metadata Labels
def extract_tags(metadata):
    """Extract tags from metadata dictionary, safely handling different structures."""
    if not isinstance(metadata, dict):
        return []
    
    # Try to extract tags from 'raindrop_tags' key (adjust based on actual structure)
    tags = metadata.get('raindrop_tags', [])
    
    # Ensure result is a list
    if not isinstance(tags, list):
        return []
    
    return tags

# Apply extraction to metadata column
df['existing_tags'] = df['metadata'].apply(extract_tags)

# Count existing tags
total_tags = sum(len(tags) for tags in df['existing_tags'])
tagged_items = sum(1 for tags in df['existing_tags'] if len(tags) > 0)
print(f"Found {total_tags} existing tags across {tagged_items} items ({tagged_items/len(df)*100:.1f}% of dataset).")

# Load Ontology
try:
    # Define the path to the ontology file
    ontology_path = "Ontology.yaml"  # Path to the ontology
    
    # Check if file exists
    if os.path.exists(ontology_path):
        # Read YAML file
        with open(ontology_path, 'r') as file:
            ontology = yaml.safe_load(file)
        
        # Verify successful loading
        print(f"✅ Ontology loaded successfully from {ontology_path}")
        
        # Extract labels from ontology
        # The ontology has a complex nested structure
        # Let's extract the top-level categories as our labels
        if isinstance(ontology, dict):
            # Get top-level categories (excluding special categories like ContentType)
            labels = {}
            for key, value in ontology.items():
                if isinstance(value, dict) and 'description' in value:
                    labels[key] = value['description']
            
            print(f"   - Found {len(labels)} top-level labels in the ontology")
            
            # Display a few sample labels
            print("\nSample Labels (first 5):")
            for i, (label, desc) in enumerate(list(labels.items())[:5]):
                print(f"   - {label}: {desc}")
        else:
            print("❌ Error: Ontology is not a dictionary")
            sys.exit(1)
            
    else:
        print(f"❌ Error: Ontology file not found at {ontology_path}")
        print("Please check the file path and ensure the file exists.")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error loading ontology: {e}")
    sys.exit(1)

# Prepare Ontology for XMLC
# Create a list of label names and descriptions
label_names = list(labels.keys())
label_descriptions = [labels[name] for name in label_names]

print(f"Prepared {len(label_names)} labels for XMLC processing.")

# Create a sample of the dataset for testing
sample_size = min(100, len(df))  # Use at most 100 items for testing
df_sample = df.sample(sample_size, random_state=42)
print(f"Created sample dataset with {len(df_sample)} items for testing.")

# Function to assign ontology labels to a text
def assign_labels(text, label_names, label_descriptions, top_k=5):
    """
    Assign ontology labels to a text using LOTUS semantic operations with OpenAI.
    
    Args:
        text: The text to classify
        label_names: List of label names
        label_descriptions: List of label descriptions
        top_k: Number of top labels to return
        
    Returns:
        List of assigned label names
    """
    try:
        # First try to use LOTUS LM for classification
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
            
            # Use LOTUS LM to get the response
            response = lotus.lm(prompt)
            
            # Parse the response to extract label names
            if response and response.lower() != "none":
                # Split by commas and clean up each label
                predicted_labels = [label.strip() for label in response.split(',')]
                
                # Filter to only include valid labels
                valid_labels = [label for label in predicted_labels if label in label_names]
                
                # Limit to top_k
                return valid_labels[:top_k]
            else:
                # Fallback to text matching if LM returns "None"
                return text_matching_fallback(text, label_names, label_descriptions, top_k)
                
        except Exception as lm_error:
            print(f"Warning: LM classification failed: {lm_error}")
            print("Falling back to text matching...")
            return text_matching_fallback(text, label_names, label_descriptions, top_k)
            
    except Exception as e:
        print(f"Error assigning labels: {e}")
        return []

# Fallback method using text matching
def text_matching_fallback(text, label_names, label_descriptions, top_k=5):
    """
    Fallback method using text matching when LM classification fails.
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
                import re
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

# Test the label assignment on a single item
print("\nTesting label assignment on a single item:")
sample_text = df_sample.iloc[0]['combined_text']
print(f"Sample text: {sample_text[:100]}...")  # Show first 100 chars

try:
    assigned_labels = assign_labels(sample_text, label_names, label_descriptions)
    print(f"Assigned labels: {assigned_labels}")
    print("✅ Label assignment test successful.")
except Exception as e:
    print(f"❌ Error in label assignment test: {e}")

# Process the entire sample dataset
print("\nProcessing the sample dataset...")
df_sample['assigned_labels'] = df_sample['combined_text'].apply(
    lambda text: assign_labels(text, label_names, label_descriptions)
)

# Count the number of items with assigned labels
items_with_labels = sum(1 for labels in df_sample['assigned_labels'] if len(labels) > 0)
print(f"Assigned labels to {items_with_labels} out of {len(df_sample)} items ({items_with_labels/len(df_sample)*100:.1f}% of sample).")

# Count the frequency of each label
label_counts = {}
for labels in df_sample['assigned_labels']:
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

# Display the label distribution
print("\nLabel Distribution:")
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   - {label}: {count} items ({count/len(df_sample)*100:.1f}%)")

# Save the results to a CSV file
output_path = "enriched_sample.csv"
df_sample.to_csv(output_path, index=False)
print(f"\n✅ Saved enriched sample dataset to {output_path}")

# Ask if the user wants to process the full dataset
process_full = input("\nDo you want to process the full dataset? (y/n): ")
if process_full.lower() == 'y':
    print("\nProcessing the full dataset...")
    # Use tqdm for a progress bar
    from tqdm import tqdm
    tqdm.pandas()
    
    # Process in batches to avoid memory issues
    batch_size = 500
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    # Create a new DataFrame to store the results
    df_enriched = pd.DataFrame()
    
    for i in tqdm(range(num_batches), desc="Processing batches"):
        # Get the current batch
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        df_batch = df.iloc[start_idx:end_idx].copy()
        
        # Assign labels
        df_batch['assigned_labels'] = df_batch['combined_text'].progress_apply(
            lambda text: assign_labels(text, label_names, label_descriptions)
        )
        
        # Append to the result DataFrame
        df_enriched = pd.concat([df_enriched, df_batch])
    
    # Count the number of items with assigned labels
    items_with_labels = sum(1 for labels in df_enriched['assigned_labels'] if len(labels) > 0)
    print(f"\nAssigned labels to {items_with_labels} out of {len(df_enriched)} items ({items_with_labels/len(df_enriched)*100:.1f}% of full dataset).")
    
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
    full_output_path = "enriched_full_dataset.csv"
    df_enriched.to_csv(full_output_path, index=False)
    print(f"\n✅ Saved enriched full dataset to {full_output_path}")
else:
    print("\nSkipping full dataset processing.")

print("\nScript completed successfully!")