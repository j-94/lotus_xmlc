#!/usr/bin/env python
# This script requires the lotus conda environment to be active
# Run setup_lotus_env.sh to create the environment and run_lotus_xmlc.sh to execute this script

import os
import warnings
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from html import unescape
import lotus
# Import operators directly from lotus, not from a submodule
from lotus import sem_index, sem_sim_join, sem_extract, sem_cluster_by
# from datetime import datetime # Not explicitly used, can be removed if not needed elsewhere
import numpy as np # numpy might be used indirectly by pandas or lotus, keep for safety
import shutil
import logging
import time

# --- Configuration ---
# LOTUS Models
# Consider adding environment variable fallbacks or checks
LM_MODEL = os.environ.get("LOTUS_LM_MODEL", "gpt-4o-mini") # Example fallback
RM_MODEL = os.environ.get("LOTUS_RM_MODEL", "intfloat/e5-base-v2") # Example fallback
RERANKER_MODEL = os.environ.get("LOTUS_RERANKER_MODEL", "mixedbread-ai/mxbai-rerank-large-v1") # Example fallback

# Input/Output Files
# Use os.path.join for better cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get script's directory
DATA_FILE = os.path.join(BASE_DIR, "../data-bookmark.jsonl") # Assuming data is one level up
ONTOLOGY_FILE = os.path.join(BASE_DIR, "../Ontology.yaml")    # Assuming ontology is one level up
OUTPUT_FILE = os.path.join(BASE_DIR, "enriched_bookmarks_advanced.jsonl")
SUGGESTIONS_FILE = os.path.join(BASE_DIR, "ontology_suggestions.jsonl")

# Workflow Parameters
K_CANDIDATES = 15  # Number of candidates to retrieve via sem_sim_join
RUN_ONTOLOGY_EXPANSION = True # Set to False to skip clustering and gap analysis
NUM_CLUSTERS_EXPANSION = 20 # Number of clusters for expansion analysis
ENFORCE_HIERARCHY = False # Set to True to add parent labels based on child assignment (requires logic below)

# Index Directories (Place within the script's directory or a dedicated 'output'/'temp' dir)
INDEX_BASE_DIR = os.path.join(BASE_DIR, "index_output") # Create a dedicated dir
ONTOLOGY_INDEX_DIR = os.path.join(INDEX_BASE_DIR, "ontology_label_index_advanced")
CLUSTER_INDEX_DIR = os.path.join(INDEX_BASE_DIR, "cluster_index_advanced")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def clean_text(text):
    """Cleans HTML tags and excessive whitespace from text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    try:
        # Added check for empty string after potential conversion
        if not text.strip():
            return ""
        # Use 'lxml' for potentially faster parsing if installed, fallback to 'html.parser'
        try:
            soup = BeautifulSoup(text, "lxml")
        except ImportError:
            soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = unescape(text)
        text = " ".join(text.split()) # Normalize whitespace
    except Exception as e:
        # Log the specific text causing issues if possible (first 100 chars)
        logging.warning(f"Error cleaning text: {e}. Original text start: '{str(text)[:100]}...'")
        return "" # Return empty string on error
    return text

def extract_ontology_labels(ontology, path=""):
    """Recursively extracts labels, descriptions, and paths from the ontology structure."""
    labels = []
    current_path_segment = path.strip('/') # Track current level name for path building

    if isinstance(ontology, dict):
        for key, value in ontology.items():
            # Sanitize key for path element if needed (e.g., replace spaces)
            sanitized_key = key.replace(' ', '_') # Example sanitization
            new_path = f"{path}/{sanitized_key}" if path else sanitized_key

            # Case 1: Standard node with description and possibly children
            if isinstance(value, dict):
                description = value.get('description', 'No description')
                # Add the label itself
                labels.append((key, description, new_path))
                # Recursively process potential children (excluding reserved keys)
                children = {k: v for k, v in value.items() if k not in ['description', 'instances']}
                if children:
                     labels.extend(extract_ontology_labels(children, new_path))

            # Case 2: List of items under a key (e.g., list of tools under 'Libraries')
            elif isinstance(value, list) and key != 'instances': # Avoid specific keywords if needed
                for item in value:
                    if isinstance(item, dict) and 'name' in item:
                        item_name = item['name']
                        item_desc = item.get('description', 'No description available') # Consistent default
                        # Sanitize item name for path
                        sanitized_item_name = item_name.replace(' ', '_')
                        item_path = f"{new_path}/{sanitized_item_name}"
                        labels.append((item_name, item_desc, item_path))
                        # Recursively process potential nested structure within list items
                        children = {k: v for k, v in item.items() if k not in ['name', 'description']}
                        if children:
                            labels.extend(extract_ontology_labels(children, item_path))
                    elif isinstance(item, str): # Handle simple list of strings under a key
                        item_name = item
                        item_desc = f"{item}, part of {key}" # Generic description
                        sanitized_item_name = item_name.replace(' ', '_')
                        item_path = f"{new_path}/{sanitized_item_name}"
                        labels.append((item_name, item_desc, item_path))

            # Case 3: Key has a simple string value (treat as label with minimal description)
            elif isinstance(value, str) and key != 'description':
                 labels.append((key, value, new_path)) # Use value as description

    elif isinstance(ontology, list): # Handle case where root or a value is a list directly
        for index, item in enumerate(ontology):
             # Create a generic path element if needed, e.g., list_item_0, list_item_1
             item_path_segment = f"list_item_{index}"
             new_path = f"{path}/{item_path_segment}" if path else item_path_segment
             labels.extend(extract_ontology_labels(item, new_path)) # Pass current path for hierarchy

    return labels

def escape_for_prompt(text: str) -> str:
    """Escapes characters potentially problematic for LLM prompts or JSON within prompts."""
    if not isinstance(text, str):
        return ""
    # Basic escaping: backslashes and double quotes. Replace newlines with spaces.
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', '')

def get_llm_classification(row, text_col='combined_text', candidate_labels_col='candidate_labels', candidate_descs_col='candidate_descriptions', row_id_col='id'):
    """Uses LLM (sem_extract) to verify candidate labels against bookmark text."""
    row_id = row.get(row_id_col, 'N/A') # Get row ID for logging

    # Check if candidate lists exist and are lists
    if not isinstance(row.get(candidate_labels_col), list) or not row[candidate_labels_col]:
        logging.debug(f"No candidate labels found for ID {row_id}.")
        return []

    bookmark_text = row.get(text_col, "")
    candidates = row[candidate_labels_col]
    descriptions = row.get(candidate_descs_col, []) # Use .get for safety

    # Ensure descriptions is a list
    if not isinstance(descriptions, list):
        logging.warning(f"Descriptions for ID {row_id} is not a list. Treating as empty.")
        descriptions = []

    # Create formatted list of candidates with descriptions
    candidates_formatted_list = []
    if len(descriptions) == len(candidates):
        for i, (label, desc) in enumerate(zip(candidates, descriptions)):
            candidates_formatted_list.append(f"{i+1}. {escape_for_prompt(label)}: {escape_for_prompt(desc)}")
    else:
        logging.warning(f"Mismatch between candidate labels ({len(candidates)}) and descriptions ({len(descriptions)}) for ID {row_id}. Using labels only for formatting.")
        for i, label in enumerate(candidates):
            candidates_formatted_list.append(f"{i+1}. {escape_for_prompt(label)}")

    candidates_formatted = "\n".join(candidates_formatted_list)
    escaped_bookmark_text = escape_for_prompt(bookmark_text)

    # --- Corrected Prompt Definition ---
    # Use the clear, structured prompt for the LLM.
    prompt_text = f'''Given the following bookmark text:
--- TEXT START ---
{escaped_bookmark_text}
--- TEXT END ---

And the following candidate ontology labels with descriptions:
--- CANDIDATES START ---
{candidates_formatted}
--- CANDIDATES END ---

Which of these candidate labels are the MOST relevant and accurate classifications for the text, considering the ontology's purpose? Focus on applicability and core topic, not just keyword similarity. List ONLY the applicable label names (exactly as provided in the candidate list), separated ONLY by commas (e.g., LabelA, LabelC, LabelF). If none are truly applicable, respond with the single word NONE.
'''
    # --- End Corrected Prompt Definition ---

    classification_prompt = {
        "relevant_labels": prompt_text # Key matches expected output structure
    }

    try:
        # Use sem_extract which handles the call to the configured LM
        # Pass text as a list, even if it's just one item
        extracted_data = sem_extract(
             texts=[bookmark_text],
             prompt=classification_prompt
        )

        # sem_extract returns a list of dictionaries, one per input text
        if not extracted_data:
            logging.error(f"sem_extract returned empty result for ID {row_id}.")
            return []

        # Extract the relevant field from the first (and only) dictionary in the list
        raw_output = extracted_data[0].get('relevant_labels')

        if raw_output is None or not isinstance(raw_output, str) or raw_output.strip().upper() == 'NONE':
            logging.debug(f"LLM returned no applicable labels for ID {row_id}.")
            return []

        # Split by comma and strip whitespace from each potential label
        llm_chosen_labels = [label.strip() for label in raw_output.split(',') if label.strip()]

        # Validate chosen labels against original candidates (case-sensitive)
        valid_chosen_labels = [label for label in llm_chosen_labels if label in candidates]

        # Log if LLM potentially hallucinated labels not in the original candidate list
        if len(valid_chosen_labels) != len(llm_chosen_labels):
             hallucinated = set(llm_chosen_labels) - set(valid_chosen_labels)
             logging.warning(f"LLM may have returned labels not in the candidate list for ID {row_id}. Hallucinated/Mismatched: {list(hallucinated)}. Original LLM output: '{raw_output}', Candidates: {candidates}, Filtered: {valid_chosen_labels}")

        return valid_chosen_labels

    except Exception as e:
        logging.error(f"Error during LLM classification for ID {row_id}: {e}", exc_info=True) # Add traceback
        return [] # Return empty list on error

def add_parent_labels(selected_labels, label_to_path_map, all_ontology_labels):
    """
    Adds parent labels based on hierarchical paths defined in label_to_path_map.
    Requires a robust way to map path components back to valid labels.
    """
    if not ENFORCE_HIERARCHY or not selected_labels:
        return list(selected_labels) # Return original list if hierarchy is off or no labels

    final_labels = set(selected_labels)
    ontology_label_set = set(all_ontology_labels) # For quick checking if a parent exists

    labels_to_process = list(selected_labels) # Start with the initially selected labels

    processed_paths = set() # Avoid redundant processing of the same path

    while labels_to_process:
        label = labels_to_process.pop(0)
        path = label_to_path_map.get(label)

        if path and path not in processed_paths:
            processed_paths.add(path)
            parts = path.strip('/').split('/')
            # Iterate through path components to find parents
            current_path = ""
            for i in range(len(parts) - 1): # Iterate up to the second-to-last part (parent)
                part = parts[i]
                current_path = f"{current_path}/{part}" if current_path else part
                # Find the label corresponding to this parent path segment
                # This requires a reverse lookup or assumption that path segment == label name
                parent_label = None
                # Find the label in the master map whose path *ends* with this segment's path
                # This is inefficient; ideally, precompute parent relationships
                for lbl, pth in label_to_path_map.items():
                    if pth == current_path:
                         # Check if this parent label is a valid label in the ontology
                        if lbl in ontology_label_set:
                            parent_label = lbl
                            break # Found the label for this path part

                if parent_label and parent_label not in final_labels:
                    # Check if the parent label is a valid label overall
                     if parent_label in ontology_label_set:
                        final_labels.add(parent_label)
                        # If we add a parent, maybe we need to process its parents too?
                        # To avoid cycles and keep it simple, maybe only add direct parents.
                        # For full hierarchy, add parent_label back to labels_to_process
                        # labels_to_process.append(parent_label) # Uncomment for full recursive parent addition
                    # else:
                         # logging.warning(f"Path part '{part}' mapped to '{parent_label}' which is not in the ontology label set. Path: {path}")

    return sorted(list(final_labels)) # Return sorted list for consistency


# --- Main Workflow ---

def main():
    start_time = time.time()
    logging.info("--- Starting Advanced XMLC Workflow ---")

    # SECTION 0: Initial Setup
    logging.info("Configuring environment...")
    # Ensure API keys are set (Lotus framework might handle this internally based on env vars)
    # Example explicit check:
    if "OPENAI_API_KEY" not in os.environ:
        logging.warning("OPENAI_API_KEY environment variable not set. LLM calls may fail.")
        # raise ValueError("OpenAI API key not found in environment variables.") # Or just warn

    # Configure LOTUS Framework (uses settings, models specified above)
    try:
        lotus.settings.configure(lm=LM_MODEL, rm=RM_MODEL, reranker=RERANKER_MODEL)
        logging.info(f"LOTUS configured with: lm={LM_MODEL}, rm={RM_MODEL}, reranker={RERANKER_MODEL}")
    except Exception as e:
        logging.error(f"Failed to configure Lotus: {e}", exc_info=True)
        return # Cannot proceed if Lotus config fails

    warnings.filterwarnings("ignore", category=UserWarning, module='bs4') # Example: Ignore specific warnings
    os.makedirs(INDEX_BASE_DIR, exist_ok=True) # Ensure base index dir exists

    # SECTION 1: Load Bookmark Data
    logging.info(f"Loading bookmark data from {DATA_FILE}...")
    try:
        df = pd.read_json(DATA_FILE, lines=True)
        if 'id' not in df.columns:
             # Generate a unique ID, ensuring it doesn't clash if merging later
             df['id'] = [f"bookmark_{i}" for i in range(len(df))]
        # Ensure 'id' column is string type for consistency, if needed
        df['id'] = df['id'].astype(str)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Data file not found: {DATA_FILE}")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        return

    initial_rows = len(df)

    # SECTION 2: Clean Data
    logging.info("Cleaning data...")
    df["cleaned_title"] = df["title"].apply(clean_text)
    df["cleaned_content"] = df["content"].apply(clean_text)

    # Drop rows where both title and content are effectively empty after cleaning
    rows_before_empty_drop = len(df)
    df = df[df["cleaned_title"].str.strip().astype(bool) | df["cleaned_content"].str.strip().astype(bool)]
    rows_after_empty_drop = len(df)
    logging.info(f"Dropped {rows_before_empty_drop - rows_after_empty_drop} rows with effectively empty title and content.")

    # Handle potential NaNs introduced by cleaning errors (redundant if clean_text returns "")
    # df["cleaned_title"].fillna("", inplace=True)
    # df["cleaned_content"].fillna("", inplace=True)

    # Deduplicate based on URL, keeping the most recent (if created_at exists)
    if "created_at" in df.columns and "url" in df.columns:
        logging.info("Deduplicating based on URL, keeping most recent...")
        df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
        # Handle potential NaT values if conversion fails
        df.sort_values(by=["url", "created_at_dt"], ascending=[True, False], inplace=True, na_position='last')
        rows_before_dedup = len(df)
        df.drop_duplicates(subset=["url"], keep="first", inplace=True)
        rows_after_dedup = len(df)
        logging.info(f"Removed {rows_before_dedup - rows_after_dedup} duplicate entries based on URL.")
        df = df.drop(columns=["created_at_dt"]) # Clean up temporary column
    elif "url" in df.columns:
        logging.info("Deduplicating based on URL (no date info)...")
        rows_before_dedup = len(df)
        df.drop_duplicates(subset=["url"], keep="first", inplace=True)
        rows_after_dedup = len(df)
        logging.info(f"Removed {rows_before_dedup - rows_after_dedup} duplicate entries based on URL.")
    else:
        logging.warning("Cannot deduplicate by URL as 'url' column is missing.")
        rows_after_dedup = len(df) # Set for consistent logging later


    # SECTION 3: Enhanced Data Preparation (LLM Summary)
    logging.info("Generating content summaries using LLM...")
    try:
        # Define summarization prompt structure for sem_extract
        summary_prompt = {"summary": "Summarize the core topic of this text in 1 concise sentence, focusing on the main subject or technology discussed."}

        # Prepare texts for batch processing (handle empty/NaN content)
        texts_to_summarize = df['cleaned_content'].fillna("").tolist()
        valid_texts_indices = [i for i, text in enumerate(texts_to_summarize) if text.strip()]
        texts_for_llm = [texts_to_summarize[i] for i in valid_texts_indices]

        summaries = [""] * len(df) # Initialize summaries list
        if texts_for_llm:
            # Batch processing with sem_extract if possible
            extracted_summaries = sem_extract(texts_for_llm, prompt=summary_prompt)
            # Map summaries back to the original DataFrame index
            summary_results = [s.get('summary', '') for s in extracted_summaries]
            for i, idx in enumerate(valid_texts_indices):
                summaries[idx] = summary_results[i]
        else:
             logging.warning("No valid content found to generate summaries.")

        df['content_summary'] = summaries
        num_summarized = sum(1 for s in summaries if s)
        logging.info(f"Generated summaries for {num_summarized} rows.")

    except Exception as e:
        logging.error(f"Failed to generate summaries: {e}", exc_info=True)
        df['content_summary'] = "" # Fallback

    # Combine fields for different use cases
    df['summary_and_title'] = df['cleaned_title'].fillna('') + " | Summary: " + df['content_summary'].fillna('')
    df['combined_text'] = df['cleaned_title'].fillna('') + " " + df['cleaned_content'].fillna('')

    # SECTION 4: Load and Prepare Ontology
    logging.info(f"Loading ontology from {ONTOLOGY_FILE}...")
    try:
        with open(ONTOLOGY_FILE, "r", encoding='utf-8') as f: # Specify encoding
            ontology_structure = yaml.safe_load(f)
        logging.info("Ontology loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Ontology file not found: {ONTOLOGY_FILE}")
        return
    except yaml.YAMLError as e:
        logging.error(f"Error parsing ontology YAML: {e}", exc_info=True)
        return
    except Exception as e:
        logging.error(f"Error loading ontology file: {e}", exc_info=True)
        return

    logging.info("Extracting labels from ontology...")
    labels_collection = extract_ontology_labels(ontology_structure)
    if not labels_collection:
        logging.error("No labels extracted from the ontology. XMLC cannot proceed.")
        return # Critical step failed

    # Create DataFrame and ensure uniqueness
    labels_df = pd.DataFrame(labels_collection, columns=["label", "description", "path"])
    labels_df["label_plus_desc"] = labels_df["label"] + ": " + labels_df["description"].fillna("No description")
    # Handle potential duplicate labels - keep first occurrence? Log warning?
    if labels_df['label'].duplicated().any():
        logging.warning(f"Duplicate labels found in ontology. Keeping first occurrence. Duplicates: {labels_df[labels_df['label'].duplicated()]['label'].tolist()}")
        labels_df = labels_df.drop_duplicates(subset=['label'], keep='first').reset_index(drop=True)
    else:
        labels_df = labels_df.reset_index(drop=True) # Ensure consistent index

    logging.info(f"Extracted {len(labels_df)} unique labels from the ontology.")

    # Create map for hierarchical logic
    # Ensure labels used as keys are unique (handled above)
    label_to_path_map = pd.Series(labels_df.path.values, index=labels_df.label).to_dict()
    all_ontology_labels_list = labels_df["label"].tolist() # Get a list of all valid labels

    # Index ontology labels
    logging.info(f"Indexing ontology labels into {ONTOLOGY_INDEX_DIR}...")
    shutil.rmtree(ONTOLOGY_INDEX_DIR, ignore_errors=True) # Clean previous index
    os.makedirs(ONTOLOGY_INDEX_DIR, exist_ok=True)     # Ensure directory exists
    try:
        # Use a field that includes both label and description for better semantic matching
        indexed_labels_df = sem_index(labels_df, field="label_plus_desc", index_dir=ONTOLOGY_INDEX_DIR)
        logging.info("Ontology label indexing completed.")
    except Exception as e:
        logging.error(f"Failed to index ontology labels: {e}", exc_info=True)
        return # Cannot proceed without index

    # SECTION 5: Candidate Label Generation (Semantic Similarity)
    logging.info(f"Generating Top-{K_CANDIDATES} candidate labels using semantic similarity...")
    try:
        # Use the combined summary and title for potentially better signal
        candidate_matches_df = sem_sim_join(
            df,
            indexed_labels_df, # Use the indexed dataframe from sem_index
            left_field="summary_and_title",
            right_field="label_plus_desc", # Match field used for indexing
            k=K_CANDIDATES,
            reranker=RERANKER_MODEL, # Use the configured reranker
            index_dir=ONTOLOGY_INDEX_DIR # Point to the index directory
        )
        logging.info(f"Semantic similarity join completed. Found {len(candidate_matches_df)} potential matches overall.")

        # Check if join result is empty
        if candidate_matches_df.empty:
            logging.warning("Semantic similarity join returned no matches.")
            # Initialize empty candidate columns
            df['candidate_labels'] = [[] for _ in range(len(df))]
            df['candidate_descriptions'] = [[] for _ in range(len(df))]
            df['candidate_scores'] = [[] for _ in range(len(df))]
        else:
            # Aggregate candidates per bookmark
            # Ensure correct column names from sem_sim_join (usually 'id', 'label', 'description', 'path', 'label_plus_desc', '_score')
            # Make sure the 'id' column from the left dataframe (df) is preserved in the join result. Lotus usually does this.
            # Check required columns exist in candidate_matches_df
            required_cols = ['id', 'label', 'description', '_score']
            if not all(col in candidate_matches_df.columns for col in required_cols):
                 logging.error(f"Similarity join result missing required columns. Found: {candidate_matches_df.columns}. Expected: {required_cols}")
                 raise ValueError("Similarity join result malformed.")

            candidate_labels_agg = candidate_matches_df.sort_values(
                by=["id", "_score"], ascending=[True, False] # Sort by original ID and score
            ).groupby("id").agg(
                # Aggregate descriptions along with labels and scores
                candidate_labels=pd.NamedAgg(column="label", aggfunc=list),
                candidate_descriptions=pd.NamedAgg(column="description", aggfunc=list),
                candidate_scores=pd.NamedAgg(column="_score", aggfunc=list)
            ).reset_index()

            # Merge candidates back into the main dataframe
            df = pd.merge(df, candidate_labels_agg, on="id", how="left")

            # Fill NaNs for bookmarks with no matches - use apply with lambda for safety
            df['candidate_labels'] = df['candidate_labels'].apply(lambda x: x if isinstance(x, list) else [])
            df['candidate_descriptions'] = df['candidate_descriptions'].apply(lambda x: x if isinstance(x, list) else [])
            df['candidate_scores'] = df['candidate_scores'].apply(lambda x: x if isinstance(x, list) else [])
            logging.info("Candidate labels aggregated.")

    except Exception as e:
        logging.error(f"Error during semantic similarity join: {e}", exc_info=True)
        # Add empty columns to allow script to continue if desired
        df['candidate_labels'] = [[] for _ in range(len(df))]
        df['candidate_descriptions'] = [[] for _ in range(len(df))]
        df['candidate_scores'] = [[] for _ in range(len(df))]
        # Consider returning if this critical step fails: return

    # SECTION 6: LLM Reasoning & Classification
    logging.info("Performing LLM-based classification on candidate labels...")
    start_llm_time = time.time()

    # Prepare data for apply: Use combined_text which includes title and full content
    # Ensure necessary columns exist
    apply_cols = ['combined_text', 'candidate_labels', 'candidate_descriptions', 'id']
    if not all(col in df.columns for col in apply_cols):
         logging.error(f"Missing columns required for LLM classification. Needed: {apply_cols}, Have: {df.columns}")
         # Handle error appropriately, e.g., add empty columns or return
         df['llm_verified_labels'] = [[] for _ in range(len(df))]
    else:
        # Using df.apply for potential conciseness, though performance is similar to iterrows for external calls
        # Wrap the call in a lambda function
        llm_results = df.apply(
            lambda row: get_llm_classification(
                row,
                text_col='combined_text',
                candidate_labels_col='candidate_labels',
                candidate_descs_col='candidate_descriptions',
                row_id_col='id'
            ),
            axis=1 # Apply function row-wise
        )
        df['llm_verified_labels'] = llm_results.tolist() # Convert Series result back to list if needed

        # # --- Alternative using iterrows (as originally shown, potentially easier to debug) ---
        # llm_results = []
        # total_rows = len(df)
        # log_interval = max(1, total_rows // 20) # Log progress roughly 20 times
        # for index, row in df.iterrows():
        #     # Log progress occasionally
        #     if index % log_interval == 0 or index == total_rows - 1:
        #         logging.info(f"Processing LLM classification for row {index + 1}/{total_rows}...")
        #     llm_results.append(get_llm_classification(
        #         row,
        #         text_col='combined_text',
        #         candidate_labels_col='candidate_labels',
        #         candidate_descs_col='candidate_descriptions',
        #         row_id_col='id'
        #     ))
        # df['llm_verified_labels'] = llm_results
        # # --- End iterrows alternative ---

    end_llm_time = time.time()
    logging.info(f"LLM classification completed in {end_llm_time - start_llm_time:.2f} seconds.")
    num_classified = len(df[df['llm_verified_labels'].apply(len) > 0])
    logging.info(f"LLM assigned labels to {num_classified} bookmarks out of {len(df)}.")


    # SECTION 7: Hierarchical Consistency & Finalization
    logging.info("Finalizing labels...")
    if ENFORCE_HIERARCHY:
        logging.info("Enforcing hierarchical consistency (adding parent labels)...")
        # Pass the list of all valid ontology labels to the function for checking
        df['final_ontology_labels'] = df['llm_verified_labels'].apply(
            lambda labels: add_parent_labels(labels, label_to_path_map, all_ontology_labels_list)
        )
        logging.info("Parent labels added based on hierarchy (if logic implemented and applicable).")
    else:
        # Ensure the column exists even if hierarchy is off
        df['final_ontology_labels'] = df['llm_verified_labels']
        logging.info("Hierarchical consistency enforcement skipped.")

    # Ensure final_ontology_labels is always a list
    df['final_ontology_labels'] = df['final_ontology_labels'].apply(lambda x: x if isinstance(x, list) else [])


    # SECTION 8: Ontology Expansion Analysis (Optional)
    ontology_suggestions = []
    # Initialize cluster ID column with a default value (-1 often means noise or unclustered)
    df['_cluster_id'] = -1

    if RUN_ONTOLOGY_EXPANSION:
        logging.info("--- Starting Ontology Expansion Analysis ---")
        try:
            # Ensure data has content suitable for clustering
            # Use 'summary_and_title' as it's likely cleaner than 'combined_text'
            field_for_clustering = "summary_and_title"
            df_to_cluster = df[df[field_for_clustering].str.strip().astype(bool)].copy() # Avoid clustering empty texts

            if df_to_cluster.empty:
                 logging.warning("No data suitable for clustering found. Skipping Ontology Expansion.")
            else:
                # Index data for clustering
                logging.info(f"Indexing {len(df_to_cluster)} items for clustering into {CLUSTER_INDEX_DIR}...")
                shutil.rmtree(CLUSTER_INDEX_DIR, ignore_errors=True)
                os.makedirs(CLUSTER_INDEX_DIR, exist_ok=True)
                df_indexed_for_cluster = sem_index(df_to_cluster, field=field_for_clustering, index_dir=CLUSTER_INDEX_DIR)

                # Cluster
                logging.info(f"Clustering data into {NUM_CLUSTERS_EXPANSION} clusters using '{field_for_clustering}'...")
                # sem_cluster_by returns cluster assignments for the input dataframe
                cluster_ids = sem_cluster_by(
                    df_indexed_for_cluster, # Use the indexed data frame
                    num_clusters=NUM_CLUSTERS_EXPANSION,
                    field=field_for_clustering, # Field used for indexing/clustering
                    index_dir=CLUSTER_INDEX_DIR # Point to the index
                )
                # Map cluster IDs back to the original DataFrame using the index
                # Note: sem_cluster_by might return a Series or directly modify the input df. Check lotus docs.
                # Assuming it returns a Series aligned with df_indexed_for_cluster's index:
                df_to_cluster['_cluster_id'] = cluster_ids
                # Update the main dataframe with cluster IDs
                df.update(df_to_cluster[['_cluster_id']]) # Updates based on matching index

                logging.info(f"Clustering completed. Cluster IDs assigned. Found {len(df['_cluster_id'].unique()) -1} clusters (excluding noise).")

                # Analyze clusters for potential new entities/gaps
                logging.info("Analyzing clusters for potential ontology gaps...")
                unique_clusters = sorted([c for c in df['_cluster_id'].unique() if c != -1]) # Get actual cluster IDs

                for cluster_id in unique_clusters:
                    cluster_df = df[df['_cluster_id'] == cluster_id]
                    logging.info(f"Analyzing Cluster {cluster_id} ({len(cluster_df)} items)...")

                    # Sample text from cluster (e.g., concat a few summaries or titles)
                    sample_size = min(10, len(cluster_df)) # Sample up to 10 items
                    sample_texts = ". ".join(cluster_df[field_for_clustering].fillna('').sample(sample_size, random_state=42).tolist())
                    if not sample_texts.strip():
                        logging.warning(f"Cluster {cluster_id} sample text is empty. Skipping analysis.")
                        continue

                    # Get unique labels already assigned within this cluster
                    assigned_labels_in_cluster = sorted(list(set(label for sublist in cluster_df['final_ontology_labels'] for label in sublist)))
                    assigned_labels_str = ", ".join(assigned_labels_in_cluster) if assigned_labels_in_cluster else "None"

                    # Escape inputs for the prompt
                    # --- Corrected Escaping ---
                    escaped_sample_texts = escape_for_prompt(sample_texts)
                    escaped_assigned_labels = escape_for_prompt(assigned_labels_str)
                    # --- End Corrected Escaping ---

                    # Define prompts using triple quotes and f-strings
                    potential_entities_prompt_text = f'''Based ONLY on the following text, list specific, potentially non-obvious software tools, libraries, algorithms, datasets, or concrete technical concepts mentioned that might be suitable candidates for adding to a technical knowledge ontology. Focus on specific nouns/noun phrases. Avoid generic terms or broad categories. List terms separated ONLY by commas, or respond with the single word NONE if nothing specific stands out.
--- TEXT START ---
{escaped_sample_texts}
--- TEXT END ---'''

                    gap_suggestions_prompt_text = f'''Consider the main topics in the text below. The following ontology labels were assigned to items in this cluster: [{escaped_assigned_labels}]. Are there any clear, important thematic areas or core concepts present in the text that are NOT well covered by these existing labels? If yes, suggest 1-2 potential NEW label names and brief descriptions relevant to the text's core subject. Format EACH suggestion as 'Suggested New Label Name: Brief description of what it covers'. Separate multiple suggestions with a newline. If no significant gaps are apparent, respond with the single word NONE.
--- TEXT START ---
{escaped_sample_texts}
--- TEXT END ---'''

                    current_prompt = {
                        "potential_new_entities": potential_entities_prompt_text,
                        "thematic_gap_suggestions": gap_suggestions_prompt_text
                    }

                    try:
                        # Pass sample_texts as a list to sem_extract
                        extracted_suggestions = sem_extract([sample_texts], prompt=current_prompt) # Pass text in a list

                        if extracted_suggestions:
                            suggestions = extracted_suggestions[0] # Get dict from list
                            suggestions['cluster_id'] = cluster_id
                            # suggestions['sample_texts_analyzed'] = sample_texts # Optional: store sample text (can be large)
                            suggestions['assigned_labels_in_cluster'] = assigned_labels_in_cluster
                            suggestions['cluster_size'] = len(cluster_df)

                            # Clean up LLM output (e.g., remove "NONE" strings if necessary)
                            entities_raw = suggestions.get("potential_new_entities", "")
                            gaps_raw = suggestions.get("thematic_gap_suggestions", "")

                            if isinstance(entities_raw, str) and entities_raw.strip().upper() == "NONE":
                                suggestions["potential_new_entities"] = []
                            elif isinstance(entities_raw, str):
                                suggestions["potential_new_entities"] = [e.strip() for e in entities_raw.split(',') if e.strip()]

                            if isinstance(gaps_raw, str) and gaps_raw.strip().upper() == "NONE":
                                suggestions["thematic_gap_suggestions"] = []
                            elif isinstance(gaps_raw, str):
                                # Split by newline for multiple suggestions
                                suggestions["thematic_gap_suggestions"] = [g.strip() for g in gaps_raw.split('\n') if g.strip()]

                            ontology_suggestions.append(suggestions)
                        else:
                            logging.warning(f"sem_extract returned no suggestion data for cluster {cluster_id}")

                    except Exception as e:
                         logging.warning(f"Could not run gap analysis LLM prompt for cluster {cluster_id}: {e}", exc_info=True)

                logging.info(f"Ontology expansion analysis completed. Generated suggestions for {len(ontology_suggestions)} clusters.")

                # Save suggestions
                if ontology_suggestions:
                    try:
                        suggestions_df = pd.DataFrame(ontology_suggestions)
                        # Reorder columns for clarity
                        cols_order = ['cluster_id', 'cluster_size', 'potential_new_entities', 'thematic_gap_suggestions', 'assigned_labels_in_cluster']
                        suggestions_df = suggestions_df[[col for col in cols_order if col in suggestions_df.columns]]
                        suggestions_df.to_json(SUGGESTIONS_FILE, orient="records", lines=True, date_format="iso", default_handler=str, force_ascii=False, indent=2) # Added indent for readability
                        logging.info(f"Ontology suggestions saved to {SUGGESTIONS_FILE}")
                    except Exception as e:
                        logging.error(f"Error saving ontology suggestions: {e}", exc_info=True)
                else:
                    logging.info("No ontology suggestions were generated to save.")

        except Exception as e:
            logging.error(f"Error during Ontology Expansion Analysis: {e}", exc_info=True)
            # Ensure cluster_id column is still -1 if analysis failed midway
            if '_cluster_id' not in df.columns:
                 df['_cluster_id'] = -1
    else:
        logging.info("Ontology Expansion Analysis skipped.")
        # Ensure the _cluster_id column exists if skipped
        if '_cluster_id' not in df.columns:
             df['_cluster_id'] = -1

    # Rename cluster ID column for final output *after* potential use
    df = df.rename(columns={'_cluster_id': 'cluster_id'})


    # SECTION 9: Store Enriched Data
    logging.info(f"Saving enriched data to {OUTPUT_FILE}...")
    try:
        # Select and order relevant columns for output
        columns_to_keep = [
            'id', 'url', 'title', 'content', # Original fields
            'created_at', # Original timestamp if available
            'cleaned_title', 'cleaned_content', 'content_summary', 'summary_and_title', # Processed text fields
            'candidate_labels', 'candidate_scores', 'candidate_descriptions', # Intermediate results
            'llm_verified_labels', # LLM refined labels
            'final_ontology_labels', # Final labels after hierarchy (if applied)
            'cluster_id' # Cluster assignment from expansion analysis
        ]
        # Ensure all selected columns exist in the DataFrame, add missing ones with default values if necessary
        output_columns = []
        for col in columns_to_keep:
            if col in df.columns:
                output_columns.append(col)
            else:
                logging.warning(f"Column '{col}' not found in DataFrame. It will be excluded from the output.")
                # Optionally add it with default null/empty values:
                # df[col] = None # or pd.NA, [], "", etc. depending on expected type
                # output_columns.append(col)

        output_df = df[output_columns]

        # Save to JSON Lines format
        output_df.to_json(
            OUTPUT_FILE,
            orient="records",
            lines=True,
            date_format="iso", # Standard format for datetime
            default_handler=str, # Handle potential non-serializable types
            force_ascii=False # Ensure UTF-8 characters are saved correctly
        )
        logging.info(f"Enriched data saved successfully. Final Shape: {output_df.shape}")
    except Exception as e:
        logging.error(f"Error saving final data: {e}", exc_info=True)

    # SECTION 10: Finish Task & Summarize
    end_time = time.time()
    logging.info("\n--- Execution Summary ---")
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    logging.info(f"LOTUS models configured: lm={LM_MODEL}, rm={RM_MODEL}, reranker={RERANKER_MODEL}")
    logging.info(f"Input data: {DATA_FILE}")
    logging.info(f"Ontology: {ONTOLOGY_FILE}")
    logging.info(f"Initial data rows: {initial_rows}")
    logging.info(f"Data rows after cleaning/deduplication: {rows_after_dedup}")
    logging.info(f"Ontology loaded with {len(labels_df)} unique labels.")
    logging.info(f"XMLC enrichment stages: LLM Summary -> sem_sim_join (K={K_CANDIDATES}) -> LLM Classification")
    logging.info(f"LLM assigned labels to {num_classified} bookmarks.")
    logging.info(f"Hierarchical consistency enforced: {ENFORCE_HIERARCHY}")
    logging.info(f"Ontology expansion analysis run: {RUN_ONTOLOGY_EXPANSION}")
    if RUN_ONTOLOGY_EXPANSION and ontology_suggestions:
        logging.info(f"Ontology suggestions saved to: {SUGGESTIONS_FILE}")
    elif RUN_ONTOLOGY_EXPANSION:
         logging.info("Ontology expansion analysis run, but no suggestions were generated.")
    logging.info(f"Enriched data saved to: {OUTPUT_FILE}")
    if 'output_df' in locals(): # Check if output_df was created
        logging.info(f"Final data shape: {output_df.shape}")
    else:
        logging.warning("Final output DataFrame was not generated due to errors.")
    logging.info("--- Workflow Completed ---")

if __name__ == "__main__":
    main()