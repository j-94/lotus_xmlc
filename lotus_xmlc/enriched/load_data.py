# -*- coding: utf-8 -*-
import pandas as pd
import json
import logging
import yaml # Import YAML parser
from . import utils # Use relative import within the 'enriched' package

logger = logging.getLogger(__name__)

def load_and_clean_bookmarks(data_file_path: str, initial_limit: int = None) -> pd.DataFrame:
    """Loads bookmarks from a JSONL file, cleans text fields, and adds initial columns.

    Args:
        data_file_path: Path to the input Dataset.jsonl file.
        initial_limit: Optional number of rows to load for testing.

    Returns:
        A pandas DataFrame with loaded and initially cleaned bookmark data.
    """
    logger.info(f"Loading bookmark data from {data_file_path}...")
    bookmarks_data = []
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if initial_limit is not None and i >= initial_limit:
                    logger.warning(f"Limiting input data to first {initial_limit} rows for testing.")
                    break
                try:
                    bookmarks_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {i+1}: {e}")
        if not bookmarks_data:
            logger.error(f"No valid data loaded from {data_file_path}. Exiting.")
            return pd.DataFrame() # Return empty DataFrame if no data

        df = pd.DataFrame(bookmarks_data)
        logger.info(f"Loaded {len(df)} bookmarks.")

        # Ensure essential columns exist, fill missing with defaults
        if 'id' not in df.columns: # Assuming 'id' is crucial
            logger.warning("'id' column missing, generating sequential IDs.")
            df['id'] = range(len(df))
        else:
             # Ensure ID is string type, consistent with original usage
             df['id'] = df['id'].astype(str)

        df['url'] = df.get('url', pd.Series(index=df.index, dtype=str)).fillna("")
        df['title'] = df.get('title', pd.Series(index=df.index, dtype=str)).fillna("")
        df['content'] = df.get('content', pd.Series(index=df.index, dtype=str)).fillna("")

        # --- Section 2: Basic Cleaning (Title and Content) ---
        logger.info("Performing initial text cleaning...")
        df['cleaned_title'] = df['title'].apply(utils.clean_text)
        df['cleaned_content'] = df['content'].apply(utils.clean_text)
        logger.info("Initial text cleaning complete.")

        # --- Section 3: Deduplication (Optional - Keeping simple for now) ---
        # Original script didn't have explicit deduplication here, relying on IDs?
        # Add deduplication logic here if needed, e.g.:
        # initial_rows = len(df)
        # df = df.drop_duplicates(subset=['url', 'cleaned_title', 'cleaned_content'])
        # logger.info(f"Removed {initial_rows - len(df)} duplicate rows.")

        # Add empty columns expected by later stages
        df['content_summary'] = ""
        df['summary_and_title'] = ""
        df['combined_text'] = ""
        df['candidate_labels'] = [[] for _ in range(len(df))]
        df['candidate_scores'] = [[] for _ in range(len(df))]
        df['candidate_descriptions'] = [[] for _ in range(len(df))]
        df['llm_verified_labels'] = [[] for _ in range(len(df))]
        df['final_ontology_labels'] = [[] for _ in range(len(df))]

        return df

    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file_path}")
        return pd.DataFrame() # Return empty DataFrame on file not found
    except Exception as e:
        logger.error(f"Error loading or cleaning data: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on other errors

# --- Ontology Loading --- #
def load_ontology(ontology_path: str) -> pd.DataFrame | None:
    """Loads ontology labels and descriptions from a YAML file.

    Uses the structure defined in Ontology.yaml and the helper function
    `utils.extract_ontology_labels` to create a flat DataFrame.

    Args:
        ontology_path: Path to the Ontology.yaml file.

    Returns:
        A pandas DataFrame with 'label' and 'description' columns, or None on error.
    """
    logger.info(f"Loading ontology from YAML file: {ontology_path}...")
    try:
        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology_data = yaml.safe_load(f)

        if not ontology_data:
            logger.error(f"Ontology file is empty or invalid: {ontology_path}")
            return None

        # Use the utility function to extract flat list (label, description, path)
        extracted_labels = utils.extract_ontology_labels(ontology_data)

        if not extracted_labels:
            logger.error("No labels could be extracted from the ontology structure.")
            return None

        # Convert list of tuples to DataFrame
        # Columns will be: 0=label, 1=description, 2=path
        ontology_df = pd.DataFrame(extracted_labels, columns=['label', 'description', 'path'])

        # Keep only necessary columns for candidate generation
        if 'label' not in ontology_df.columns or 'description' not in ontology_df.columns:
             logger.error("Extracted ontology data missing 'label' or 'description' columns.")
             return None

        ontology_df = ontology_df[['label', 'description']].copy()
        # Optional: Drop duplicates just in case extraction logic produced some
        ontology_df.drop_duplicates(subset=['label'], inplace=True)

        logger.info(f"Successfully loaded {len(ontology_df)} unique labels from ontology.")
        return ontology_df

    except FileNotFoundError:
        logger.error(f"Ontology YAML file not found: {ontology_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing ontology YAML file {ontology_path}: {e}", exc_info=True)
        return None
    except ImportError:
        logger.error("PyYAML library not found. Please install it (`pip install pyyaml`) to load YAML ontology.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during ontology processing: {e}", exc_info=True)
        return None 