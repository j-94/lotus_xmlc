# -*- coding: utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
from html import unescape
import logging

# Configure logging for utils module
logger = logging.getLogger(__name__)

def clean_text(text):
    """Cleans HTML tags and excessive whitespace from text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    try:
        if not text.strip():
            return ""
        # Use lxml if available, fallback to html.parser
        try:
            soup = BeautifulSoup(text, "lxml")
        except ImportError:
            soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = unescape(text)
        text = " ".join(text.split())
    except Exception as e:
        logger.warning(f"Error cleaning text: {e}. Original text start: '{str(text)[:100]}...'")
        return "" # Return empty string on error
    return text

def extract_ontology_labels(ontology, path=""):
    """Recursively extracts leaf labels and descriptions from 'instances' lists
       within the ontology structure.
    """
    labels = []
    if isinstance(ontology, dict):
        for key, value in ontology.items():
            current_path_segment = key.replace(' ', '_')
            new_path = f"{path}/{current_path_segment}" if path else current_path_segment

            # Target the 'instances' lists specifically
            if key == 'instances' and isinstance(value, list):
                for item in value:
                    label = None
                    description = "No description"
                    item_path_segment = ""

                    if isinstance(item, dict):
                        # Handles structures like: - LabelName: {description: ..., type: ...}
                        if len(item) == 1:
                            label = list(item.keys())[0]
                            details = item[label]
                            if isinstance(details, dict):
                                description = details.get('description', f'{label}')
                            else: # Handle cases where value isn't a dict (though unlikely based on YAML)
                                description = f'{label}'
                            item_path_segment = label.replace(' ', '_')
                    elif isinstance(item, str):
                        # Handles simple string instances
                        label = item
                        description = f"{label} (instance)"
                        item_path_segment = label.replace(' ', '_')

                    if label:
                        # Construct path based on the path leading *to* the instances list
                        instance_path = f"{path}/{item_path_segment}" if path else item_path_segment
                        labels.append((label, description, instance_path))

            # Recursively search deeper dictionaries
            elif isinstance(value, dict):
                 labels.extend(extract_ontology_labels(value, new_path))
            # Recursively search deeper lists (that aren't 'instances')
            elif isinstance(value, list):
                 # Pass the path of the list's key (e.g., subcategories)
                 labels.extend(extract_ontology_labels(value, new_path))

    # Handle case where the structure passed in is a list (e.g. list under subcategories)
    elif isinstance(ontology, list):
        for item in ontology:
            # Pass the current path down when recursing into list items
            labels.extend(extract_ontology_labels(item, path))

    return labels

def escape_for_prompt(text: str) -> str:
    """Escapes characters potentially problematic for LLM prompts."""
    if not isinstance(text, str):
        return ""
    # Escape backslashes first, then quotes, then newlines/carriage returns
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', '')

def add_parent_labels(selected_labels: list, label_to_path_map: dict, all_ontology_labels: list, enforce_hierarchy: bool) -> list:
    """Adds parent labels based on hierarchical paths if enforce_hierarchy is True."""
    if not enforce_hierarchy or not selected_labels:
        return sorted(list(set(selected_labels))) # Return sorted unique list

    final_labels = set(selected_labels)
    ontology_label_set = set(all_ontology_labels)
    labels_to_process = list(selected_labels)
    processed_paths = set()

    while labels_to_process:
        label = labels_to_process.pop(0)
        path = label_to_path_map.get(label)

        if path and path not in processed_paths:
            processed_paths.add(path)
            parts = path.strip('/').split('/')
            current_path = ""
            # Iterate through parent path segments
            for i in range(len(parts) - 1):
                part = parts[i]
                current_path = f"{current_path}/{part}" if current_path else part
                parent_label = None
                # Find the label corresponding to the current parent path
                for lbl, pth in label_to_path_map.items():
                    if pth == current_path and lbl in ontology_label_set:
                        parent_label = lbl
                        break
                # If a valid parent label is found and not already included, add it
                if parent_label and parent_label not in final_labels:
                    final_labels.add(parent_label)
                    # Optional: Add parent to processing queue if recursive addition is desired
                    # labels_to_process.append(parent_label)

    return sorted(list(final_labels))

# New function to apply clean_text to specified DataFrame columns
def clean_dataframe_text(df: pd.DataFrame, columns_to_clean: list) -> pd.DataFrame:
    """Applies the clean_text function to specified columns of a DataFrame."""
    logger.info(f"Cleaning text in columns: {columns_to_clean}")
    df_copy = df.copy()
    for col in columns_to_clean:
        if col in df_copy.columns:
            # Apply clean_text and create new cleaned column names
            cleaned_col_name = f"cleaned_{col}"
            logger.debug(f"Applying clean_text to column '{col}', saving as '{cleaned_col_name}'")
            df_copy[cleaned_col_name] = df_copy[col].apply(clean_text)
            logger.debug(f"Finished cleaning column '{col}'.")
        else:
            logger.warning(f"Column '{col}' not found in DataFrame during cleaning. Skipping.")
    logger.info("DataFrame text cleaning complete.")
    return df_copy 