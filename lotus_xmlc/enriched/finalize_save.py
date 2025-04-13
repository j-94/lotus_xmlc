# -*- coding: utf-8 -*-
import pandas as pd
import logging
import json
from . import utils # For add_parent_labels

logger = logging.getLogger(__name__)

def finalize_and_save(df: pd.DataFrame, output_file_path: str, label_to_path_map: dict, all_ontology_labels: list, enforce_hierarchy: bool):
    """Applies optional hierarchical consistency and saves the final DataFrame.

    Args:
        df: The processed DataFrame with 'llm_verified_labels'.
        output_file_path: Path to save the final JSONL output.
        label_to_path_map: Dictionary mapping ontology labels to paths.
        all_ontology_labels: List of all valid ontology labels.
        enforce_hierarchy: Boolean flag from config to enable hierarchy enforcement.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty, skipping finalization and saving.")
        return

    # SECTION 9: Apply Hierarchy & Final Output
    logger.info("Applying final label processing...")
    if enforce_hierarchy:
        logger.info("Enforcing hierarchy: adding parent labels...")
        if not label_to_path_map or not all_ontology_labels:
             logger.warning("Cannot enforce hierarchy: Missing label_to_path_map or all_ontology_labels. Skipping hierarchy step.")
             # Copy LLM verified labels directly to final if hierarchy data is missing
             df['final_ontology_labels'] = df['llm_verified_labels']
        else:
            df['final_ontology_labels'] = df['llm_verified_labels'].apply(
                lambda labels: utils.add_parent_labels(
                    selected_labels=labels,
                    label_to_path_map=label_to_path_map,
                    all_ontology_labels=all_ontology_labels,
                    enforce_hierarchy=True # Pass True here explicitly
                )
            )
        logger.info("Hierarchy enforcement complete.")
    else:
        logger.info("Skipping hierarchy enforcement as per configuration.")
        # If hierarchy is not enforced, the final labels are just the LLM verified ones
        df['final_ontology_labels'] = df['llm_verified_labels']

    # Ensure final labels are always lists (even if empty)
    df['final_ontology_labels'] = df['final_ontology_labels'].apply(lambda x: x if isinstance(x, list) else [])

    # --- Save Results ---
    logger.info(f"Saving enriched data to {output_file_path}...")
    try:
        # Select and order columns for the final output (optional, but good practice)
        output_columns = [
            'id', 'url', 'title', 'content', 'created_at', # Original fields
            'cleaned_title', 'cleaned_content', # Cleaned text
            'content_summary', 'summary_and_title', 'combined_text', # Summarization/Combined
            'candidate_labels', 'candidate_scores', 'candidate_descriptions', # Candidates
            'llm_verified_labels', # LLM output
            'final_ontology_labels' # Final selected labels
        ]
        # Filter to only columns that actually exist in the DataFrame to avoid KeyErrors
        output_columns = [col for col in output_columns if col in df.columns]
        output_df = df[output_columns]

        # Save as JSONL
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for record in output_df.to_dict(orient='records'):
                # Ensure list fields are actually lists before dumping
                for key in ['candidate_labels', 'candidate_scores', 'candidate_descriptions', 'llm_verified_labels', 'final_ontology_labels']:
                    if key in record and not isinstance(record[key], list):
                        record[key] = [] # Default to empty list if not a list
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Successfully saved {len(output_df)} records to {output_file_path}.")

    except KeyError as ke:
         logger.error(f"Missing expected column during saving: {ke}. Check DataFrame columns.", exc_info=True)
    except Exception as e:
        logger.error(f"Error saving results to {output_file_path}: {e}", exc_info=True) 