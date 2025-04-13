# -*- coding: utf-8 -*-
import pandas as pd
import logging
import time # Add time import
import lotus # Import lotus for settings access
from . import utils # Import utility functions like escape_for_prompt

logger = logging.getLogger(__name__)


def _get_llm_classification_for_row(bookmark_text: str, candidates: list, descriptions: list, row_id_for_log: str = "N/A") -> list:
    """Internal helper to get LLM classification for a single row."""
    if not candidates:
        logger.debug(f"[LLM Classify] No candidates for ID {row_id_for_log}.")
        return []

    # Retrieve the configured LM instance from settings
    try:
        llm_instance = lotus.settings.lm
        if not llm_instance:
            raise ValueError("Lotus LM not configured in settings.")
    except AttributeError:
         logger.error("[LLM Classify] lotus.settings.lm not found. Ensure lotus is configured before calling classification.")
         return [] # Cannot proceed without LM
    except ValueError as e:
        logger.error(f"[LLM Classify] Issue with configured LM: {e}")
        return []

    # Create formatted list of candidates with descriptions
    candidates_formatted_list = []
    # Ensure descriptions list matches candidates length, pad if necessary or log warning
    if len(descriptions) != len(candidates):
        logger.warning(f"[LLM Classify] Mismatch candidates ({len(candidates)})/descriptions ({len(descriptions)}) for ID {row_id_for_log}. Using labels only for prompt.")
        descriptions = ["No description"] * len(candidates) # Pad descriptions

    for i, (label, desc) in enumerate(zip(candidates, descriptions)):
        candidates_formatted_list.append(f"{i+1}. {utils.escape_for_prompt(label)}: {utils.escape_for_prompt(desc)}")

    candidates_formatted = "\n".join(candidates_formatted_list)
    escaped_bookmark_text = utils.escape_for_prompt(bookmark_text)

    # Construct the specific prompt for this task
    prompt_text = f'''Given the following bookmark text:
--- TEXT START ---
{escaped_bookmark_text}
--- TEXT END ---

And the following candidate ontology labels with descriptions:
--- CANDIDATES START ---
{candidates_formatted}
--- CANDIDATES END ---

Which of these candidate labels are the MOST relevant and accurate classifications for the text? List ONLY the applicable label names (exactly as provided), separated ONLY by commas (e.g., LabelA, LabelC, LabelF). If none are applicable, respond with NONE.
'''
    try:
        # Manually call the configured LM
        # Assuming the LM instance is callable or has a standard method like `complete` or `__call__`
        # Based on original script: llm_instance(prompts=[prompt_text])
        completion_results = llm_instance(prompts=[prompt_text]) # Using __call__ based on lotus examples

        # Process the result (assuming result is a list of LMOutput or similar)
        if not completion_results or not hasattr(completion_results[0], 'text'):
             logger.error(f"[LLM Classify] LLM completion failed or returned unexpected format for ID {row_id_for_log}.")
             return []

        raw_output = completion_results[0].text

        if raw_output is None or not isinstance(raw_output, str) or raw_output.strip().upper() == 'NONE':
            logger.debug(f"[LLM Classify] LLM returned no applicable labels for ID {row_id_for_log}. Output: '{raw_output}'")
            return []

        llm_chosen_labels = [label.strip() for label in raw_output.split(',') if label.strip()]
        valid_chosen_labels = [label for label in llm_chosen_labels if label in candidates]

        if len(valid_chosen_labels) != len(llm_chosen_labels):
             hallucinated = set(llm_chosen_labels) - set(valid_chosen_labels)
             logger.warning(f"[LLM Classify] LLM hallucinated labels for ID {row_id_for_log}. Hallucinated: {list(hallucinated)}. Filtered: {valid_chosen_labels}")

        return valid_chosen_labels

    except Exception as e:
        logger.error(f"[LLM Classify] Error during LLM classification for ID {row_id_for_log}: {e}", exc_info=True)
        return [] # Return empty list on error

def perform_llm_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Performs LLM classification on candidate labels for each row in the DataFrame.

    Iterates through the DataFrame and uses the configured LLM via `lotus.settings.lm`
    to verify candidate labels against the combined text.

    Args:
        df: DataFrame containing 'id', 'combined_text', 'candidate_labels',
            and 'candidate_descriptions'.

    Returns:
        The DataFrame updated with LLM verified labels in the 'llm_verified_labels' column.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty, skipping LLM classification.")
        return df

    required_cols = ['id', 'combined_text', 'candidate_labels', 'candidate_descriptions']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns for LLM classification: {required_cols}. Skipping.")
        return df

    # SECTION 7: LLM-based Classification (Manual Loop)
    logger.info("Starting LLM-based classification loop...")
    verified_labels_list = []
    total_rows = len(df)
    start_time = time.time() # Record start time

    # Check if LM is configured before starting the loop
    try:
        if not lotus.settings.lm:
            logger.error("Lotus LM not configured in settings. Cannot proceed with LLM classification.")
            # Add empty lists to avoid breaking downstream processing
            df['llm_verified_labels'] = [[] for _ in range(len(df))]
            return df
    except AttributeError:
        logger.error("lotus.settings.lm not found. Ensure lotus is configured. Cannot proceed with LLM classification.")
        df['llm_verified_labels'] = [[] for _ in range(len(df))]
        return df

    for index, row in df.iterrows():
        row_id = row.get('id', f"index_{index}") # Use ID if available, else index
        # Reduced logging verbosity here, focus on periodic updates
        # logger.debug(f"Processing row {index + 1}/{total_rows} (ID: {row_id}) for LLM classification...")

        candidates = row['candidate_labels']
        descriptions = row['candidate_descriptions']
        text_for_prompt = row['combined_text']

        # Ensure candidates and descriptions are lists
        if not isinstance(candidates, list): candidates = []
        if not isinstance(descriptions, list): descriptions = []

        verified_labels = _get_llm_classification_for_row(
            bookmark_text=text_for_prompt,
            candidates=candidates,
            descriptions=descriptions,
            row_id_for_log=str(row_id)
        )
        verified_labels_list.append(verified_labels)

        # Log progress periodically with time estimates
        processed_rows = index + 1
        if processed_rows % 20 == 0 or processed_rows == total_rows: # Update every 20 rows or at the end
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_row = elapsed_time / processed_rows if processed_rows > 0 else 0
            estimated_total_time = avg_time_per_row * total_rows
            eta = estimated_total_time - elapsed_time

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta)) if eta > 0 else "00:00:00"

            logger.info(
                f"LLM Classification progress: {processed_rows}/{total_rows} rows processed. "
                f"Elapsed: {elapsed_str}. ETA: {eta_str}."
            )


    df['llm_verified_labels'] = verified_labels_list
    end_time = time.time() # Record end time
    total_elapsed = end_time - start_time
    total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
    logger.info(f"LLM-based classification loop finished. Total time: {total_elapsed_str}.")
    return df 