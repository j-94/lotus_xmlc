# -*- coding: utf-8 -*-
import pandas as pd
import logging
# import lotus # Removed - accessors should be registered by run_pipeline.py
# from lotus.sem_ops.sem_extract import SemExtractDataFrame # Removed - Not needed if accessor called directly

# Import specific error types if needed for catching
from litellm.exceptions import APIConnectionError, APIError

logger = logging.getLogger(__name__)

def generate_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Generates content summaries using the lotus sem_extract accessor.

    Args:
        df: The input DataFrame containing 'cleaned_content' and 'cleaned_title'.

    Returns:
        The DataFrame with added 'content_summary', 'summary_and_title',
        and 'combined_text' columns.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty, skipping summarization.")
        return df

    # SECTION 4: Content Summarization (Using df.sem_extract)
    logger.info("Generating content summaries using df.sem_extract...")
    summary_prompt = "Summarize the following text in 1-2 concise sentences, capturing the main topic:"
    try:
        # Call the accessor directly on the DataFrame
        result_df = df.sem_extract(
            input_cols=["cleaned_content"],
            output_cols={"content_summary": summary_prompt}
        )

        # sem_extract returns a *new* DataFrame including original columns +
        # the new extracted columns. We can just use this result directly
        # if it contains all necessary original columns, or selectively assign.
        # For safety, let's assign the new column back to the original df.
        if 'content_summary' in result_df.columns:
            df['content_summary'] = result_df['content_summary']
            logger.info("Content summaries generated.")
        else:
            logger.error("sem_extract did not produce the expected 'content_summary' column.")
            df['content_summary'] = "Error: Summarization Failed (Missing Column)"

        # Combine title and summary
        df['summary_and_title'] = df['cleaned_title'].fillna("") + " | Summary: " + df['content_summary'].fillna("")

        # Create combined text field for candidate generation
        df['combined_text'] = df['summary_and_title'].fillna("") + " " + df['cleaned_content'].fillna("")
        logger.info("Combined text fields created.")

    except APIConnectionError as conn_err:
        logger.error(f"API Connection Error during summarization: {conn_err}. Check API key and network.", exc_info=True)
        df['content_summary'] = "Error: API Connection Failed"
        # Also update fallback columns
        df['summary_and_title'] = df['cleaned_title'].fillna("") + " | Summary: " + df['content_summary'].fillna("")
        df['combined_text'] = df['summary_and_title'].fillna("") + " " + df['cleaned_content'].fillna("")
    except APIError as api_err:
        logger.error(f"API Error during summarization: {api_err}. Check request/model.", exc_info=True)
        df['content_summary'] = f"Error: API Failed ({api_err.status_code})"
        # Also update fallback columns
        df['summary_and_title'] = df['cleaned_title'].fillna("") + " | Summary: " + df['content_summary'].fillna("")
        df['combined_text'] = df['summary_and_title'].fillna("") + " " + df['cleaned_content'].fillna("")
    except AttributeError as ae:
         # Catch if the accessor isn't registered OR if called incorrectly
         if "'DataFrame' object has no attribute 'sem_extract'" in str(ae):
             logger.error(f"Lotus accessor 'sem_extract' not found. Ensure accessor registration is working. Error: {ae}", exc_info=True)
         elif "'APIConnectionError' object has no attribute 'choices'" in str(ae):
             logger.error(f"Caught AttributeError on APIConnectionError - likely LLM call failed. Check API key/network. Error: {ae}", exc_info=True)
             df['content_summary'] = "Error: Summarization Failed (API Connection Issue)"
         else:
             logger.error(f"Attribute error during summarization: {ae}", exc_info=True)
             df['content_summary'] = "Error: Summarization Failed (AttributeError)"
         # Ensure downstream columns exist even if summarization fails
         df['summary_and_title'] = df['cleaned_title'].fillna("") + " | Summary: " + df['content_summary'].fillna("")
         df['combined_text'] = df['summary_and_title'].fillna("") + " " + df['cleaned_content'].fillna("")
         # Don't return here, let the pipeline continue if desired, but summary is marked as error
         return df # Return df with error state
    except Exception as e:
        # Catch other potential errors
        logger.error(f"Unexpected error during content summarization: {e}", exc_info=True)
        df['content_summary'] = f"Error: Summarization Failed (Unexpected: {type(e).__name__})"
        # Also update fallback columns
        df['summary_and_title'] = df['cleaned_title'].fillna("") + " | Summary: " + df['content_summary'].fillna("")
        df['combined_text'] = df['summary_and_title'].fillna("") + " " + df['cleaned_content'].fillna("")

    # Ensure downstream columns exist even if summarization had errors but didn't raise an exception handled above
    if 'content_summary' not in df.columns:
        df['content_summary'] = "Error: Summarization Failed (Unknown)"
    if 'summary_and_title' not in df.columns:
        df['summary_and_title'] = df['cleaned_title'].fillna("") + " | Summary: " + df['content_summary'].fillna("")
    if 'combined_text' not in df.columns:
         df['combined_text'] = df['summary_and_title'].fillna("") + " " + df['cleaned_content'].fillna("")

    return df 