#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runs the enriched bookmark processing pipeline."""
import os
import sys
import logging
import time
import pandas as pd
import lotus
from dotenv import load_dotenv
# Import model classes and cache utilities
from lotus.models import LM, SentenceTransformersRM
from lotus.cache import CacheFactory, CacheConfig, CacheType
from lotus.vector_store import FaissVS # Import FaissVS

# Load environment variables from .env file
load_dotenv()

# Set up logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# Determine the base directory based on script location
# Assuming the script is in the 'enriched' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Add root directory to sys.path for module resolution
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- Move Lotus Configuration Here ---
logger.info("Configuring Lotus...")
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # 1. Define Cache Configuration
    # Using SQLite cache in the project's .cache directory
    cache_file_path = os.path.join(ROOT_DIR, ".cache", "lotus_op_cache.db")
    cache_config = CacheConfig(
        cache_type=CacheType.SQLITE,
        cache_db_path=cache_file_path, # Specify path for SQLite
        max_size=10000 # Example size
    )
    operator_cache = CacheFactory.create_cache(cache_config)
    logger.info(f"Lotus operator cache configured: Type=SQLITE, Path={cache_file_path}")

    # 2. Instantiate Models
    # Instantiate LM with the cache object
    lm_instance = LM(
        model="openai/gpt-3.5-turbo-0125",
        api_key=openai_api_key, # Pass API key during LM instantiation
        cache=operator_cache
    )
    # Instantiate RM (does not seem to accept cache directly)
    rm_instance = SentenceTransformersRM(model="intfloat/e5-base-v2")
    # Instantiate VS
    vs_instance = FaissVS() # Instantiate the Faiss Vector Store

    # 3. Configure Lotus Settings with instantiated models
    lotus.settings.configure(
        lm=lm_instance,
        rm=rm_instance,
        vs=vs_instance, # Pass the instantiated VS
        enable_cache=True # Keep global cache enabled for operators
    )
    logger.info("Lotus configured successfully with instantiated models.")
    # Verification (log configured types)
    lm_type = type(lotus.settings.lm).__name__ if lotus.settings.lm else None
    rm_type = type(lotus.settings.rm).__name__ if lotus.settings.rm else None
    vs_type = type(lotus.settings.vs).__name__ if lotus.settings.vs else None # Note: VS might still be str
    logger.info(f"Types after configure - LM: {lm_type}, RM: {rm_type}, VS: {vs_type}")

except Exception as e:
    logger.error(f"Fatal: Failed to configure Lotus settings: {e}", exc_info=True)
    sys.exit(1) # Exit if configuration fails
# --- End Lotus Configuration ---

# Import project modules *after* Lotus is configured
# from enriched.utils import text_utils # Assuming utilities are in utils.py
from enriched.cache import check_or_run # Import from the new cache.py file
# Make sure accessors are registered by importing the classes *after* configuration
from lotus.sem_ops.sem_extract import SemExtractDataFrame
from lotus.sem_ops.sem_sim_join import SemSimJoinDataframe

# Import pipeline steps
from enriched import load_data, summarize_data, candidate_generation, llm_classification, finalize_save # Corrected import

# Removed RM/VS type imports
# from lotus.models import RM # Removed
# from lotus.vector_store import VS # Removed

def run_pipeline(input_path: str, output_path: str, ontology_path: str, num_rows: int = None, use_cache: bool = True):
    """Main function to run the XMLC pipeline."""
    start_time = time.time()
    logger.info("Starting Modular XMLC Workflow...")

    # === Configuration ===
    run_id = f"run_{int(time.time())}"
    # Define cache paths relative to the ROOT_DIR
    load_cache_path = os.path.join(ROOT_DIR, ".cache", f"load_data_{os.path.basename(input_path)}.pkl")
    # clean_cache_path = os.path.join(ROOT_DIR, ".cache", f"clean_data_{run_id}.pkl") # REMOVED
    summary_cache_path = os.path.join(ROOT_DIR, ".cache", f"summary_data_{run_id}.pkl")
    ontology_cache_path = os.path.join(ROOT_DIR, ".cache", f"ontology_data_{os.path.basename(ontology_path)}.pkl")
    candidate_cache_path = os.path.join(ROOT_DIR, ".cache", f"candidate_data_{run_id}.pkl")
    llm_cache_path = os.path.join(ROOT_DIR, ".cache", f"llm_classify_data_{run_id}.pkl")

    os.makedirs(os.path.join(ROOT_DIR, ".cache"), exist_ok=True)

    # Removed RM/VS instance variables
    # rm_instance: RM = None
    # vs_instance: VS = None

    # === Pipeline Steps ===

    # 1. Load Data
    logger.info(f"Step 1: Loading and cleaning data from {input_path}...") # Updated log msg
    df = check_or_run(
        cache_path=load_cache_path,
        # func=load_data.load_bookmarks, # Incorrect function name
        func=load_data.load_and_clean_bookmarks, # Correct function
        use_cache=use_cache,
        force_rerun=not use_cache, # Rerun if cache disabled
        # Pass arguments matching the function signature
        data_file_path=input_path,
        initial_limit=num_rows # Renamed from num_rows
    )
    if df is None or df.empty:
        logger.error("Failed to load data. Exiting.")
        return
    logger.info(f"Loaded and cleaned {len(df)} bookmarks.") # Updated log msg
    if num_rows:
        logger.warning(f"Limiting processing to the first {num_rows} rows for testing.")

    # 2. Basic Cleaning - REMOVED (Now done in load_and_clean_bookmarks)
    # logger.info("Step 2: Cleaning text data...")
    # df = check_or_run(
    #     cache_path=clean_cache_path,
    #     func=clean_dataframe_text, # Use the imported function
    #     use_cache=use_cache,
    #     force_rerun=not use_cache,
    #     df=df,
    #     columns_to_clean=['title', 'content'] # Specify columns
    # )
    # if df is None or df.empty:
    #     logger.error("Failed during text cleaning. Exiting.")
    #     return
    # logger.info("Text cleaning complete.")

    # 3. Generate Summaries (Uses df.sem_extract) - Now Step 2
    logger.info("Step 2: Generating content summaries...") # Renumbered
    df = check_or_run(
        cache_path=summary_cache_path,
        func=summarize_data.generate_summaries,
        use_cache=use_cache,
        force_rerun=not use_cache,
        df=df,
        # text_column='content', # Summarize uses cleaned_content now
        # Note: Relies on globally configured lotus.settings.lm
    )
    if df is None:
        logger.error("Failed during summary generation. Exiting.")
        return
    # Handle cases where summarization might partially fail but adds the columns
    if 'content_summary' not in df.columns:
        logger.warning("'content_summary' column not found after generate_summaries. Proceeding without it.")
        df['content_summary'] = "" # Add empty column to prevent downstream errors
    if 'summary_and_title' not in df.columns:
        # df['summary_and_title'] = df['title'] # Fallback
        df['summary_and_title'] = df['cleaned_title'] # Use cleaned title
    if 'combined_text' not in df.columns:
        # df['combined_text'] = df['title'] + " " + df['content'] # Recreate if needed
        df['combined_text'] = df['cleaned_title'] + " " + df['cleaned_content'] # Use cleaned cols
    logger.info("Summary generation step complete.")

    # 4. Load Ontology - Now Step 3
    logger.info(f"Step 3: Loading ontology from {ontology_path}...") # Renumbered
    ontology_df = check_or_run(
        cache_path=ontology_cache_path,
        func=load_data.load_ontology,
        use_cache=use_cache,
        force_rerun=not use_cache,
        ontology_path=ontology_path
    )
    if ontology_df is None or ontology_df.empty:
        logger.error("Failed to load ontology. Exiting.")
        return
    logger.info(f"Loaded {len(ontology_df)} ontology labels.")

    # Removed RM/VS instance retrieval and checks

    # 5. Candidate Generation (Uses df.sem_sim_join) - Now Step 4
    logger.info("Step 4: Generating candidate labels...") # Renumbered
    # No longer pass rm/vs instances
    df = check_or_run(
        cache_path=candidate_cache_path,
        func=candidate_generation.generate_candidates,
        use_cache=use_cache,
        force_rerun=not use_cache,
        df=df,
        ontology_df=ontology_df,
        k_candidates=5 # Example: Get top 5 candidates
        # rm=rm_instance, # Removed
        # vs=vs_instance  # Removed
    )
    if df is None:
        logger.error("Failed during candidate generation. Exiting.")
        return
    # Check if candidate columns were added
    if 'candidate_labels' not in df.columns:
        logger.warning("'candidate_labels' column not found after generate_candidates. LLM classification might be affected.")
        # Add empty lists to prevent errors in next step if possible
        df['candidate_labels'] = [[] for _ in range(len(df))]
        df['candidate_scores'] = [[] for _ in range(len(df))]
        df['candidate_descriptions'] = [[] for _ in range(len(df))]

    logger.info("Candidate generation complete.")

    # 6. LLM Classification (Uses df.sem_extract) - Now Step 5
    logger.info("Step 5: Performing LLM-based classification...") # Renumbered
    df = check_or_run(
        cache_path=llm_cache_path,
        # func=llm_classification.classify_bookmarks_with_llm, # Incorrect function name
        func=llm_classification.perform_llm_classification, # Correct function name
        use_cache=use_cache,
        force_rerun=not use_cache,
        df=df
        # Relies on globally configured lotus.settings.lm
    )
    if df is None:
        logger.error("Failed during LLM classification. Exiting.")
        return
    logger.info("LLM classification complete.")

    # 7. Save Results - Now Step 6
    logger.info(f"Step 6: Saving enriched data to {output_path}...") # Renumbered
    # save_data.save_enriched_data(df, output_path)
    # Note: finalize_and_save requires more arguments (map, labels, hierarchy flag)
    # These are not currently available in this simplified pipeline structure.
    # For now, let's just save the current state without final hierarchy logic.
    # We might need a simpler save function or adjust the pipeline later.
    try:
        output_df = df # Use the df as is for now
        # Define essential columns (adapt as needed)
        essential_columns = [
            'id', 'url', 'title', 'content', 'created_at', # Original fields
            'cleaned_title', 'cleaned_content', # Cleaned text
            'content_summary', 'summary_and_title', 'combined_text', # Summarization/Combined
            'candidate_labels', 'candidate_scores', 'candidate_descriptions', # Candidates
            'llm_verified_labels' # LLM output (final labels for now)
        ]
        output_columns = [col for col in essential_columns if col in output_df.columns]
        output_df = output_df[output_columns]

        logger.info(f"Saving columns: {output_columns} to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in output_df.to_dict(orient='records'):
                # Ensure list fields are valid lists
                for key in ['candidate_labels', 'candidate_scores', 'candidate_descriptions', 'llm_verified_labels']:
                     if key in record and not isinstance(record[key], list):
                         record[key] = []
                import json # Local import for safety
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Enriched data saved successfully to {output_path} without final hierarchy step.")

    except Exception as e:
         logger.error(f"Error during simple save in Step 6: {e}", exc_info=True) # Renumbered

    # finalize_save.finalize_and_save(df, output_path) # Original call - replaced with simpler save

    end_time = time.time()
    logger.info(f"Modular XMLC Workflow finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Example usage:
    # python -m enriched.run_pipeline

    # --- Define Paths Relative to ROOT_DIR ---
    # DEFAULT_INPUT = os.path.join(ROOT_DIR, 'data', 'bookmarks_small.jsonl')
    DEFAULT_INPUT = os.path.join(ROOT_DIR, 'Dataset.jsonl') # Use the correct full dataset name
    DEFAULT_OUTPUT = os.path.join(ROOT_DIR, 'enriched', 'enriched_bookmarks_accessor.jsonl')
    # DEFAULT_ONTOLOGY = os.path.join(ROOT_DIR, 'data', 'ontology.jsonl')
    DEFAULT_ONTOLOGY = os.path.join(ROOT_DIR, 'Ontology.yaml') # Use the correct YAML ontology name
    NUM_ROWS_TEST = None # Process all rows

    input_file = os.getenv('INPUT_FILE', DEFAULT_INPUT)
    output_file = os.getenv('OUTPUT_FILE', DEFAULT_OUTPUT)
    ontology_file = os.getenv('ONTOLOGY_FILE', DEFAULT_ONTOLOGY)

    logger.info(f"Using Input file: {input_file}")
    logger.info(f"Using Output file: {output_file}")
    logger.info(f"Using Ontology file: {ontology_file}")
    if NUM_ROWS_TEST:
         logger.warning(f"--- RUNNING IN TEST MODE: PROCESSING ONLY {NUM_ROWS_TEST} ROWS ---  ")

    run_pipeline(
        input_path=input_file,
        output_path=output_file,
        ontology_path=ontology_file,
        num_rows=NUM_ROWS_TEST,
        use_cache=False # Disable cache for testing
    ) 