# -*- coding: utf-8 -*-
import pandas as pd
import logging
import lotus # Import lotus to access settings
# from lotus.sem_ops.sem_sim_join import SemSimJoinDataframe # Removed - Not needed if accessor called directly
# from lotus.models import RM # Removed - No longer passing RM/VS explicitly
# from lotus.vector_store import VS # Removed - No longer passing RM/VS explicitly

logger = logging.getLogger(__name__)

def generate_candidates(df: pd.DataFrame, ontology_df: pd.DataFrame, k_candidates: int) -> pd.DataFrame:
    """Generates candidate ontology labels for each bookmark using semantic similarity.

    Uses the `df.sem_sim_join` accessor, which relies on globally configured lotus.settings.

    Args:
        df: DataFrame with bookmark data, must contain 'id' and 'combined_text'.
        ontology_df: DataFrame with ontology data, must contain 'label' and 'description'.
        k_candidates: The number of top candidates to retrieve (K).

    Returns:
        The input DataFrame `df` updated with candidate information in
        'candidate_labels', 'candidate_scores', and 'candidate_descriptions' columns.
        Returns the original DataFrame if inputs are invalid or an error occurs.
    """
    if df.empty or ontology_df.empty:
        logger.warning("Input DataFrame or Ontology DataFrame is empty. Skipping candidate generation.")
        return df

    # Ensure required columns exist before proceeding
    required_left_cols = ['combined_text']
    required_right_cols = ['label', 'description']
    if not all(col in df.columns for col in required_left_cols):
        logger.error(f"Missing required columns in left DataFrame: {required_left_cols}. Skipping candidate generation.")
        return df
    if not all(col in ontology_df.columns for col in required_right_cols):
        logger.error(f"Missing required columns in right DataFrame: {required_right_cols}. Skipping candidate generation.")
        return df

    # Don't need explicit RM/VS check here as sem_sim_join uses settings directly

    # SECTION 6: Candidate Generation (Using df.sem_sim_join)
    logger.info(f"Generating top {k_candidates} candidates using df.sem_sim_join...")
    try:
        # --- DIAGNOSTIC LOGGING --- START ---
        logger.info("--- Pre-sem_sim_join Diagnostics ---")
        try:
            rm_setting = lotus.settings.rm
            vs_setting = lotus.settings.vs
            logger.info(f"lotus.settings.rm: {type(rm_setting)} - {rm_setting}")
            logger.info(f"lotus.settings.vs: {type(vs_setting)} - {vs_setting}")
        except Exception as diag_e:
            logger.error(f"Error accessing lotus.settings during diagnostics: {diag_e}")
        logger.info("--- Pre-sem_sim_join Diagnostics --- END ---")
        # --- DIAGNOSTIC LOGGING --- END ---

        # Call the accessor directly on the DataFrame
        # It implicitly uses lotus.settings.rm and lotus.settings.vs
        result_df = df.sem_sim_join(
            other=ontology_df,
            K=k_candidates,
            left_on="combined_text",
            right_on="label"
        )

        # The result_df contains columns like _left_id, _right_id, _scores{score_suffix}, and columns from both DFs
        logger.info("Aggregating candidate generation results...")

        # Expected columns in result_df: _left_id, _right_id, _scores, label, description (from other)
        required_result_cols = ['label', '_scores', '_left_id', 'description']
        if not all(col in result_df.columns for col in required_result_cols):
             logger.error(f"sem_sim_join result missing expected columns ('label', '_scores', '_left_id', 'description'). Got columns: {result_df.columns}")
             return df # Cannot proceed without essential results

        # SemSimJoinDataframe joins based on index, so group by the preserved left index (_left_id)
        grouped = result_df.sort_values(by=['_left_id', '_scores'], ascending=[True, False]).groupby('_left_id')

        # Aggregate candidates, scores, and descriptions
        candidates = grouped['label'].apply(list)
        scores = grouped['_scores'].apply(list)
        descriptions = grouped['description'].apply(list)

        # Create a mapping from the original index (_left_id) to the aggregated lists
        candidate_map = candidates.to_dict()
        score_map = scores.to_dict()
        description_map = descriptions.to_dict()

        # Map aggregated results back to the original DataFrame using its index
        # Ensure the index exists before mapping
        if df.index.name is None:
             logger.debug("DataFrame index has no name, mapping results directly.")
             map_index = df.index
        else:
            logger.debug(f"Mapping results based on index name: {df.index.name}")
            map_index = df.index

        df['candidate_labels'] = map_index.map(candidate_map).fillna("").apply(lambda x: x if isinstance(x, list) else [])
        df['candidate_scores'] = map_index.map(score_map).fillna("").apply(lambda x: x if isinstance(x, list) else [])
        df['candidate_descriptions'] = map_index.map(description_map).fillna("").apply(lambda x: x if isinstance(x, list) else [])

        logger.info("Candidate generation complete.")

    except ValueError as ve:
        # Catch the specific ValueError we saw
        if "The retrieval model must be an instance of RM" in str(ve):
             logger.error(f"ValueError in sem_sim_join: RM/VS not configured in lotus.settings when called. Check settings propagation. Error: {ve}", exc_info=True)
        else:
            logger.error(f"ValueError during candidate generation: {ve}", exc_info=True)
        return df # Return original df without candidates
    except AttributeError as ae:
        # Catch if the accessor isn't registered OR if called incorrectly
        if "'DataFrame' object has no attribute 'sem_sim_join'" in str(ae):
            logger.error(f"Lotus accessor 'sem_sim_join' not found. Ensure accessor registration is working. Error: {ae}", exc_info=True)
        else:
             logger.error(f"Attribute error during candidate generation: {ae}", exc_info=True)
        return df # Return original df without candidates
    except KeyError as ke:
        logger.error(f"Missing expected column during candidate aggregation: {ke}. Check join results.", exc_info=True)
        return df
    except Exception as e:
        logger.error(f"Error during candidate generation with sem_sim_join: {e}", exc_info=True)
        return df # Return original df without candidates

    return df 