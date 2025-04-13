# -*- coding: utf-8 -*-
import yaml
import pandas as pd
import logging
from . import utils # Use relative import within the 'enriched' package

logger = logging.getLogger(__name__)

def load_and_prepare_ontology(ontology_file_path: str) -> tuple[pd.DataFrame | None, dict | None, list | None]:
    """Loads the ontology from a YAML file, extracts labels/descriptions/paths,
       and prepares it as a DataFrame and a label-to-path map.

    Args:
        ontology_file_path: Path to the Ontology.yaml file.

    Returns:
        A tuple containing:
        - ontology_df: DataFrame with columns ['label', 'description', 'path'].
        - label_to_path_map: Dictionary mapping labels to their hierarchical paths.
        - all_labels: A list of all unique labels extracted.
        Returns (None, None, None) if loading or processing fails.
    """
    logger.info(f"Loading ontology from {ontology_file_path}...")
    try:
        with open(ontology_file_path, 'r', encoding='utf-8') as f:
            ontology_data = yaml.safe_load(f)
        if not ontology_data:
            logger.error(f"Ontology file {ontology_file_path} is empty or invalid.")
            return None, None, None

        # SECTION 5: Load Ontology & Prepare for Matching
        logger.info("Extracting labels, descriptions, and paths from ontology...")
        extracted_data = utils.extract_ontology_labels(ontology_data)
        if not extracted_data:
            logger.warning("No labels extracted from the ontology structure.")
            # Return empty structures instead of None if extraction yields nothing?
            # For now, returning None to signal potential issue.
            return None, None, None

        # Create DataFrame from extracted data
        ontology_df = pd.DataFrame(extracted_data, columns=['label', 'description', 'path'])
        ontology_df['label'] = ontology_df['label'].astype(str) # Ensure string type
        ontology_df['description'] = ontology_df['description'].astype(str).fillna('No description')
        ontology_df['path'] = ontology_df['path'].astype(str).fillna('')

        # Ensure unique labels if necessary (the extractor might produce duplicates if structure allows)
        initial_count = len(ontology_df)
        ontology_df = ontology_df.drop_duplicates(subset=['label'])
        if len(ontology_df) < initial_count:
            logger.warning(f"Removed {initial_count - len(ontology_df)} duplicate labels from ontology.")

        # Create helper structures
        label_to_path_map = pd.Series(ontology_df.path.values, index=ontology_df.label).to_dict()
        all_labels = ontology_df['label'].tolist()

        logger.info(f"Loaded and prepared {len(ontology_df)} unique ontology labels.")
        return ontology_df, label_to_path_map, all_labels

    except FileNotFoundError:
        logger.error(f"Ontology file not found: {ontology_file_path}")
        return None, None, None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing ontology YAML file {ontology_file_path}: {e}", exc_info=True)
        return None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during ontology processing: {e}", exc_info=True)
        return None, None, None 