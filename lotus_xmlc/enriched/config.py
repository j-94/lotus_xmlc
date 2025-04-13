# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Ensures API keys etc. are available environment-wide
load_dotenv()

# --- LOTUS Model Configuration ---
# Fetch model names from environment variables with defaults
LM_MODEL = os.environ.get("LOTUS_LM_MODEL", "gpt-4o-mini")
RM_MODEL = os.environ.get("LOTUS_RM_MODEL", "intfloat/e5-base-v2")
RERANKER_MODEL = os.environ.get("LOTUS_RERANKER_MODEL", "mixedbread-ai/mxbai-rerank-large-v1")

# --- File Paths ---
# Determine the base directory of this config file, to construct other paths relatively
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct paths relative to the base directory
# Assumes 'Dataset.jsonl' and 'Ontology.yaml' are one level up from 'enriched'
DATA_FILE = os.path.join(BASE_DIR, "../Dataset.jsonl")
ONTOLOGY_FILE = os.path.join(BASE_DIR, "../Ontology.yaml")
# Output files will be placed within the 'enriched' directory
OUTPUT_FILE = os.path.join(BASE_DIR, "enriched_bookmarks_accessor.jsonl")
SUGGESTIONS_FILE = os.path.join(BASE_DIR, "ontology_suggestions_accessor.jsonl") # For potential future use

# --- Workflow Parameters ---
K_CANDIDATES = 15 # Number of candidates to generate via semantic similarity
ENFORCE_HIERARCHY = False # Whether to add parent labels to LLM-verified labels
# RUN_ONTOLOGY_EXPANSION = True # Keep commented out, relates to unused clustering
# NUM_CLUSTERS_EXPANSION = 20 # Keep commented out

# --- Logging & Warnings ---
import logging
import warnings
from bs4 import MarkupResemblesLocatorWarning

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Filter specific warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # General future warnings

# --- Indexing (Commented out as not directly used by accessors) ---
# INDEX_BASE_DIR = os.path.join(BASE_DIR, "index_output")
# ONTOLOGY_INDEX_DIR = os.path.join(INDEX_BASE_DIR, "ontology_label_index_accessor")
# CLUSTER_INDEX_DIR = os.path.join(INDEX_BASE_DIR, "cluster_index_accessor") 