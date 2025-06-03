"""
Debug script for XMLC-LOTUS-real.ipynb
This script contains the code from the notebook, modified to run in a Python script.
"""

import os
import warnings
import sys

# Set API key directly
os.environ['OPENAI_API_KEY'] = "sk-dummy-key-for-testing"
print("✅ OpenAI API key set in the environment.")

# Monkey patch numpy.lib.stride_tricks to add broadcast_to
# This is a workaround for the missing function in the current numpy version
import numpy as np
if not hasattr(np.lib.stride_tricks, 'broadcast_to'):
    print("Adding broadcast_to to numpy.lib.stride_tricks")
    from numpy.lib.stride_tricks import as_strided
    
    def broadcast_to(array, shape, subok=False):
        """Broadcast an array to a new shape.
        
        This is a simplified implementation for compatibility.
        """
        array = np.array(array, copy=False, subok=subok)
        shape = tuple(shape)
        
        if array.shape == shape:
            return array
            
        # Get the broadcast shape
        strides = list(array.strides)
        for i, (old_dim, new_dim) in enumerate(zip(array.shape, shape)):
            if old_dim == 1 and new_dim != 1:
                strides[i] = 0
                
        # Add new dimensions
        for _ in range(len(shape) - len(array.shape)):
            strides.append(0)
            
        return as_strided(array, shape=shape, strides=strides)
        
    np.lib.stride_tricks.broadcast_to = broadcast_to

# Try to import core libraries
try:
    # Core libraries
    import pandas as pd
    import numpy as np
    import yaml
    from bs4 import BeautifulSoup
    import html
    import json
    from datetime import datetime
    import re
    
    # LOTUS framework
    import lotus
    
    # Suppress common warnings for cleaner execution logs
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Suppress BeautifulSoup warnings about URLs and filenames
    from bs4 import MarkupResemblesLocatorWarning
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    
    print("✅ Core libraries successfully imported.")
except ImportError as e:
    print(f"❌ Error importing libraries: {e}")
    print("Please install missing packages using pip:")
    print("pip install pandas numpy pyyaml beautifulsoup4 html5lib lotus-ai")
    sys.exit(1)

# Debug LOTUS model location
print("--- Debugging LOTUS Model Location ---")
print("dir(lotus):", dir(lotus))
if hasattr(lotus, 'models'):
    print("dir(lotus.models):", dir(lotus.models))
    # Check if Models are directly in lotus.models
    print(f"LanguageModel in lotus.models: {hasattr(lotus.models, 'LanguageModel')}")
    print(f"RetrievalModel in lotus.models: {hasattr(lotus.models, 'RetrievalModel')}")
    print(f"RerankerModel in lotus.models: {hasattr(lotus.models, 'RerankerModel')}")
else:
    print("lotus does NOT have 'models' attribute")

# Check if Models are directly in lotus
print(f"LanguageModel in lotus: {hasattr(lotus, 'LanguageModel')}")
print(f"RetrievalModel in lotus: {hasattr(lotus, 'RetrievalModel')}")
print(f"RerankerModel in lotus: {hasattr(lotus, 'RerankerModel')}")
print("--- End Model Location Debug ---")

# Configure LOTUS Framework
try:
    # Find the correct model classes based on the debug output
    if hasattr(lotus, 'models'):
        if hasattr(lotus.models, 'LM'):
            # 1. Initialize Language Model (LM)
            lm = lotus.models.LM("gpt-4o-mini")  # Balance capability/cost
            print("Using lotus.models.LM")
        elif hasattr(lotus.models, 'LanguageModel'):
            # Alternative class name
            lm = lotus.models.LanguageModel("gpt-4o-mini")
            print("Using lotus.models.LanguageModel")
        else:
            print("Could not find LM or LanguageModel in lotus.models")
            sys.exit(1)
            
        if hasattr(lotus.models, 'SentenceTransformersRM'):
            # 2. Initialize Retrieval Model (RM)
            rm = lotus.models.SentenceTransformersRM("intfloat/e5-base-v2")
            print("Using lotus.models.SentenceTransformersRM")
        elif hasattr(lotus.models, 'RetrievalModel'):
            # Alternative class name
            rm = lotus.models.RetrievalModel("intfloat/e5-base-v2")
            print("Using lotus.models.RetrievalModel")
        else:
            print("Could not find SentenceTransformersRM or RetrievalModel in lotus.models")
            sys.exit(1)
            
        if hasattr(lotus.models, 'CrossEncoderReranker'):
            # 3. Initialize Reranker Model
            reranker = lotus.models.CrossEncoderReranker("mixedbread-ai/mxbai-rerank-large-v1")
            print("Using lotus.models.CrossEncoderReranker")
        elif hasattr(lotus.models, 'RerankerModel'):
            # Alternative class name
            reranker = lotus.models.RerankerModel("mixedbread-ai/mxbai-rerank-large-v1")
            print("Using lotus.models.RerankerModel")
        else:
            print("Could not find CrossEncoderReranker or RerankerModel in lotus.models")
            sys.exit(1)
    else:
        # Try direct imports from lotus
        if hasattr(lotus, 'LM'):
            lm = lotus.LM("gpt-4o-mini")
            print("Using lotus.LM")
        elif hasattr(lotus, 'LanguageModel'):
            lm = lotus.LanguageModel("gpt-4o-mini")
            print("Using lotus.LanguageModel")
        else:
            print("Could not find LM or LanguageModel in lotus")
            sys.exit(1)
            
        if hasattr(lotus, 'SentenceTransformersRM'):
            rm = lotus.SentenceTransformersRM("intfloat/e5-base-v2")
            print("Using lotus.SentenceTransformersRM")
        elif hasattr(lotus, 'RetrievalModel'):
            rm = lotus.RetrievalModel("intfloat/e5-base-v2")
            print("Using lotus.RetrievalModel")
        else:
            print("Could not find SentenceTransformersRM or RetrievalModel in lotus")
            sys.exit(1)
            
        if hasattr(lotus, 'CrossEncoderReranker'):
            reranker = lotus.CrossEncoderReranker("mixedbread-ai/mxbai-rerank-large-v1")
            print("Using lotus.CrossEncoderReranker")
        elif hasattr(lotus, 'RerankerModel'):
            reranker = lotus.RerankerModel("mixedbread-ai/mxbai-rerank-large-v1")
            print("Using lotus.RerankerModel")
        else:
            print("Could not find CrossEncoderReranker or RerankerModel in lotus")
            sys.exit(1)

    # Register models with LOTUS
    if hasattr(lotus, 'configure'):
        lotus.configure(lm=lm, rm=rm, reranker=reranker)
        print("Using lotus.configure")
    elif hasattr(lotus, 'settings'):
        # Try to set models through settings
        if hasattr(lotus.settings, 'set_lm'):
            lotus.settings.set_lm(lm)
            print("Using lotus.settings.set_lm")
        else:
            print("Could not find set_lm in lotus.settings")
            
        if hasattr(lotus.settings, 'set_rm'):
            lotus.settings.set_rm(rm)
            print("Using lotus.settings.set_rm")
        else:
            print("Could not find set_rm in lotus.settings")
            
        if hasattr(lotus.settings, 'set_reranker'):
            lotus.settings.set_reranker(reranker)
            print("Using lotus.settings.set_reranker")
        else:
            print("Could not find set_reranker in lotus.settings")
    else:
        # Try to set global variables directly
        if hasattr(lotus, 'lm'):
            lotus.lm = lm
            print("Set lotus.lm directly")
        if hasattr(lotus, 'rm'):
            lotus.rm = rm
            print("Set lotus.rm directly")
        if hasattr(lotus, 'reranker'):
            lotus.reranker = reranker
            print("Set lotus.reranker directly")
        
        if not (hasattr(lotus, 'lm') or hasattr(lotus, 'rm') or hasattr(lotus, 'reranker')):
            print("Could not find configure in lotus or alternative configuration methods")
            sys.exit(1)

    # Verify configuration
    print(f"✅ LOTUS configured successfully with:")
    # Access model_name if it exists, otherwise handle potential AttributeError
    print(f"   - Language Model: {getattr(lm, 'model_name', 'N/A')}")
    print(f"   - Retrieval Model: {getattr(rm, 'model_name', 'N/A')}")
    print(f"   - Reranker Model: {getattr(reranker, 'model_name', 'N/A')}")

except Exception as e:
    print(f"❌ Error configuring LOTUS: {e}")
    print("Please check your API key and ensure LOTUS is properly installed.")
    print("Installation: pip install lotus-ai")
    sys.exit(1)

# Load Bookmark Data
try:
    # Define the path to the JSONL file
    data_path = "Dataset.jsonl"  # Path to the dataset
    
    # Check if file exists
    if os.path.exists(data_path):
        # Read JSONL file into a Pandas DataFrame
        df = pd.read_json(data_path, lines=True)
        
        # Verify successful loading
        print(f"✅ Data loaded successfully from {data_path}")
        print(f"   - Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display basic info
        print("\nDataFrame Info:")
        print(df.info())
        
        print("\nSample Data (first 2 rows):")
        print(df.head(2))
        
        # Check for expected columns
        expected_columns = ['id', 'url', 'source', 'title', 'content', 'created_at', 'domain', 'metadata']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\n⚠️ Warning: Missing expected columns: {missing_columns}")
        else:
            print("\n✅ All expected columns are present.")
            
    else:
        print(f"❌ Error: File not found at {data_path}")
        print("Please check the file path and ensure the file exists.")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

print("\nScript completed successfully!")