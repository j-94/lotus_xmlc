#!/usr/bin/env python3
"""
XMLC-LOTUS Script - Main Entry Point
This script sets up the environment and runs the XMLC-LOTUS process.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def ensure_data_file():
    """Ensure data-bookmark.jsonl exists (symlink to Dataset.jsonl if needed)"""
    base_dir = Path(__file__).parent.absolute()
    data_file = base_dir / "data-bookmark.jsonl"
    dataset_file = base_dir / "Dataset.jsonl"
    
    if not data_file.exists() and dataset_file.exists():
        print(f"Creating symlink from Dataset.jsonl to data-bookmark.jsonl...")
        os.symlink(dataset_file, data_file)
        print("Symlink created.")
    elif not dataset_file.exists():
        print("ERROR: Dataset.jsonl not found!")
        return False
    return True

def ensure_directory(path):
    """Ensure a directory exists"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def run_xmlc_script():
    """Run the XMLC-LOTUS script"""
    base_dir = Path(__file__).parent.absolute()
    script_path = base_dir / "enriched" / "xmlc-lotus.py"
    
    # Ensure output directories exist
    ensure_directory(base_dir / "enriched" / "index_output")
    ensure_directory(base_dir / "enriched" / "index_output" / "ontology_label_index_advanced")
    ensure_directory(base_dir / "enriched" / "index_output" / "cluster_index_advanced")
    
    print(f"Running XMLC-LOTUS script: {script_path}")
    # Set the OPENAI_API_KEY if not already set
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key with: export OPENAI_API_KEY=your-api-key")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                                check=True, 
                                cwd=base_dir)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        return False

def main():
    """Main entry point"""
    print("=== XMLC-LOTUS Runner ===")
    
    # Ensure data file exists
    if not ensure_data_file():
        return 1
    
    # Run the XMLC script
    if not run_xmlc_script():
        return 1
    
    print("=== XMLC-LOTUS process completed successfully ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())