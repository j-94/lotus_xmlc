# XMLC-LOTUS: Bookmark Data Enrichment with Ontology Labels

This project implements a pipeline for enriching bookmark data (JSONL) using a custom ontology (YAML) and the LOTUS framework. The goal is to prepare data for eXtreme Multi-Label Classification (XMLC) and potential Knowledge Graph construction.

## Overview

The project consists of the following components:

1. **Interactive Script**: `xmlc_processor.py` - Processes a sample of the dataset and allows for optional full dataset processing.
2. **Batch Processor**: `batch_processor.py` - Processes the entire dataset without user interaction.
3. **Jupyter Notebook**: `XMLC-LOTUS-real.ipynb` - The original notebook with detailed explanations.

## Data Files

- `Dataset.jsonl` - The input dataset containing bookmark data.
- `Ontology.yaml` - The ontology file containing the label hierarchy.
- `enriched_sample.csv` - The output file containing the sample dataset with assigned labels.
- `enriched_dataset.csv` - The output file containing the full dataset with assigned labels (created by the batch processor).

## Usage

### Interactive Script

```bash
python xmlc_processor.py
```

This script will:
1. Load and clean the dataset
2. Extract the ontology labels
3. Process a sample of the dataset
4. Optionally process the full dataset if requested

### Batch Processor

```bash
python batch_processor.py
```

This script will:
1. Load and clean the dataset
2. Extract the ontology labels
3. Process the entire dataset in batches
4. Save the results to `enriched_dataset.csv`

## Label Assignment Algorithm

The label assignment algorithm uses a two-tier approach:

### Primary Method: OpenAI-powered Classification

The primary method uses LOTUS's language model (powered by OpenAI) to perform classification:

1. A prompt is constructed with:
   - The text to classify
   - Available labels and their descriptions
   - Instructions for classification
2. The language model analyzes the text and returns the most relevant labels
3. The response is parsed to extract valid label names

### Fallback Method: Text Matching

If the language model fails (e.g., due to API issues or no matches found), a text matching approach is used:

1. For each label in the ontology:
   - Extract significant terms from the label description
   - Extract significant terms from the label name (handling underscores and camel case)
   - Check if any of these terms appear in the bookmark text
   - If matches are found, assign the label to the bookmark

## Results

The script produces a CSV file with the following columns:
- All original columns from the input dataset
- `cleaned_title` - The cleaned title text
- `cleaned_content` - The cleaned content text
- `combined_text` - The combined title and content text
- `existing_tags` - Any tags that were already present in the metadata
- `assigned_labels` - The labels assigned by the algorithm

## Future Improvements

1. **Embedding-Based Matching**: Replace the text matching approach with an embedding-based approach for more accurate label assignment.
2. **Hierarchical Label Assignment**: Leverage the hierarchical structure of the ontology to assign more specific labels.
3. **Label Correlation Analysis**: Analyze correlations between labels to improve the assignment algorithm.
4. **User Interface**: Create a web interface for exploring the enriched dataset.
5. **Knowledge Graph Construction**: Use the assigned labels to construct a knowledge graph of the bookmarks.

## Dependencies

- pandas
- numpy
- pyyaml
- beautifulsoup4
- html5lib
- lotus-ai
- tqdm

## API Key Requirements

This project requires an OpenAI API key to use the language model capabilities of LOTUS. You can provide your API key in one of two ways:

1. **Environment Variable**: Set the `OPENAI_API_KEY` environment variable before running the scripts:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Interactive Input**: The interactive script (`xmlc_processor.py`) will prompt you for your API key if it's not found in the environment.

Note: The batch processor (`batch_processor.py`) requires the API key to be set as an environment variable.

## License

This project is licensed under the MIT License - see the LICENSE file for details.