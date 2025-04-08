# This file will help fix the import and configure issues in the notebook

import nbformat as nbf
import re # Import re for potential use, although the simple replace should work

notebook_path = 'XMLC-LOTUS-real.ipynb'

# Read the notebook
try:
    with open(notebook_path, 'r') as f:
        notebook = nbf.read(f, as_version=4)
except FileNotFoundError:
    print(f"Error: Notebook file not found at {notebook_path}")
    exit()
except Exception as e:
    print(f"Error reading notebook: {e}")
    exit()

import_fixed = False
configure_fixed = False

# Iterate through cells to fix issues
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        # Fix the import issue in the library import cell
        if 'from lotus.settings import configure' in cell['source']:
            cell['source'] = cell['source'].replace(
                'from lotus.settings import configure',
                '# configure removed - use lotus.configure() instead' # More informative comment
            )
            import_fixed = True
            print("Found and fixed the import statement.")

        # Fix the configure usage in the configuration cell
        if 'lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)' in cell['source']:
            cell['source'] = cell['source'].replace(
                'lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)',
                'lotus.configure(lm=lm, rm=rm, reranker=reranker)' # Use top-level configure
            )
            configure_fixed = True
            print("Found and fixed the configure() call.")

# Write the updated notebook
try:
    with open(notebook_path, 'w') as f:
        nbf.write(notebook, f)
    if import_fixed or configure_fixed:
        print(f"Successfully updated {notebook_path}.")
    else:
        print("No changes needed for import or configure calls.")
except Exception as e:
    print(f"Error writing updated notebook: {e}")
