"""
Debug script for LOTUS LM integration.
This script focuses on understanding the LOTUS LM response format and fixing the integration.
"""

import os
import sys
import json
import traceback

# Set the OpenAI API key
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    print("❌ Error: OpenAI API key not found in environment variables.")
    print("Please set your OPENAI_API_KEY environment variable before running this script.")
    print("Example: export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)
else:
    print("✅ OpenAI API key found in environment variables.")

# Import LOTUS
try:
    import lotus
    print("✅ LOTUS framework imported successfully.")
    
    # Print LOTUS version
    print(f"LOTUS version: {lotus.__version__ if hasattr(lotus, '__version__') else 'Unknown'}")
    
    # Import the models
    from lotus.models import LM, SentenceTransformersRM, CrossEncoderReranker
    print("✅ LOTUS models imported successfully.")
    
    # Initialize language model
    lm = LM()
    print("✅ Language Model initialized.")
    
    # Initialize retrieval model
    rm = SentenceTransformersRM()
    print("✅ Retrieval Model initialized.")
    
    # Initialize reranker
    reranker = CrossEncoderReranker()
    print("✅ Reranker Model initialized.")
    
    # Configure LOTUS settings
    lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)
    print("✅ LOTUS configured through settings.")
    
except ImportError as e:
    print(f"❌ Error importing LOTUS: {e}")
    sys.exit(1)

# Test the language model with a simple prompt
print("\n--- Testing LOTUS LM with a simple prompt ---")
simple_prompt = "Hello, how are you?"
try:
    print(f"Prompt: {simple_prompt}")
    response = lm(simple_prompt)
    print(f"Response type: {type(response)}")
    print(f"Response: {response}")
except Exception as e:
    print(f"❌ Error with simple prompt: {e}")
    print(traceback.format_exc())

# Test with a classification prompt
print("\n--- Testing LOTUS LM with a classification prompt ---")
classification_prompt = """
You are an expert classifier. Your task is to assign the most relevant labels to the following text.

TEXT:
This is a test about artificial intelligence and machine learning.

AVAILABLE LABELS:
1. AI_Machine_Learning: Concepts, tools, and applications related to Artificial Intelligence and Machine Learning.
2. Software_Development: Concepts, tools, and practices for building software.
3. Project_Business_Development: Ideas and strategies related to building products and businesses.

INSTRUCTIONS:
1. Analyze the text carefully.
2. Select up to 2 labels that best match the content of the text.
3. Return ONLY the label names in a comma-separated list.
4. If no labels match, return "None".

SELECTED LABELS:
"""

try:
    print(f"Prompt: {classification_prompt}")
    response = lm(classification_prompt)
    print(f"Response type: {type(response)}")
    print(f"Response: {response}")
    
    # Try to access the 'choices' attribute that's causing the error
    try:
        if hasattr(response, 'choices'):
            print(f"Response has 'choices' attribute: {response.choices}")
        else:
            print("Response does not have 'choices' attribute")
            
            # If it's a dictionary, check its keys
            if isinstance(response, dict):
                print(f"Response keys: {response.keys()}")
            
            # If it's an object, check its attributes
            print(f"Response dir: {dir(response)}")
    except Exception as attr_error:
        print(f"❌ Error accessing response attributes: {attr_error}")
        
except Exception as e:
    print(f"❌ Error with classification prompt: {e}")
    print(traceback.format_exc())

# Inspect the LM class
print("\n--- Inspecting LOTUS LM class ---")
print(f"LM class: {LM}")
print(f"LM instance: {lm}")
print(f"LM dir: {dir(lm)}")

# Try to understand the expected response format
print("\n--- Trying to understand the expected response format ---")
try:
    # Check if there's a method to get the raw response
    if hasattr(lm, 'get_raw_response'):
        raw_response = lm.get_raw_response(simple_prompt)
        print(f"Raw response: {raw_response}")
    else:
        print("LM does not have 'get_raw_response' method")
        
    # Check if there's a method to get the model
    if hasattr(lm, 'model'):
        print(f"LM model: {lm.model}")
    else:
        print("LM does not have 'model' attribute")
        
    # Check if we can access the underlying OpenAI client
    if hasattr(lm, 'client'):
        print(f"LM client: {lm.client}")
    else:
        print("LM does not have 'client' attribute")
        
except Exception as e:
    print(f"❌ Error inspecting LM: {e}")
    print(traceback.format_exc())

# Try to create a custom wrapper for the LM
print("\n--- Creating a custom wrapper for LOTUS LM ---")
try:
    def safe_lm_call(prompt):
        """Safely call the LM and handle the response."""
        try:
            response = lm(prompt)
            
            # If the response is a string, return it directly
            if isinstance(response, str):
                return response
                
            # If the response has a 'choices' attribute, extract the text
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'text'):
                    return response.choices[0].text
                elif hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
                    
            # If the response is a dictionary with 'choices' key
            if isinstance(response, dict) and 'choices' in response:
                if 'text' in response['choices'][0]:
                    return response['choices'][0]['text']
                elif 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                    return response['choices'][0]['message']['content']
                    
            # If we can't extract the text, return the response as a string
            return str(response)
            
        except Exception as e:
            print(f"❌ Error in safe_lm_call: {e}")
            return None
            
    # Test the safe wrapper
    print("Testing safe_lm_call with simple prompt...")
    safe_response = safe_lm_call(simple_prompt)
    print(f"Safe response: {safe_response}")
    
    print("Testing safe_lm_call with classification prompt...")
    safe_classification_response = safe_lm_call(classification_prompt)
    print(f"Safe classification response: {safe_classification_response}")
    
except Exception as e:
    print(f"❌ Error creating custom wrapper: {e}")
    print(traceback.format_exc())

print("\nDebugging completed.")