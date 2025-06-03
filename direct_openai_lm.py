"""
Direct OpenAI LM implementation for LOTUS.
This script creates a custom LM class that directly uses the OpenAI API.
"""

import os
import sys
import traceback
from openai import OpenAI

# Set the OpenAI API key
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    print("❌ Error: OpenAI API key not found in environment variables.")
    print("Please set your OPENAI_API_KEY environment variable before running this script.")
    print("Example: export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)
else:
    print("✅ OpenAI API key found in environment variables.")
    os.environ['OPENAI_API_KEY'] = api_key

# Import LOTUS
try:
    import lotus
    from lotus.models import LM, SentenceTransformersRM, CrossEncoderReranker
    print("✅ LOTUS imported successfully.")
    
    # Create a direct OpenAI LM class
    class DirectOpenAILM(LM):
        """
        Direct OpenAI LM implementation for LOTUS.
        This class bypasses the LOTUS LM implementation and directly uses the OpenAI API.
        """
        
        def __init__(self, model="gpt-3.5-turbo", temperature=0.3, max_tokens=1000):
            """Initialize the DirectOpenAILM."""
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.client = OpenAI(api_key=api_key)
            print(f"✅ DirectOpenAILM initialized with model: {model}")
        
        def __call__(self, prompt, **kwargs):
            """Call the OpenAI API directly."""
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
                
                # Extract the response text
                response_text = response.choices[0].message.content.strip()
                return response_text
                
            except Exception as e:
                print(f"❌ Error calling OpenAI API: {e}")
                print(traceback.format_exc())
                return "I'm sorry, I couldn't process that request."
    
    # Test the DirectOpenAILM class
    print("\n--- Testing DirectOpenAILM ---")
    try:
        # Initialize the DirectOpenAILM
        direct_lm = DirectOpenAILM()
        
        # Test with a simple prompt
        simple_prompt = "Hello, how are you?"
        print(f"Testing with simple prompt: {simple_prompt}")
        response = direct_lm(simple_prompt)
        print(f"Response: {response}")
        
        # Test with a classification prompt
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
        
        print(f"\nTesting with classification prompt...")
        response = direct_lm(classification_prompt)
        print(f"Classification response: {response}")
        
    except Exception as e:
        print(f"❌ Error testing DirectOpenAILM: {e}")
        print(traceback.format_exc())
    
    # Configure LOTUS settings with the DirectOpenAILM
    print("\n--- Configuring LOTUS with DirectOpenAILM ---")
    try:
        # Initialize components
        direct_lm = DirectOpenAILM()
        rm = SentenceTransformersRM()
        reranker = CrossEncoderReranker()
        
        # Configure settings
        lotus.settings.configure(lm=direct_lm, rm=rm, reranker=reranker)
        print("✅ LOTUS configured with DirectOpenAILM")
        
    except Exception as e:
        print(f"❌ Error configuring LOTUS with DirectOpenAILM: {e}")
        print(traceback.format_exc())
    
except ImportError as e:
    print(f"❌ Error importing LOTUS: {e}")
    sys.exit(1)

print("\nDirectOpenAILM setup completed.")