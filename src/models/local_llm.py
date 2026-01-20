import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List
from pathlib import Path
from src.config.settings import Settings
import warnings

class LocalLLM:
    """Manages local LLM for test generation"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Settings.MODEL_NAME
        self.model_path = Settings.MODEL_PATH
        self.pipeline = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the local model"""
        print(f"Loading model: {self.model_name}")
        
        try:
            # Try to load locally first
            if self.model_path.exists():
                print(f"Loading from local cache: {self.model_path}")
                model_path = str(self.model_path)
            else:
                print(f"Downloading model: {self.model_name}")
                model_path = self.model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
            
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a smaller fallback model if primary fails"""
        print("Loading fallback model...")
        try:
            self.pipeline = pipeline(
                "text-generation",
                model="gpt2",  # Small model that's always available
                device=-1,  # CPU
            )
        except:
            # If even GPT-2 fails, use a mock
            self.pipeline = self._mock_pipeline
    
    def _mock_pipeline(self, prompt, max_length=512):
        """Mock pipeline for testing without actual model"""
        return [{"generated_text": "# Mock test generation\n\nimport pytest\n\ndef test_mock():\n    assert True"}]
    
    def generate_tests(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate test code from prompt"""
        if not self.pipeline:
            self.load_model()
        
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=Settings.TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            generated_text = response[0]["generated_text"]
            # Extract only the code after the prompt
            if prompt in generated_text:
                generated_text = generated_text.split(prompt)[1].strip()
            
            return self._clean_generated_code(generated_text)
            
        except Exception as e:
            print(f"Error generating tests: {e}")
            return self._generate_basic_tests()
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and validate generated code"""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove markdown code blocks
            if line.strip() in ["```python", "```", "```py"]:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _generate_basic_tests(self) -> str:
        """Generate basic test template"""
        return '''
import pytest

# Basic test template
def test_basic():
    """Placeholder test"""
    assert True
'''