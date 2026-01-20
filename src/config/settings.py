import os
from pathlib import Path
from typing import Optional

class Settings:
    """Application settings and configurations"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    SRC_DIR = BASE_DIR / "src"
    TESTS_DIR = BASE_DIR / "generated_tests"
    COVERAGE_DIR = BASE_DIR / "coverage_reports"
    
    # Model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "codellama/CodeLlama-7b-Instruct-hf")
    MODEL_PATH: Path = BASE_DIR / "models" / MODEL_NAME
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.1
    
    # Test settings
    TARGET_COVERAGE: int = 100
    MAX_ITERATIONS: int = 10
    TEST_TIMEOUT: int = 30
    
    # Docker settings
    DOCKER_IMAGE: str = "python:3.10-slim"
    DOCKER_TIMEOUT: int = 60
    
    # File patterns
    PYTHON_FILES = ["*.py"]
    TEST_FILE_PREFIX = "test_"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [cls.TESTS_DIR, cls.COVERAGE_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)