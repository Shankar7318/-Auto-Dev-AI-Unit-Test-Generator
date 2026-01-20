import json
from typing import Dict, Any
from pathlib import Path
from src.sandbox.docker_manager import DockerManager
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TestRunner:
    """Runs tests and collects results"""
    
    def __init__(self):
        self.docker = DockerManager()
        self.results = []
    
    def execute(self, source_code: str, test_code: str) -> Dict[str, Any]:
        """Execute tests and return results"""
        
        logger.info("Running tests in Docker...")
        docker_result = self.docker.run_tests(source_code, test_code)
        
        # Save test file
        self._save_test_file(test_code)
        
        # Parse coverage if available
        coverage_data = self._parse_coverage()
        
        result = {
            "success": docker_result["success"],
            "output": docker_result["output"],
            "coverage": coverage_data,
            "timestamp": "now",  # Add actual timestamp
        }
        
        self.results.append(result)
        return result
    
    def _save_test_file(self, test_code: str):
        """Save generated test file"""
        test_dir = Settings.TESTS_DIR
        test_count = len(list(test_dir.glob("test_*.py"))) + 1
        
        test_file = test_dir / f"test_generated_{test_count}.py"
        test_file.write_text(test_code)
        
        logger.info(f"Test saved: {test_file}")
    
    def _parse_coverage(self) -> Dict[str, Any]:
        """Parse coverage report from JSON file"""
        coverage_file = Settings.COVERAGE_DIR / "coverage.json"
        
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Failed to parse coverage JSON")
        
        return {}