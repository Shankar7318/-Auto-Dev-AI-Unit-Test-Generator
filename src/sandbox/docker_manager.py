import docker
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DockerManager:
    """Manages Docker containers for safe test execution"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.timeout = Settings.DOCKER_TIMEOUT
    
    def run_tests(self, source_code: str, test_code: str) -> Dict[str, Any]:
        """Execute tests in Docker container"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file
            source_file = Path(tmpdir) / "source.py"
            source_file.write_text(source_code)
            
            # Create test file
            test_file = Path(tmpdir) / "test_source.py"
            test_code = self._prepare_test_code(test_code, "source")
            test_file.write_text(test_code)
            
            # Create requirements.txt
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("pytest\npytest-cov\ncoverage")
            
            # Run container
            result = self._run_container(tmpdir)
            
            return result
    
    def _prepare_test_code(self, test_code: str, module_name: str) -> str:
        """Prepare test code for execution"""
        
        # Add import for the source module
        import_statement = f"import {module_name}\n"
        
        # Ensure pytest is imported
        if "import pytest" not in test_code:
            import_statement += "import pytest\n\n"
        
        return import_statement + test_code
    
    def _run_container(self, tmpdir: str) -> Dict[str, Any]:
        """Run Docker container with tests"""
        
        try:
            # Build Dockerfile
            dockerfile_content = f"""
FROM {Settings.DOCKER_IMAGE}
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
"""
            
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text(dockerfile_content)
            
            # Build image
            image, build_logs = self.client.images.build(
                path=tmpdir,
                tag="auto-dev-test",
                rm=True,
                timeout=self.timeout
            )
            
            # Run container
            container = self.client.containers.run(
                image.id,
                command="python -m pytest test_source.py --cov=source --cov-report=json --cov-report=term",
                detach=False,
                remove=True,
                stdout=True,
                stderr=True,
                mem_limit="256m",
                network_mode="none",  # No network access
            )
            
            # Parse output
            output = container.decode('utf-8') if isinstance(container, bytes) else str(container)
            
            return {
                "success": "FAILED" not in output and "ERROR" not in output,
                "output": output,
                "raw_output": output,
            }
            
        except docker.errors.ContainerError as e:
            logger.error(f"Container error: {e}")
            return {
                "success": False,
                "output": str(e),
                "error": "ContainerError",
            }
        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            return {
                "success": False,
                "output": str(e),
                "error": "ExecutionError",
            }
