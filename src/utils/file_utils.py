from pathlib import Path
from typing import List
import shutil

def ensure_directory(path: Path) -> Path:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def clean_directory(path: Path) -> None:
    """Clean directory contents"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory"""
    return list(directory.glob("**/*.py"))