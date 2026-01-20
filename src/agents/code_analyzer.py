import ast
import astunparse
from typing import Dict, List, Any
from dataclasses import dataclass
from src.config import constants

@dataclass
class FunctionInfo:
    name: str
    args: List[str]
    returns: str
    docstring: str
    lines: tuple

@dataclass
class ClassInfo:
    name: str
    methods: List[FunctionInfo]
    bases: List[str]

class CodeAnalyzer:
    """Analyzes Python source code and extracts structural information"""
    
    def __init__(self):
        self.tree = None
        self.source_code = ""
        
    def analyze(self, source_code: str) -> Dict[str, Any]:
        """Main analysis method"""
        self.source_code = source_code
        
        try:
            self.tree = ast.parse(source_code)
            return self._extract_all_info()
        except SyntaxError as e:
            raise ValueError(f"Syntax error in source code: {e}")
    
    def _extract_all_info(self) -> Dict[str, Any]:
        """Extract all information from parsed code"""
        return {
            "functions": self._extract_functions(),
            "classes": self._extract_classes(),
            "branches": self._extract_branches(),
            "exceptions": self._extract_exceptions(),
            "imports": self._extract_imports(),
            "docstrings": self._extract_docstrings(),
        }
    
    def _extract_functions(self) -> List[Dict]:
        """Extract function definitions"""
        functions = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "returns": self._get_return_type(node),
                    "docstring": ast.get_docstring(node) or "",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                }
                functions.append(func_info)
        
        return functions
    
    def _extract_classes(self) -> List[Dict]:
        """Extract class definitions"""
        classes = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    "bases": [astunparse.unparse(base).strip() for base in node.bases],
                    "docstring": ast.get_docstring(node) or "",
                }
                classes.append(class_info)
        
        return classes
    
    def _extract_branches(self) -> List[Dict]:
        """Extract conditional branches (if/elif/else)"""
        branches = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.If):
                branch_info = {
                    "line": node.lineno,
                    "condition": astunparse.unparse(node.test).strip(),
                    "type": "if",
                }
                branches.append(branch_info)
        
        return branches
    
    def _extract_exceptions(self) -> List[Dict]:
        """Extract exception handling blocks"""
        exceptions = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    exc_info = {
                        "line": handler.lineno,
                        "exception": astunparse.unparse(handler.type).strip() if handler.type else "all",
                        "type": "try/except",
                    }
                    exceptions.append(exc_info)
        
        return exceptions
    
    def _extract_imports(self) -> List[str]:
        """Extract import statements"""
        imports = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend(f"{module}.{alias.name}" for alias in node.names)
        
        return imports
    
    def _extract_docstrings(self) -> Dict[str, str]:
        """Extract docstrings from functions and classes"""
        docstrings = {}
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstrings[node.name] = docstring
        
        return docstrings
    
    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """Extract return type annotation if present"""
        if node.returns:
            return astunparse.unparse(node.returns).strip()
        return "Any"