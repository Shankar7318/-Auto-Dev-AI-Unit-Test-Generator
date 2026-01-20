

import ast
from typing import Dict, List, Any, Optional, Tuple
import re


class CodeParser:
    """
    Python code parser using AST
    
    Features:
    - Extract functions and classes
    - Analyze control flow (branches, loops, exceptions)
    - Calculate code metrics
    - Generate testable paths
    """
    
    def __init__(self):
        self.tree = None
        self.source_lines = []
        
    def parse(self, source_code: str) -> Dict[str, Any]:
        """
        Parse Python source code and return comprehensive analysis
        
        Args:
            source_code: Python source code as string
            
        Returns:
            Dictionary with parsed information
        """
        self.source_lines = source_code.split('\n')
        
        try:
            self.tree = ast.parse(source_code)
            return self._analyze_code()
            
        except SyntaxError as e:
            raise ValueError(f"Syntax error in source code: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse code: {e}")
    
    def _analyze_code(self) -> Dict[str, Any]:
        """Analyze the parsed AST tree"""
        
        functions = self._extract_functions()
        classes = self._extract_classes()
        imports = self._extract_imports()
        branches = self._extract_branches()
        exceptions = self._extract_exceptions()
        
        complexity_score = self._calculate_complexity()
        
        return {
            "summary": {
                "total_lines": len(self.source_lines),
                "functions_count": len(functions),
                "classes_count": len(classes),
                "branches_count": len(branches),
                "imports_count": len(imports),
                "complexity_score": complexity_score,
            },
            "functions": functions,
            "classes": classes,
            "branches": branches,
            "imports": imports,
            "exceptions": exceptions,
            "testable_paths": self._generate_testable_paths(functions, branches, exceptions),
            "edge_cases": self._generate_edge_cases(functions),
        }
    
    def _extract_functions(self) -> List[Dict[str, Any]]:
        """Extract function definitions"""
        functions = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = {
                    "name": node.name,
                    "args": self._extract_function_args(node),
                    "returns": self._extract_return_type(node),
                    "docstring": ast.get_docstring(node) or "",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                }
                functions.append(func_info)
        
        return functions
    
    def _extract_function_args(self, node) -> List[str]:
        """Extract function arguments"""
        args = []
        if hasattr(node, 'args'):
            for arg in node.args.args:
                args.append(arg.arg)
        return args
    
    def _extract_return_type(self, node) -> str:
        """Extract return type annotation"""
        if hasattr(node, 'returns') and node.returns:
            try:
                return ast.unparse(node.returns)
            except:
                return "Any"
        return "Any"
    
    def _extract_classes(self) -> List[Dict[str, Any]]:
        """Extract class definitions"""
        classes = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                
                class_info = {
                    "name": node.name,
                    "methods": methods,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "docstring": ast.get_docstring(node) or "",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                }
                classes.append(class_info)
        
        return classes
    
    def _extract_imports(self) -> List[Dict[str, Any]]:
        """Extract import statements"""
        imports = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "module": alias.name,
                        "alias": alias.asname,
                        "is_from_import": False,
                        "line": node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "is_from_import": True,
                        "line": node.lineno,
                    })
        
        return imports
    
    def _extract_branches(self) -> List[Dict[str, Any]]:
        """Extract conditional branches (if/elif/else)"""
        branches = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.If):
                try:
                    condition = ast.unparse(node.test)
                except:
                    condition = "if condition"
                
                branch_info = {
                    "line": node.lineno,
                    "condition": condition,
                    "type": "if",
                }
                branches.append(branch_info)
            
            elif isinstance(node, (ast.For, ast.While)):
                loop_type = "for" if isinstance(node, ast.For) else "while"
                branch_info = {
                    "line": node.lineno,
                    "condition": loop_type,
                    "type": "loop",
                }
                branches.append(branch_info)
        
        return branches
    
    def _extract_exceptions(self) -> List[Dict[str, Any]]:
        """Extract exception handling blocks"""
        exceptions = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type:
                        try:
                            exc_type = ast.unparse(handler.type)
                        except:
                            exc_type = "Exception"
                    else:
                        exc_type = "BaseException"
                    
                    exc_info = {
                        "line": handler.lineno,
                        "exception_type": exc_type,
                        "type": "try/except",
                    }
                    exceptions.append(exc_info)
        
        return exceptions
    
    def _calculate_complexity(self) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _generate_testable_paths(self, functions: List[Dict], branches: List[Dict], 
                               exceptions: List[Dict]) -> List[Dict[str, Any]]:
        """Generate testable execution paths through the code"""
        
        testable_paths = []
        
        for func in functions:
            paths = []
            
            # Base path
            paths.append({
                "description": f"Base execution of {func['name']}",
                "conditions": [],
                "line_coverage": [func['line_start']],
                "expected_outcome": "Returns expected value",
            })
            
            # Add branch paths
            func_branches = [b for b in branches if func['line_start'] <= b['line'] <= func['line_end']]
            for branch in func_branches[:5]:  # Limit to 5 branches per function
                paths.append({
                    "description": f"{func['name']}: {branch['type']} branch at line {branch['line']}",
                    "condition": branch['condition'],
                    "line_coverage": [func['line_start'], branch['line']],
                    "expected_outcome": f"Executes {branch['type']} branch",
                })
            
            # Add exception paths
            func_exceptions = [e for e in exceptions if func['line_start'] <= e['line'] <= func['line_end']]
            for exc in func_exceptions[:3]:  # Limit to 3 exceptions per function
                paths.append({
                    "description": f"{func['name']}: Raises {exc['exception_type']}",
                    "condition": f"Trigger {exc['exception_type']}",
                    "line_coverage": [func['line_start'], exc['line']],
                    "expected_outcome": f"Raises {exc['exception_type']}",
                })
            
            testable_paths.append({
                "function": func['name'],
                "paths": paths,
                "total_paths": len(paths),
            })
        
        return testable_paths
    
    def _generate_edge_cases(self, functions: List[Dict]) -> List[Dict[str, Any]]:
        """Generate potential edge cases for testing"""
        
        edge_cases = []
        
        for func in functions:
            cases = []
            func_name = func['name']
            
            # Always add null input case
            cases.append({
                "type": "null_input",
                "description": f"Pass None to {func_name}",
                "test_input": "None for parameters",
                "expected": "TypeError or handled appropriately",
            })
            
            # Add edge cases based on function name or signature
            args = func.get('args', [])
            for arg in args:
                arg_lower = arg.lower()
                if any(keyword in arg_lower for keyword in ['list', 'array', 'collection']):
                    cases.append({
                        "type": "empty_collection",
                        "description": f"Empty collection input for {arg} in {func_name}",
                        "test_input": f"Empty list/dict/set for {arg}",
                        "expected": "Handles empty collection",
                    })
                
                if any(keyword in arg_lower for keyword in ['limit', 'size', 'count', 'index']):
                    cases.append({
                        "type": "boundary_value",
                        "description": f"Boundary values for {arg} in {func_name}",
                        "test_input": f"0, 1, max_value for {arg}",
                        "expected": "Handles boundaries correctly",
                    })
            
            if cases:
                edge_cases.append({
                    "function": func_name,
                    "cases": cases,
                })
        
        return edge_cases


# Helper functions for external use

def parse_code(source_code: str) -> Dict[str, Any]:
    """Convenience function to parse code"""
    parser = CodeParser()
    return parser.parse(source_code)


def parse_file(filepath: str) -> Dict[str, Any]:
    """Convenience function to parse file"""
    import ast
    with open(filepath, 'r') as f:
        source_code = f.read()
    parser = CodeParser()
    return parser.parse(source_code)


def extract_functions(source_code: str) -> List[Dict[str, Any]]:
    """Extract only function information"""
    parser = CodeParser()
    result = parser.parse(source_code)
    return result.get("functions", [])


def extract_classes(source_code: str) -> List[Dict[str, Any]]:
    """Extract only class information"""
    parser = CodeParser()
    result = parser.parse(source_code)
    return result.get("classes", [])


def calculate_complexity(source_code: str) -> int:
    """Calculate overall code complexity"""
    parser = CodeParser()
    result = parser.parse(source_code)
    return result.get("summary", {}).get("complexity_score", 0)


if __name__ == "__main__":
    # Test the parser
    test_code = '''
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

class Calculator:
    """Simple calculator class"""
    
    def multiply(self, x: int, y: int) -> int:
        """Multiply two numbers"""
        return x * y
'''
    
    parser = CodeParser()
    result = parser.parse(test_code)
    
    print("Parsing Result:")
    print(f"Functions: {result['summary']['functions_count']}")
    print(f"Classes: {result['summary']['classes_count']}")
    print(f"Complexity: {result['summary']['complexity_score']}")