"""
Test Generator Agent for Auto-Dev
Generates unit tests using local LLM with advanced prompting and validation
"""

from typing import Dict, Any, Optional, List
from src.models.local_llm import LocalLLM
from src.utils.logger import get_logger
import ast
import re

logger = get_logger(__name__)


class TestGenerator:
    """Generates unit tests using local LLM with validation and improvement"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Test Generator
        
        Args:
            model_name: Optional model name to use (default from config)
        """
        self.llm = LocalLLM(model_name)
        self.generated_tests = []
        self.iteration_history = []
    
    def generate(self, source_code: str, analysis: Dict[str, Any], 
                 uncovered_lines: Optional[List[int]] = None,
                 previous_errors: Optional[List[str]] = None,
                 iteration: int = 1) -> str:
        """
        Generate test code for given source code
        
        Args:
            source_code: Source code to test
            analysis: Code analysis results
            uncovered_lines: Lines not yet covered (for iterative improvement)
            previous_errors: Errors from previous test runs
            iteration: Current iteration number
            
        Returns:
            Generated test code
        """
        
        # Build context-aware prompt
        prompt = self._build_context_prompt(
            source_code, analysis, uncovered_lines, previous_errors, iteration
        )
        
        logger.info(f"Generating tests (iteration {iteration})...")
        
        # Generate test code
        test_code = self.llm.generate_tests(prompt)
        
        # Validate and improve generated code
        validated_code = self._validate_and_improve(test_code, source_code, analysis)
        
        # Track generation
        test_record = {
            "iteration": iteration,
            "code": validated_code,
            "prompt_length": len(prompt),
            "uncovered_lines": uncovered_lines or [],
            "previous_errors": previous_errors or [],
        }
        self.generated_tests.append(test_record)
        self.iteration_history.append({
            "iteration": iteration,
            "test_length": len(validated_code),
            "functions_count": self._count_test_functions(validated_code),
        })
        
        logger.info(f"Generated {self._count_test_functions(validated_code)} test functions")
        
        return validated_code
    
    def _build_context_prompt(self, source_code: str, analysis: Dict[str, Any],
                            uncovered_lines: Optional[List[int]],
                            previous_errors: Optional[List[str]],
                            iteration: int) -> str:
        """Build context-aware prompt for LLM"""
        
        # Format analysis for prompt
        analysis_text = self._format_detailed_analysis(analysis)
        
        # Build uncovered lines context
        uncovered_context = self._build_uncovered_context(uncovered_lines, analysis)
        
        # Build error context
        error_context = self._build_error_context(previous_errors, iteration)
        
        # Build instruction context based on iteration
        instruction_context = self._build_instruction_context(iteration)
        
        prompt = f"""You are an expert Python testing assistant. Generate comprehensive pytest unit tests for the following code.

ITERATION: {iteration}

SOURCE CODE:
```python
{source_code}
CODE ANALYSIS:
{analysis_text}
{uncovered_context}
{error_context}

INSTRUCTIONS:
{instruction_context}

IMPORTANT REQUIREMENTS:

Import ALL necessary modules (pytest, unittest.mock if needed)

Write test functions that start with 'test_'

Each test should have a descriptive docstring

Test both normal cases and edge cases

Mock external dependencies properly

Handle exceptions appropriately

Ensure 100% code coverage

Use pytest fixtures where beneficial

Follow Python testing best practices

Generate ONLY the complete test code with proper imports and function definitions:
"""
        

        
        return prompt
    
    def _format_detailed_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results with detailed information"""
        
        lines = []
        
        # Functions section
        if analysis.get("functions"):
            lines.append("FUNCTIONS TO TEST:")
            for func in analysis["functions"][:10]:  # Limit to first 10 functions
                func_info = []
                func_info.append(f"Name: {func.get('name', 'unknown')}")
                
                # Add signature if available
                if "signature" in func:
                    sig = func["signature"]
                    args = sig.get("args", [])
                    defaults = sig.get("defaults", {})
                    returns = sig.get("returns", "None")
                    
                    # Build argument string
                    arg_strs = []
                    for arg in args:
                        if arg in defaults:
                            arg_strs.append(f"{arg}={defaults[arg]}")
                        else:
                            arg_strs.append(arg)
                    
                    func_info.append(f"Signature: {func['name']}({', '.join(arg_strs)}) -> {returns}")
                
                # Add other info
                if "line_start" in func and "line_end" in func:
                    func_info.append(f"Lines: {func['line_start']}-{func['line_end']}")
                
                if "docstring" in func and func["docstring"]:
                    func_info.append(f"Description: {func['docstring'][:100]}...")
                
                lines.append(f"- {' | '.join(func_info)}")
        
        # Classes section
        if analysis.get("classes"):
            lines.append("\nCLASSES TO TEST:")
            for cls in analysis["classes"][:5]:  # Limit to first 5 classes
                cls_info = []
                cls_info.append(f"Name: {cls.get('name', 'unknown')}")
                
                if "methods" in cls:
                    cls_info.append(f"Methods: {', '.join(cls['methods'][:5])}")
                
                if "line_start" in cls and "line_end" in cls:
                    cls_info.append(f"Lines: {cls['line_start']}-{cls['line_end']}")
                
                lines.append(f"- {' | '.join(cls_info)}")
        
        # Branches section
        if analysis.get("branches"):
            lines.append("\nBRANCHES TO COVER:")
            for branch in analysis["branches"][:10]:  # Limit to first 10 branches
                branch_info = f"Line {branch.get('line', '?')}: {branch.get('type', 'branch')}"
                if "condition" in branch:
                    branch_info += f" - {branch['condition'][:50]}"
                lines.append(f"- {branch_info}")
        
        # Edge cases section
        if analysis.get("edge_cases"):
            lines.append("\nSUGGESTED EDGE CASES:")
            for edge_case in analysis["edge_cases"][:5]:
                if "function" in edge_case and "cases" in edge_case:
                    for case in edge_case["cases"][:2]:
                        if "description" in case:
                            lines.append(f"- {edge_case['function']}: {case['description']}")
        
        return "\n".join(lines)
    
    def _build_uncovered_context(self, uncovered_lines: Optional[List[int]], 
                               analysis: Dict[str, Any]) -> str:
        """Build context about uncovered lines"""
        
        if not uncovered_lines:
            return "\nCOVERAGE STATUS: No specific uncovered lines (first iteration)"
        
        # Group uncovered lines by function/class
        uncovered_by_function = {}
        for line in uncovered_lines[:20]:  # Limit to first 20 lines
            # Find which function/class this line belongs to
            for func in analysis.get("functions", []):
                if func.get("line_start") <= line <= func.get("line_end"):
                    func_name = func.get("name", "unknown")
                    if func_name not in uncovered_by_function:
                        uncovered_by_function[func_name] = []
                    uncovered_by_function[func_name].append(line)
                    break
        
        lines = ["\nUNCOVERED LINES (need tests):"]
        
        if uncovered_by_function:
            for func_name, lines_list in uncovered_by_function.items():
                lines_str = ", ".join(str(l) for l in sorted(lines_list)[:10])
                lines.append(f"- {func_name}: lines {lines_str}")
        else:
            lines.append(f"- Lines: {', '.join(str(l) for l in uncovered_lines[:20])}")
            if len(uncovered_lines) > 20:
                lines.append(f"- ... and {len(uncovered_lines) - 20} more lines")
        
        return "\n".join(lines)
    
    def _build_error_context(self, previous_errors: Optional[List[str]], 
                           iteration: int) -> str:
        """Build context about previous errors"""
        
        if not previous_errors or iteration == 1:
            return ""
        
        lines = ["\nPREVIOUS TEST ERRORS (fix these):"]
        
        # Categorize errors
        error_categories = {
            "import": [],
            "syntax": [],
            "assertion": [],
            "runtime": [],
            "coverage": [],
            "other": []
        }
        
        for error in previous_errors[:5]:  # Limit to last 5 errors
            error_lower = error.lower()
            if "import" in error_lower:
                error_categories["import"].append(error)
            elif "syntax" in error_lower or "indent" in error_lower:
                error_categories["syntax"].append(error)
            elif "assert" in error_lower:
                error_categories["assertion"].append(error)
            elif "runtime" in error_lower or "exception" in error_lower:
                error_categories["runtime"].append(error)
            elif "coverage" in error_lower:
                error_categories["coverage"].append(error)
            else:
                error_categories["other"].append(error)
        
        # Add categorized errors
        for category, errors in error_categories.items():
            if errors:
                lines.append(f"\n{category.upper()} ERRORS:")
                for i, error in enumerate(errors[:3], 1):
                    lines.append(f"  {i}. {error[:100]}{'...' if len(error) > 100 else ''}")
        
        return "\n".join(lines)
    
    def _build_instruction_context(self, iteration: int) -> str:
        """Build iteration-specific instructions"""
        
        base_instructions = [
            "1. Import necessary modules (pytest, unittest.mock, etc.)",
            "2. Write test functions starting with 'test_'",
            "3. Add descriptive docstrings to each test",
            "4. Test all functions and methods",
            "5. Include edge cases and boundary conditions",
            "6. Mock external dependencies properly",
            "7. Handle expected exceptions",
            "8. Use appropriate assertions",
            "9. Ensure tests are independent",
            "10. Follow pytest conventions",
        ]
        
        # Add iteration-specific guidance
        if iteration == 1:
            base_instructions.append("11. Focus on basic functionality first")
            base_instructions.append("12. Create comprehensive test suite")
        elif iteration <= 3:
            base_instructions.append("11. Fix previous errors from failed tests")
            base_instructions.append("12. Improve test coverage")
        else:
            base_instructions.append("11. Optimize and refactor existing tests")
            base_instructions.append("12. Add integration tests if needed")
        
        return "\n".join(f"- {instr}" for instr in base_instructions)
    
    def _validate_and_improve(self, test_code: str, source_code: str, 
                            analysis: Dict[str, Any]) -> str:
        """Validate and improve generated test code"""
        
        # Basic validation
        if not test_code or len(test_code.strip()) < 50:
            logger.warning("Generated test code too short, using fallback")
            return self._generate_fallback_tests(source_code, analysis)
        
        # Parse and validate syntax
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {e}")
            test_code = self._fix_syntax_errors(test_code)
        
        # Ensure required imports
        test_code = self._ensure_imports(test_code, source_code)
        
        # Ensure test functions exist
        if not self._has_test_functions(test_code):
            test_code = self._add_basic_tests(test_code, analysis)
        
        # Fix common issues
        test_code = self._fix_common_issues(test_code)
        
        # Format code
        test_code = self._format_code(test_code)
        
        return test_code
    
    def _ensure_imports(self, test_code: str, source_code: str) -> str:
        """Ensure required imports are present"""
        
        required_imports = ["import pytest"]
        
        # Check if we need unittest.mock
        if "mock" in test_code.lower() or "patch" in test_code.lower():
            required_imports.append("from unittest.mock import Mock, patch, MagicMock")
        
        # Check source code imports
        source_imports = self._extract_imports_from_source(source_code)
        for imp in source_imports:
            if imp not in test_code:
                # Add import if module is used in tests
                module_name = imp.split()[1] if "import" in imp else imp
                if module_name in test_code:
                    required_imports.append(imp)
        
        # Add missing imports
        imports_section = ""
        for imp in required_imports:
            if imp not in test_code:
                imports_section += f"{imp}\n"
        
        if imports_section:
            # Insert imports at the beginning
            lines = test_code.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    lines.insert(i, imports_section.strip())
                    break
            test_code = '\n'.join(lines)
        
        return test_code
    
    def _extract_imports_from_source(self, source_code: str) -> List[str]:
        """Extract import statements from source code"""
        
        imports = []
        lines = source_code.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        
        return imports
    
    def _has_test_functions(self, test_code: str) -> bool:
        """Check if code contains test functions"""
        
        patterns = [
            r'def test_',
            r'class Test',
            r'@pytest\.fixture',
        ]
        
        for pattern in patterns:
            if re.search(pattern, test_code):
                return True
        
        return False
    
    def _add_basic_tests(self, test_code: str, analysis: Dict[str, Any]) -> str:
        """Add basic test functions if none exist"""
        
        tests_to_add = []
        
        # Add tests for functions
        for func in analysis.get("functions", [])[:3]:
            func_name = func.get("name", "")
            if func_name:
                test_func = f"""
def test_{func_name}():
    \"\"\"Test {func_name} function\"\"\"
    # TODO: Add actual test implementation
    assert True
"""
                tests_to_add.append(test_func)
        
        # Add tests for classes
        for cls in analysis.get("classes", [])[:2]:
            cls_name = cls.get("name", "")
            if cls_name:
                test_class = f"""
def test_{cls_name.lower()}():
    \"\"\"Test {cls_name} class\"\"\"
    # TODO: Add actual test implementation
    assert True
"""
                tests_to_add.append(test_class)
        
        if tests_to_add:
            test_code += "\n\n" + "\n\n".join(tests_to_add)
        
        return test_code
    
    def _fix_common_issues(self, test_code: str) -> str:
        """Fix common issues in generated test code"""
        
        # Fix missing self in class methods
        if "class Test" in test_code:
            # Simple fix for missing self parameters
            lines = test_code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def test_') and '(self' not in line:
                    # Check if it's inside a class
                    for j in range(i-1, max(0, i-10), -1):
                        if 'class Test' in lines[j]:
                            # Add self parameter
                            lines[i] = lines[i].replace('(', '(self, ')
                            break
        
        # Fix missing assert statements
        lines = test_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def test_'):
                # Check next few lines for assert
                has_assert = False
                for j in range(i+1, min(len(lines), i+10)):
                    if 'assert' in lines[j] or 'raise' in lines[j]:
                        has_assert = True
                        break
                
                if not has_assert:
                    # Add a basic assert
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    lines.insert(i+1, f"{indent_str}    assert True")
        
        return '\n'.join(lines)
    
    def _format_code(self, test_code: str) -> str:
        """Format test code for consistency"""
        
        # Remove extra blank lines
        lines = test_code.split('\n')
        formatted_lines = []
        blank_line_count = 0
        
        for line in lines:
            if line.strip() == '':
                blank_line_count += 1
                if blank_line_count <= 2:  # Keep max 2 consecutive blank lines
                    formatted_lines.append(line)
            else:
                blank_line_count = 0
                formatted_lines.append(line)
        
        # Ensure trailing newline
        formatted_code = '\n'.join(formatted_lines)
        if not formatted_code.endswith('\n'):
            formatted_code += '\n'
        
        return formatted_code
    
    def _generate_fallback_tests(self, source_code: str, analysis: Dict[str, Any]) -> str:
        """Generate fallback tests when LLM fails"""
        
        fallback_code = '''"""
Auto-generated test suite
"""

import pytest


def test_basic():
    """Basic test to ensure test suite runs"""
    assert True
'''
        
        # Add tests for each function
        for func in analysis.get("functions", [])[:5]:
            func_name = func.get("name", "")
            if func_name:
                fallback_code += f'''

def test_{func_name}():
    """Test {func_name} function - needs implementation"""
    # TODO: Implement proper tests for {func_name}
    # Test parameters: {func.get("signature", {}).get("args", [])}
    # Return type: {func.get("signature", {}).get("returns", "unknown")}
    assert True
'''
        
        return fallback_code
    
    def _fix_syntax_errors(self, test_code: str) -> str:
        """Attempt to fix common syntax errors"""
        
        # Fix common indentation issues
        lines = test_code.split('\n')
        fixed_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('def ') or stripped.startswith('class '):
                # Reset indent for new function/class
                indent_level = 0
            elif stripped.endswith(':'):
                indent_level += 4
            elif stripped.startswith('return ') or stripped.startswith('assert '):
                # These should be indented
                pass
            
            # Apply indentation
            fixed_line = (' ' * indent_level) + stripped
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _count_test_functions(self, test_code: str) -> int:
        """Count number of test functions in code"""
        
        count = 0
        lines = test_code.split('\n')
        
        for line in lines:
            if re.match(r'^\s*def test_', line):
                count += 1
        
        return count
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about test generation"""
        
        total_tests = sum(rec.get("test_length", 0) for rec in self.iteration_history)
        total_functions = sum(rec.get("functions_count", 0) for rec in self.iteration_history)
        
        return {
            "total_iterations": len(self.generated_tests),
            "total_test_code_length": total_tests,
            "total_test_functions": total_functions,
            "average_test_length": total_tests / len(self.generated_tests) if self.generated_tests else 0,
            "iteration_details": self.iteration_history,
        }
    
    def reset(self):
        """Reset generator state"""
        self.generated_tests = []
        self.iteration_history = []


# Helper function for external use
def generate_tests_for_code(source_code: str, analysis: Dict[str, Any], 
                          model_name: Optional[str] = None) -> str:
    """
    Convenience function to generate tests for code
    
    Args:
        source_code: Source code to test
        analysis: Code analysis results
        model_name: Optional model name
        
    Returns:
        Generated test code
    """
    generator = TestGenerator(model_name)
    return generator.generate(source_code, analysis)


if __name__ == "__main__":
    # Example usage
    test_source = '''
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
    
    # Create analysis
    test_analysis = {
        "functions": [
            {
                "name": "add",
                "signature": {
                    "args": ["a", "b"],
                    "returns": "int"
                },
                "line_start": 1,
                "line_end": 3,
                "docstring": "Add two numbers"
            },
            {
                "name": "divide",
                "signature": {
                    "args": ["a", "b"],
                    "returns": "float"
                },
                "line_start": 5,
                "line_end": 10,
                "docstring": "Divide two numbers"
            }
        ]
    }
    
    generator = TestGenerator()
    tests = generator.generate(test_source, test_analysis)
    
    print("Generated tests:")
    print(tests)
    print(f"\nStats: {generator.get_generation_stats()}")