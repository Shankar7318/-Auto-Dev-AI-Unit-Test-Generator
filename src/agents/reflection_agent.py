import re
from typing import Dict, List, Any, Optional
from src.models.local_llm import LocalLLM
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ReflectionAgent:
    """Analyzes test failures and suggests fixes"""
    
    def __init__(self):
        self.llm = LocalLLM()
        self.reflection_history = []
    
    def reflect(self, test_code: str, error_output: str, 
                coverage_gaps: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        """Analyze failure and generate fix plan"""
        
        logger.info("Reflecting on test failure...")
        
        # Analyze error
        error_analysis = self._analyze_error(error_output)
        
        # Generate fix
        fix_suggestion = self._generate_fix_suggestion(
            test_code, error_analysis, coverage_gaps, source_code
        )
        
        reflection = {
            "error_analysis": error_analysis,
            "fix_suggestion": fix_suggestion,
            "new_test_code": fix_suggestion.get("fixed_test_code", test_code),
            "changes_made": fix_suggestion.get("changes", []),
        }
        
        self.reflection_history.append(reflection)
        return reflection
    
    def _analyze_error(self, error_output: str) -> Dict[str, Any]:
        """Parse and analyze error output"""
        
        error_types = {
            "AssertionError": "Test assertion failed",
            "ImportError": "Module import failed",
            "SyntaxError": "Syntax error in test code",
            "NameError": "Undefined name used",
            "TypeError": "Wrong data type used",
            "AttributeError": "Attribute doesn't exist",
            "ValueError": "Incorrect value",
            "KeyError": "Missing dictionary key",
            "IndexError": "List index out of range",
        }
        
        # Extract error type and message
        error_type = "UnknownError"
        error_message = error_output
        
        for err_type in error_types.keys():
            if err_type in error_output:
                error_type = err_type
                # Extract the line with error
                lines = error_output.split('\n')
                for line in lines:
                    if err_type in line:
                        error_message = line.strip()
                        break
                break
        
        # Extract line number if available
        line_number = None
        line_match = re.search(r'line (\d+)', error_output)
        if line_match:
            line_number = int(line_match.group(1))
        
        return {
            "type": error_type,
            "message": error_message,
            "line_number": line_number,
            "description": error_types.get(error_type, "Unknown error"),
        }
    
    def _generate_fix_suggestion(self, test_code: str, error_analysis: Dict[str, Any],
                                coverage_gaps: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        """Generate fix suggestion using LLM"""
        
        prompt = self._build_reflection_prompt(
            test_code, error_analysis, coverage_gaps, source_code
        )
        
        llm_response = self.llm.generate_tests(prompt, max_tokens=1024)
        
        # Extract fixed code from response
        fixed_code = self._extract_fixed_code(llm_response, test_code)
        
        # Analyze changes
        changes = self._analyze_changes(test_code, fixed_code)
        
        return {
            "fixed_test_code": fixed_code,
            "changes": changes,
            "analysis": error_analysis,
        }
    
    def _build_reflection_prompt(self, test_code: str, error_analysis: Dict[str, Any],
                                coverage_gaps: Dict[str, Any], source_code: str) -> str:
        """Build reflection prompt for LLM"""
        
        missing_lines = coverage_gaps.get("missing_lines", [])
        missing_branches = coverage_gaps.get("missing_branches", [])
        
        prompt = f"""Fix this failing test:

ORIGINAL TEST CODE:
```python
{test_code}
ERROR:
{error_analysis['type']}: {error_analysis['message']}

SOURCE CODE BEING TESTED:

python
{source_code}
"""

        if missing_lines:
            prompt += f"\nMISSING LINES TO COVER: {missing_lines}"
        
        if missing_branches:
            prompt += f"\nMISSING BRANCHES: {len(missing_branches)} branches not covered"
        
        prompt += """

INSTRUCTIONS:
1. Fix the test to pass
2. Cover missing lines if possible
3. Keep the same test structure
4. Add comments explaining fixes
5. Return ONLY the fixed test code

FIXED TEST CODE:
```python
"""
        
        return prompt
    
    def _extract_fixed_code(self, llm_response: str, original_code: str) -> str:
        """Extract fixed code from LLM response"""
        
        # Try to find code block
        if "```python" in llm_response:
            parts = llm_response.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()
        
        # If no code block, return response as-is
        return llm_response.strip()
    
    def _analyze_changes(self, original: str, fixed: str) -> List[str]:
        """Analyze what changed between original and fixed code"""
        
        changes = []
        original_lines = original.split('\n')
        fixed_lines = fixed.split('\n')
        
        # Simple line comparison (for demo)
        if len(original_lines) != len(fixed_lines):
            changes.append("Number of lines changed")
        
        # Check for added imports
        original_imports = [l for l in original_lines if l.strip().startswith('import') or l.strip().startswith('from')]
        fixed_imports = [l for l in fixed_lines if l.strip().startswith('import') or l.strip().startswith('from')]
        
        for imp in fixed_imports:
            if imp not in original_imports:
                changes.append(f"Added import: {imp}")
        
        return changes
