"""Constant values used throughout the application"""

# Agent states
class AgentState:
    ANALYZING = "analyzing"
    GENERATING_TESTS = "generating_tests"
    RUNNING_TESTS = "running_tests"
    ANALYZING_COVERAGE = "analyzing_coverage"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"

# Error messages
ERROR_MESSAGES = {
    "NO_SOURCE_CODE": "No source code provided",
    "PARSE_ERROR": "Failed to parse source code",
    "TEST_GENERATION_FAILED": "Failed to generate tests",
    "TEST_EXECUTION_FAILED": "Test execution failed",
    "COVERAGE_ANALYSIS_FAILED": "Coverage analysis failed",
    "MAX_ITERATIONS_REACHED": "Maximum iterations reached without achieving 100% coverage",
}

# LLM prompts
class Prompts:
    ANALYSIS = """Analyze this Python code and provide structured information:
Code:
{code}

Provide output in this format:
FUNCTIONS:
- function_name: description
CLASSES:
- class_name: methods
BRANCHES:
- line X: if/else condition
EXCEPTIONS:
- line Y: try/except block
EDGE_CASES:
- description"""

    TEST_GENERATION = """Generate pytest tests for this code:
Source Code:
{source_code}

Analysis:
{analysis}

Uncovered Lines: {uncovered_lines}
Previous Errors: {previous_errors}

Generate only Python test code:"""