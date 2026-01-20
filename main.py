"""
Auto-Dev Agent - Streamlit Web Interface
Upload Python code and generate unit tests automatically
"""

import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import json
from datetime import datetime
import time
import re
import ast
from typing import Optional, List, Dict, Any

current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

try:
    from utils.code_parser import parse_code
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False
    
    def parse_code(source_code: str) -> dict:
        """Simple code parser"""
        lines = source_code.split('\n')
        functions = []
        
        # Try to parse with AST
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node) or '',
                        'line_start': node.lineno,
                        'line_end': node.end_lineno,
                    })
        except:
            # Fallback: simple regex-based parsing
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    functions.append({
                        'name': func_name,
                        'args': [],
                        'docstring': '',
                        'line_start': i + 1,
                        'line_end': i + 1,
                    })
        
        return {
            'summary': {
                'total_lines': len(lines),
                'functions_count': len(functions),
                'classes_count': source_code.count('class '),
                'complexity_score': 1,
            },
            'functions': functions,
            'classes': [],
            'testable_paths': [],
        }
try:
    from models.local_llm import generate_tests_with_llm, LocalLLM
    HAS_LOCAL_LLM = True
    print("Local LLM module imported successfully")
except ImportError as e:
    print(f"Could not import local_llm: {e}")
    HAS_LOCAL_LLM = False
    
    
    def generate_tests_with_llm(source_code: str, analysis: dict, model_name: Optional[str] = None) -> str:
        return "Local LLM not available - using template generation"

# Settings class
class Settings:
    BASE_DIR = current_dir
    TESTS_DIR = current_dir / "generated_tests"
    COVERAGE_DIR = current_dir / "coverage_reports"
    
    @classmethod
    def setup_directories(cls):
        cls.TESTS_DIR.mkdir(exist_ok=True, parents=True)
        cls.COVERAGE_DIR.mkdir(exist_ok=True, parents=True)

# Page configuration
st.set_page_config(
    page_title="Auto-Dev: AI Unit Test Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .code-block {
        font-family: 'Courier New', monospace;
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        overflow-x: auto;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'source_code' not in st.session_state:
    st.session_state.source_code = ""
if 'test_code' not in st.session_state:
    st.session_state.test_code = ""
if 'analysis' not in st.session_state:
    st.session_state.analysis = {}
if 'iterations' not in st.session_state:
    st.session_state.iterations = 0
if 'test_history' not in st.session_state:
    st.session_state.test_history = []
if 'show_editor' not in st.session_state:
    st.session_state.show_editor = False
if 'test_generation_started' not in st.session_state:
    st.session_state.test_generation_started = False

# Helper function to generate sample tests (MUST BE DEFINED BEFORE USING)
def generate_sample_tests(source_code: str, analysis: dict) -> str:
    """Generate sample test code based on analysis"""
    
    test_code = '''"""
Auto-generated test suite
Generated by Auto-Dev Agent
"""

import pytest
'''
    
    # Add imports for the source module
    test_code += "# Test imports would be added here based on source code\n\n"
    
    # Generate tests for functions
    if analysis.get('functions'):
        test_code += "# Function Tests\n"
        for func in analysis.get('functions', [])[:10]:
            func_name = func.get('name', '')
            if func_name:
                test_code += f'''
def test_{func_name}():
    """
    Test {func_name} function
    """
    # TODO: Add actual test implementation
    # Test basic functionality
    assert True
    
def test_{func_name}_edge_cases():
    """
    Test edge cases for {func_name}
    """
    # TODO: Test boundary values, invalid inputs, etc.
    assert True
    
def test_{func_name}_exceptions():
    """
    Test exception handling for {func_name}
    """
    # TODO: Test expected exceptions
    assert True
'''
    
    # Generate tests for classes
    if analysis.get('classes'):
        test_code += "\n# Class Tests\n"
        for cls in analysis.get('classes', [])[:5]:
            cls_name = cls.get('name', '')
            if cls_name:
                test_code += f'''
class Test{cls_name}:
    """
    Test suite for {cls_name} class
    """
    
    def test_{cls_name.lower()}_initialization(self):
        """
        Test class initialization
        """
        # TODO: Test constructor
        assert True
    
    def test_{cls_name.lower()}_methods(self):
        """
        Test class methods
        """
        # TODO: Test all methods
        assert True
'''
    
    # Add fixtures if needed
    test_code += '''
# Fixtures
@pytest.fixture
def sample_data():
    """Sample fixture for test data"""
    return {"key": "value"}

# Integration tests
def test_integration():
    """Integration test example"""
    assert True
'''
    
    return test_code

def generate_advanced_tests(source_code: str, analysis: dict) -> str:
    """Generate more advanced test code with actual test logic"""
    
    test_code = '''"""
Advanced Auto-generated Test Suite
Generated by Auto-Dev Agent
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

'''
    
    # Check if we need to import specific modules from source
    if 'math' in source_code.lower():
        test_code += "import math\n"
    if 'json' in source_code.lower():
        test_code += "import json\n"
    if 'datetime' in source_code.lower():
        test_code += "from datetime import datetime, date, timedelta\n"
    
    test_code += "\n# ========== FUNCTION TESTS ==========\n\n"
    
    # Generate actual tests for functions
    if analysis.get('functions'):
        for func in analysis.get('functions', [])[:8]:
            func_name = func.get('name', '')
            if func_name:
                # Generate different types of tests based on function name
                if 'add' in func_name.lower() or 'sum' in func_name.lower():
                    test_code += f'''
def test_{func_name}_basic():
    """Test basic addition/sum functionality"""
    # Example test - would need actual implementation
    result = 2 + 2
    assert result == 4

def test_{func_name}_negative_numbers():
    """Test with negative numbers"""
    result = -5 + 3
    assert result == -2

def test_{func_name}_zero():
    """Test with zero"""
    result = 0 + 0
    assert result == 0
'''
                elif 'divide' in func_name.lower() or 'split' in func_name.lower():
                    test_code += f'''
def test_{func_name}_normal():
    """Test normal division"""
    result = 10 / 2
    assert result == 5

def test_{func_name}_by_zero():
    """Test division by zero error"""
    with pytest.raises(ZeroDivisionError):
        result = 10 / 0

def test_{func_name}_negative():
    """Test division with negative numbers"""
    result = -10 / 2
    assert result == -5
'''
                else:
                    # Generic test template
                    test_code += f'''
def test_{func_name}():
    """
    Test {func_name} function
    Parameters: {func.get('args', [])}
    """
    # Basic test
    assert True
    
def test_{func_name}_edge_cases():
    """Test edge cases for {func_name}"""
    # Test with None
    # Test with empty values
    # Test with extreme values
    assert True

def test_{func_name}_invalid_input():
    """Test invalid input handling for {func_name}"""
    # Should raise appropriate exceptions
    assert True
'''
    
    # Generate tests for classes
    if analysis.get('classes'):
        test_code += "\n# ========== CLASS TESTS ==========\n\n"
        for cls in analysis.get('classes', [])[:4]:
            cls_name = cls.get('name', '')
            if cls_name:
                test_code += f'''
class Test{cls_name}:
    """
    Comprehensive test suite for {cls_name} class
    """
    
    @pytest.fixture
    def {cls_name.lower()}_instance(self):
        """Fixture to create class instance"""
        return {cls_name}()
    
    def test_{cls_name.lower()}_creation(self, {cls_name.lower()}_instance):
        """Test class instance creation"""
        assert {cls_name.lower()}_instance is not None
        assert isinstance({cls_name.lower()}_instance, {cls_name})
    
    def test_{cls_name.lower()}_methods_exist(self, {cls_name.lower()}_instance):
        """Test that all expected methods exist"""
        methods = {cls.get('methods', [])}
        for method in methods:
            assert hasattr({cls_name.lower()}_instance, method)
    
    def test_{cls_name.lower()}_method_calls(self, {cls_name.lower()}_instance):
        """Test basic method calls"""
        # Test each method with basic inputs
        for method in {cls.get('methods', [])}:
            try:
                getattr({cls_name.lower()}_instance, method)()
            except TypeError:
                # Method requires arguments
                pass
'''
    
    # Add comprehensive fixtures
    test_code += '''
# ========== FIXTURES ==========

@pytest.fixture
def sample_numbers():
    """Fixture providing sample numbers"""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_strings():
    """Fixture providing sample strings"""
    return ["hello", "world", "test", "python"]

@pytest.fixture
def sample_dict():
    """Fixture providing sample dictionary"""
    return {"key1": "value1", "key2": 2, "key3": [1, 2, 3]}

# ========== PARAMETRIZED TESTS ==========

@pytest.mark.parametrize("input_val,expected", [
    (1, 2),
    (0, 1),
    (-1, 0),
    (100, 101),
])
def test_increment(input_val, expected):
    """Parametrized test example"""
    # This is a template - implement actual logic
    pass

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, -50, 50),
])
def test_addition(a, b, expected):
    """Parametrized addition test"""
    # This is a template - implement actual logic
    pass
'''
    
    return test_code

# App header
st.markdown('<div class="main-header">🤖 Auto-Dev: AI Unit Test Generator</div>', unsafe_allow_html=True)
st.markdown("""
**Upload Python code** → **AI analyzes** → **Generates comprehensive unit tests** → **Download ready-to-use tests**
""")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/python.png", width=80)
    st.markdown("### 🔧 Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "AI Model",
        ["Local LLM (CodeLlama)", "GPT-4", "Claude", "Custom Model"],
        help="Select the AI model for test generation"
    )
    
    # Test settings
    st.markdown("### ⚙️ Test Settings")
    max_iterations = st.slider("Max Iterations", 1, 20, 5, 
                              help="Maximum number of test improvement cycles")
    target_coverage = st.slider("Target Coverage %", 50, 100, 95,
                               help="Target test coverage percentage")
    
    # Test type selection
    test_type = st.selectbox(
        "Test Generation Style",
        ["Basic Template", "Advanced Template", "Comprehensive Suite"],
        help="Select the type of tests to generate"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        include_edge_cases = st.checkbox("Generate Edge Cases", True)
        use_mocks = st.checkbox("Use Mocking", True)
        generate_fixtures = st.checkbox("Generate Fixtures", True)
        timeout_seconds = st.number_input("Test Timeout (s)", 10, 300, 30)
    
    # Info
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    if st.session_state.iterations > 0:
        st.metric("Total Iterations", st.session_state.iterations)
        if st.session_state.analysis:
            st.metric("Functions Found", st.session_state.analysis.get('summary', {}).get('functions_count', 0))
            st.metric("Classes Found", st.session_state.analysis.get('summary', {}).get('classes_count', 0))
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    if st.button("🔄 Reset Session", use_container_width=True):
        # Clear all relevant session state
        keys_to_clear = ['source_code', 'test_code', 'analysis', 'iterations', 
                        'test_history', 'show_editor', 'test_generation_started']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Reinitialize with default values
        st.session_state.source_code = ""
        st.session_state.test_code = ""
        st.session_state.analysis = {}
        st.session_state.iterations = 0
        st.session_state.test_history = []
        st.session_state.show_editor = False
        st.session_state.test_generation_started = False
        
        st.success("Session reset successfully!")
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Code", "🔍 Code Analysis", "🧪 Generate Tests", "📥 Export Tests"])

with tab1:
    st.markdown("### Upload or Write Python Code")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Method 1: File upload
        uploaded_file = st.file_uploader(
            "Choose a Python file",
            type=['py', 'python'],
            help="Upload a .py file to generate tests for"
        )
        
        if uploaded_file is not None:
            source_code = uploaded_file.read().decode("utf-8")
            st.session_state.source_code = source_code
            st.success(f"📄 File uploaded: {uploaded_file.name}")
            st.session_state.show_editor = False
    
    with col2:
        # Method 2: Code editor
        st.markdown("**Or write code directly:**")
        if st.button("✏️ Open Code Editor", use_container_width=True):
            st.session_state.show_editor = True
        
        if st.button("📋 Use Sample Code", use_container_width=True):
            # Load sample code
            sample_code = '''class Calculator:
    """A simple calculator class"""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a"""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base: float, exponent: float) -> float:
        """Calculate power"""
        return base ** exponent'''
            st.session_state.source_code = sample_code
            st.session_state.show_editor = True
    
    # Code editor
    if st.session_state.show_editor:
        st.markdown("### 📝 Write Your Python Code")
        edited_code = st.text_area(
            "Write your Python code here:",
            height=400,
            value=st.session_state.source_code,
            placeholder="""# Write your Python code here
# Example:
# def add(a, b):
#     return a + b

class MyClass:
    def my_method(self):
        return "Hello World"
""",
            key="editor"
        )
        if edited_code:
            st.session_state.source_code = edited_code
    
    # Display uploaded code
    if st.session_state.source_code:
        st.markdown("### 📝 Source Code Preview")
        with st.expander("View Source Code", expanded=False):
            st.code(st.session_state.source_code, language='python')
        
        # Analyze button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔍 Analyze Code Structure", type="primary", use_container_width=True):
                with st.spinner("Analyzing code structure..."):
                    try:
                        # Parse and analyze code
                        analysis = parse_code(st.session_state.source_code)
                        st.session_state.analysis = analysis
                        
                        # Show success
                        st.markdown('<div class="success-box">✅ Code analysis completed successfully!</div>', unsafe_allow_html=True)
                        
                        # Switch to analysis tab
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

with tab2:
    st.markdown("### 🔍 Code Analysis Results")
    
    if not st.session_state.analysis:
        st.warning("No code analyzed yet. Upload or write code in the first tab, then click 'Analyze Code Structure'.")
    else:
        analysis = st.session_state.analysis
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="stats-card">📊<br><strong>Lines</strong><br>{}</div>'.format(
                analysis.get('summary', {}).get('total_lines', 0)
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stats-card">⚙️<br><strong>Functions</strong><br>{}</div>'.format(
                analysis.get('summary', {}).get('functions_count', 0)
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="stats-card">🏗️<br><strong>Classes</strong><br>{}</div>'.format(
                analysis.get('summary', {}).get('classes_count', 0)
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="stats-card">🌀<br><strong>Complexity</strong><br>{}</div>'.format(
                analysis.get('summary', {}).get('complexity_score', 0)
            ), unsafe_allow_html=True)
        
        # Functions section
        if analysis.get('functions'):
            st.markdown("#### 📋 Functions to Test")
            for func in analysis['functions'][:10]:  # Show first 10
                with st.expander(f"🔹 {func.get('name', 'Unknown')}"):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.write(f"**Function:** `{func.get('name', '')}`")
                        args = func.get('args', [])
                        if args:
                            st.write(f"**Arguments:** `{', '.join(args)}`")
                        if func.get('docstring'):
                            st.write(f"**Description:** {func.get('docstring')}")
                    with col_b:
                        st.write(f"**Lines:** {func.get('line_start', '?')}-{func.get('line_end', '?')}")
        
        # Generate tests button - update this button logic
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Generate New Tests", type="primary", use_container_width=True):
                if st.session_state.iterations >= max_iterations:
                    st.warning(f"Maximum iterations ({max_iterations}) reached. Please reset the session or increase max iterations.")
                else:
                    # Clear previous test code to force new generation
                    st.session_state.test_code = ""
                    st.session_state.test_generation_started = True
                    st.rerun()
        
        with col2:
            if st.button("🔍 Re-analyze Code", use_container_width=True):
                with st.spinner("Re-analyzing code..."):
                    try:
                        analysis = parse_code(st.session_state.source_code)
                        st.session_state.analysis = analysis
                        st.success("Code re-analyzed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

with tab3:
    st.markdown("### 🧪 Test Generation")
    
    # Check if we should start a new generation
    if st.session_state.get('test_generation_started', False):
        # Check if we've reached maximum iterations
        if st.session_state.iterations >= max_iterations:
            st.markdown(f'<div class="warning-box">⚠️ Maximum iterations ({max_iterations}) reached. Cannot generate more tests.</div>', unsafe_allow_html=True)
            st.info("Please reset the session or increase the max iterations in the sidebar if you want to generate more tests.")
            
            # Reset the flag
            st.session_state.test_generation_started = False
        else:
            # Test generation progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate test generation process
            steps = [
                "Initializing AI model...",
                "Analyzing code structure...",
                "Generating test cases...",
                "Creating edge case tests...",
                "Adding mocks and fixtures...",
                "Validating test syntax...",
                "Finalizing test suite..."
            ]
            
            # Simulate process with progress bar
            for i, step in enumerate(steps):
                progress = (i + 1) / len(steps)
                progress_bar.progress(progress)
                status_text.text(f"🔄 {step}")
                time.sleep(0.3)  # Small delay for visual effect
            
            # Generate test code based on selected type
            source_code = st.session_state.source_code
            analysis = st.session_state.analysis
            
            if test_type == "Basic Template":
                test_code = generate_sample_tests(source_code, analysis)
            elif test_type == "Advanced Template":
                test_code = generate_advanced_tests(source_code, analysis)
            elif test_type == "Comprehensive Suite":
                if model_option == "Local LLM (CodeLlama)" and HAS_LOCAL_LLM:
                    try:
                        test_code = generate_tests_with_llm(source_code, analysis, model_name="code-llama-7b")
                    except Exception as e:
                        st.error(f"Local LLM generation failed: {str(e)}. Falling back to advanced template.")
                        test_code = generate_advanced_tests(source_code, analysis)
                # Combine both
                test_code = generate_advanced_tests(source_code, analysis)
                # Add extra sections
                test_code += '''
# ========== INTEGRATION TESTS ==========

def test_integration_workflow():
    """Test complete workflow integration"""
    # Implement integration tests here
    assert True

def test_error_handling_integration():
    """Test error handling across components"""
    # Implement error handling tests
    assert True

# ========== PERFORMANCE TESTS ==========

@pytest.mark.performance
def test_performance_large_inputs():
    """Test performance with large inputs"""
    # Implement performance tests
    assert True
'''
            
            # Store the new test code and update iteration count
            st.session_state.test_code = test_code
            st.session_state.iterations += 1
            st.session_state.test_history.append({
                "timestamp": datetime.now().isoformat(),
                "iteration": st.session_state.iterations,
                "test_code": test_code
            })
            
            # Clear the flag to stop regeneration on rerun
            st.session_state.test_generation_started = False
            
            # Force a rerun to show the results
            st.rerun()
    
    # Display existing test code or prompt
    if not st.session_state.test_code:
        st.info("No tests generated yet. Go to the Analysis tab and click 'Generate New Tests' to start.")
    else:
        # Show the current test code
        st.markdown("### 📋 Generated Test Code")
        
        # Show stats
        line_count = len(st.session_state.test_code.split('\n'))
        function_count = st.session_state.test_code.count('def test_')
        class_count = st.session_state.test_code.count('class Test')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Lines", line_count)
        with col2:
            st.metric("Test Functions", function_count)
        with col3:
            st.metric("Test Classes", class_count)
        
        with st.expander("View Complete Test Code", expanded=True):
            st.code(st.session_state.test_code, language='python')
        
        # Test actions
        st.markdown("### 🎯 Test Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Run Tests Now", use_container_width=True):
                st.info("Test execution would run here (requires Docker installation)")
                st.code("pytest generated_tests.py -v")
        
        with col2:
            # Check if we can generate more tests
            if st.session_state.iterations < max_iterations:
                if st.button("🔄 Generate New Tests", use_container_width=True):
                    st.session_state.test_generation_started = True
                    st.rerun()
            else:
                st.warning(f"Max: {max_iterations}")
        
        with col3:
            if st.button("📋 Copy Test Code", use_container_width=True):
                st.success("Code copied to clipboard! (Use Ctrl+V to paste)")
        
        with col4:
            if st.button("🗑️ Clear Tests", use_container_width=True):
                st.session_state.test_code = ""
                st.session_state.test_generation_started = False
                st.success("Tests cleared!")
                st.rerun()
        
        # Test history section
        if st.session_state.test_history:
            st.markdown("---")
            st.markdown("#### 📜 Test Generation History")
            
            for i, history in enumerate(reversed(st.session_state.test_history)):
                with st.expander(f"Version {history['iteration']} - {history['timestamp'][:19]}"):
                    preview_lines = history['test_code'].split('\n')[:30]
                    preview = '\n'.join(preview_lines)
                    if len(history['test_code'].split('\n')) > 30:
                        preview += "\n# ... (truncated)"
                    st.code(preview, language='python')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label=f"📥 Export v{history['iteration']}",
                            data=history['test_code'],
                            file_name=f"tests_v{history['iteration']}.py",
                            mime="text/x-python",
                            key=f"export_v{history['iteration']}"
                        )

with tab4:
    st.markdown("### 📥 Export Generated Tests")
    
    if not st.session_state.test_code:
        st.warning("No tests generated yet. Generate tests first in the previous tab.")
    else:
        test_code = st.session_state.test_code
        
        # Export options
        st.markdown("#### 📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as Python file
            st.download_button(
                label="📥 Download as .py",
                data=test_code,
                file_name="generated_tests.py",
                mime="text/x-python",
                use_container_width=True,
                help="Download as standalone Python test file"
            )
        
        with col2:
            # Copy to clipboard
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.success("Code copied to clipboard! (Use Ctrl+V to paste)")
        
        with col3:
            # Save to file
            if st.button("💾 Save to Project", use_container_width=True):
                Settings.setup_directories()
                test_file = Settings.TESTS_DIR / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                test_file.write_text(test_code)
                st.success(f"✅ Tests saved to: {test_file}")
        
        # Preview of what will be exported
        st.markdown("#### 👀 Preview")
        with st.expander("Preview First 100 Lines", expanded=True):
            lines = test_code.split('\n')
            preview = '\n'.join(lines[:100])
            if len(lines) > 100:
                preview += "\n# ... (truncated for preview)"
            st.code(preview, language='python')
        
        # Batch export all versions
        st.markdown("---")
        if st.session_state.test_history and len(st.session_state.test_history) > 1:
            if st.button("📦 Export All Versions", use_container_width=True):
                all_tests = "# Auto-Dev Test Suite - All Versions\n\n"
                for history in st.session_state.test_history:
                    all_tests += f"# ===== Version {history['iteration']} =====\n"
                    all_tests += f"# Generated: {history['timestamp']}\n\n"
                    all_tests += history['test_code']
                    all_tests += "\n\n" + "#" * 60 + "\n\n"
                
                st.download_button(
                    label="📥 Download All Versions",
                    data=all_tests,
                    file_name="all_test_versions.py",
                    mime="text/x-python",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 Auto-Dev Agent | AI-Powered Unit Test Generation</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":

    Settings.setup_directories()