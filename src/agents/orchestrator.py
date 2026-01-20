from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from src.agents.code_analyzer import CodeAnalyzer
from src.agents.test_generator import TestGenerator
from src.agents.test_runner import TestRunner
from src.agents.coverage_analyzer import CoverageAnalyzer
from src.agents.reflection_agent import ReflectionAgent
from src.config.constants import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AgentState(TypedDict):
    """State definition for LangGraph workflow"""
    source_code: str
    test_code: str
    analysis: Dict[str, Any]
    coverage_data: Dict[str, Any]
    coverage_analysis: Dict[str, Any]
    test_results: Dict[str, Any]
    errors: List[str]
    iteration: int
    max_iterations: int
    status: str
    reflection: Optional[Dict[str, Any]]

class AutoDevOrchestrator:
    """Main orchestrator for the agentic workflow"""
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        
        # Initialize agents
        self.code_analyzer = CodeAnalyzer()
        self.test_generator = TestGenerator()
        self.test_runner = TestRunner()
        self.coverage_analyzer = CoverageAnalyzer()
        self.reflection_agent = ReflectionAgent()
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Build LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_code", self._analyze_code)
        workflow.add_node("generate_tests", self._generate_tests)
        workflow.add_node("run_tests", self._run_tests)
        workflow.add_node("analyze_coverage", self._analyze_coverage)
        workflow.add_node("reflect", self._reflect)
        
        # Set entry point
        workflow.set_entry_point("analyze_code")
        
        # Add edges
        workflow.add_edge("analyze_code", "generate_tests")
        workflow.add_edge("generate_tests", "run_tests")
        workflow.add_edge("run_tests", "analyze_coverage")
        
        # Conditional edge based on coverage
        workflow.add_conditional_edges(
            "analyze_coverage",
            self._should_continue,
            {
                "continue": "reflect",
                "complete": END
            }
        )
        
        workflow.add_edge("reflect", "generate_tests")
        
        return workflow.compile()
    
    def run(self, source_code: str) -> Dict[str, Any]:
        """Run the complete workflow"""
        
        logger.info("Starting Auto-Dev workflow...")
        
        # Initial state
        initial_state: AgentState = {
            "source_code": source_code,
            "test_code": "",
            "analysis": {},
            "coverage_data": {},
            "coverage_analysis": {},
            "test_results": {},
            "errors": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": AgentState.ANALYZING,
            "reflection": None,
        }
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Compile results
        results = self._compile_results(final_state)
        
        logger.info(f"Workflow completed. Coverage: {results['final_coverage']}%")
        
        return results
    
    def _analyze_code(self, state: AgentState) -> AgentState:
        """Analyze source code"""
        logger.info("Analyzing source code...")
        
        try:
            analysis = self.code_analyzer.analyze(state["source_code"])
            state["analysis"] = analysis
            state["status"] = AgentState.GENERATING_TESTS
        except Exception as e:
            state["errors"].append(f"Analysis error: {str(e)}")
            state["status"] = AgentState.FAILED
        
        return state
    
    def _generate_tests(self, state: AgentState) -> AgentState:
        """Generate test code"""
        logger.info(f"Generating tests (iteration {state['iteration'] + 1})...")
        
        try:
            uncovered_lines = state.get("coverage_analysis", {}).get("missing_lines", [])
            previous_errors = state.get("errors", [])
            
            test_code = self.test_generator.generate(
                state["source_code"],
                state["analysis"],
                uncovered_lines,
                previous_errors[-3:] if previous_errors else None
            )
            
            state["test_code"] = test_code
            state["status"] = AgentState.RUNNING_TESTS
        except Exception as e:
            state["errors"].append(f"Test generation error: {str(e)}")
            state["status"] = AgentState.FAILED
        
        return state
    
    def _run_tests(self, state: AgentState) -> AgentState:
        """Run tests in Docker"""
        logger.info("Running tests...")
        
        try:
            test_results = self.test_runner.execute(
                state["source_code"],
                state["test_code"]
            )
            
            state["test_results"] = test_results
            
            if not test_results["success"]:
                state["errors"].append("Tests failed")
            
            state["status"] = AgentState.ANALYZING_COVERAGE
        except Exception as e:
            state["errors"].append(f"Test execution error: {str(e)}")
            state["status"] = AgentState.FAILED
        
        return state
    
    def _analyze_coverage(self, state: AgentState) -> AgentState:
        """Analyze coverage results"""
        logger.info("Analyzing coverage...")
        
        try:
            coverage_data = state["test_results"].get("coverage", {})
            coverage_analysis = self.coverage_analyzer.analyze(coverage_data)
            
            state["coverage_analysis"] = coverage_analysis
            state["coverage_data"] = coverage_data
            
            if coverage_analysis["is_complete"]:
                state["status"] = AgentState.COMPLETED
            else:
                state["status"] = AgentState.REFLECTING
            
            state["iteration"] += 1
            
        except Exception as e:
            state["errors"].append(f"Coverage analysis error: {str(e)}")
            state["status"] = AgentState.FAILED
        
        return state
    
    def _reflect(self, state: AgentState) -> AgentState:
        """Reflect on failures and coverage gaps"""
        logger.info("Reflecting on test results...")
        
        # Check if max iterations reached
        if state["iteration"] >= state["max_iterations"]:
            state["errors"].append("Max iterations reached")
            state["status"] = AgentState.FAILED
            return state
        
        try:
            reflection = self.reflection_agent.reflect(
                state["test_code"],
                state["test_results"].get("output", ""),
                state["coverage_analysis"],
                state["source_code"]
            )
            
            state["reflection"] = reflection
            state["test_code"] = reflection["new_test_code"]
            state["status"] = AgentState.GENERATING_TESTS
            
        except Exception as e:
            state["errors"].append(f"Reflection error: {str(e)}")
            state["status"] = AgentState.FAILED
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue or end"""
        
        if state["status"] == AgentState.FAILED:
            return "complete"
        
        if state["coverage_analysis"].get("is_complete", False):
            return "complete"
        
        if state["iteration"] >= state["max_iterations"]:
            return "complete"
        
        return "continue"
    
    def _compile_results(self, state: AgentState) -> Dict[str, Any]:
        """Compile final results"""
        
        coverage_pct = state["coverage_analysis"].get("summary", {}).get("percent_covered", 0)
        
        return {
            "success": state["status"] == AgentState.COMPLETED,
            "final_coverage": coverage_pct,
            "iterations": state["iteration"],
            "test_code": state["test_code"],
            "analysis": state["analysis"],
            "coverage_analysis": state["coverage_analysis"],
            "errors": state["errors"],
            "status": state["status"],
            "has_reflection": state["reflection"] is not None,
        }