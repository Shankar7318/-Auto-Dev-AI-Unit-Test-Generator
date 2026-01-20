import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CoverageAnalyzer:
    """Analyzes coverage reports and identifies gaps"""
    
    def __init__(self):
        self.coverage_dir = Settings.COVERAGE_DIR
    
    def analyze(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coverage and return gap analysis"""
        
        if not coverage_data:
            return self._empty_coverage_analysis()
        
        try:
            files = coverage_data.get("files", {})
            
            if not files:
                return self._empty_coverage_analysis()
            
            # Get first file (we only test one file at a time)
            file_name = list(files.keys())[0]
            file_coverage = files[file_name]
            
            analysis = {
                "summary": self._get_summary(file_coverage),
                "missing_lines": self._get_missing_lines(file_coverage),
                "missing_branches": self._get_missing_branches(file_coverage),
                "coverage_percentage": file_coverage.get("summary", {}).get("percent_covered", 0),
                "is_complete": False,
            }
            
            # Check if coverage is complete
            analysis["is_complete"] = (
                analysis["coverage_percentage"] >= Settings.TARGET_COVERAGE and
                not analysis["missing_lines"] and
                not analysis["missing_branches"]
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Coverage analysis error: {e}")
            return self._empty_coverage_analysis()
    
    def _get_summary(self, coverage: Dict[str, Any]) -> Dict[str, float]:
        """Extract coverage summary"""
        summary = coverage.get("summary", {})
        
        return {
            "lines_covered": summary.get("covered_lines", 0),
            "lines_total": summary.get("num_statements", 0),
            "branches_covered": summary.get("covered_branches", 0),
            "branches_total": summary.get("num_branches", 0),
            "percent_covered": summary.get("percent_covered", 0),
        }
    
    def _get_missing_lines(self, coverage: Dict[str, Any]) -> List[int]:
        """Get lines not covered"""
        missing_lines = []
        
        # Parse line-by-line coverage
        lines = coverage.get("executed_lines", [])
        missing = coverage.get("missing_lines", [])
        
        return missing
    
    def _get_missing_branches(self, coverage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get branches not covered"""
        missing_branches = []
        
        branch_data = coverage.get("branch_data", {})
        for line, branches in branch_data.items():
            for branch_idx, (taken, total) in enumerate(branches):
                if taken < total:
                    missing_branches.append({
                        "line": int(line),
                        "branch": branch_idx,
                        "taken": taken,
                        "total": total,
                    })
        
        return missing_branches
    
    def _empty_coverage_analysis(self) -> Dict[str, Any]:
        """Return empty analysis when no coverage data"""
        return {
            "summary": {
                "lines_covered": 0,
                "lines_total": 0,
                "branches_covered": 0,
                "branches_total": 0,
                "percent_covered": 0,
            },
            "missing_lines": [],
            "missing_branches": [],
            "coverage_percentage": 0,
            "is_complete": False,
        }