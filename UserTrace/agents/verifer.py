from typing import Dict, Any, Optional, List, Set
from .base import BaseAgent
import logging
import sys
import json
from performance import PerformanceMetrics

# -------------------- Logging Configuration --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VerifierAgent")


# -------------------- Verifier Agent --------------------
class VerifierAgent(BaseAgent):
    """Agent for verifying and providing feedback on Use Case quality."""
    
    def __init__(self, config_path: Optional[str] = None, performance_metrics: Optional[PerformanceMetrics] = None):
        super().__init__("VerifierAgent", config_path=config_path, performance_metrics=performance_metrics)
        
        system_prompt = """You are a Verifier agent responsible for evaluating Use Case quality and providing structured feedback for iterative improvement.

EVALUATION CRITERIA FOR USE CASES:
1. CLARITY: Use case name and description clearly convey the user goal and business value
2. ACTOR IDENTIFICATION: Primary and secondary actors are correctly identified and realistic
3. PRECONDITION COMPLETENESS: All necessary preconditions are specified and testable
4. FLOW LOGIC: Main flow represents a realistic, step-by-step user-system interaction
5. ALTERNATIVE COVERAGE: Alternative flows address realistic error and exception scenarios
6. POSTCONDITION VERIFICATION: Postconditions are observable and represent successful outcomes
7. BUSINESS ALIGNMENT: Use case aligns with business objectives and user needs
8. IMPLEMENTATION FEASIBILITY: Use case can be realistically implemented with given components

SPECIFIC EVALUATION POINTS:
- Use Case Name: Action-oriented, clear user goal (e.g., "Process Order", not "Order Processing")
- Primary Actor: Represents the main initiator with clear motivation
- Secondary Actors: Include all necessary external systems and supporting roles
- Preconditions: Specific, testable conditions (not vague statements)
- Main Flow: 5-10 steps of clear actor-system interactions
- Alternative Flows: Cover realistic error scenarios with proper handling
- Postconditions: Observable system state changes and business outcomes

FEEDBACK FRAMEWORK:
- NEEDS_REVISION: Whether the use case requires significant improvement
- NEEDS_CONTEXT: Whether additional business/domain knowledge is needed
- SPECIFIC SUGGESTIONS: Concrete recommendations for improvement
- QUALITY SCORE: Numerical assessment (1-10 scale)

OUTPUT FORMATS:
You may respond in either JSON or XML tag format:

JSON FORMAT:
{
    "needs_revision": boolean,
    "needs_context": boolean, 
    "context_suggestion": "specific knowledge gaps to address",
    "suggestion": "concrete improvement recommendations",
    "score": integer (1-10)
}

XML TAG FORMAT:
<NEED_REVISION>true/false</NEED_REVISION>
<MORE_CONTEXT>true/false</MORE_CONTEXT>
<SUGGESTION_CONTEXT>specific knowledge gaps</SUGGESTION_CONTEXT>
<SUGGESTION>improvement recommendations</SUGGESTION>"""

        self.add_to_memory("system", system_prompt)

    def process(self, use_case_json: Dict[str, Any]) -> str:
        """Evaluate a Use Case and provide structured feedback."""
        
        # Extract key components for evaluation
        use_case_name = use_case_json.get("use_case_name", "")
        description = use_case_json.get("description", "")
        primary_actor = use_case_json.get("primary_actor", "")
        secondary_actors = use_case_json.get("secondary_actors", [])
        preconditions = use_case_json.get("preconditions", [])
        main_flow = use_case_json.get("main_flow", [])
        alternative_flows = use_case_json.get("alternative_flows", [])
        postconditions = use_case_json.get("postconditions", [])
        key_files = use_case_json.get("key_files", [])
        
        prompt = f"""Evaluate the following Use Case against quality criteria and provide structured feedback.

USE CASE TO EVALUATE:
{json.dumps(use_case_json, indent=2, ensure_ascii=False)}

EVALUATION CHECKLIST:
1. CLARITY (1-10): Is the use case name action-oriented and description clear?
2. ACTOR IDENTIFICATION (1-10): Are primary and secondary actors realistic and well-defined?
3. PRECONDITION COMPLETENESS (1-10): Are all necessary preconditions specified and testable?
4. FLOW LOGIC (1-10): Does the main flow represent realistic user-system interactions?
5. ALTERNATIVE COVERAGE (1-10): Do alternative flows address realistic error scenarios?
6. POSTCONDITION VERIFICATION (1-10): Are postconditions observable and meaningful?
7. BUSINESS ALIGNMENT (1-10): Does the use case align with business objectives?
8. IMPLEMENTATION FEASIBILITY (1-10): Can this be implemented with the given components?

SPECIFIC ASSESSMENT POINTS:
- Use Case Name: Is it action-oriented and clear? (Current: "{use_case_name}")
- Primary Actor: Is it realistic and well-motivated? (Current: "{primary_actor}")
- Secondary Actors: Are all necessary actors included? ({len(secondary_actors)} actors listed)
- Preconditions: Are they specific and testable? ({len(preconditions)} conditions listed)
- Main Flow: Does it have 5-10 clear interaction steps? ({len(main_flow)} steps listed)
- Alternative Flows: Do they cover realistic error scenarios? ({len(alternative_flows)} flows listed)
- Postconditions: Are they observable outcomes? ({len(postconditions)} conditions listed)
- Key Files: Are the implementation files identified? ({len(key_files)} files listed)

FEEDBACK REQUIREMENTS:
- If overall score < 7: needs_revision = true
- If business context or domain knowledge seems lacking: needs_context = true
- Provide specific, actionable suggestions for improvement
- Focus on use case completeness and business value

Provide your evaluation in JSON format with specific recommendations."""

        self.add_to_memory("user", prompt)
        response = self.generate_response()
        self.clear_memory()
        self.add_to_memory("system", self.memory[0]["content"])  # Re-add system prompt
        
        return response