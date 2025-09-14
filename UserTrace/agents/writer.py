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
logger = logging.getLogger("WriterAgent")

# -------------------- Writer Agent --------------------
class WriterAgent(BaseAgent):
    """Agent for generating Use Cases from Intermediate Requirements (IRs)."""
    
    def __init__(self, config_path: Optional[str] = None, performance_metrics: Optional[PerformanceMetrics] = None):
        super().__init__("WriterAgent", config_path=config_path, performance_metrics=performance_metrics)
        
        system_prompt = """You are a Writer agent specialized in transforming technical Intermediate Requirements (IRs) into structured Use Cases. Your role is to create formal use case specifications that capture user interactions with the system.

CORE PRINCIPLES:
1. USER-ACTOR FOCUS: Identify primary and secondary actors who interact with the system
2. GOAL-ORIENTED: Each use case represents a specific user goal or business objective
3. INTERACTION MODELING: Describe the step-by-step interaction between actors and system
4. CONDITION SPECIFICATION: Define clear preconditions and postconditions
5. SCENARIO COVERAGE: Include main success scenario and alternative flows

USE CASE STRUCTURE:
- Use Case Name: Clear, action-oriented name describing the user goal
- Description: Brief summary of what the use case accomplishes
- Primary Actor: Main user or system that initiates the use case
- Secondary Actors: Other users, systems, or external entities involved
- Preconditions: System state and conditions that must be true before execution
- Main Flow: Step-by-step description of the normal execution path
- Alternative Flows: Variations and exception handling scenarios
- Postconditions: System state after successful completion

TRANSFORMATION GUIDELINES:
- Convert technical components into system behaviors visible to users
- Transform dependencies into actor interactions and system responses
- Abstract implementation details into user-observable actions
- Identify decision points and alternative paths
- Focus on business value and user outcomes"""

        self.add_to_memory("system", system_prompt)

    def _compose_intent(self,
                       idx: int,
                       files: Set[str],
                       file_requirements: Dict[str, Dict[str, Any]],
                       comp_requirements: Dict[str, Dict[str, Any]],
                       file_graph: Dict[str, set],
                       ext_knowledge: Optional[str] = "") -> Dict[str, Any]:
        """Compose a user requirement from file cluster IRs."""
        
        # Identify central files based on connectivity
        deg = {f: len(file_graph.get(f, set()) & files) for f in files}
        central_files = [f for f, _ in sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))][:5]
        
        # Aggregate file summaries
        file_summaries = []
        for f in files:
            if f in file_requirements:
                file_summaries.append({
                    "file": f,
                    "summary": file_requirements[f]["summary"]
                })
        
        # Sample key components
        sample_components = []
        for f in list(files)[:3]:  # Limit to 3 files to avoid context overflow
            file_comps = file_requirements.get(f, {}).get("components", [])
            for cid in file_comps[:3]:  # Top 3 components per file
                if cid in comp_requirements:
                    sample_components.append({
                        "id": cid,
                        "type": comp_requirements[cid].get("type", "component"),
                        "summary": comp_requirements[cid].get("summary", "")
                    })
        
        # Build external knowledge context
        ext_context = ""
        if ext_knowledge:
            ext_context = f"\n\nEXTERNAL BUSINESS KNOWLEDGE:\n{ext_knowledge}"
        
        prompt = f"""Transform the following technical Intermediate Requirements into a structured Use Case specification.

CLUSTER INFORMATION:
- Cluster ID: {idx}
- Files in cluster: {len(files)} files
- Central files: {central_files}
- All files: {sorted(list(files))}

FILE INTERMEDIATE REQUIREMENTS:
{json.dumps(file_summaries, indent=2)}

KEY COMPONENT EXAMPLES:
{json.dumps(sample_components, indent=2)}
{ext_context}

TRANSFORMATION TASK:
1. Analyze the technical IRs to identify the primary user goal and business capability
2. Determine the main actors (users, external systems) who interact with this functionality
3. Define the preconditions required for the use case to execute
4. Describe the main flow of interactions between actors and the system
5. Identify alternative flows and exception scenarios
6. Specify the postconditions after successful completion

OUTPUT REQUIREMENTS (STRICT JSON FORMAT):
{{
    "use_case_name": "Clear, action-oriented name (e.g., 'Process Customer Order', 'Generate Report')",
    "description": "Brief 2-3 sentence summary of what this use case accomplishes and its business value",
    "primary_actor": "Main user or system that initiates this use case (e.g., 'Customer', 'Administrator', 'External API')",
    "secondary_actors": [
        "List of other actors involved",
        "Include external systems, databases, services",
        "Include other user roles that participate"
    ],
    "preconditions": [
        "System state conditions that must be true before execution",
        "Required data or resources that must be available",
        "User authentication or authorization requirements",
        "External system availability requirements"
    ],
    "main_flow": [
        "1. Actor performs initial action or request",
        "2. System validates input and preconditions", 
        "3. System processes the request using internal components",
        "4. System interacts with external actors/systems if needed",
        "5. System updates data and state as required",
        "6. System provides response or confirmation to actor",
        "7. Use case ends successfully"
    ],
    "alternative_flows": [
        "A1: If validation fails, system displays error message and returns to step 1",
        "A2: If external system unavailable, system queues request and notifies actor",
        "A3: If processing fails, system logs error and provides fallback response"
    ],
    "postconditions": [
        "System state changes after successful completion",
        "Data that has been created, updated, or deleted",
        "Notifications or communications sent to actors",
        "Business rules or constraints that are satisfied"
    ],
    "key_files": ["List of up to 5 most important files that implement this use case"]
}}

IMPORTANT GUIDELINES:
- Focus on user-observable behaviors and interactions
- Use business terminology that stakeholders understand
- Ensure each step in main_flow is a clear actor-system interaction
- Make preconditions and postconditions testable and verifiable
- Include realistic alternative flows for common error scenarios

Generate ONLY the JSON object, no additional text or formatting."""

        self.add_to_memory("user", prompt)
        raw_response = self.generate_response()
        self.clear_memory()
        self.add_to_memory("system", self.memory[0]["content"])  # Re-add system prompt
        
        return self._safe_parse_intent(raw_response, files, central_files)

    def _safe_parse_intent(self, raw: str, files: Set[str], central: List[str]) -> Dict[str, Any]:
        """Safely parse the use case JSON with fallback handling."""
        try:
            # Try to extract JSON from response
            json_start = raw.find('{')
            json_end = raw.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(raw)
        except Exception as e:
            logger.warning(f"Failed to parse use case JSON: {e}")
            # Fallback to structured use case data
            lines = raw.strip().splitlines()
            use_case_name = lines[0][:60] if lines else "Generated Use Case"
            data = {
                "use_case_name": use_case_name,
                "description": "Use case description not available",
                "primary_actor": "User",
                "secondary_actors": [],
                "preconditions": ["System is available"],
                "main_flow": ["1. Actor initiates action", "2. System processes request", "3. System provides response"],
                "alternative_flows": ["A1: If error occurs, system displays error message"],
                "postconditions": ["System state is updated"],
                "key_files": central or list(files)[:5]
            }
        
        # Ensure required use case fields exist
        data.setdefault("use_case_name", "Generated Use Case")
        data.setdefault("description", "Use case description not available")
        data.setdefault("primary_actor", "User")
        data.setdefault("secondary_actors", [])
        data.setdefault("preconditions", ["System is available"])
        data.setdefault("main_flow", ["1. Actor initiates action", "2. System processes request", "3. System provides response"])
        data.setdefault("alternative_flows", ["A1: If error occurs, system displays error message"])
        data.setdefault("postconditions", ["System state is updated"])
        data.setdefault("key_files", central or list(files)[:5])
        
        # Add metadata
        data["files"] = sorted(list(files))
        
        return data

    def process(self, *args, **kwargs) -> Any:
        """Process method for generating use cases."""
        # This method can be used for batch processing if needed
        return "WriterAgent ready for intent composition"