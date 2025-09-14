from typing import Dict, Any, Optional, List
from .base import BaseAgent
import logging
import sys
from performance import PerformanceMetrics

# -------------------- Logging Configuration --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CodeReviewAgent")

# -------------------- Code Reviewer Agent --------------------
class CodeReviewAgent(BaseAgent):
    """Agent responsible for generating high-quality intermediate requirements (IRs) for code components."""
    
    def __init__(self, config_path: Optional[str] = None, performance_metrics: Optional[PerformanceMetrics] = None):
        super().__init__("CodeReviewer", config_path=config_path, performance_metrics=performance_metrics)
        
        # Enhanced base prompt for IR generation
        self.base_prompt = """You are a Code Reviewer agent specialized in generating Intermediate Requirements (IRs) from code components. Your role is to bridge the gap between low-level code implementation and high-level business requirements.

CORE PRINCIPLES:
1. Focus on WHAT the code does from a user/business perspective, not HOW it's implemented
2. Describe the business value and functional purpose
3. Identify key inputs, outputs, and transformations
4. Note important constraints, preconditions, and side effects
5. Use clear, business-oriented language while maintaining technical accuracy

CONTEXT AWARENESS:
- Leverage dependency information to understand component relationships
- Consider how this component fits into the larger system architecture
- Reference dependent components when explaining functionality
- Identify integration points and data flows

OUTPUT REQUIREMENTS:
- Generate concise, actionable intermediate requirements
- Focus on functional behavior and business logic
- Avoid implementation details unless they affect external behavior
- Maintain consistency with dependent component requirements"""

        self.add_to_memory("system", self.base_prompt)
        
        # Component-specific prompts
        self.class_prompt = """For CLASS components, focus on:
1. BUSINESS ENTITY: What real-world concept or business entity does this class represent?
2. RESPONSIBILITIES: What are the primary responsibilities and capabilities?
3. STATE MANAGEMENT: What key data does it manage and how?
4. COLLABORATIONS: How does it interact with other components?
5. LIFECYCLE: What are the key states and transitions?

Generate an IR that describes the class as a business component, not a code construct."""

        self.method_prompt = """For METHOD/FUNCTION components, focus on:
1. BUSINESS OPERATION: What business operation or process does this method perform?
2. INPUT REQUIREMENTS: What data or conditions are required?
3. BUSINESS LOGIC: What transformations or decisions are made?
4. OUTPUT DELIVERY: What results or side effects are produced?
5. ERROR CONDITIONS: What business rules or constraints are enforced?

Generate an IR that describes the method as a business process step."""

        self.file_prompt = """For FILE-level summaries, focus on:
1. COHESIVE CAPABILITY: What unified business capability does this file provide?
2. COMPONENT ORCHESTRATION: How do the internal components work together?
3. EXTERNAL INTERFACES: What services does it provide to other parts of the system?
4. DATA FLOWS: What are the key data inputs and outputs?
5. BUSINESS CONTEXT: How does this file contribute to overall business objectives?

Generate an IR that describes the file as a business module."""

    def _get_custom_prompt(self, component_type: str) -> str:
        """Get component-specific prompt."""
        if component_type in ["class", "interface", "enum"]:
            return self.class_prompt
        elif component_type in ["method", "function", "constructor"]:
            return self.method_prompt
        elif component_type == "file":
            return self.file_prompt
        else:
            return ""

    def extract_docstring(self, response: str) -> str:
        """Extract the IR from the LLM response."""
        start_tag = "<DOCSTRING>"
        end_tag = "</DOCSTRING>"
        
        try:
            start_idx = response.index(start_tag) + len(start_tag)
            end_idx = response.index(end_tag)
            return response[start_idx:end_idx].strip()
        except ValueError:
            logger.warning("No DOCSTRING XML tags found, returning full response")
            return response.strip()

    def _process_component_ir(self, component_code: str, component_context: Dict[str, Any]) -> str:
        """Generate IR for a single component."""
        component_type = component_context.get("component_type", "")
        component_id = component_context.get("component_id", "")
        
        # Build dependency context
        deps_info = ""
        neighbors = component_context.get("neighbors", [])
        if neighbors:
            deps_info = "\nDEPENDENCY CONTEXT:\n"
            for neighbor in neighbors[:5]:  # Limit to top 5 dependencies
                deps_info += f"- {neighbor.get('type', 'component')} '{neighbor.get('id', '')}': {neighbor.get('hint', 'No description available')}\n"
        
        task_description = f"""Generate an Intermediate Requirement (IR) for the following code component.

COMPONENT INFORMATION:
- ID: {component_id}
- Type: {component_type}
- File: {component_context.get('file_path', '')}
{deps_info}

{self._get_custom_prompt(component_type)}

SOURCE CODE:
```
{component_code}
```

TASK:
1. Analyze the code component in its business context
2. Consider the dependency relationships and how they inform the component's purpose
3. Generate a clear, business-oriented IR that describes WHAT this component does
4. Focus on functional requirements, not implementation details
5. Ensure the IR is actionable and testable from a business perspective

OUTPUT FORMAT:
Wrap your IR in XML tags: <DOCSTRING>Your IR here</DOCSTRING>

The IR should be 3-6 sentences that clearly describe the business functionality and requirements."""

        self.add_to_memory("user", task_description)
        full_response = self.generate_response()
        self.clear_memory()  # Clear memory for next component
        self.add_to_memory("system", self.base_prompt)  # Re-add system prompt
        
        return self.extract_docstring(full_response)

    def process_file_summary(self, file_context: Dict[str, Any]) -> str:
        """Generate file-level IR by aggregating component IRs."""
        file_path = file_context.get("file_path", "")
        neighbors = file_context.get("neighbors", [])
        component_summaries = file_context.get("component_summaries", [])
        
        # Build component context
        comp_info = ""
        if component_summaries:
            comp_info = "\nCOMPONENT SUMMARIES:\n"
            for comp in component_summaries:
                comp_info += f"- {comp.get('type', 'component')} '{comp.get('id', '')}': {comp.get('summary', '')}\n"
        
        # Build neighbor context
        neighbor_info = ""
        if neighbors:
            neighbor_info = f"\nNEIGHBOR FILES: {', '.join(neighbors[:5])}"
        
        task_description = f"""Generate a file-level Intermediate Requirement (IR) by synthesizing component IRs.

FILE INFORMATION:
- Path: {file_path}
{neighbor_info}
{comp_info}

{self.file_prompt}

TASK:
1. Analyze how the components within this file work together
2. Identify the unified business capability this file provides
3. Consider the file's role in the broader system architecture
4. Generate a cohesive IR that describes the file's business purpose

OUTPUT:
Provide a 4-8 sentence IR that describes the file's business functionality and its role in the system."""

        self.add_to_memory("user", task_description)
        response = self.generate_response()
        self.clear_memory()
        self.add_to_memory("system", self.base_prompt)
        
        return response.strip()

    def process(self, component_code: str, file_content: Optional[str], component_context: Dict[str, Any], comp_type: str) -> str:
        """Main processing method for generating IRs."""
        if comp_type in ["class", "interface", "enum", "method", "function", "constructor"] and component_code is not None:
            return self._process_component_ir(component_code, component_context)
        elif comp_type == "file" and file_content is not None:
            return self.process_file_summary(component_context)
        else:
            return "Unable to process: invalid component type or missing content"