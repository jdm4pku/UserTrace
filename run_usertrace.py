#!/usr/bin/env python3
"""
UserTrace: A comprehensive system for generating use cases from code repositories
and maintaining traceability links between use cases and code components.

This implementation integrates four specialized agents (Code Reviewer, Searcher, Writer, Verifier)
across three phases: repository structuring, IR derivation, and UR generation.
"""

import os
import sys
import json
import logging
import argparse
import random
import time
import ast
import re
import builtins
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, Any, Set, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

try:
    import requests
except ImportError:
    requests = None

try:
    import javalang
    import javalang.ast
    from javalang.tree import (
        CompilationUnit, ClassDeclaration, InterfaceDeclaration, EnumDeclaration,
        MethodDeclaration, ConstructorDeclaration, FieldDeclaration, VariableDeclarator,
        FormalParameter, MemberReference, MethodInvocation, SuperMethodInvocation,
        ClassCreator, ReferenceType, BasicType, LocalVariableDeclaration,
        MethodReference, ArrayCreator, Cast, BinaryOperation
    )
except ImportError:
    javalang = None

try:
    import igraph as ig
    import leidenalg as la
except ImportError:
    ig = None
    la = None

try:
    from colorama import Fore, Back, Style, init
    from tqdm import tqdm
except ImportError:
    # Fallback implementations
    class Fore:
        CYAN = GREEN = YELLOW = RED = ""
    class Style:
        BRIGHT = RESET_ALL = ""
    def init(): pass
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.n = 0
        def update(self, n=1): 
            self.n += n
        def close(self): pass

# -------------------- Logging Configuration --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("user_trace")

# -------------------- Constants --------------------
EXCLUDED_QUALIFIERS = {"this", "super"}
JAVA_LANG_IMPLICIT = "java.lang"
BUILTIN_TYPES = {name for name in dir(builtins)}
STANDARD_MODULES = {
    'abc', 'argparse', 'array', 'asyncio', 'base64', 'collections', 'copy', 'csv', 
    'datetime', 'enum', 'functools', 'glob', 'io', 'itertools', 'json', 'logging', 
    'math', 'os', 'pathlib', 'random', 're', 'shutil', 'string', 'sys', 'time', 
    'typing', 'uuid', 'warnings', 'xml'
}
EXCLUDED_NAMES = {'self', 'cls'}

# -------------------- Data Classes --------------------
@dataclass
class JavaCodeComponent:
    """Represents one code component in a Java repo."""
    id: str
    node: Any
    component_type: str  # class|interface|enum|method|constructor
    file_path: str
    relative_path: str
    depends_on: Set[str] = field(default_factory=set)
    source_code: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    has_docstring: bool = False
    docstring: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "component_type": self.component_type,
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "depends_on": sorted(self.depends_on),
            "source_code": self.source_code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "has_docstring": self.has_docstring,
            "docstring": self.docstring,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "JavaCodeComponent":
        return JavaCodeComponent(
            id=data["id"],
            node=None,
            component_type=data["component_type"],
            file_path=data["file_path"],
            relative_path=data["relative_path"],
            depends_on=set(data.get("depends_on", [])),
            source_code=data.get("source_code"),
            start_line=data.get("start_line", 0),
            end_line=data.get("end_line", 0),
            has_docstring=data.get("has_docstring", False),
            docstring=data.get("docstring", ""),
        )

@dataclass
class PyCodeComponent:
    """Represents a single code component in a Python codebase."""
    id: str
    node: ast.AST
    component_type: str
    file_path: str
    relative_path: str
    depends_on: Set[str] = field(default_factory=set)
    source_code: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    has_docstring: bool = False
    docstring: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'component_type': self.component_type,
            'file_path': self.file_path,
            'relative_path': self.relative_path,
            'depends_on': list(self.depends_on),
            'start_line': self.start_line,
            'end_line': self.end_line,
            'has_docstring': self.has_docstring,
            'docstring': self.docstring
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PyCodeComponent':
        return PyCodeComponent(
            id=data['id'],
            node=None,
            component_type=data['component_type'],
            file_path=data['file_path'],
            relative_path=data['relative_path'],
            depends_on=set(data.get('depends_on', [])),
            start_line=data.get('start_line', 0),
            end_line=data.get('end_line', 0),
            has_docstring=data.get('has_docstring', False),
            docstring=data.get('docstring', "")
        )

@dataclass
class ImportContext:
    """Stores package/imports for resolving simple names."""
    package: str
    regular_imports: Dict[str, str] = field(default_factory=dict)
    on_demand_imports: List[str] = field(default_factory=list)
    static_imports: Dict[str, str] = field(default_factory=dict)
    static_on_demand_types: List[str] = field(default_factory=list)

# -------------------- Performance Monitoring --------------------
@dataclass
class TokenUsage:
    """Track token usage for LLM calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage from a single call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)

@dataclass
class PerformanceMetrics:
    """Track performance metrics for the UserTrace system."""
    start_time: float = 0.0
    end_time: float = 0.0
    total_runtime: float = 0.0
    api_wait_time: float = 0.0  # Time spent waiting due to API errors
    effective_runtime: float = 0.0  # Total runtime minus API wait time
    
    # Token usage by agent
    reviewer_tokens: TokenUsage = field(default_factory=TokenUsage)
    searcher_tokens: TokenUsage = field(default_factory=TokenUsage)
    writer_tokens: TokenUsage = field(default_factory=TokenUsage)
    verifier_tokens: TokenUsage = field(default_factory=TokenUsage)
    
    # Call counts
    reviewer_calls: int = 0
    searcher_calls: int = 0
    writer_calls: int = 0
    verifier_calls: int = 0
    
    # Error tracking
    api_errors: int = 0
    retry_attempts: int = 0
    
    def start_timing(self):
        """Start timing the operation."""
        self.start_time = time.time()
    
    def end_timing(self):
        """End timing and calculate durations."""
        self.end_time = time.time()
        self.total_runtime = self.end_time - self.start_time
        self.effective_runtime = self.total_runtime - self.api_wait_time
    
    def add_api_wait_time(self, wait_time: float):
        """Add time spent waiting due to API errors."""
        self.api_wait_time += wait_time
    
    def get_total_tokens(self) -> TokenUsage:
        """Get total token usage across all agents."""
        total = TokenUsage()
        for agent_tokens in [self.reviewer_tokens, self.searcher_tokens, 
                           self.writer_tokens, self.verifier_tokens]:
            total.add_usage(agent_tokens.input_tokens, agent_tokens.output_tokens)
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        total_tokens = self.get_total_tokens()
        return {
            "timing": {
                "total_runtime_seconds": round(self.total_runtime, 2),
                "effective_runtime_seconds": round(self.effective_runtime, 2),
                "api_wait_time_seconds": round(self.api_wait_time, 2),
                "start_time": self.start_time,
                "end_time": self.end_time
            },
            "token_usage": {
                "total": {
                    "input_tokens": total_tokens.input_tokens,
                    "output_tokens": total_tokens.output_tokens,
                    "total_tokens": total_tokens.total_tokens
                },
                "by_agent": {
                    "reviewer": {
                        "input_tokens": self.reviewer_tokens.input_tokens,
                        "output_tokens": self.reviewer_tokens.output_tokens,
                        "total_tokens": self.reviewer_tokens.total_tokens,
                        "calls": self.reviewer_calls
                    },
                    "searcher": {
                        "input_tokens": self.searcher_tokens.input_tokens,
                        "output_tokens": self.searcher_tokens.output_tokens,
                        "total_tokens": self.searcher_tokens.total_tokens,
                        "calls": self.searcher_calls
                    },
                    "writer": {
                        "input_tokens": self.writer_tokens.input_tokens,
                        "output_tokens": self.writer_tokens.output_tokens,
                        "total_tokens": self.writer_tokens.total_tokens,
                        "calls": self.writer_calls
                    },
                    "verifier": {
                        "input_tokens": self.verifier_tokens.input_tokens,
                        "output_tokens": self.verifier_tokens.output_tokens,
                        "total_tokens": self.verifier_tokens.total_tokens,
                        "calls": self.verifier_calls
                    }
                }
            },
            "error_stats": {
                "api_errors": self.api_errors,
                "retry_attempts": self.retry_attempts
            }
        }

# -------------------- LLM Base Classes --------------------
class BaseLLM(ABC):
    """Base class for all LLM implementations."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 4096) -> Tuple[str, TokenUsage]:
        """Generate response from messages. Returns (response, token_usage)."""
        pass
    
    def format_message(self, role: str, content: str) -> Dict[str, str]:
        """Format a message for the LLM."""
        return {"role": role, "content": content}

class MockLLM(BaseLLM):
    """Mock LLM for testing purposes."""
    
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 4096) -> Tuple[str, TokenUsage]:
        """Generate a mock response with token usage."""
        if not messages:
            return "Mock response", TokenUsage(50, 20, 70)
        
        # Calculate mock token usage based on message length
        total_input_chars = sum(len(msg["content"]) for msg in messages)
        input_tokens = max(10, total_input_chars // 4)  # Rough approximation: 4 chars per token
        
        last_message = messages[-1]["content"].lower()
        
        # Mock responses based on content with token usage
        if "docstring" in last_message or "component" in last_message:
            response = """<DOCSTRING>
This component handles core functionality for the system.
It processes input data and returns processed results.
Used in the main workflow for data transformation.
</DOCSTRING>"""
            output_tokens = len(response) // 4
            return response, TokenUsage(input_tokens, output_tokens, input_tokens + output_tokens)
        
        elif "file" in last_message and "summary" in last_message:
            response = "This file provides essential functionality for data processing and system integration."
            output_tokens = len(response) // 4
            return response, TokenUsage(input_tokens, output_tokens, input_tokens + output_tokens)
        
        elif "intent" in last_message or "requirement" in last_message or "use case" in last_message:
            response = """{
    "use_case_name": "Process Data Files",
    "description": "This use case allows users to upload, validate, and process data files through the system, transforming raw data into structured output formats.",
    "primary_actor": "Data Analyst",
    "secondary_actors": [
        "File Storage System",
        "Data Validation Service",
        "Notification System"
    ],
    "preconditions": [
        "User is authenticated and authorized",
        "System is operational and accessible",
        "File storage system is available",
        "Data processing engine is running"
    ],
    "main_flow": [
        "1. Data Analyst selects data file to upload",
        "2. System validates file format and size constraints",
        "3. System uploads file to storage and confirms receipt",
        "4. System initiates data processing workflow",
        "5. Data Validation Service checks data integrity and format",
        "6. System transforms data according to business rules",
        "7. System generates processed output in specified format",
        "8. System notifies Data Analyst of completion",
        "9. Data Analyst downloads or accesses processed results"
    ],
    "alternative_flows": [
        "A1: If file format is invalid, system displays error message and prompts for correct format",
        "A2: If file size exceeds limit, system rejects upload and suggests file compression",
        "A3: If data validation fails, system logs errors and provides detailed error report",
        "A4: If processing fails, system rolls back changes and notifies administrator"
    ],
    "postconditions": [
        "Data file is successfully processed and stored",
        "Processed data is available in specified output format",
        "Processing log is created with timestamp and status",
        "User notification is sent confirming completion"
    ],
    "key_files": ["main.py", "processor.py", "validator.py"]
}"""
            output_tokens = len(response) // 4
            return response, TokenUsage(input_tokens, output_tokens, input_tokens + output_tokens)
        
        elif "verify" in last_message or "evaluate" in last_message:
            response = """{
    "needs_revision": false,
    "needs_context": false,
    "context_suggestion": "",
    "suggestion": "",
    "score": 8
}"""
            output_tokens = len(response) // 4
            return response, TokenUsage(input_tokens, output_tokens, input_tokens + output_tokens)
        
        elif "search" in last_message:
            response = "External knowledge: This relates to common data processing patterns in enterprise software systems."
            output_tokens = len(response) // 4
            return response, TokenUsage(input_tokens, output_tokens, input_tokens + output_tokens)
        
        response = "Mock response for: " + last_message[:100]
        output_tokens = len(response) // 4
        return response, TokenUsage(input_tokens, output_tokens, input_tokens + output_tokens)

# -------------------- Agent Base Class --------------------
class BaseAgent(ABC):
    """Base class for all agents in the UserTrace system."""
    
    def __init__(self, name: str, config_path: Optional[str] = None, performance_metrics: Optional[PerformanceMetrics] = None):
        self.name = name
        self._memory: List[Dict[str, Any]] = []
        self.performance_metrics = performance_metrics
        
        # For this implementation, we'll use MockLLM
        # In production, this would load from config and create appropriate LLM
        self.llm = MockLLM()
        self.llm_params = {
            "max_output_tokens": 4096,
            "temperature": 0.1,
            "model": "mock-model"
        }
    
    def add_to_memory(self, role: str, content: str) -> None:
        """Add a message to the agent's memory."""
        assert content is not None and content != "", "Content cannot be empty"
        self._memory.append(self.llm.format_message(role, content))
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self._memory = []
    
    @property
    def memory(self) -> List[Dict[str, Any]]:
        """Get the agent's memory."""
        return self._memory.copy()
    
    def generate_response(self, messages: Optional[List[Dict[str, Any]]] = None, retry_on_error: bool = True) -> str:
        """Generate a response using the agent's LLM and memory with performance tracking."""
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                call_start_time = time.time()
                
                response, token_usage = self.llm.generate(
                    messages=messages if messages is not None else self._memory,
                    temperature=self.llm_params["temperature"],
                    max_tokens=self.llm_params["max_output_tokens"]
                )
                
                # Record successful call metrics
                if self.performance_metrics:
                    self._record_successful_call(token_usage)
                
                return response
                
            except Exception as e:
                retry_count += 1
                wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30 seconds
                
                if self.performance_metrics:
                    self.performance_metrics.api_errors += 1
                    self.performance_metrics.retry_attempts += 1
                    self.performance_metrics.add_api_wait_time(wait_time)
                
                if retry_count > max_retries or not retry_on_error:
                    logger.error(f"LLM call failed after {retry_count} attempts: {e}")
                    # Return a fallback response
                    return f"Error: Unable to generate response after {retry_count} attempts"
                
                logger.warning(f"LLM call failed (attempt {retry_count}/{max_retries}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
        
        return "Error: Maximum retries exceeded"
    
    def _record_successful_call(self, token_usage: TokenUsage):
        """Record metrics for a successful LLM call."""
        if not self.performance_metrics:
            return
            
        # Record token usage based on agent type
        if self.name.lower() == "codereviewagent" or "reviewer" in self.name.lower():
            self.performance_metrics.reviewer_tokens.add_usage(token_usage.input_tokens, token_usage.output_tokens)
            self.performance_metrics.reviewer_calls += 1
        elif self.name.lower() == "searchagent" or "search" in self.name.lower():
            self.performance_metrics.searcher_tokens.add_usage(token_usage.input_tokens, token_usage.output_tokens)
            self.performance_metrics.searcher_calls += 1
        elif self.name.lower() == "writeragent" or "writer" in self.name.lower():
            self.performance_metrics.writer_tokens.add_usage(token_usage.input_tokens, token_usage.output_tokens)
            self.performance_metrics.writer_calls += 1
        elif self.name.lower() == "verifieragent" or "verifier" in self.name.lower():
            self.performance_metrics.verifier_tokens.add_usage(token_usage.input_tokens, token_usage.output_tokens)
            self.performance_metrics.verifier_calls += 1
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process the input and generate output."""
        pass

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

# -------------------- Search Agent --------------------
class SearchAgent(BaseAgent):
    """Agent for retrieving external business knowledge to enrich requirement generation."""
    
    def __init__(self, config_path: Optional[str] = None, performance_metrics: Optional[PerformanceMetrics] = None):
        super().__init__("SearchAgent", config_path=config_path, performance_metrics=performance_metrics)
        
        # Mock configuration - in production this would come from config file
        self.search_cfg = {
            "provider": "mock",
            "api_key": "mock_key",
            "endpoint": "mock_endpoint",
            "k": 5
        }
        
        self.provider = self.search_cfg.get("provider")
        self.api_key = self.search_cfg.get("api_key")
        self.endpoint = self.search_cfg.get("endpoint")
        self.k = int(self.search_cfg.get("k", 5))

    def search(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Search for external knowledge related to the query."""
        k = k or self.k
        results: List[Dict[str, Any]] = []
        
        # Mock search results for demonstration
        if self.provider == "mock":
            mock_results = [
                {
                    "title": f"Business Process Documentation for {query[:30]}",
                    "link": "https://example.com/docs1",
                    "snippet": f"Industry best practices for {query} include standardized workflows, data validation, and user experience optimization."
                },
                {
                    "title": f"Technical Implementation Guide - {query[:30]}",
                    "link": "https://example.com/docs2", 
                    "snippet": f"Common patterns for {query} involve modular design, error handling, and performance considerations."
                },
                {
                    "title": f"User Requirements Analysis - {query[:30]}",
                    "link": "https://example.com/docs3",
                    "snippet": f"User expectations for {query} typically focus on reliability, ease of use, and clear feedback mechanisms."
                }
            ]
            results = mock_results[:k]
        
        # In production, this would make actual API calls to search engines
        # try:
        #     if self.provider == "serpapi" and requests:
        #         # SerpAPI implementation
        #         pass
        #     elif self.provider == "bing" and requests:
        #         # Bing Search API implementation  
        #         pass
        # except Exception as e:
        #     logger.warning(f"Search API error: {e}")
        
        summary = self._summarize_results(query, results)
        return {"query": query, "results": results, "summary": summary}

    def _summarize_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Summarize search results to provide business context."""
        if not results:
            return f"No external knowledge found for: {query}"
        
        bullet_points = "\n".join([
            f"- {r.get('title', 'Unknown')}: {r.get('snippet', 'No description')}" 
            for r in results[:3]
        ])
        
        prompt = f"""Synthesize the following search results into actionable business knowledge for requirement generation.

SEARCH QUERY: {query}

SEARCH RESULTS:
{bullet_points}

TASK:
Extract key business insights, industry standards, and user expectations that should inform requirement generation. Focus on:
1. Common business workflows and processes
2. Industry best practices and standards  
3. User experience expectations
4. Integration patterns and constraints
5. Quality and performance considerations

Provide 3-5 concise sentences that capture the most relevant business context."""

        self.add_to_memory("user", prompt)
        response = self.generate_response()
        self.clear_memory()
        
        return response.strip()

    def process(self, *args, **kwargs) -> Any:
        """Process method for compatibility with BaseAgent."""
        query = kwargs.get("query") or (args[0] if args else "")
        return self.search(query)

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

# -------------------- Progress Visualizer --------------------
class ProgressVisualizer:
    """Visualizes the progress of IR generation in the terminal."""
    
    def __init__(self, components: Dict[str, Any], sorted_order: List[str], repo_type: str):
        init()  # Initialize colorama
        self.components = components
        self.sorted_order = sorted_order
        self.repo_type = repo_type
        self.processed = set()
        self.current = None
        self.progress_bar = None
        self.start_time = time.time()

    def initialize(self):
        """Initialize the visualization."""
        self._clear_screen()
        self._print_header()
        self.progress_bar = tqdm(
            total=len(self.sorted_order),
            desc="Generating IRs",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

    def update(self, component_id: str = None, status: str = "processing"):
        """Update the visualization."""
        if component_id is not None:
            self.current = component_id
            if status == "completed":
                self.processed.add(component_id)
                self.progress_bar.update(1)

    def finalize(self):
        """Finalize the visualization."""
        if self.progress_bar:
            self.progress_bar.close()
        
        elapsed = time.time() - self.start_time
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        
        print(f"\n{Fore.GREEN}IR Generation Complete!{Style.RESET_ALL}")
        print(f"Total components processed: {len(self.processed)}/{len(self.sorted_order)}")
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    def show_dependency_stats(self):
        """Show dependency statistics."""
        total_deps = sum(len(getattr(comp, 'depends_on', set())) for comp in self.components.values())
        avg_deps = total_deps / len(self.components) if self.components else 0
        
        print(f"\n{Fore.CYAN}Dependency Graph Statistics:{Style.RESET_ALL}")
        print(f"Total components: {len(self.components)}")
        print(f"Average dependencies per component: {avg_deps:.2f}")

    def _clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_header(self):
        """Print header information."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}UserTrace IR Generation{Style.RESET_ALL}\n")
        print(f"Processing {len(self.sorted_order)} components in dependency order")

# -------------------- Utility Functions --------------------
def save_json(path: str, obj: Any):
    """Save object to JSON file with proper directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Any:
    """Load object from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def connected_components_undirected(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    """Find connected components in an undirected graph."""
    seen: Set[str] = set()
    comps: List[Set[str]] = []
    
    for node in graph.keys():
        if node in seen:
            continue
        
        q = deque([node])
        group: Set[str] = set([node])
        seen.add(node)
        
        while q:
            u = q.popleft()
            for v in graph[u]:
                if v not in seen:
                    seen.add(v)
                    group.add(v)
                    q.append(v)
        
        comps.append(group)
    
    return comps

def detect_file_communities_leiden(
    file_graph: Dict[str, Set[str]],
    edge_details: Optional[Dict[Tuple[str, str], List[Tuple[str, str]]]] = None,
    resolution: float = 1.0,
    iterations: Optional[int] = None,
    seed: Optional[int] = 42,
) -> List[Set[str]]:
    """
    Detect file communities using Leiden algorithm.
    Falls back to connected components if Leiden is unavailable.
    """
    try:
        import igraph as ig
        import leidenalg as la
    except Exception as e:
        logger.warning(f"Leiden unavailable ({e}); falling back to connected components.")
        comms = connected_components_undirected(file_graph)
        return sorted(comms, key=lambda g: (-len(g), sorted(list(g))[0] if g else ""))
    
    # Build vertex list and index mapping
    vertices: Set[str] = set(file_graph.keys())
    for u, nbrs in file_graph.items():
        vertices.update(nbrs)
    
    vlist = sorted(vertices)
    vidx = {v: i for i, v in enumerate(vlist)}
    
    # Build edges and weights
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    
    def get_weight(a: str, b: str) -> float:
        if not edge_details:
            return 1.0
        return float(len(edge_details.get((a, b), edge_details.get((b, a), []))))
    
    added = set()
    for a in vlist:
        for b in file_graph.get(a, set()):
            if a == b:
                continue
            
            e = (a, b) if a < b else (b, a)
            if e in added:
                continue
            
            added.add(e)
            ia, ib = vidx[e[0]], vidx[e[1]]
            edges.append((ia, ib))
            weights.append(get_weight(e[0], e[1]))
    
    # Create graph and run Leiden
    g = ig.Graph(n=len(vlist), edges=edges, directed=False)
    if weights:
        g.es["weight"] = weights
    
    partition_cls = la.RBConfigurationVertexPartition
    kwargs = {"resolution_parameter": resolution}
    if weights:
        kwargs["weights"] = g.es["weight"]
    if seed is not None:
        kwargs["seed"] = seed
    if iterations is not None:
        kwargs["n_iterations"] = iterations
    
    try:
        part = la.find_partition(g, partition_cls, **kwargs)
    except TypeError:
        part = la.find_partition(g, partition_cls, weights=g.es["weight"] if weights else None)
    
    # Convert back to file names
    comms: List[Set[str]] = []
    for comm in part:
        files = {vlist[i] for i in comm}
        comms.append(files)
    
    comms = sorted(comms, key=lambda s: (-len(s), sorted(list(s))[0] if s else ""))
    return comms

def build_file_graph_from_components(components: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], List[Tuple[str, str]]]]:
    """Build file-level dependency graph from component dependencies."""
    comp_to_file: Dict[str, str] = {}
    for cid, comp in components.items():
        f = getattr(comp, "relative_path", None) or getattr(comp, "file_path", "")
        comp_to_file[cid] = f
    
    file_graph: Dict[str, Set[str]] = defaultdict(set)
    edge_details: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    
    for cid, comp in components.items():
        src_file = comp_to_file.get(cid, "")
        if not src_file:
            continue
        
        for dep in getattr(comp, "depends_on", set()) or []:
            if dep not in components:
                continue
            
            dst_file = comp_to_file.get(dep, "")
            if not dst_file or dst_file == src_file:
                continue
            
            a, b = sorted([src_file, dst_file])
            file_graph[a].add(b)
            file_graph[b].add(a)
            edge_details[(a, b)].append((cid, dep))
    
    # Ensure all files are in the graph
    for f in set(comp_to_file.values()):
        _ = file_graph[f]
    
    return file_graph, edge_details

def _first_line(text: str, limit: int = 200) -> str:
    """Extract first line of text with length limit."""
    if not text:
        return ""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return (line[:limit]).strip()

def build_graph_from_components(components: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Build dependency graph from components."""
    graph = {}
    for comp_id, comp in components.items():
        depends_on = getattr(comp, 'depends_on', set()) or set()
        graph[comp_id] = depends_on
    return graph

def dependency_first_dfs(graph: Dict[str, Set[str]]) -> List[str]:
    """Perform topological sort using DFS."""
    visited = set()
    temp_visited = set()
    result = []
    
    def dfs(node):
        if node in temp_visited:
            return  # Cycle detected, skip
        if node in visited:
            return
        
        temp_visited.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor in graph:  # Only visit nodes that exist
                dfs(neighbor)
        temp_visited.remove(node)
        visited.add(node)
        result.append(node)
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return result

# -------------------- Parser Implementations --------------------
class JavaDependencyParser:
    """Simplified Java parser for demonstration."""
    
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.components: Dict[str, JavaCodeComponent] = {}
        self.modules: Set[str] = set()
        self.classes: Set[str] = set()

    def parse_repository(self) -> Dict[str, JavaCodeComponent]:
        """Parse Java repository (simplified implementation)."""
        logger.info(f"Parsing Java repository at {self.repo_path}")
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if not file.endswith(".java"):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                
                try:
                    self._parse_file_simple(file_path, relative_path)
                except Exception as e:
                    logger.warning(f"Error parsing {file_path}: {e}")
        
        logger.info(f"Found {len(self.components)} Java components")
        return self.components

    def _parse_file_simple(self, file_path: str, relative_path: str):
        """Simplified Java file parsing."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            lines = content.splitlines()
            package = self._extract_package(content)
            
            # Simple class detection
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("public class ") or line.startswith("class "):
                    class_name = line.split()[2 if "public" in line else 1]
                    class_id = f"{package}.{class_name}" if package else class_name
                    
                    self.components[class_id] = JavaCodeComponent(
                        id=class_id,
                        node=None,
                        component_type="class",
                        file_path=file_path,
                        relative_path=relative_path,
                        source_code=content,
                        start_line=i + 1,
                        end_line=len(lines),
                        has_docstring=False,
                        docstring=""
                    )
                    self.classes.add(class_id)
                    break
                    
        except Exception as e:
            logger.warning(f"Error parsing Java file {file_path}: {e}")

    def _extract_package(self, content: str) -> str:
        """Extract package declaration."""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("package "):
                return line.replace("package ", "").replace(";", "").strip()
        return ""

    def save_dependency_graph(self, output_path: str):
        """Save dependency graph to JSON."""
        data = {cid: c.to_dict() for cid, c in self.components.items()}
        save_json(output_path, data)

    def load_dependency_graph(self, input_path: str) -> Dict[str, JavaCodeComponent]:
        """Load dependency graph from JSON."""
        raw = load_json(input_path)
        self.components = {cid: JavaCodeComponent.from_dict(d) for cid, d in raw.items()}
        return self.components

class PyDependencyParser:
    """Python dependency parser."""
    
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.components: Dict[str, PyCodeComponent] = {}
        self.modules: Set[str] = set()

    def parse_repository(self) -> Dict[str, PyCodeComponent]:
        """Parse Python repository."""
        logger.info(f"Parsing Python repository at {self.repo_path}")
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if not file.endswith(".py"):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                module_path = self._file_to_module_path(relative_path)
                self.modules.add(module_path)
                
                try:
                    self._parse_file(file_path, relative_path, module_path)
                except Exception as e:
                    logger.warning(f"Error parsing {file_path}: {e}")
        
        logger.info(f"Found {len(self.components)} Python components")
        return self.components

    def _file_to_module_path(self, file_path: str) -> str:
        """Convert file path to module path."""
        path = file_path[:-3] if file_path.endswith(".py") else file_path
        return path.replace(os.path.sep, ".")

    def _parse_file(self, file_path: str, relative_path: str, module_path: str):
        """Parse a Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            self._collect_components(tree, file_path, relative_path, module_path, source)
            
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error parsing {file_path}: {e}")

    def _collect_components(self, tree: ast.AST, file_path: str, relative_path: str, module_path: str, source: str):
        """Collect components from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_id = f"{module_path}.{node.name}"
                has_docstring = self._has_docstring(node)
                docstring = self._get_docstring(node) if has_docstring else ""
                
                component = PyCodeComponent(
                    id=class_id,
                    node=node,
                    component_type="class",
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=self._get_source_segment(source, node),
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    has_docstring=has_docstring,
                    docstring=docstring
                )
                self.components[class_id] = component
                
                # Collect methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_id = f"{class_id}.{item.name}"
                        method_has_docstring = self._has_docstring(item)
                        method_docstring = self._get_docstring(item) if method_has_docstring else ""
                        
                        method_component = PyCodeComponent(
                            id=method_id,
                            node=item,
                            component_type="method",
                            file_path=file_path,
                            relative_path=relative_path,
                            source_code=self._get_source_segment(source, item),
                            start_line=item.lineno,
                            end_line=getattr(item, "end_lineno", item.lineno),
                            has_docstring=method_has_docstring,
                            docstring=method_docstring
                        )
                        self.components[method_id] = method_component
                        
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only collect top-level functions
                if hasattr(node, 'parent') and isinstance(getattr(node, 'parent', None), ast.Module):
                    func_id = f"{module_path}.{node.name}"
                    has_docstring = self._has_docstring(node)
                    docstring = self._get_docstring(node) if has_docstring else ""
                    
                    component = PyCodeComponent(
                        id=func_id,
                        node=node,
                        component_type="function",
                        file_path=file_path,
                        relative_path=relative_path,
                        source_code=self._get_source_segment(source, node),
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        has_docstring=has_docstring,
                        docstring=docstring
                    )
                    self.components[func_id] = component

    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if node has a docstring."""
        if hasattr(node, 'body') and node.body:
            first = node.body[0]
            return (isinstance(first, ast.Expr) and 
                   isinstance(first.value, ast.Constant) and 
                   isinstance(first.value.value, str))
        return False

    def _get_docstring(self, node: ast.AST) -> str:
        """Extract docstring from node."""
        if self._has_docstring(node):
            return node.body[0].value.value
        return ""

    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """Get source code segment for node."""
        try:
            if hasattr(ast, "get_source_segment"):
                segment = ast.get_source_segment(source, node)
                if segment is not None:
                    return segment
            
            lines = source.split("\n")
            start_line = node.lineno - 1
            end_line = getattr(node, "end_lineno", node.lineno) - 1
            return "\n".join(lines[start_line:end_line + 1])
        except Exception:
            return ""

    def save_dependency_graph(self, output_path: str):
        """Save dependency graph to JSON."""
        data = {cid: c.to_dict() for cid, c in self.components.items()}
        save_json(output_path, data)

    def load_dependency_graph(self, input_path: str) -> Dict[str, PyCodeComponent]:
        """Load dependency graph from JSON."""
        raw = load_json(input_path)
        self.components = {cid: PyCodeComponent.from_dict(d) for cid, d in raw.items()}
        return self.components

# -------------------- IR Generation Functions --------------------
def generate_component_ir(component, dep_graph, rev_graph, visualizer, components, reviewer: CodeReviewAgent):
    """Generate Intermediate Requirement (IR) for a single component."""
    comp_id = component.id
    comp_type = component.component_type
    file_path = component.file_path
    component_code = component.source_code or ""
    
    deps = dep_graph.get(comp_id, [])
    revs = rev_graph.get(comp_id, [])
    
    # Build neighbor hints with existing IRs
    neighbor_hints = []
    for d in deps:
        dtype = "component"
        dfile = ""
        hint = ""
        if d in components:
            dc = components[d]
            dtype = getattr(dc, "component_type", "component")
            dfile = getattr(dc, "relative_path", getattr(dc, "file_path", ""))
            if getattr(dc, "has_docstring", False) and getattr(dc, "docstring", ""):
                hint = _first_line(dc.docstring)
        neighbor_hints.append({
            "id": d,
            "type": dtype,
            "file": dfile,
            "hint": hint
        })
    
    context = {
        "component_id": comp_id,
        "component_type": getattr(component, "component_type", "component"),
        "file_path": getattr(component, "relative_path", file_path),
        "depends_on": deps,
        "depended_by": revs,
        "neighbors": neighbor_hints
    }
    
    visualizer.update(comp_id, "processing")
    
    # Generate IR using Code Reviewer agent
    file_content = None
    summary = reviewer.process(component_code, file_content, context, comp_type)
    
    # Update component with generated IR
    component.docstring = summary
    component.has_docstring = True
    
    visualizer.update(comp_id, "completed")
    return summary

def generate_component_level_irs(repo_path, sorted_components, components, dep_graph, rev_graph, visualizer, overwrite, reviewer: CodeReviewAgent):
    """Generate IRs for all components in dependency order."""
    comp_requirements: Dict[str, Dict[str, Any]] = {}
    
    for component_id in sorted_components:
        component = components.get(component_id)
        if not component:
            logger.warning(f"Component {component_id} not found in parsed components")
            continue
        
        # Skip constructors as they typically don't need separate requirements
        if getattr(component, "component_type", "") == "constructor":
            logger.info(f"Skipping {component_id} - constructor methods don't need separate IRs")
            visualizer.update(component_id, "completed")
            continue
        
        logger.info(f"Generating IR for {component_id}")
        summary_text = generate_component_ir(component, dep_graph, rev_graph, visualizer, components, reviewer)
        
        comp_requirements[component_id] = {
            "id": component_id,
            "type": getattr(component, "component_type", "component"),
            "file": getattr(component, "relative_path", getattr(component, "file_path", "")),
            "summary": summary_text
        }
    
    return comp_requirements

def generate_file_level_irs(file_graph, file_to_components, comp_requirements, reviewer: CodeReviewAgent):
    """Generate file-level IRs by aggregating component IRs."""
    file_requirements = {}
    
    for fpath in file_graph.keys():
        # Collect component summaries for this file
        parts = []
        for cid in file_to_components.get(fpath, []):
            if cid in comp_requirements:
                parts.append({
                    "id": cid,
                    "type": comp_requirements[cid]["type"],
                    "summary": comp_requirements[cid]["summary"]
                })
        
        # Get neighboring files
        neighbors = sorted(list(file_graph.get(fpath, set())))
        
        # Build context for file-level IR generation
        ctx = {
            "file_path": fpath,
            "neighbors": neighbors,
            "component_summaries": parts
        }
        
        # Generate file-level IR
        file_summary = reviewer.process_file_summary(ctx)
        
        file_requirements[fpath] = {
            "file": fpath,
            "summary": file_summary,
            "components": file_to_components.get(fpath, [])
        }
    
    return file_requirements

# -------------------- UR Generation Functions --------------------
def _safe_json_loads(text: str) -> dict:
    """Safely parse JSON text, return empty dict on failure."""
    try:
        return json.loads(text)
    except Exception:
        return {}

def parse_intent_verifier_response(resp: str) -> dict:
    """Parse verifier response in JSON or XML tag format."""
    import re
    
    # Try JSON first
    data = _safe_json_loads(resp)
    if data:
        return {
            "needs_revision": bool(data.get("needs_revision", False)),
            "needs_context": bool(data.get("needs_context", False)),
            "context_suggestion": data.get("context_suggestion", "") or data.get("context", "") or "",
            "suggestion": data.get("suggestion", "") or "",
            "score": int(data.get("score", 0)) if str(data.get("score", "")).isdigit() else None,
        }
    
    # Fall back to XML tags
    def _tag(tag):
        m = re.search(fr"<{tag}>(.*?)</{tag}>", resp, re.S | re.I)
        return m.group(1).strip() if m else ""
    
    need_rev_raw = _tag("NEED_REVISION").lower()
    more_ctx_raw = _tag("MORE_CONTEXT").lower()
    
    return {
        "needs_revision": (need_rev_raw == "true") or ("need" in need_rev_raw),
        "needs_context": (more_ctx_raw == "true") or ("context" in more_ctx_raw),
        "context_suggestion": _tag("SUGGESTION_CONTEXT"),
        "suggestion": _tag("SUGGESTION"),
        "score": None,
    }

def build_external_query_for_files(files: List[str], file_requirements: Dict[str, Dict[str, Any]], max_files: int = 3) -> str:
    """Build search query from file requirements."""
    chunks = []
    for f in list(files)[:max_files]:
        fr = file_requirements.get(f, {})
        chunks.append(os.path.basename(f) + " " + (fr.get("summary", "")[:120]))
    return " ".join(chunks)[:300]

def verify_intent(verifier: VerifierAgent, intent_obj: Dict[str, Any]) -> str:
    """Verify intent quality using Verifier agent."""
    return verifier.process(intent_obj)

def iterate_intent_for_community(
    idx: int,
    files: Set[str],
    file_requirements: Dict[str, Dict[str, Any]],
    component_requirements: Dict[str, Dict[str, Any]],
    file_graph: Dict[str, Set[str]],
    writer: WriterAgent,
    verifier: VerifierAgent,
    searcher: Optional[SearchAgent] = None,
    *,
    max_verifier_rejections: int = 3,
    max_search_attempts: int = 2
) -> Tuple[Dict[str, Any], str]:
    """
    Iteratively generate and refine use cases for a file community.
    Implements the verify-then-feedback mechanism.
    """
    ext_knowledge = ""
    verifier_rejection_count = 0
    search_attempts = 0
    feedback_text = ""
    improvement_prompts: List[str] = []
    
    while True:
        # Prepare advice from previous iterations
        advice = "\n".join(improvement_prompts[-2:])
        ext_for_writer = (ext_knowledge + ("\n\nVerifier advice:\n" + advice if advice else "")).strip()
        
        # Generate intent using Writer agent
        intent = writer._compose_intent(
            idx=idx,
            files=files,
            file_requirements=file_requirements,
            comp_requirements=component_requirements,
            file_graph=file_graph,
            ext_knowledge=ext_for_writer
        )
        intent["intent_id"] = f"intent_{idx}"
        
        # Verify intent quality
        feedback_text = verify_intent(verifier, intent)
        vr = parse_intent_verifier_response(feedback_text)
        
        # Check if verification passed or max rejections reached
        if not vr.get("needs_revision", False) or verifier_rejection_count >= max_verifier_rejections:
            return intent, feedback_text
        
        verifier_rejection_count += 1
        
        # Handle context needs
        if vr.get("needs_context", False) and searcher is not None and search_attempts < max_search_attempts:
            # Search for external knowledge
            query = build_external_query_for_files(sorted(list(files)), file_requirements)
            sr = searcher.search(query)
            ext_knowledge = (sr.get("summary", "") or "") + (
                "\n\n" + vr.get("context_suggestion", "") if vr.get("context_suggestion") else ""
            )
            search_attempts += 1
            continue
        else:
            # Add improvement suggestion
            suggestion = vr.get("suggestion", "")
            if suggestion:
                improvement_prompts.append(suggestion)
            else:
                improvement_prompts.append(
                    "Sharpen title; clarify inputs/outputs and guardrails; add 3-5 measurable acceptance criteria."
                )
            continue

# -------------------- Main UserTrace Class --------------------
class UserTrace:
    """
    Main UserTrace system that orchestrates the three phases:
    1. Repository Structuring
    2. IR Derivation  
    3. UR Generation
    """
    
    def __init__(self, repo_path: str, repo_type: str, config_path: str, output_dir: str, result_dir: str):
        self.repo_path = repo_path
        self.repo_name = os.path.basename(os.path.dirname(repo_path)) if os.path.basename(repo_path) == "code" else os.path.basename(repo_path)
        self.repo_type = repo_type
        self.config_path = config_path
        self.output_dir = os.path.join(output_dir, self.repo_name)
        self.result_dir = os.path.join(result_dir, self.repo_name)
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Initialize performance metrics
        self.performance_metrics = PerformanceMetrics()
        
        # Initialize components
        self.components: Dict[str, Any] = {}
        self.component_graph: Dict[str, Set[str]] = {}
        self.file_graph: Dict[str, Set[str]] = {}
        self.file_edge_details: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        
        # Initialize agents with performance metrics
        self.reviewer = CodeReviewAgent(config_path=config_path, performance_metrics=self.performance_metrics)
        self.searcher = SearchAgent(config_path=config_path, performance_metrics=self.performance_metrics)
        self.writer = WriterAgent(config_path=config_path, performance_metrics=self.performance_metrics)
        self.verifier = VerifierAgent(config_path=config_path, performance_metrics=self.performance_metrics)
        
        logger.info(f"UserTrace initialized for repository: {self.repo_name}, Language: {repo_type}")
    
    def phase1_repository_structuring(self, overwrite_dependency_graph: bool = False) -> Dict[str, Any]:
        """
        Phase 1: Repository Structuring
        Parse repository and build component and file dependency graphs.
        """
        logger.info("=== Phase 1: Repository Structuring ===")
        
        dependency_graph_path = os.path.join(self.output_dir, "dependency_graph.json")
        
        # Choose parser based on repository type
        if self.repo_type in ("Java", "C", "C++"):
            parser_engine = JavaDependencyParser(self.repo_path)
        else:
            parser_engine = PyDependencyParser(self.repo_path)
        
        # Load or build dependency graph
        if os.path.exists(dependency_graph_path) and not overwrite_dependency_graph:
            logger.info(f"Loading dependency graph from {dependency_graph_path}")
            self.components = parser_engine.load_dependency_graph(dependency_graph_path)
        else:
            logger.info("Parsing repository to build dependency graph")
            self.components = parser_engine.parse_repository()
            parser_engine.save_dependency_graph(dependency_graph_path)
        
        # Build component dependency graph
        self.component_graph = build_graph_from_components(self.components)
        
        # Build file dependency graph
        self.file_graph, self.file_edge_details = build_file_graph_from_components(self.components)
        
        # Save file dependency graph
        save_json(
            os.path.join(self.output_dir, "file_dependency_graph.json"),
            {k: sorted(list(v)) for k, v in self.file_graph.items()}
        )
        
        logger.info(f"Repository structuring complete: {len(self.components)} components, {len(self.file_graph)} files")
        return self.components
    
    def phase2_ir_derivation(self, overwrite_component_summary: bool = False, order_mode: str = "typo") -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Phase 2: IR Derivation
        Generate Intermediate Requirements for components and files in dependency order.
        """
        logger.info("=== Phase 2: IR Derivation ===")
        
        # Build dependency and reverse dependency graphs
        dep_graph: Dict[str, List[str]] = {k: sorted(list(v)) for k, v in self.component_graph.items()}
        rev_graph: Dict[str, List[str]] = {k: [] for k in dep_graph.keys()}
        for u, vs in dep_graph.items():
            for v in vs:
                rev_graph.setdefault(v, []).append(u)
        
        # Get topological order
        logger.info("Computing topological order for components")
        sorted_components = dependency_first_dfs(self.component_graph)
        logger.info(f"Sorted {len(sorted_components)} components")
        
        # Apply ordering strategy
        if order_mode == 'random':
            random.shuffle(sorted_components)
        elif order_mode == 'file':
            file_to_components = defaultdict(list)
            for cid in sorted_components:
                c = self.components.get(cid)
                if c:
                    file_to_components[c.file_path].append(cid)
            file_paths = list(file_to_components.keys())
            random.shuffle(file_paths)
            sorted_components = [cid for fp in file_paths for cid in file_to_components[fp]]
        
        # Initialize visualizer
        visualizer = ProgressVisualizer(self.components, sorted_components, self.repo_type)
        visualizer.initialize()
        visualizer.show_dependency_stats()
        
        # Generate component-level IRs
        component_requirements_path = os.path.join(self.output_dir, "component_requirements.json")
        if os.path.exists(component_requirements_path) and not overwrite_component_summary:
            comp_requirements = load_json(component_requirements_path)
        else:
            logger.info("Generating component-level IRs")
            comp_requirements = generate_component_level_irs(
                self.repo_path, sorted_components, self.components, 
                dep_graph, rev_graph, visualizer, overwrite_component_summary, self.reviewer
            )
            save_json(component_requirements_path, comp_requirements)
        
        # Build file to components mapping
        file_to_components: Dict[str, List[str]] = defaultdict(list)
        for cid, item in comp_requirements.items():
            f = item.get("file")
            if f:
                file_to_components[f].append(cid)
        
        # Generate file-level IRs
        logger.info("Generating file-level IRs")
        file_requirements = generate_file_level_irs(
            self.file_graph, file_to_components, comp_requirements, self.reviewer
        )
        save_json(os.path.join(self.output_dir, "file_requirements.json"), file_requirements)
        
        visualizer.finalize()
        logger.info("IR derivation complete")
        
        return comp_requirements, file_requirements
    
    def phase3_ur_generation(self, file_requirements: Dict[str, Dict[str, Any]], component_requirements: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Phase 3: Use Case Generation
        Generate Use Cases by clustering files and abstracting IRs into structured use case specifications.
        """
        logger.info("=== Phase 3: Use Case Generation ===")
        
        # Detect file communities using Leiden algorithm
        logger.info("Detecting file communities using Leiden algorithm")
        communities = detect_file_communities_leiden(
            self.file_graph, 
            edge_details=self.file_edge_details, 
            resolution=1.0, 
            iterations=None, 
            seed=42
        )
        save_json(os.path.join(self.output_dir, "file_communities.json"), [sorted(list(c)) for c in communities])
        logger.info(f"Detected {len(communities)} file communities")
        
        # Generate Use Cases with writer-verifier iterative loop
        logger.info("Generating Use Cases with iterative refinement")
        verified_intents: List[Dict[str, Any]] = []
        
        # Build file to components mapping
        file_to_components = defaultdict(list)
        for cid, item in component_requirements.items():
            f = item.get("file")
            if f:
                file_to_components[f].append(cid)
        
        # Process each community
        for i, files in enumerate(communities, start=1):
            logger.info(f"Processing community {i}/{len(communities)} with {len(files)} files")
            
            intent, feedback = iterate_intent_for_community(
                idx=i,
                files=files,
                file_requirements=file_requirements,
                component_requirements=component_requirements,
                file_graph=self.file_graph,
                writer=self.writer,
                verifier=self.verifier,
                searcher=self.searcher,
                max_verifier_rejections=3,
                max_search_attempts=2
            )
            
            # Add component and file mappings
            comps: List[str] = []
            for f in files:
                comps.extend(file_to_components.get(f, []))
            
            intent["components"] = sorted(set(comps))
            intent["files"] = sorted(list(files))
            intent["verification"] = feedback
            intent["community_id"] = i
            
            verified_intents.append(intent)
            logger.info(f"Generated UR: {intent.get('title', 'Untitled')}")
        
        logger.info("UR generation complete")
        return verified_intents
    
    def generate_traceability_links(self, use_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate traceability links between use cases and code components/files."""
        logger.info("Generating traceability links")
        
        traceability = {}
        
        for uc in use_cases:
            uc_id = uc["intent_id"]
            traceability[uc_id] = {
                "use_case_name": uc.get("use_case_name", ""),
                "description": uc.get("description", ""),
                "primary_actor": uc.get("primary_actor", ""),
                "secondary_actors": uc.get("secondary_actors", []),
                "files": uc.get("files", []),
                "components": uc.get("components", []),
                "community_id": uc.get("community_id", 0),
                "preconditions": uc.get("preconditions", []),
                "main_flow": uc.get("main_flow", []),
                "alternative_flows": uc.get("alternative_flows", []),
                "postconditions": uc.get("postconditions", [])
            }
        
        return traceability
    
    def run_complete_pipeline(self, overwrite_dependency_graph: bool = False, overwrite_component_summary: bool = False, order_mode: str = "typo") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run the complete UserTrace pipeline across all three phases.
        
        Returns:
            Tuple of (use_cases, traceability_links)
        """
        # Start performance monitoring
        self.performance_metrics.start_timing()
        
        logger.info("Starting UserTrace pipeline with performance monitoring...")
        
        # Phase 1: Repository Structuring
        phase1_start = time.time()
        self.phase1_repository_structuring(overwrite_dependency_graph)
        phase1_time = time.time() - phase1_start
        logger.info(f"Phase 1 completed in {phase1_time:.2f} seconds")
        
        # Phase 2: IR Derivation
        phase2_start = time.time()
        comp_requirements, file_requirements = self.phase2_ir_derivation(overwrite_component_summary, order_mode)
        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2 completed in {phase2_time:.2f} seconds")
        
        # Phase 3: Use Case Generation
        phase3_start = time.time()
        use_cases = self.phase3_ur_generation(file_requirements, comp_requirements)
        phase3_time = time.time() - phase3_start
        logger.info(f"Phase 3 completed in {phase3_time:.2f} seconds")
        
        # Generate traceability links
        traceability_links = self.generate_traceability_links(use_cases)
        
        # End performance monitoring
        self.performance_metrics.end_timing()
        
        # Save final results including performance metrics
        save_json(os.path.join(self.result_dir, "use_cases.json"), use_cases)
        save_json(os.path.join(self.result_dir, "traceability_links.json"), traceability_links)
        save_json(os.path.join(self.result_dir, "performance_metrics.json"), self.performance_metrics.to_dict())
        
        # Log comprehensive performance summary
        self._log_performance_summary(use_cases, phase1_time, phase2_time, phase3_time)
        
        return use_cases, traceability_links
    
    def _log_performance_summary(self, use_cases: List[Dict[str, Any]], phase1_time: float, phase2_time: float, phase3_time: float):
        """Log comprehensive performance summary."""
        metrics = self.performance_metrics
        total_tokens = metrics.get_total_tokens()
        
        logger.info("="*80)
        logger.info("USERTRACE PIPELINE PERFORMANCE SUMMARY")
        logger.info("="*80)
        
        # Timing Summary
        logger.info(f"📊 TIMING SUMMARY:")
        logger.info(f"   Total Runtime: {metrics.total_runtime:.2f} seconds")
        logger.info(f"   Effective Runtime: {metrics.effective_runtime:.2f} seconds")
        logger.info(f"   API Wait Time: {metrics.api_wait_time:.2f} seconds")
        logger.info(f"   Phase 1 (Repository Structuring): {phase1_time:.2f} seconds")
        logger.info(f"   Phase 2 (IR Derivation): {phase2_time:.2f} seconds")
        logger.info(f"   Phase 3 (Use Case Generation): {phase3_time:.2f} seconds")
        
        # Token Usage Summary
        logger.info(f"\n🔤 TOKEN USAGE SUMMARY:")
        logger.info(f"   Total Input Tokens: {total_tokens.input_tokens:,}")
        logger.info(f"   Total Output Tokens: {total_tokens.output_tokens:,}")
        logger.info(f"   Total Tokens: {total_tokens.total_tokens:,}")
        
        # Agent-specific metrics
        logger.info(f"\n🤖 AGENT PERFORMANCE:")
        logger.info(f"   Code Reviewer: {metrics.reviewer_calls} calls, {metrics.reviewer_tokens.total_tokens:,} tokens")
        logger.info(f"   Search Agent: {metrics.searcher_calls} calls, {metrics.searcher_tokens.total_tokens:,} tokens")
        logger.info(f"   Writer Agent: {metrics.writer_calls} calls, {metrics.writer_tokens.total_tokens:,} tokens")
        logger.info(f"   Verifier Agent: {metrics.verifier_calls} calls, {metrics.verifier_tokens.total_tokens:,} tokens")
        
        # Error Statistics
        if metrics.api_errors > 0 or metrics.retry_attempts > 0:
            logger.info(f"\n⚠️  ERROR STATISTICS:")
            logger.info(f"   API Errors: {metrics.api_errors}")
            logger.info(f"   Retry Attempts: {metrics.retry_attempts}")
        
        # Output Summary
        logger.info(f"\n📋 OUTPUT SUMMARY:")
        logger.info(f"   Generated Use Cases: {len(use_cases)}")
        logger.info(f"   Components Processed: {len(self.components)}")
        logger.info(f"   Files Analyzed: {len(self.file_graph)}")
        
        # Efficiency Metrics
        if total_tokens.total_tokens > 0:
            tokens_per_use_case = total_tokens.total_tokens / len(use_cases) if use_cases else 0
            logger.info(f"\n⚡ EFFICIENCY METRICS:")
            logger.info(f"   Tokens per Use Case: {tokens_per_use_case:.0f}")
            logger.info(f"   Use Cases per Minute: {len(use_cases) / (metrics.effective_runtime / 60):.1f}")
        
        logger.info(f"\n💾 Results saved to: {self.result_dir}")
        logger.info("="*80)

# -------------------- CLI Interface --------------------
def get_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="UserTrace: Generate use cases and traceability links from code repositories"
    )
    parser.add_argument(
        "--repo_path", type=str, default="dataset/smos/code",
        help="Path to the code directory"
    )
    parser.add_argument(
        "--repo_type", type=str, default="Java", choices=["Java", "Python"],
        help="Type of the repository (Java or Python)"
    )
    parser.add_argument(
        "--config_path", type=str, default="config/agent_config.yaml",
        help="Path to the configuration file for LLM agents"
    )
    parser.add_argument(
        "--order_mode", type=str, default="typo", choices=["typo", "file", "random"],
        help="Order mode for processing components: typo, file, or random"
    )
    parser.add_argument(
        "--overwrite_dependency_graph", action='store_true',
        help="Overwrite existing dependency graph if it exists"
    )
    parser.add_argument(
        "--overwrite_component_summary", action='store_true',
        help="Overwrite existing component IRs if they exist"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/",
        help="Path to save intermediate outputs"
    )
    parser.add_argument(
        "--result_dir", type=str, default="result/",
        help="Path to save final results"
    )
    parser.add_argument(
        "--verbose_performance", action='store_true',
        help="Display detailed performance metrics during execution"
    )
    return parser

def main():
    """Main entry point for UserTrace system."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Initialize UserTrace system
    user_trace = UserTrace(
        repo_path=args.repo_path,
        repo_type=args.repo_type,
        config_path=args.config_path,
        output_dir=args.output_dir,
        result_dir=args.result_dir
    )
    
    # Run complete pipeline
    try:
        use_cases, traceability_links = user_trace.run_complete_pipeline(
            overwrite_dependency_graph=args.overwrite_dependency_graph,
            overwrite_component_summary=args.overwrite_component_summary,
            order_mode=args.order_mode
        )
        
        print("\n" + "="*60)
        print("UserTrace Pipeline Completed Successfully!")
        print("="*60)
        print(f"Generated {len(use_cases)} use cases")
        print(f"Results saved to: {user_trace.result_dir}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"UserTrace pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def create_sample_repository():
    """Create a sample repository for testing."""
    sample_dir = "sample_repo"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a simple Python file
    sample_py = """
class DataProcessor:
    '''Processes data for the application.'''
    
    def __init__(self, config):
        self.config = config
    
    def process_data(self, data):
        '''Process input data and return results.'''
        # Validate input
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Transform data
        processed = []
        for item in data:
            processed.append(self.transform_item(item))
        
        return processed
    
    def transform_item(self, item):
        '''Transform a single data item.'''
        return item.upper() if isinstance(item, str) else str(item)

def main_process():
    '''Main processing function.'''
    processor = DataProcessor({'mode': 'production'})
    data = ['hello', 'world', 123]
    result = processor.process_data(data)
    return result
"""
    
    with open(os.path.join(sample_dir, "processor.py"), "w") as f:
        f.write(sample_py)
    
    return sample_dir

def run_sample():
    """Run UserTrace on a sample repository."""
    print("Creating sample repository...")
    sample_dir = create_sample_repository()
    
    try:
        print("Running UserTrace on sample repository...")
        user_trace = UserTrace(
            repo_path=sample_dir,
            repo_type="Python",
            config_path="config/agent_config.yaml",  # This would be optional
            output_dir="sample_output",
            result_dir="sample_results"
        )
        
        use_cases, traceability_links = user_trace.run_complete_pipeline()
        
        print(f"\nSample run completed!")
        print(f"Generated {len(use_cases)} use cases")
        print(f"Results saved to: {user_trace.result_dir}")
        
        # Print performance summary
        metrics = user_trace.performance_metrics
        total_tokens = metrics.get_total_tokens()
        print(f"\nPerformance Summary:")
        print(f"  Total Runtime: {metrics.total_runtime:.2f} seconds")
        print(f"  Effective Runtime: {metrics.effective_runtime:.2f} seconds")
        print(f"  Total Tokens Used: {total_tokens.total_tokens:,}")
        print(f"  API Calls: {metrics.reviewer_calls + metrics.searcher_calls + metrics.writer_calls + metrics.verifier_calls}")
        
        # Print first use case as example
        if use_cases:
            print(f"\nExample Use Case:")
            print(f"Name: {use_cases[0].get('use_case_name', 'N/A')}")
            print(f"Primary Actor: {use_cases[0].get('primary_actor', 'N/A')}")
            print(f"Description: {use_cases[0].get('description', 'N/A')[:200]}...")
        
    except Exception as e:
        print(f"Sample run failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)

def test_performance_monitoring():
    """Test the performance monitoring functionality."""
    print("Testing performance monitoring...")
    
    # Create a simple performance metrics instance
    metrics = PerformanceMetrics()
    metrics.start_timing()
    
    # Simulate some work
    time.sleep(0.1)
    
    # Simulate token usage
    metrics.reviewer_tokens.add_usage(100, 50)
    metrics.reviewer_calls += 1
    
    metrics.writer_tokens.add_usage(200, 100)
    metrics.writer_calls += 1
    
    # Simulate API error
    metrics.api_errors += 1
    metrics.add_api_wait_time(0.05)
    
    metrics.end_timing()
    
    # Print results
    print("Performance Metrics Test Results:")
    print(f"  Total Runtime: {metrics.total_runtime:.3f} seconds")
    print(f"  Effective Runtime: {metrics.effective_runtime:.3f} seconds")
    print(f"  API Wait Time: {metrics.api_wait_time:.3f} seconds")
    
    total_tokens = metrics.get_total_tokens()
    print(f"  Total Tokens: {total_tokens.total_tokens}")
    print(f"  API Errors: {metrics.api_errors}")
    
    # Test JSON serialization
    metrics_dict = metrics.to_dict()
    print(f"  JSON serialization: {'✓ Success' if metrics_dict else '✗ Failed'}")
    
    print("Performance monitoring test completed!\n")

if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == "--sample":
    #     run_sample()
    # elif len(sys.argv) > 1 and sys.argv[1] == "--test-performance":
    #     test_performance_monitoring()
    # else:
    main()