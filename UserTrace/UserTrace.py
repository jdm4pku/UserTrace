import os
import sys
import json
import time
import random
from collections import defaultdict
from typing import Dict, Any, Set, List, Tuple, Optional, Union
from performance import PerformanceMetrics, TokenUsage
import logging
from dependency_analyzer.java_parser import JavaDependencyParser
from dependency_analyzer.python_parser import PyDependencyParser
from agents.codereviewer import CodeReviewAgent
from agents.searcher import SearchAgent
from agents.writer import WriterAgent
from agents.verifer import VerifierAgent
from visualizer.progress import ProgressVisualizer
from utils import save_json, load_json, build_graph_from_components, build_file_graph_from_components, dependency_first_dfs, detect_file_communities_leiden


# -------------------- Logging Configuration --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("user_trace")

def _first_line(text: str, limit: int = 200) -> str:
    """Extract first line of text with length limit."""
    if not text:
        return ""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return (line[:limit]).strip()

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
