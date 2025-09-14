from dataclasses import dataclass, field
import time
from typing import Dict, Any, Set, List, Tuple, Optional, Union

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