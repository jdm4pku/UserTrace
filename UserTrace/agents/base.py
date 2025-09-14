from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import os
from pathlib import Path

from .llm.factory import LLMFactory
from .llm.base import BaseLLM

class BaseAgent(ABC):
    """Base class for all agents in the docstring generation system."""
    
    def __init__(self, name: str, config_path: Optional[str] = None):
        """Initialize the base agent.
        
        Args:
            name: The name of the agent
            config_path: Optional path to the configuration file
        """
        self.name = name
        self._memory: list[Dict[str, Any]] = []
        
        # Initialize LLM and parameters from config
        self.llm, self.llm_params = self._initialize_llm(name, config_path)

    
    def _initialize_llm(self, agent_name: str, config_path: Optional[str] = None) -> tuple[BaseLLM, Dict[str, Any]]:
        """Initialize the LLM for this agent.
        
        Args:
            agent_name: Name of the agent
            config_path: Optional path to the configuration file
            
        Returns:
            Tuple of (Initialized LLM instance, LLM parameters dictionary)
        """
        # Load configuration
        if config_path is None:
            config_path = "config/agent_config.yaml"
            print(f"Using default config from {config_path}")
            
        config = LLMFactory.load_config(config_path)
        
        # Check for agent-specific configuration
        agent_config = config.get("agent_llms", {}).get(agent_name.lower())
        
        # Use agent-specific config if available, otherwise use default
        llm_config = agent_config if agent_config else config.get("llm", {})
        
        # Verify api_key is provided in config
        if ("api_key" not in llm_config or not llm_config["api_key"]) and (llm_config["type"] not in ["huggingface", "local"]):
            raise ValueError("API key must be specified directly in the config file")

        # Extract LLM parameters
        llm_params = {
            "max_output_tokens": llm_config.get("max_output_tokens", 4096),
            "temperature": llm_config.get("temperature", 0.1),
            "model": llm_config.get("model"),
        }

        return LLMFactory.create_llm(llm_config), llm_params
    
    def add_to_memory(self, role: str, content: str) -> None:
        """Add a message to the agent's memory.
        
        Args:
            role: The role of the message sender (e.g., 'system', 'user', 'assistant')
            content: The content of the message
        """
        assert content is not None and content != "", "Content cannot be empty"
        self._memory.append(self.llm.format_message(role, content))
    
    def refresh_memory(self, new_memory: list[Dict[str, Any]]) -> None:
        """Replace the current memory with new memory.
        
        Args:
            new_memory: The new memory to replace the current memory
        """
        self._memory = [
            self.llm.format_message(msg["role"], msg["content"])
            for msg in new_memory
        ]
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self._memory = []
    
    @property
    def memory(self) -> list[Dict[str, Any]]:
        """Get the agent's memory.
        
        Returns:
            The agent's memory as a list of message dictionaries
        """
        return self._memory.copy()
    
    def generate_response(self, messages: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response using the agent's LLM and memory.
        
        Args:
            messages: Optional list of messages to use instead of memory
            
        Returns:
            Generated response text
        """
        return self.llm.generate(
            messages=messages if messages is not None else self._memory,
            temperature=self.llm_params["temperature"],
            max_tokens=self.llm_params["max_output_tokens"]
        )
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process the input and generate output.
        
        This method should be implemented by each specific agent.
        """
        pass 