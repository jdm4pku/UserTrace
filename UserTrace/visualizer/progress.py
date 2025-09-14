import sys
import time
import os
from typing import Dict, List, Set, Optional
from colorama import Fore, Back, Style, init
from tqdm import tqdm

class ProgressVisualizer:
    """Visualizes the progress of docstring generation in the terminal."""
    
    def __init__(self, components: Dict[str, any], sorted_order: List[str],repo_type):
        """
        Initialize the progress visualizer.
        
        Args:
            components: Dictionary of code components
            sorted_order: List of component IDs in topological order
        """
        init()  # Initialize colorama
        self.components = components
        self.sorted_order = sorted_order
        self.repo_type = repo_type
        self.processed = set()  # Set of processed component IDs
        self.current = None  # Current component being processed
        self.progress_bar = None
        self.start_time = time.time()

    def initialize(self):
        """Initialize the visualization and show the initial state."""
        self._clear_screen()
        self._print_header()
        
        # Create progress bar
        self.progress_bar = tqdm(
            total=len(self.sorted_order),
            desc="Generating docstrings",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        # Print initial component status
        self._print_component_status()

    def update(self, component_id: str = None, status: str = "processing"):
        """
        Update the visualization with the current component status.
        
        Args:
            component_id: ID of the component being processed (or None)
            status: Status of the component ('processing', 'completed', or 'error')
        """
        if component_id is not None:
            self.current = component_id
            
            if status == "completed":
                self.processed.add(component_id)
                self.progress_bar.update(1)
                
        # Update the visualization
        self._print_component_status()
    
    def finalize(self):
        """Finalize the visualization and show summary statistics."""
        if self.progress_bar:
            self.progress_bar.close()
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        
        self._clear_screen()
        self._print_header()
        
        # Print summary
        print(f"\n{Fore.GREEN}Docstring Generation Complete!{Style.RESET_ALL}")
        print(f"Total components processed: {len(self.processed)}/{len(self.sorted_order)}")
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print("\nComponents by type:")
        
        # Count components by type
        type_counts = {"function": 0, "method": 0, "class": 0}
        for comp_id in self.processed:
            comp_type = self.components[comp_id].component_type
            type_counts[comp_type] += 1
        
        for comp_type, count in type_counts.items():
            print(f"  {comp_type.capitalize()}: {count}")
        
        print("\nGeneration complete. Results saved to repository files.")
    
    def _clear_screen(self):
        """Clear the terminal screen."""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
    
    def _print_header(self):
        """Print the header with title and information."""
        title = "Topological Docstring Generator"
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}\n")
        print(f"Generating docstrings for {len(self.sorted_order)} code components in dependency order")
        print(f"Components will be processed in topological order to ensure all dependencies")
        print(f"have docstrings before dependent components.")
    
    def _print_component_status(self):
        """Print the current status of components in the dependency graph."""
        if not self.current:
            return
        
        # Get the current component and its info
        current_comp = self.components.get(self.current)
        if not current_comp:
            return
            
        # Print current component information
        comp_type = current_comp.component_type.capitalize()
        file_path = current_comp.relative_path
        
        # Create a simplified name for display
        parts = self.current.split('.')
        if len(parts) > 2 and current_comp.component_type == "method":
            # For methods, show Class.method
            name = f"{parts[-2]}.{parts[-1]}"
        else:
            # For functions and classes, show just the name
            name = parts[-1]
        
        # Print status line
        print(f"\n{Fore.YELLOW}Currently processing: {Style.RESET_ALL}{comp_type} '{name}' in {file_path}")
        
        # Print dependency information
        if current_comp.depends_on:
            deps = [dep_id for dep_id in current_comp.depends_on if dep_id in self.components]
            if deps:
                print(f"{Fore.CYAN}Dependencies:{Style.RESET_ALL}")
                for dep_id in deps:
                    dep = self.components.get(dep_id)
                    if not dep:
                        continue
                        
                    # Format the dependency name similarly
                    parts = dep_id.split('.')
                    if len(parts) > 2 and dep.component_type == "method":
                        dep_name = f"{parts[-2]}.{parts[-1]}"
                    else:
                        dep_name = parts[-1]
                    
                    # Color based on processing status
                    if dep_id in self.processed:
                        status_color = Fore.GREEN
                        status_text = "(processed)"
                    else:
                        status_color = Fore.RED
                        status_text = "(not yet processed)"
                        
                    print(f"  {status_color}{dep.component_type.capitalize()} '{dep_name}' {status_text}{Style.RESET_ALL}")
        
        # Add some space after the component status
        print()
    
    def show_dependency_stats(self):
        """Show statistics about the dependency graph."""
        # Calculate dependency metrics
        total_deps = sum(len(self.components[comp_id].depends_on) for comp_id in self.components)
        max_deps = max((len(self.components[comp_id].depends_on), comp_id) for comp_id in self.components)
        
        avg_deps = total_deps / len(self.components) if self.components else 0
        
        # Count components by type
        if self.repo_type in ("Java", "C", "C++"):
            types = {"class": 0, "interface": 0, "enum": 0, "method": 0, "constructor": 0}
        else:
            types = {"function": 0, "method": 0, "class": 0}
        
        for comp_id in self.components:
            comp_type = self.components[comp_id].component_type
            types[comp_type] += 1
        if self.repo_type in ("Java", "C", "C++"):
            print(f"\n{Fore.CYAN}Dependency Graph Statistics:{Style.RESET_ALL}")
            print(f"Total components: {len(self.components)}")
            print(f"  Classes: {types['class']}")
            print(f"  Interfaces: {types['interface']}")
            print(f"  Enums: {types['enum']}")
            print(f"  Methods: {types['method']}")
            print(f"  Constructors: {types['constructor']}")
        else: # python projects
            print(f"\n{Fore.CYAN}Dependency Graph Statistics:{Style.RESET_ALL}")
            print(f"Total components: {len(self.components)}")
            print(f"  Functions: {types['function']}")
            print(f"  Methods: {types['method']}")
            print(f"  Classes: {types['class']}")

        print(f"Average dependencies per component: {avg_deps:.2f}")
        print(f"Max dependencies: {max_deps[0]} (in component '{max_deps[1]}')")
        
        # Print information about cycles if available
        print(f"\nComponents will be processed in topological order.")
        print() 