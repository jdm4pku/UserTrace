import argparse
from UserTrace.UserTrace import UserTrace

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


if __name__=="__main__":
    main()