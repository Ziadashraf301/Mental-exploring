from src.pipeline.train_pipeline import run_pipeline
  
if __name__ == "__main__":
    """
    Run the pipeline using settings from config.yaml
    To use a different config file: python run_pipeline.py path/to/train_config.yaml
    """
    import sys
    import os 

    # Check if custom config path provided
    config_path = sys.argv[1] if len(sys.argv) > 1 else "src/config/train_config.yaml"

    print(f"\n🚀 Starting pipeline with configuration: {config_path}\n")
    
    # Run pipeline
    results = run_pipeline(config_path)
    
    # Print final results
    if 'best_model' in results:
        print(f"\n{'='*70}")
        print(f"🎯 Best Model: {results['best_model'][0]}")
        print(f"🎯 Test Accuracy: {results['best_model'][1]*100:.2f}%")
        print(f"{'='*70}\n")