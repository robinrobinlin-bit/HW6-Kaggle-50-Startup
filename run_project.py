import os
import sys

# Add src folder to python path to resolve module imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from main import run_pipeline

if __name__ == "__main__":
    run_pipeline()
