#!/usr/bin/env python3
"""Setup script to install dependencies and download spaCy model."""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and report result."""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    # Install requirements
    success = run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing requirements from requirements.txt"
    )
    
    if success:
        # Download spaCy model
        run_command(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            "Downloading spaCy en_core_web_sm model"
        )
        print("\n✓ Setup complete! Your RAG project is ready to use.")
    else:
        print("\n✗ Setup failed. Please check the error messages above.")
        sys.exit(1)
