#!/bin/bash

# Install all Python packages from requirements.txt
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Optional but recommended: Confirm uvicorn is available
pip install uvicorn
