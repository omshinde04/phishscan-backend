#!/bin/bash

echo "🔧 Installing dependencies required before downloading spaCy model"
pip install spacy
pip install uvicorn

echo "🌐 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

echo "✅ spaCy language model installed"
