# Speech Analysis API 🎤📊

[![GitHub Actions](https://github.com/AnMaster15/Memo_nlp.git/actions/workflows/cloudbuild.yaml/badge.svg)](https://github.com/AnMaster15/Memo_nlp.git/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FastAPI service for analyzing speech patterns to detect cognitive decline markers.

## Features
- Audio feature extraction (pitch, pauses, speech rate)
- Linguistic analysis (hesitation patterns, vocabulary diversity)
- Batch processing support
- Google Cloud Run deployment

## Local Setup
```bash
git clonehttps://github.com/AnMaster15/Memo_nlp.git
cd speech-analysis-api

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --reload