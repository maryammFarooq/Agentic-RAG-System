# Agentic RAG System

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-18%2B-339933?logo=node.js&logoColor=white)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?logo=react&logoColor=black)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![Pipelines](https://img.shields.io/badge/Pipelines-4%20Strategies-orange)
![Dataset](https://img.shields.io/badge/Dataset-CRAG%20Task%201%262-blue)

This repository presents a comparative study of multiple Retrieval‑Augmented Generation (RAG) techniques applied to a noisy, multi‑domain web dataset. The objective is to evaluate how different RAG approaches perform when document retrieval is not perfect and contains irrelevant or incomplete information.

### Implemented strategies
- RAG Fusion  
- HyDE (Hypothetical Document Embeddings)  
- CRAG (Corrective RAG)  
- Graph RAG  

Detailed assignment rules and restrictions can be found in `ASSIGNMENT.md`.

## Table of Contents
- Overview
- Technology Stack
- Getting Started
- Dataset Preparation
- Configuration Guide
- Execution Steps
- Evaluation Results
- Directory Layout
- Key Notes

## Overview

The application constructs a unified vector index from CRAG dataset snippets. This shared index is reused by all four pipelines to ensure a consistent and fair performance comparison. The evaluation script calculates performance metrics for each pipeline and stores the outcomes in `evaluation_results.json`.

## Technology Stack

Core technologies used in this project include:

- Python (3.9 or newer)
- FastAPI for backend services
- React with Vite for the frontend interface
- Sentence‑transformer based embedding pipeline (configured via `config/config.yaml`)

## Getting Started

From the root directory, install the required Python dependencies:

```
pip install -r requirements.txt
```

Install frontend dependencies:

```
cd frontend
npm install
cd ..
```

Create the configuration file from the template:

```
copy config\config.example.yaml config\config.yaml
```

If your environment does not support the copy command, manually duplicate the file.

## Dataset Preparation

This project uses the **CRAG Task 1 & Task 2 development v4 dataset**.

Steps:

1. Download the compressed dataset file:
   https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2

2. Extract the archive.

3. Move the extracted file to:

```
dataset/crag_task_1_and_2_dev_v4.jsonl
```

More information about the dataset format is available in `docs/dataset.md`.

## Configuration Guide

Update the following required fields in `config/config.yaml`:

- `dataset_path` → location of the CRAG dataset file
- `embedding_model` → for example: all‑MiniLM‑L6‑v2
- `generation_model` → the LLM you plan to use
- `top_k` → number of retrieved chunks per query

## Execution Steps

### Run the evaluation pipeline

```
python run_evaluation.py
```

This process will:

- Create or load the shared index (`dataset/global_index.pkl`)
- Execute all four RAG pipelines on the development dataset
- Compute evaluation metrics
- Save results to `evaluation_results.json`

### Start the backend service

```
python backend/main.py
```

API endpoint:

- POST `/query`

Default backend address:

```
http://localhost:8000
```

### Launch the frontend

```
cd frontend
npm run dev
```

Open the URL displayed by Vite (typically `http://localhost:5173`).

## Evaluation Results

All performance outputs are saved in:

```
evaluation_results.json
```

Each pipeline reports:

- accuracy
- number of correct predictions
- total evaluated queries

## Directory Layout

```
Assignment2/
│
├── backend/
├── config/
├── dataset/
├── docs/
├── frontend/
├── src/
│   └── pipelines/
├── run_evaluation.py
├── evaluation_results.json
├── ASSIGNMENT.md
└── README.md
```

## Key Notes

- Follow all constraints listed in `ASSIGNMENT.md`.
- Maintain the required directory structure.
- Never commit secrets or API keys inside `config/config.yaml`.
- For LLM usage, do not use OpenAI APIs. Instead, use Groq, Gemini free tier, or other free/local alternatives as required by the assignment.
