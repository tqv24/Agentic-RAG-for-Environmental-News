# Advanced RAG for Environmental News

## Project Overview

This project implements an **Advanced Retrieval-Augmented Generation (RAG) Assistant** for answering questions based on a curated dataset of **environmental news articles**. It extends previous modules by integrating:

- A **multi-agent LangGraph workflow**.
- A **LoRA fine-tuned summarization model**.
- A **Streamlit-based frontend for live evaluation**.
- A **qualitative comparison** of multiple LLM systems.


![alt text](<image.png>)
Project structure:
```bash
stitching/
│
├── stitch.ipynb                  # Main development notebook
├── results.pdf                   # Final report or evaluation results
│
└── unit_test/                    # All runtime and deployment files
    ├── lora_model/              # Folder containing the fine-tuned LoRA model
    ├── news.csv                 # Environmental news dataset
    ├── .env                     # API keys (not included in version control)
    ├── requirements.txt         # Python dependencies
    ├── Dockerfile               # Docker setup for deployment
```

---

## Key System Components

### 1. Document Retrieval with ChromaDB + Pinecone

- Processes the environmental news dataset: [`news.csv`](https://drive.google.com/file/d/1tXx6sEmIV127Jm5VVXhBFY9uy9NCI0P9/view?usp=sharing).
- Splits documents into **250-token chunks** with **50 overlap** using `RecursiveCharacterTextSplitter`.
- Stores the embeddings using:
  - `ChromaDB` for local document retrieval.
  - `Pinecone` for remote indexing with `text-embedding-3-small`.

### 2. Multi-Agent Workflow with LangGraph

The system defines a graph-based workflow with the following agents:

- **Retriever Agent** → Retrieves top-k relevant documents.
- **Grader Agent** → Uses GPT-3.5-turbo to score document relevance.
- **Summarizer Agent (LoRA)** → Generates document summaries using a **LoRA fine-tuned Seq2Seq model** (BART).
- **Answer Generator** → Synthesizes the final response from summaries.
- **Rewriter Agent** → Refines the query if relevance is low.

If insufficient relevance is detected, the workflow loops and regenerates.

### 3. Multi-Model Evaluation

The system compares four different models:

| Model                        | Description                                    |
|-----------------------------|------------------------------------------------|
| **Base LLM**                | GPT-3.5 without document context               |
| **Basic RAG**               | GPT-3.5 with top-5 documents via Pinecone        |
| **Advanced RAG (Base)**     | Multi-agent workflow, summaries by GPT         |
| **Advanced RAG (LoRA)**     | Full workflow with summaries via LoRA model    |

Each answer is evaluated on:
- Relevance
- Factual accuracy
- Question resolution

---

## How to Run

### 1. Clone the Project

```bash
git clone <your-repo-url>
cd stitching
``` 

### 2. Setup API Keys
Create a .env file:
```ini
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

### 3. Run with Docker (Recommended)
Build and run the container:
```bash
docker build -t streamlit-llm-eval .
docker run --env-file .env -p 8501:8501 streamlit-llm-eval
```

### 4. Or Run the Unit Test locally (Dev Mode)
```bash
pip install -r requirements.txt
streamlit run main.py
```


