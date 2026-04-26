# SympScan: Advanced Medical RAG & Knowledge Graph System

## Overview
SympScan is an intelligent medical assistant designed to provide clinical facts, diagnostic insights, and treatment protocols by leveraging a **Hybrid Dual-Indexing RAG** (Retrieval-Augmented Generation) pipeline and a **Knowledge Graph**. The system transitions from a simple retriever to a sophisticated "Medical Knowledge Engine" that synthesizes information from both unstructured document chunks and structured entity relationships.

The pipeline integrates:
1.  **Hybrid Search**: Combining BM25 keyword search with FAISS-based semantic vector embeddings.
2.  **Knowledge Graph**: A Neo4j-powered graph database to capture explicit relationships between diseases, symptoms, medications, and precautions.
3.  **Agentic Workflow**: Pre-retrieval query rewriting, expansion, and HyDE (Hypothetical Document Embeddings), followed by post-retrieval extractive compression and reranking.

## System Architecture

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | LangChain / Ollama | Managing LLM chains and tool integration. |
| **Vector Database** | FAISS | High-performance semantic similarity search. |
| **Graph Database** | Neo4j | Retrieving structured medical entities and relationships. |
| **LLM Interface** | Qwen-2.5 / Llama 3 | Intent detection, entity extraction, and final response synthesis. |
| **Data Processing** | PySpark | Efficient transformation and Parquet storage of medical datasets. |

[Image of a RAG pipeline architecture including Vector DB and Knowledge Graph]

## File Description

| File Name | Description |
|---|---|
| `Raw_Dataset_PreProcess.py` | Uses **PySpark** to clean and transform raw medical CSVs into a structured Parquet dataset. |
| `Hybrid_Dual_Indexing.py` | Implements semantic chunking and BM25 indexing for dual-path retrieval. |
| `Knowledge_Graph.py` | Constructs and queries a **Neo4j** graph to map Disease $\rightarrow$ Medication/Precaution relations. |
| `Vector_Database.py` | Manages the **FAISS** HNSW index for efficient vector storage and retrieval. |
| `PreRetrival_and_PostRetrieval.py` | Handles query rewriting, HyDE generation, and extractive context compression. |
| `Retrieval.py` | The main engine that merges Hybrid and Graph results using **RRF** and **Cross-Encoders**. |
| `Augmented_Generation.py` | The core RAG logic; handles prompt engineering, JSON validation, and chat history summarization. |
| `Inference.py` | A **Streamlit** dashboard providing a professional UI for real-time medical analysis. |

## Methodology & Analysis

### 1. Hybrid Retrieval Strategy
The system utilizes a "Dual-Path" approach. The `Hybrid_Dual_Indexing.py` script ensures that technical medical terms (captured by BM25) and contextual meanings (captured by `all-MiniLM-L6-v2`) are both considered. Results are then reranked using a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) to ensure top-tier relevance.

### 2. Knowledge Graph Synergy
While the vector database provides descriptive context, the `Knowledge_Graph.py` component provides hard clinical links. For example, if "Hypertension" is detected, the graph immediately pulls associated "Medications" and "Precautions" as verified facts, which are prioritized in the final prompt.

### 3. Reliability & Validation
* **Format Guardrails**: The system enforces strict JSON outputs for consistent UI rendering.
* **Self-Correction**: If the LLM produces an invalid format, the `Generation` loop triggers a rectification prompt.
* **Evaluation Scores**: Includes internal metrics for **Response Confidence** and **Retrieval Helpfulness**.

## Installation & Setup

### Prerequisites
* **Database**: Neo4j Desktop or AuraDB instance (configured with APOC).
* **Local LLM**: [Ollama](https://ollama.com/) installed and running.
* **Environment**: Python 3.10+

### Setup

1.  **Initialize LLMs:**
    ```bash
    ollama pull qwen2.5:0.5b-instruct-q5_k_m
    ollama pull llama3
    ```

2.  **Data Preprocessing:**
    ```bash
    python Raw_Dataset_PreProcess.py
    python Hybrid_Dual_Indexing.py
    ```

3.  **Database Population:** Ensure Neo4j is running
    ```bash
    python Vector_Database.py
    python Knowledge_Graph.py
    ```

4.  **Launch Interface:**
    ```bash
    streamlit run Inference.py
    ```

## License
This project is licensed under the **CC-BY (Creative Commons Attribution)** license.

## Citation
Do, Chi Khoa (2026). *SympScan: Advanced Medical RAG & Knowledge Graph System*.

## Acknowledgements

This README structure is inspired by data documentation guidelines from:

- [Queen’s University README Template](https://guides.library.queensu.ca/ReadmeTemplate)  
- [Cornell University Data Sharing README Guide](https://data.research.cornell.edu/data-management/sharing/readme/)  


This project utilizes the **SympScan - Symptomps to Disease Dataset**, available on Kaggle:

- [SympScan - Symptomps to Disease](https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease)

## Contact
For inquiries regarding the architecture or medical dataset integration, contact [dochikhoa2006@gmail.com](mailto:dochikhoa2006@gmail.com).