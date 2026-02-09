# PR Article Auto-Classification System

### Introduction to NLP Term Project, Winter 2026
TAESUNG LEE(Group 1)

# About this Project
Automatic classification system for PR articles into product categories using a **hybrid approach** (Rule-based + Retrieval).

**Key Achievement**: **97.25% accuracy** on high-performance categories, outperforming LLM-based methods by **+49.33%p**.


# Experiment Settings

## models
- **Sentence-Transformers**: all-MiniLM-L6-v2 (384-dim embeddings)
- **FAISS**: IndexFlatIP (Inner Product for cosine similarity)
- **LLaMA-2-7B**: For baseline comparison (8-bit quantization)

## Datasets
- **Source**: Corporate PR articles (2025-01)
- **Size**: 500 articles
- **Categories**: 10 product categories (TV, Audio, Vehicle/Autos, Advanced Tech, Home Appliance, IT Product, HVAC, Company, Signage, Robot)
- **Languages**: Multilingual (English, Korean, Arabic, Greek, etc.)
- **Note**: Company names anonymized for public release

### Rule-based Classifier
```python
rules = {
    "TV": ["oled", "qled", "tv", "webos", "4k", "8k"],
    "Audio": ["speaker", "soundbar", "xboom", "tone free"],
    "Vehicle/Autos": ["vehicle", "automotive", "sdv"],
    "Advanced Tech": ["ai", "robot", "thinq", "ces 2025"],
}
```

### Retrieval Classifier
1. Embed text using Sentence-Transformers
2. Search similar articles using FAISS (top-k=10)
3. Vote by category labels
4. Return most common category


## Results

![f1.jpg](figures/f1.jpg)
![recall.jpg](figures/recall.jpg)

For details about the experiment, refer to `presentation.pdf`.

# How to run

## Install dependencies
pip install -r requirements.txt

## Build FAISS Index
### Build index from data 
python src/main_hybrid_simple.py --build_index --data data/A_company_202501_clean.jsonl --index_dir index --encoder sentence-transformers/all-MiniLM-L6-v2

Or with line breaks for readability:
python src/main_hybrid_simple.py --build_index \
    --data data/A_company_202501_clean.jsonl \
    --index_dir index \
    --encoder sentence-transformers/all-MiniLM-L6-v2

## Prerequisites
You need following packages installed on your environment to run the demo: refer to `requirements.txt and requirements_langchain.txt'.

### LangChain Agentic Flow Requirements
langchain>=0.1.0
langchain-community>=0.0.20
langchain-core>=0.1.0
transformers>=4.36.0
torch>=2.1.0
sentence-transformers>=2.2.2
faiss-gpu>=1.7.2
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
bitsandbytes>=0.41.0
accelerate>=0.25.0

## Run Evaluation

### Evaluate on high-performance categories
python src/evaluate_high_performance.py --data data/A_company_202501_clean.jsonl --index_dir index

## Calculate detailed metrics
python src/calculate_metrics.py runs/results_hybrid_high_performance.csv

### Expected output:
Overall Accuracy: 0.9725 (106/109)
Macro F1: 0.7824
Weighted F1: 0.9768

## Demo Examples
results_hybrid_high_performance.csv  # Hybrid results (97.25%)
results_baseline_176.csv       # Baseline results (47.92%)
results_agentic_176.csv        # Agentic results (47.92%)

### 1. LLM Limitations
- **Discovery**: LLaMA-2-7B has severe TV bias (46.8% of data)
- **Evidence**: Baseline and Agentic have identical performance (47.92%)
- **Lesson**: LLM is not always the answer for domain-specific tasks

### 2. Rule-based Effectiveness
- **Discovery**: Keywords work better than LLM for product classification
- **Evidence**: Rule-based (68.0%) > LLM (47.92%)
- **Lesson**: Domain knowledge > LLM for clear categories

### 3. Selective Automation
- **Discovery**: Not all categories need same treatment
- **Strategy**: Focus on high-confidence categories
- **Impact**: 56% automation with 97.25% accuracy

## Limitations
- The system works best for **4 high-performance categories** with 97.25% accuracy
- **6 low-performance categories** have lower accuracy (32-57%) and require manual review
- The rule-based classifier requires **manual keyword updates** when new products are released
- The system was tested only with **PR articles** and may not generalize to other domains
