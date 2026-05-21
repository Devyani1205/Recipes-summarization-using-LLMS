# Recipe Summarization Using Large Language Models

A comprehensive implementation of abstractive recipe summarization using state-of-the-art transformer architectures and Large Language Models (LLMs). This project evaluates multiple summarization models on cooking recipe datasets to generate concise, coherent, and semantically accurate summaries while preserving critical cooking instructions and ingredient information.

---

# 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Technical Architecture](#-technical-architecture)
- [Implemented Models](#-implemented-models)
- [Evaluation Metrics](#-evaluation-metrics)
- [Model Comparison Table](#-model-comparison-table)
- [Technical Implementation Details](#-technical-implementation-details)
- [Dependencies & Installation](#-dependencies--installation)
- [Project Structure](#-project-structure)
- [Usage Instructions](#-usage-instructions)
- [Model Inference Examples](#-model-inference-examples)
- [Evaluation & Results](#-evaluation--results)
- [GPU Configuration](#-gpu-configuration)
- [Training Configuration](#-training-configuration)
- [Inference Optimization](#-inference-optimization)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [References](#-references)
- [Author](#-author)
- [License](#-license)

---

# 🎯 Project Overview

This project implements and compares multiple transformer-based architectures for abstractive text summarization of cooking recipes using Large Language Models.

The goal is to generate concise summaries that preserve:

- Important cooking instructions
- Ingredients and measurements
- Cooking sequence and preparation flow
- Semantic coherence
- Contextual understanding of recipes

The project includes implementation and evaluation of multiple pre-trained transformer architectures including BRIO, PEGASUS, and Llama-based models.

The notebooks demonstrate:

- End-to-end summarization workflows
- Transformer model loading and inference
- ROUGE-based evaluation
- Comparative performance analysis
- GPU-accelerated inference
- Statistical and qualitative comparison of generated summaries

---

# 🚀 Objectives

## Primary Goals

- Implement abstractive recipe summarization using LLMs
- Compare multiple transformer architectures
- Evaluate models using ROUGE metrics
- Analyze summarization quality and semantic preservation
- Optimize inference performance on GPU environments

---

## Secondary Goals

- Explore transfer learning in text summarization
- Compare encoder-decoder vs decoder-only architectures
- Study recipe-domain summarization challenges
- Evaluate inference efficiency and memory utilization
- Generate production-ready summarization pipelines

---

# 🏗️ Technical Architecture

The system follows a modular transformer-based summarization pipeline.

```text
┌────────────────────────────────────┐
│ Raw Recipe Dataset                │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ Text Cleaning & Preprocessing     │
│ - Tokenization                    │
│ - Normalization                   │
│ - Length Handling                 │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ Transformer Tokenizer             │
│ - BRIO Tokenizer                  │
│ - PEGASUS Tokenizer               │
│ - Llama Tokenizer                 │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ Model Inference                   │
│ - Beam Search Decoding            │
│ - Sampling Strategies             │
│ - Sequence Generation             │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ Generated Recipe Summary          │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ Evaluation Pipeline               │
│ - ROUGE Metrics                   │
│ - Statistical Analysis            │
│ - Performance Comparison          │
└────────────────────────────────────┘
```

---

# 🤖 Implemented Models

## 1. BRIO-CNNDM

### Notebook
```bash
brio-cnndm-uncased.ipynb
```

### Architecture
- Better Rank of Information Optimization (BRIO)
- Encoder-decoder transformer architecture
- Fine-tuned on CNN/DailyMail dataset

### Key Features
- Ranking-based optimization
- Improved factual consistency
- Strong abstractive summarization performance
- High-quality semantic preservation

### Advantages
- Excellent coherence
- Strong summary ranking mechanism
- Better handling of long-form recipes

---

## 2. PEGASUS-Large

### Notebook
```bash
pegasus-large-recipes.ipynb
```

### Architecture
- Pre-training with Extracted Gap Sentences
- Transformer encoder-decoder architecture

### Key Features
- Specialized for summarization tasks
- Gap sentence generation objective
- Strong long-document understanding

### Advantages
- High ROUGE performance
- Excellent abstractive generation
- Strong contextual understanding

---

## 3. PEGASUS-Base

### Notebook
```bash
pegasusXBaseSummary.ipynb
```

### Features
- Lightweight PEGASUS implementation
- Faster inference compared to large models
- Lower GPU memory consumption

### Advantages
- Faster execution
- Efficient deployment
- Suitable for resource-constrained environments

---

## 4. Llama-3 Based Summarization

### Notebook
```bash
summllama3.ipynb
```

### Architecture
- Decoder-only transformer LLM
- Instruction-following large language model

### Features
- Generative summarization
- Flexible prompt engineering
- Context-aware summarization

### Advantages
- Human-like summaries
- Strong semantic understanding
- Flexible generation capabilities

---

## 5. Baseline Summarization Models

### Notebook
```bash
receipe-summerization.ipynb
```

### Features
- Baseline encoder-decoder implementations
- Comparative benchmarking
- Traditional summarization reference models

---

# 📊 Evaluation Metrics

The project evaluates model performance using ROUGE metrics.

---

## ROUGE Metrics Explained

### ROUGE-1
Measures unigram overlap between generated summary and reference summary.

Used for:
- Word-level similarity
- Basic content overlap
- Recall-oriented evaluation

---

### ROUGE-2
Measures bigram overlap.

Used for:
- Phrase-level similarity
- Contextual continuity
- Sequential understanding

---

### ROUGE-L
Measures longest common subsequence.

Used for:
- Sentence structure preservation
- Fluency evaluation
- Grammatical similarity

---

### ROUGE-Lsum
Measures longest common subsequence at summary level.

Used for:
- Overall summary coherence
- Structural quality assessment

---

# 📈 Model Comparison Table

| Model | Architecture | Parameters | Inference Speed | Memory Usage | Summary Quality | ROUGE Performance |
|---|---|---|---|---|---|---|
| BRIO-CNNDM | Encoder-Decoder | ~550M | Fast | Medium | High | Excellent |
| PEGASUS-Large | Encoder-Decoder | ~568M | Medium | High | Very High | Outstanding |
| PEGASUS-Base | Encoder-Decoder | ~223M | Very Fast | Low | Medium | Good |
| Llama-3 | Decoder-only LLM | ~7B | Slower | Very High | Excellent | Very High |
| Baseline Models | Mixed | Variable | Fast | Low | Moderate | Baseline |

---

# 🔧 Technical Implementation Details

## Core Libraries

```python
transformers==4.45.1
datasets==2.4.0
torch
evaluate==0.4.3
rouge-score==0.1.2
pandas>=2.2.2
numpy>=1.26.4
plotly>=5.22.0
statsmodels>=0.14.2
py7zr==0.22.0
huggingface-hub>=0.25.1
tokenizers>=0.20.0
```

---

## BRIO Implementation Example

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline

model_name = "facebook/brio-cnndm-uncased"

# Load tokenizer
 tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create summarization pipeline
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer
)

recipe_text = """
Mix flour, eggs, milk, sugar and butter.
Bake for 30 minutes at 180 degrees.
"""

summary = summarizer(
    recipe_text,
    max_length=150,
    min_length=50,
    do_sample=False
)

print(summary)
```

---

## PEGASUS Implementation Example

```python
from transformers import pipeline

model_name = "google/pegasus-large"

summarizer = pipeline(
    "summarization",
    model=model_name
)

summary = summarizer(
    recipe_text,
    max_length=150,
    min_length=40
)

print(summary)
```

---

## Llama-3 Summarization Example

```python
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b"

# Load tokenizer
 tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = f"Summarize this recipe: {recipe_text}"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

outputs = model.generate(
    inputs.input_ids,
    max_length=200
)

summary = tokenizer.decode(outputs[0])

print(summary)
```

---

# 📦 Dependencies & Installation

## Prerequisites

- Python 3.10+
- CUDA-enabled GPU (Recommended)
- Jupyter Notebook
- Hugging Face account (optional for gated models)

---

## Clone Repository

```bash
git clone https://github.com/Devyani1205/Recipes-summarization-using-LLMS.git

cd Recipes-summarization-using-LLMS
```

---

## Create Virtual Environment

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install transformers datasets evaluate rouge-score torch pandas numpy plotly statsmodels py7zr
```

---

## Jupyter Installation

```bash
pip install jupyter notebook
```

---

# 📁 Project Structure

```bash
Recipes-summarization-using-LLMS/
│
├── brio-cnndm-uncased.ipynb
├── pegasus-large-recipes.ipynb
├── pegasusXBaseSummary.ipynb
├── summllama3.ipynb
├── receipe-summerization.ipynb
├── project report LLM.pdf
└── README.md
```

---

# 🚀 Usage Instructions

## Running Jupyter Notebooks

```bash
jupyter notebook
```

Run any notebook individually:

- `brio-cnndm-uncased.ipynb`
- `pegasus-large-recipes.ipynb`
- `pegasusXBaseSummary.ipynb`
- `summllama3.ipynb`
- `receipe-summerization.ipynb`

---

## Running BRIO Summarization

```python
summary = summarizer(
    recipe_text,
    max_length=150,
    min_length=50,
    do_sample=False
)
```

---

## Running PEGASUS Summarization

```python
summary = summarizer(
    recipe_text,
    max_length=120,
    min_length=40
)
```

---

## Running Llama-Based Summarization

```python
prompt = f"Summarize this recipe: {recipe_text}"
```

---

# 📊 Evaluation & Results

## Evaluation Workflow

```text
Input Recipe
     │
     ▼
Generate Summary
     │
     ▼
Compare with Reference Summary
     │
     ▼
Calculate ROUGE Metrics
     │
     ▼
Store Results
     │
     ▼
Generate Comparative Analysis
```

---

## Evaluation Parameters

```python
num_beams = 4
length_penalty = 2.0
early_stopping = True
no_repeat_ngram_size = 3
max_length = 150
min_length = 40
```

---

## Performance Analysis Areas

- Semantic preservation
- Cooking instruction retention
- Ingredient preservation
- Fluency and coherence
- Compression quality
- Context understanding

---

# 🖥️ GPU Configuration

## Hardware Used

| Component | Specification |
|---|---|
| GPU | Tesla P100-PCIE-16GB |
| VRAM | 16GB |
| CUDA Version | 12.6 |
| Driver Version | 560.35.03 |
| Framework | PyTorch |

---

## GPU Optimization

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
```

---

# ⚙️ Training Configuration

## Hyperparameters

```python
learning_rate = 3e-5
batch_size = 8
num_epochs = 3
warmup_steps = 500
weight_decay = 0.01
```

---

## Sequence Lengths

```python
BRIO_MAX_LENGTH = 1024
PEGASUS_MAX_LENGTH = 1024
LLAMA_MAX_LENGTH = 2048
```

---

# 🚀 Inference Optimization

## Generation Configuration

```python
from transformers import GenerationConfig

config = GenerationConfig(
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True,
    no_repeat_ngram_size=3,
    temperature=1.0,
    top_p=0.95
)
```

---

## Batch Processing

```python
batch_size = 8
```

Used for:
- Faster inference
- Better GPU utilization
- Efficient summarization pipelines

---

# 🛠️ Troubleshooting

## GPU Out of Memory

Solutions:

- Reduce batch size
- Use smaller models
- Enable gradient checkpointing
- Use mixed precision inference

---

## Tokenization Errors

Solutions:

- Match tokenizer with model version
- Handle recipe special characters
- Truncate long inputs

---

## Dependency Conflicts

```bash
pip install --upgrade transformers datasets huggingface-hub
```

---

## CUDA Issues

Verify GPU:

```python
import torch
print(torch.cuda.is_available())
```

---

# 🔮 Future Improvements

Potential future enhancements include:

- Fine-tuning on larger recipe datasets
- Human evaluation pipelines
- BLEU and BERTScore integration
- Reinforcement learning for summarization
- Quantized model deployment
- ONNX optimization
- Real-time web application deployment
- Multi-language recipe summarization
- Retrieval-Augmented Generation (RAG)
- Custom fine-tuned recipe LLMs

---

# 📚 References

## Research Papers

- PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization
- BRIO: Bringing Order to Abstractive Summarization
- Attention Is All You Need

---

## Documentation

- Hugging Face Transformers
- PyTorch Documentation
- ROUGE Metric Documentation
- Hugging Face Datasets
- Evaluate Library

---

# 👤 Author

## Devyani

GitHub:
```text
@Devyani1205
```

Repository:
```text
Recipes-summarization-using-LLMS
```

---

# 📄 License

This project is provided for educational and research purposes.

---

# 🤝 Contributing

Contributions are welcome.

You can contribute by:

- Reporting issues
- Suggesting improvements
- Optimizing inference pipelines
- Improving evaluation methodologies
- Adding new transformer architectures
- Enhancing documentation

---

# 📅 Project Status

| Attribute | Value |
|---|---|
| Status | Complete |
| Language | Python |
| Environment | Jupyter Notebook |
| Framework | Transformers + PyTorch |
| Domain | NLP / Text Summarization |
| Last Updated | November 2024 |

---

# ⭐ Summary

This project demonstrates a comprehensive comparative study of transformer-based abstractive summarization models applied to recipe summarization tasks.

The implementation showcases:

- Multiple state-of-the-art LLM architectures
- GPU-accelerated inference
- ROUGE-based evaluation
- Transformer pipeline integration
- Production-level summarization workflows
- Comparative NLP experimentation

The notebooks provide a complete end-to-end implementation for understanding and experimenting with modern LLM-based text summarization systems.

# Recipes-Summarization-Using-LLM

I have web-scraped 14,000 recipes from allrecipes.com using the Beautiful Soup library and generated 
corresponding summaries using the LAMA model. The following is the structure of the dataset


CONCLUSION:--
LAMA 3B Paramater model has get best rouge 1,2,L in all among all open source LLM model because 
It utilizes multiple transformer layers with self-attention mechanisms to capture contextual relationships 
between words in a sequence. With 3 billion parameters, it balances scalability and efficiency, offering 
strong performance for task


