# VR_MiniProject2
This repository contains all code, data, and analysis for our Visual Recognition Mini Project 2, focused on VQA (Visual Question Answering) using transformer-based models like ViLT and BLIP.

## Directory Structure
``` python
VR_MiniProject2/
├── curated_dataset.csv         # Directory containing QnA CSV files
├── notebooks/                    # Jupyter notebooks for training, evaluation, and visualization
│   ├── API_Call.ipynb            # Dataset Creation
│   ├── blip-lora-ft.ipynb       # Fine-tuning and evaluating BLIP model
│   ├── fine-tuning-vilt-1.ipynb # ViLT-1 training with custom label mapping
│   ├── fine-tuning-vilt-2.ipynb # ViLT-2 training with cosine similarity-based label replacement
│   ├── baseline-evaluation-vilt.ipynb # ViLT baseline evaluation
│   └── blip_baseline.ipynb      # BLIP Baseline evaluation
├── MS2024018/                   # Inference and requirements directory
│   ├── inference.py             # Inference script
│   └── requirements.txt         # Dependencies file
├── Report.pdf                   # Project report
└── README.md                    # Project overview
```


## How to Run

1. Clone the MS2024018 folder in this repo
2. Install dependencies with `pip install -r requirements.txt`
3. Run inference.py
