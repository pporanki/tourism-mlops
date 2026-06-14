# Visit with Us — Wellness Tourism Package MLOps Pipeline

End-to-end MLOps pipeline that predicts whether a customer will purchase the
newly introduced **Wellness Tourism Package**, automated with **GitHub Actions**
and **Hugging Face Hub** (datasets, model hub, and Spaces).

## Folder structure

```
tourism-mlops/
├── .github/
│   └── workflows/
│       └── pipeline.yml             # CI/CD: data → prep → train → deploy
├── tourism_project/
│   ├── data/
│   │   ├── tourism.csv              # raw dataset
│   │   ├── data_register.py         # register raw data on HF dataset space
│   │   └── prep.py                  # clean, split, upload train/test to HF
│   ├── model_building/
│   │   └── train.py                 # tune XGBoost, log to MLflow, register model
│   └── deployment/
│       ├── app.py                   # Streamlit front-end
│       ├── Dockerfile               # HF Space (Docker SDK) image
│       ├── requirements.txt         # deployment dependencies
│       └── hosting.py               # push deployment files to HF Space
├── requirements.txt                 # pipeline dependencies
├── .env.example                     # template for HF_USERNAME / HF_TOKEN
├── .env                             # your real credentials (git-ignored)
├── .gitignore
└── README.md
```

## Pipeline stages

| Stage | Script | What it does |
|-------|--------|--------------|
| Data Registration | `data/data_register.py` | Uploads raw `tourism.csv` to the HF dataset space |
| Data Preparation | `data/prep.py` | Loads from HF, cleans, stratified split, uploads `train.csv`/`test.csv` |
| Model Building | `model_building/train.py` | GridSearchCV-tuned XGBoost, MLflow tracking, registers best model on HF model hub |
| Deployment | `deployment/hosting.py` | Pushes Streamlit app + Dockerfile to a HF Space |

## Configuration

The pipeline reads two values from the environment:

| Variable | Description |
|----------|-------------|
| `HF_USERNAME` | Your Hugging Face username (used to build all repo IDs) |
| `HF_TOKEN` | A Hugging Face access token with **write** permission |

**Locally** these come from a `.env` file loaded by `python-dotenv`.
**In GitHub Actions** they come from **repository Secrets** (Settings → Secrets
and variables → Actions). The code path is identical (`os.getenv(...)`), so
nothing changes between local and CI.

```bash
cp .env.example .env      # then edit .env and paste your username + write token
```

> `.env` is git-ignored and never pushed. Only `.env.example` (the template) is committed.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in HF_USERNAME and HF_TOKEN

python tourism_project/data/data_register.py
python tourism_project/data/prep.py
python tourism_project/model_building/train.py
python tourism_project/deployment/hosting.py
```

## Links

- **GitHub repository:** `https://github.com/prudvikrishna/tourism-mlops`
- **Hugging Face dataset:** `https://huggingface.co/datasets/prudvikrishna/tourism`
- **Hugging Face model:** `https://huggingface.co/prudvikrishna/tourism-package-model`
- **Hugging Face Space (Streamlit):** `https://huggingface.co/spaces/prudvikrishna/tourism-package-predictor`
