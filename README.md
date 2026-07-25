# Workflow-CI

Repository ini menjalankan re-training model kelayakan kredit menggunakan
MLflow Project dan GitHub Actions.

## Struktur repository

```text
Workflow-CI/
├── .github/
│   └── workflows/
│       └── retrain.yml
├── MLProject/
│   ├── modelling.py
│   ├── conda.yaml
│   ├── requirements.txt
│   ├── MLproject
│   └── credit-eligibility_preprocessing.xlsx
├── .gitignore
└── README.md
```

GitHub Actions hanya membaca file workflow yang berada di
`.github/workflows/`.
