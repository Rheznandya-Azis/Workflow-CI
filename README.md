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

## Menjalankan secara lokal

```bash
python -m pip install -r MLProject/requirements.txt

mlflow run ./MLProject \
  --entry-point main \
  --env-manager local \
  -P data_path="MLProject/credit-eligibility_preprocessing.xlsx" \
  -P test_size=0.20 \
  -P output_dir="MLProject/outputs"
```

## Menjalankan GitHub Actions

Workflow dapat dijalankan:

1. Secara manual melalui tab **Actions** lalu **Run workflow**.
2. Secara otomatis ketika terdapat push ke branch `main` yang mengubah file
   model, konfigurasi MLflow Project, dependensi, dataset, atau workflow.

Hasil model dan evaluasi tersedia sebagai artifact pada setiap workflow run.
