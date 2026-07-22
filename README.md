# Dynamic Pricing Validation Lab

A deployment-ready Streamlit application built from the original **Dynamic Pricing Strategies for Retail** notebook project. It lets users validate retail pricing files, score valid rows with a demonstration Random Forest model, apply auditable price-change guardrails, inspect portfolio patterns, and download results.

> This is a demonstration decision-support tool. The source dataset is synthetic, and its `Adjusted_Price` field is project-generated rather than an observed profit-optimal price. Recommendations must not be published automatically.

## What users can do

- Start with the included 10,000-row project dataset or upload a CSV.
- Download a ready-to-fill input template.
- Validate required columns, data types, dates, positive prices, duplicates, extreme ratios, and unseen categories.
- See blocking errors and warnings at row level.
- Score valid rows while retaining invalid rows in the exported audit file.
- Cap each price recommendation with a configurable ±1% to ±30% guardrail.
- Test a single what-if scenario.
- Review recommendation charts and the model card.
- Download the validation report and complete scored results.

## Run locally

Python 3.12 is recommended.

```bash
git clone https://github.com/mohdsaeedafri/dynamic-pricing-validation-lab.git
cd dynamic-pricing-validation-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The trained artifact is committed for fast startup. To reproduce it:

```bash
python scripts/train_model.py
```

## Input schema

| Column | Requirement | Accepted content | Use |
|---|---|---|---|
| `PRICE_RETAIL` | Required | Positive number | Reference/list price |
| `PRICE_CURRENT` | Required | Positive number | Current selling price |
| `RunDate` | Required | Parseable date | Effective/observation date and season |
| `DEPARTMENT` | Required | Non-empty text | Product department |
| `CATEGORY` | Required | Non-empty text | Product category |
| `SKU` | Optional | Text or number | Identifier and duplicate check |
| `PRODUCT_NAME` | Optional | Text | Display label |
| `BRAND` | Optional | Text | Export context |
| `PROMOTION` | Optional | Text | Model feature; defaults to `Regular` |
| `SHIPPING_LOCATION` | Optional | Text | Export context |

Common variants such as `retail_price`, `current price`, `effective_date`, and different capitalization are standardized automatically. The sidebar template is the safest starting point.

## Tests

```bash
python -m compileall -q streamlit_app.py src scripts
python -m pytest
```

The test suite covers feature engineering, schema validation, warnings and blocking rules, model guardrails, artifact inference, and a Streamlit startup smoke test. The GitHub Actions workflow runs these checks on every push and pull request.

## Free deployment

The repository is arranged for Streamlit Community Cloud:

- Entrypoint: `streamlit_app.py`
- Python: `3.12`
- Dependencies: `requirements.txt`
- Configuration: `.streamlit/config.toml`
- Secrets: none required

Follow [DEPLOYMENT.md](DEPLOYMENT.md) for the exact deployment and verification checklist.

## Repository layout

```text
.
├── streamlit_app.py
├── artifacts/
│   ├── dynamic_pricing_model.joblib
│   └── model_metadata.json
├── src/dynamic_pricing/
│   ├── constants.py
│   ├── features.py
│   ├── modeling.py
│   └── validation.py
├── scripts/train_model.py
├── tests/
├── .streamlit/config.toml
├── requirements.txt
└── Dynamic-Pricing-Strategies-for-Retail-A-Data-Driven-Approach-main/
    └── original notebook and datasets
```

## Model interpretation

The app trains a Random Forest to reproduce the repository's `Adjusted_Price` output using retail/current price, month, season, category, department, promotion, and derived features. The near-perfect synthetic holdout score is expected because the target was generated deterministically from closely related price features. It is **not** evidence of real commercial accuracy.

A production pricing model would additionally need, at minimum, units sold, product cost, inventory, competitor prices, demand elasticity, store/channel context, experimentation, bias/fairness review, monitoring, approval controls, and rollback logic.

## Privacy and operating limits

- The application does not intentionally write uploaded CSV contents to disk or a database.
- A public Community Cloud demo is not an approved destination for confidential, personal, regulated, or commercially sensitive data.
- The demo limits uploads to 50 MB and 100,000 rows to remain suitable for a free resource-constrained environment.
- Each output is a recommendation for human review, not an automated price instruction.

## Attribution

The original notebook and synthetic datasets come from the linked public repository and are provided under its MIT license. This application preserves that project as the source material while adding validation, inference, testing, and deployment layers.
