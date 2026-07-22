# Deploy on Streamlit Community Cloud

Streamlit Community Cloud is the simplest no-cost demo target for this app. Deployment needs one short account-authorized step because Streamlit must connect to a GitHub repository that you administer.

## 1. Put the completed application in your GitHub account

The deployment repository is [`mohdsaeedafri/dynamic-pricing-validation-lab`](https://github.com/mohdsaeedafri/dynamic-pricing-validation-lab). Its `main` branch must contain all of these paths:

```text
streamlit_app.py
requirements.txt
.streamlit/config.toml
artifacts/dynamic_pricing_model.joblib
artifacts/model_metadata.json
src/dynamic_pricing/
Dynamic-Pricing-Strategies-for-Retail-A-Data-Driven-Approach-main/adjusted_prices.csv
```

Confirm the files are committed and pushed:

```bash
git status
git add streamlit_app.py requirements.txt requirements-dev.txt pyproject.toml README.md DEPLOYMENT.md
git add .streamlit .github artifacts scripts src tests .gitignore
git commit -m "Add deployable Streamlit pricing validation app"
git push origin main
```

If you use a feature branch, push that branch and select the same branch during deployment.

## 2. Validate locally before deploying

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
streamlit run streamlit_app.py
```

Open the local URL printed in the terminal. Check both the built-in dataset and a copy of the downloaded upload template.

## 3. Create the free cloud app

1. Open [share.streamlit.io](https://share.streamlit.io/) and continue with GitHub.
2. Authorize access to the public repository (or grant private-repository access if you intentionally made it private).
3. Choose **Create app** and then **Yup, I have an app**.
4. Enter `mohdsaeedafri/dynamic-pricing-validation-lab` as the repository.
5. Select the `main` branch.
6. Set **Main file path** to `streamlit_app.py`.
7. Optionally choose a memorable `streamlit.app` subdomain.
8. Open **Advanced settings**, select Python `3.12`, and leave Secrets empty.
9. Click **Deploy** and watch the build log.

Official references:

- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Declare app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Community Cloud status and limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)

## 4. End-to-end acceptance checklist

After the live URL opens, verify each item:

- [ ] The header and four source-summary metrics render without an error.
- [ ] **Data quality** shows the built-in dataset validation table.
- [ ] **Recommendations** displays scored rows and the CSV download works.
- [ ] Moving the maximum-change slider changes recommendations without exceeding the guardrail.
- [ ] The single-scenario simulator returns a recommendation.
- [ ] Portfolio charts render.
- [ ] The model card explicitly labels the data and target as synthetic/project-generated.
- [ ] The input template downloads.
- [ ] An uploaded copy of the template passes blocking checks.
- [ ] Removing a required column produces a clear blocking error.
- [ ] Entering a zero/negative price blocks only that row.
- [ ] The complete output retains invalid rows with `Validation_Status` and `Validation_Issues`.

The platform health endpoint can also be checked at:

```text
https://YOUR-SUBDOMAIN.streamlit.app/_stcore/health
```

It should return `ok` while the app is awake.

## 5. Understand the free-hosting trade-offs

- Community Cloud apps without traffic hibernate; the first visitor may need to wake the app.
- CPU and memory are shared/limited, so this demo caps uploads at 100,000 rows and 50 MB.
- The app is hosted in the United States according to current Community Cloud documentation.
- Pushing to the selected GitHub branch updates the app automatically.
- Logs are available from **Manage app** to repository collaborators with the required access.

For a validation demo this is appropriate. For confidential retailer data, persistent audit history, authentication/authorization, scheduled scoring, or production SLAs, move to an approved private cloud architecture rather than extending this public demo.

## Troubleshooting

### Dependency installation fails

Verify that `requirements.txt` is in the repository root and that Python 3.12 is selected. Do not add a second environment manager such as `environment.yml` or `uv.lock` unless you intentionally replace `requirements.txt`.

### Model artifact is missing

Run and commit the generated files:

```bash
python scripts/train_model.py
git add artifacts/
git commit -m "Regenerate pricing model artifact"
git push
```

### App starts but cannot find a CSV

Confirm the original nested data file remains at:

```text
Dynamic-Pricing-Strategies-for-Retail-A-Data-Driven-Approach-main/adjusted_prices.csv
```

### Resource-limit page appears

Reduce upload size, reboot from **Manage app**, and inspect logs. For sustained high-volume use, use paid/managed infrastructure with explicit CPU, memory, security, and uptime commitments.
