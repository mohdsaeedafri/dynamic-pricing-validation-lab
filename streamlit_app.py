"""Dynamic Pricing Validation Lab — Streamlit entrypoint."""

from __future__ import annotations

from datetime import date
from io import BytesIO
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dynamic_pricing.constants import MAX_UPLOAD_ROWS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from dynamic_pricing.modeling import score_dataframe
from dynamic_pricing.validation import ValidationReport, validate_dataframe

DATA_PATH = ROOT / (
    "Dynamic-Pricing-Strategies-for-Retail-A-Data-Driven-Approach-main/adjusted_prices.csv"
)
MODEL_PATH = ROOT / "artifacts/dynamic_pricing_model.joblib"
METADATA_PATH = ROOT / "artifacts/model_metadata.json"
APP_BUILD = "2026.07.23.1"
REPOSITORY_URL = (
    "https://github.com/mansi-im-gif/"
    "Dynamic-Pricing-Strategies-for-Retail-A-Data-Driven-Approach-main"
)

st.set_page_config(
    page_title="Dynamic Pricing Validation Lab",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {
        border: 1px solid #dbe7e3;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        background: #fbfdfc;
    }
    .hero {
        padding: 1.25rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #083c36 0%, #0f766e 65%, #1f9d84 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {font-size: 2rem; margin: 0 0 0.35rem 0; color: white;}
    .hero p {font-size: 1rem; margin: 0; color: #e7fffa; max-width: 900px;}
    .small-note {color: #52635f; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Loading recommendation model…")
def load_model() -> object:
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict[str, object]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def template_csv() -> bytes:
    template = pd.DataFrame(
        [
            {
                "SKU": "ICE-001",
                "PRODUCT_NAME": "Vanilla Ice Cream 14 Oz",
                "BRAND": "Demo Brand",
                "DEPARTMENT": "Frozen Foods",
                "CATEGORY": "Ice Cream",
                "PRICE_RETAIL": 4.99,
                "PRICE_CURRENT": 4.79,
                "RunDate": "2024-07-15",
                "PROMOTION": "Summer Premium",
                "SHIPPING_LOCATION": "10001",
            },
            {
                "SKU": "BRD-002",
                "PRODUCT_NAME": "Whole Wheat Bread 20 Oz",
                "BRAND": "Demo Brand",
                "DEPARTMENT": "Bakery",
                "CATEGORY": "Bread",
                "PRICE_RETAIL": 2.99,
                "PRICE_CURRENT": 2.79,
                "RunDate": "2024-10-10",
                "PROMOTION": "Regular",
                "SHIPPING_LOCATION": "60601",
            },
            {
                "SKU": "JCE-003",
                "PRODUCT_NAME": "Orange Juice 64 Oz",
                "BRAND": "Demo Brand",
                "DEPARTMENT": "Beverages",
                "CATEGORY": "Juice",
                "PRICE_RETAIL": 3.99,
                "PRICE_CURRENT": 3.89,
                "RunDate": "2024-04-20",
                "PROMOTION": "Regular",
                "SHIPPING_LOCATION": "30301",
            },
        ]
    )
    return template.to_csv(index=False).encode("utf-8")


def dataframe_to_csv(frame: pd.DataFrame) -> bytes:
    export = frame.copy()
    if "RunDate" in export:
        export["RunDate"] = pd.to_datetime(export["RunDate"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    return export.to_csv(index=False).encode("utf-8")


def read_uploaded_csv(uploaded_file: object) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    try:
        frame = pd.read_csv(BytesIO(content), encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        frame = pd.read_csv(BytesIO(content), encoding="latin-1", dtype=str)
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(
            f"This demo accepts up to {MAX_UPLOAD_ROWS:,} rows per upload; "
            f"the file contains {len(frame):,}."
        )
    return frame


def add_identifier(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "PRODUCT_NAME" not in display:
        display["PRODUCT_NAME"] = [f"Row {index + 1}" for index in range(len(display))]
    display["PRODUCT_NAME"] = display["PRODUCT_NAME"].fillna("Unnamed product").astype(str)
    if "SKU" not in display:
        display["SKU"] = [f"ROW-{index + 1:05d}" for index in range(len(display))]
    return display


def build_complete_results(report: ValidationReport, scored: pd.DataFrame) -> pd.DataFrame:
    result_columns = [
        "Model_Prediction",
        "Recommended_Price",
        "Recommended_Change",
        "Price_Change_Pct",
        "Pricing_Action",
        "Guardrail_Applied",
    ]
    # A previously exported file can be uploaded again safely; fresh scoring
    # replaces stale result columns instead of causing an overlapping join.
    base = report.data.drop(columns=result_columns, errors="ignore")
    return base.join(scored.loc[:, result_columns], how="left")


def money(value: float) -> str:
    return f"${value:,.2f}"


def show_quality_tab(report: ValidationReport) -> None:
    st.subheader("Automated validation results")
    if report.renamed_columns:
        mappings = ", ".join(
            f"`{source}` → `{target}`" for source, target in report.renamed_columns.items()
        )
        st.info(f"Recognized and standardized column names: {mappings}")

    rules = report.rules_frame()
    st.dataframe(
        rules,
        hide_index=True,
        width="stretch",
        column_config={
            "Affected rows": st.column_config.NumberColumn(format="%d"),
            "Details": st.column_config.TextColumn(width="large"),
        },
    )

    preview_tab, invalid_tab, schema_tab = st.tabs(
        ["Validated preview", "Rows needing correction", "Accepted schema"]
    )
    with preview_tab:
        st.dataframe(report.data.head(1_000), hide_index=True, width="stretch")
        if len(report.data) > 1_000:
            st.caption("Preview limited to the first 1,000 rows; downloads contain all rows.")
    with invalid_tab:
        if report.invalid_rows.empty:
            st.success("No blocking row-level errors were found.")
        else:
            st.dataframe(
                report.invalid_rows.head(1_000), hide_index=True, width="stretch"
            )
    with schema_tab:
        schema = pd.DataFrame(
            [
                ("PRICE_RETAIL", "Required", "Positive number", "Reference/list price"),
                ("PRICE_CURRENT", "Required", "Positive number", "Current selling price"),
                ("RunDate", "Required", "Date", "Pricing effective or observation date"),
                ("DEPARTMENT", "Required", "Text", "Top-level product group"),
                ("CATEGORY", "Required", "Text", "Product category"),
                ("SKU", "Optional", "Text or number", "Row/product identifier"),
                ("PRODUCT_NAME", "Optional", "Text", "Readable product label"),
                ("BRAND", "Optional", "Text", "Brand name"),
                ("PROMOTION", "Optional", "Text", "Defaults to Regular"),
                ("SHIPPING_LOCATION", "Optional", "Text", "Store/ZIP/location identifier"),
            ],
            columns=["Column", "Requirement", "Type", "Purpose"],
        )
        st.dataframe(schema, hide_index=True, width="stretch")
        st.caption("Column matching is case-insensitive and accepts common underscore/space variants.")

    st.download_button(
        "Download validation report",
        dataframe_to_csv(report.data),
        file_name="pricing_validation_report.csv",
        mime="text/csv",
        width="content",
    )


def show_batch_recommendations(
    model: object, report: ValidationReport
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if report.valid_rows.empty:
        st.error("No valid rows are available to score. Correct the errors shown in Data quality.")
        return pd.DataFrame(), report.data.copy()

    control_a, control_b = st.columns(2)
    with control_a:
        guardrail = st.slider(
            "Maximum change from current price",
            min_value=1,
            max_value=30,
            value=15,
            step=1,
            format="%d%%",
            help="Caps each recommendation above and below the current price.",
            key="batch_guardrail",
        )
    with control_b:
        rounding = st.selectbox(
            "Price rounding",
            ["Nearest cent", "End in .99"],
            help=".99 rounding is used only when it remains inside the selected guardrail.",
            key="batch_rounding",
        )

    with st.spinner("Scoring valid rows…"):
        scored = score_dataframe(model, add_identifier(report.valid_rows), guardrail, rounding)
    complete_results = build_complete_results(report, scored)

    current_average = float(scored["PRICE_CURRENT"].mean())
    recommended_average = float(scored["Recommended_Price"].mean())
    average_change = float(scored["Price_Change_Pct"].mean())
    adjusted_count = int(scored["Pricing_Action"].ne("Hold").sum())
    guardrail_count = int(scored["Guardrail_Applied"].sum())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Average current", money(current_average))
    metric_columns[1].metric(
        "Average recommended", money(recommended_average), f"{average_change:+.2f}%"
    )
    metric_columns[2].metric("Rows changed", f"{adjusted_count:,}")
    metric_columns[3].metric("Guardrails triggered", f"{guardrail_count:,}")

    filter_a, filter_b = st.columns(2)
    actions = sorted(scored["Pricing_Action"].dropna().unique().tolist())
    categories = sorted(scored["CATEGORY"].dropna().astype(str).unique().tolist())
    selected_actions = filter_a.multiselect(
        "Filter by action", actions, default=actions, key="batch_action_filter"
    )
    selected_categories = filter_b.multiselect(
        "Filter by category", categories, default=categories, key="batch_category_filter"
    )
    filtered = scored[
        scored["Pricing_Action"].isin(selected_actions)
        & scored["CATEGORY"].astype(str).isin(selected_categories)
    ]

    display_columns = [
        column
        for column in (
            "SKU",
            "PRODUCT_NAME",
            "DEPARTMENT",
            "CATEGORY",
            "PRICE_RETAIL",
            "PRICE_CURRENT",
            "Recommended_Price",
            "Price_Change_Pct",
            "Pricing_Action",
            "Guardrail_Applied",
        )
        if column in filtered.columns
    ]
    st.dataframe(
        filtered.loc[:, display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "PRICE_RETAIL": st.column_config.NumberColumn("Retail price", format="$%.2f"),
            "PRICE_CURRENT": st.column_config.NumberColumn("Current price", format="$%.2f"),
            "Recommended_Price": st.column_config.NumberColumn(
                "Recommended price", format="$%.2f"
            ),
            "Price_Change_Pct": st.column_config.NumberColumn("Change", format="%.2f%%"),
            "Guardrail_Applied": st.column_config.CheckboxColumn("Guardrail used"),
        },
    )
    st.caption(f"Showing {len(filtered):,} of {len(scored):,} scored rows.")
    st.download_button(
        "Download complete scored file",
        dataframe_to_csv(complete_results),
        file_name="dynamic_pricing_results.csv",
        mime="text/csv",
        type="primary",
    )
    return scored, complete_results


def show_simulator(model: object, metadata: dict[str, object]) -> None:
    st.markdown("Test one pricing scenario without preparing a CSV.")
    by_department = metadata["categories_by_department"]
    promotions = metadata["known_categories"]["PROMOTION"]

    with st.form("pricing_simulator"):
        col_1, col_2, col_3 = st.columns(3)
        department = col_1.selectbox("Department", sorted(by_department))
        category = col_2.selectbox("Category", by_department[department])
        run_date = col_3.date_input("Effective date", value=date(2024, 7, 15))

        col_4, col_5, col_6 = st.columns(3)
        retail = col_4.number_input("Retail price", min_value=0.01, value=4.99, step=0.10)
        current = col_5.number_input("Current price", min_value=0.01, value=4.79, step=0.10)
        promotion = col_6.selectbox("Promotion", promotions)

        col_7, col_8 = st.columns(2)
        guardrail = col_7.slider(
            "Maximum change", 1, 30, 15, format="%d%%", key="sim_guardrail"
        )
        rounding = col_8.selectbox(
            "Rounding", ["Nearest cent", "End in .99"], key="sim_rounding"
        )
        submitted = st.form_submit_button("Calculate recommendation", type="primary")

    if submitted:
        scenario = pd.DataFrame(
            [
                {
                    "DEPARTMENT": department,
                    "CATEGORY": category,
                    "PRICE_RETAIL": retail,
                    "PRICE_CURRENT": current,
                    "RunDate": run_date.isoformat(),
                    "PROMOTION": promotion,
                }
            ]
        )
        scored = score_dataframe(model, scenario, guardrail, rounding).iloc[0]
        result_a, result_b, result_c = st.columns(3)
        result_a.metric("Current price", money(float(scored["PRICE_CURRENT"])))
        result_b.metric(
            "Recommended price",
            money(float(scored["Recommended_Price"])),
            f"{float(scored['Price_Change_Pct']):+.2f}%",
        )
        result_c.metric("Action", str(scored["Pricing_Action"]))
        if bool(scored["Guardrail_Applied"]):
            st.warning("The raw model output was capped by your maximum-change guardrail.")


def show_insights(scored: pd.DataFrame) -> None:
    if scored.empty:
        st.info("Portfolio insights appear after at least one valid row is scored.")
        return

    chart_left, chart_right = st.columns(2)
    category_summary = (
        scored.groupby("CATEGORY", as_index=False)
        .agg(
            Products=("CATEGORY", "size"),
            Average_Change_Pct=("Price_Change_Pct", "mean"),
        )
        .sort_values("Average_Change_Pct")
    )
    category_chart = px.bar(
        category_summary,
        x="Average_Change_Pct",
        y="CATEGORY",
        orientation="h",
        color="Average_Change_Pct",
        color_continuous_scale=["#b5485d", "#eef4f2", "#0f766e"],
        color_continuous_midpoint=0,
        labels={"Average_Change_Pct": "Average change (%)", "CATEGORY": "Category"},
        hover_data={"Products": True},
        title="Average recommended change by category",
    )
    category_chart.update_layout(coloraxis_showscale=False, height=440)
    chart_left.plotly_chart(category_chart, width="stretch")

    action_counts = scored["Pricing_Action"].value_counts().rename_axis("Action").reset_index(
        name="Rows"
    )
    action_chart = px.pie(
        action_counts,
        names="Action",
        values="Rows",
        hole=0.62,
        color="Action",
        color_discrete_map={"Increase": "#0f766e", "Hold": "#8fa7a1", "Decrease": "#b5485d"},
        title="Recommendation mix",
    )
    action_chart.update_layout(height=440)
    chart_right.plotly_chart(action_chart, width="stretch")

    scatter = px.scatter(
        scored,
        x="PRICE_CURRENT",
        y="Recommended_Price",
        color="Season" if "Season" in scored.columns else None,
        hover_data=[column for column in ("PRODUCT_NAME", "CATEGORY") if column in scored],
        labels={"PRICE_CURRENT": "Current price", "Recommended_Price": "Recommended price"},
        title="Current vs. recommended price",
        opacity=0.65,
    )
    low = float(min(scored["PRICE_CURRENT"].min(), scored["Recommended_Price"].min()))
    high = float(max(scored["PRICE_CURRENT"].max(), scored["Recommended_Price"].max()))
    scatter.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="No change",
            line={"color": "#667773", "dash": "dash"},
        )
    )
    scatter.update_layout(height=500)
    st.plotly_chart(scatter, width="stretch")
    st.info(
        "These charts summarize model recommendations, not projected revenue. "
        "Revenue impact cannot be estimated without units, demand response, and product cost."
    )


def show_model_card(metadata: dict[str, object]) -> None:
    metrics = metadata["metrics"]
    st.subheader("Model card")
    card_columns = st.columns(4)
    card_columns[0].metric("Model", "Random Forest")
    card_columns[1].metric("Synthetic holdout MAE", money(float(metrics["mae"])))
    card_columns[2].metric("Synthetic holdout RMSE", money(float(metrics["rmse"])))
    card_columns[3].metric("Synthetic holdout R²", f"{float(metrics['r2']):.3f}")

    st.warning(
        "The evaluation target is generated by the source project. These metrics measure how well "
        "the app reproduces that synthetic target; they do not prove profit, demand, or revenue uplift."
    )

    details_left, details_right = st.columns([1.1, 1])
    with details_left:
        st.markdown("#### Training scope")
        scope = metadata["data_scope"]
        scope_table = pd.DataFrame(
            [
                ("Full-fit rows", f"{metadata['full_fit_rows']:,}"),
                ("Date range", f"{scope['minimum_date']} to {scope['maximum_date']}"),
                ("Unique product names", f"{scope['unique_products']:,}"),
                ("Unique SKUs", f"{scope['unique_skus']:,}"),
                ("Locations", f"{scope['unique_locations']:,}"),
                ("Model version", metadata["model_version"]),
                ("scikit-learn", metadata["scikit_learn_version"]),
            ],
            columns=["Item", "Value"],
        )
        st.dataframe(scope_table, hide_index=True, width="stretch")
    with details_right:
        importance = pd.DataFrame(metadata["top_feature_importance"]).sort_values("importance")
        importance_chart = px.bar(
            importance.tail(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top model feature importance",
            color_discrete_sequence=["#0f766e"],
            labels={"importance": "Importance", "feature": "Feature"},
        )
        importance_chart.update_layout(height=390)
        st.plotly_chart(importance_chart, width="stretch")

    st.markdown("#### Known limitations")
    for limitation in metadata["limitations"]:
        st.markdown(f"- {limitation}")
    st.markdown(
        f"Source dataset and original notebook: [GitHub repository]({REPOSITORY_URL})."
    )


def main() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Dynamic Pricing Validation Lab</h1>
          <p>Validate retail pricing files, review data-quality exceptions, test a demonstration
          machine-learning recommendation, and export an auditable result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Build {APP_BUILD} · Demonstration decision-support tool · Uploaded files are processed "
        "in the active app session · Do not upload confidential or regulated data to a public demo."
    )

    required_files = [DATA_PATH, MODEL_PATH, METADATA_PATH]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        st.error("Application artifact(s) missing: " + ", ".join(missing_files))
        st.code("python scripts/train_model.py")
        st.stop()

    metadata = load_metadata()
    model = load_model()

    with st.sidebar:
        st.header("Test data")
        source_mode = st.radio(
            "Choose a source",
            ["Built-in demo", "Upload CSV"],
            help="Start with the project data or test your own schema-compatible file.",
        )
        st.download_button(
            "Download input template",
            template_csv(),
            file_name="dynamic_pricing_input_template.csv",
            mime="text/csv",
            width="stretch",
        )
        st.divider()
        st.markdown("**Minimum required columns**")
        for column in REQUIRED_COLUMNS:
            st.code(column, language=None)
        with st.expander("Optional columns"):
            st.write(", ".join(OPTIONAL_COLUMNS))
        st.caption("Upload limit: 50 MB and 100,000 rows for this free demo.")

    if source_mode == "Built-in demo":
        raw_data = load_demo_data()
        source_label = "Project demonstration dataset"
    else:
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV in the left panel, or download the template to get started.")
            st.stop()
        try:
            raw_data = read_uploaded_csv(uploaded_file)
        except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
            st.error(f"The CSV could not be loaded: {exc}")
            st.stop()
        source_label = uploaded_file.name

    report = validate_dataframe(raw_data, metadata.get("known_categories"))
    st.markdown(f"**Active source:** {source_label}")

    summary_columns = st.columns(4)
    summary_columns[0].metric("Rows received", f"{len(report.data):,}")
    summary_columns[1].metric("Rows ready to score", f"{int(report.valid_mask.sum()):,}")
    summary_columns[2].metric("Rows blocked", f"{report.invalid_row_count:,}")
    summary_columns[3].metric("Rows with warnings", f"{report.warning_row_count:,}")

    if not report.is_schema_valid:
        st.error(
            "Scoring is blocked because required columns are missing: "
            + ", ".join(report.missing_columns)
        )
    elif report.data.empty:
        st.error("Scoring is blocked because the file contains headers but no data rows.")
    elif report.invalid_row_count:
        st.warning(
            f"{report.invalid_row_count:,} row(s) contain blocking errors. Valid rows can still be scored."
        )
    else:
        st.success("The file passed all blocking schema and row checks.")

    quality_tab, recommendations_tab, insights_tab, model_tab = st.tabs(
        ["Data quality", "Recommendations", "Portfolio insights", "Model & limitations"]
    )
    with quality_tab:
        show_quality_tab(report)

    with recommendations_tab:
        batch_tab, simulator_tab = st.tabs(["Batch recommendations", "Single-scenario simulator"])
        with batch_tab:
            scored, _ = show_batch_recommendations(model, report)
        with simulator_tab:
            show_simulator(model, metadata)

    with insights_tab:
        # Reuse default auditable settings so insights are stable across tab visits.
        insight_scored = (
            score_dataframe(model, add_identifier(report.valid_rows), 15, "Nearest cent")
            if not report.valid_rows.empty
            else pd.DataFrame()
        )
        if not insight_scored.empty:
            feature_dates = pd.to_datetime(insight_scored["RunDate"], errors="coerce")
            insight_scored["Season"] = feature_dates.dt.month.map(
                {
                    1: "Winter",
                    2: "Winter",
                    3: "Spring",
                    4: "Spring",
                    5: "Spring",
                    6: "Summer",
                    7: "Summer",
                    8: "Summer",
                    9: "Fall",
                    10: "Fall",
                    11: "Fall",
                    12: "Winter",
                }
            )
        show_insights(insight_scored)

    with model_tab:
        show_model_card(metadata)

    st.divider()
    st.caption(
        "Use recommendations as hypotheses for review or controlled A/B tests. "
        "Do not publish prices automatically from this demonstration model."
    )


if __name__ == "__main__":
    main()
