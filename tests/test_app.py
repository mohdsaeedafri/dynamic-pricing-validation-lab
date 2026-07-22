from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_with_demo_data() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=30).run()

    assert not app.exception
    assert len(app.metric) >= 4
    assert any("Dynamic Pricing Validation Lab" in element.value for element in app.markdown)


def test_uploaded_csv_scores_valid_rows_and_blocks_bad_rows() -> None:
    root = Path(__file__).resolve().parents[1]
    content = (
        b"DEPARTMENT,CATEGORY,PRICE_RETAIL,PRICE_CURRENT,RunDate,SKU,SHIPPING_LOCATION\n"
        b"Bakery,Bread,2.99,2.79,2024-01-10,A,00123\n"
        b"Beverages,Juice,3.99,-1,2024-06-10,B,30301\n"
    )
    app = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=30).run()
    app.radio[0].set_value("Upload CSV").run()
    app.file_uploader[0].upload("pricing.csv", content, "text/csv").run()

    assert not app.exception
    summary = {metric.label: metric.value for metric in app.metric[:4]}
    assert summary == {
        "Rows received": "2",
        "Rows ready to score": "1",
        "Rows blocked": "1",
        "Rows with warnings": "1",
    }
    assert any(button.label == "Download complete scored file" for button in app.download_button)


def test_uploaded_csv_with_missing_schema_is_explained() -> None:
    root = Path(__file__).resolve().parents[1]
    content = b"DEPARTMENT,PRICE_RETAIL,PRICE_CURRENT,RunDate\nBakery,2.99,2.79,2024-01-10\n"
    app = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=30).run()
    app.radio[0].set_value("Upload CSV").run()
    app.file_uploader[0].upload("missing.csv", content, "text/csv").run()

    assert not app.exception
    assert any("CATEGORY" in error.value for error in app.error)


def test_header_only_csv_is_blocked() -> None:
    root = Path(__file__).resolve().parents[1]
    content = b"DEPARTMENT,CATEGORY,PRICE_RETAIL,PRICE_CURRENT,RunDate\n"
    app = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=30).run()
    app.radio[0].set_value("Upload CSV").run()
    app.file_uploader[0].upload("empty.csv", content, "text/csv").run()

    assert not app.exception
    assert any("no data rows" in error.value for error in app.error)
