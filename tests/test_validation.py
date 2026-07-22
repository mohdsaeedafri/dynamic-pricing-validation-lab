import pandas as pd

from dynamic_pricing.validation import validate_dataframe


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SKU": ["A", "B"],
            "DEPARTMENT": ["Bakery", "Beverages"],
            "CATEGORY": ["Bread", "Juice"],
            "PRICE_RETAIL": [2.99, 3.99],
            "PRICE_CURRENT": [2.79, 3.89],
            "RunDate": ["2024-01-10", "2024-06-15"],
        }
    )


def test_valid_rows_pass_blocking_checks() -> None:
    report = validate_dataframe(valid_frame())

    assert report.is_schema_valid
    assert report.valid_mask.tolist() == [True, True]
    assert report.invalid_row_count == 0
    assert report.data["Validation_Status"].tolist() == ["Valid", "Valid"]


def test_common_header_aliases_are_canonicalized() -> None:
    aliased = valid_frame().rename(
        columns={
            "PRICE_RETAIL": "Retail Price",
            "PRICE_CURRENT": "current_price",
            "RunDate": "effective date",
            "DEPARTMENT": "Department",
            "CATEGORY": "category",
        }
    )

    report = validate_dataframe(aliased)

    assert report.is_schema_valid
    assert report.valid_mask.all()
    assert report.renamed_columns["Retail Price"] == "PRICE_RETAIL"
    assert report.renamed_columns["current_price"] == "PRICE_CURRENT"
    assert report.renamed_columns["effective date"] == "RunDate"


def test_row_errors_block_only_bad_rows() -> None:
    frame = valid_frame()
    frame.loc[0, "PRICE_CURRENT"] = 0
    frame.loc[1, "RunDate"] = "not-a-date"

    report = validate_dataframe(frame)

    assert report.valid_mask.tolist() == [False, False]
    assert "greater than zero" in report.data.loc[0, "Validation_Issues"]
    assert "RunDate is missing or invalid" in report.data.loc[1, "Validation_Issues"]


def test_missing_required_column_blocks_scoring() -> None:
    report = validate_dataframe(valid_frame().drop(columns="CATEGORY"))

    assert not report.is_schema_valid
    assert report.missing_columns == ["CATEGORY"]
    assert not report.valid_mask.any()


def test_unknown_category_is_warning_not_error() -> None:
    known = {"DEPARTMENT": ["Bakery", "Beverages"], "CATEGORY": ["Bread", "Juice"]}
    frame = valid_frame()
    frame.loc[1, "CATEGORY"] = "New category"

    report = validate_dataframe(frame, known)

    assert report.valid_mask.all()
    assert report.warning_mask.tolist() == [False, True]
    assert "not seen during training" in report.data.loc[1, "Validation_Issues"]


def test_duplicate_sku_is_reported_as_warning() -> None:
    frame = valid_frame()
    frame["SKU"] = "A"

    report = validate_dataframe(frame)

    assert report.valid_mask.all()
    assert report.warning_row_count == 2

