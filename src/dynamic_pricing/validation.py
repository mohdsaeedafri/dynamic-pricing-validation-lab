"""Schema and row-level validation for user-supplied pricing data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable, Mapping

import pandas as pd

from .constants import COLUMN_ALIASES, OPTIONAL_COLUMNS, REQUIRED_COLUMNS


@dataclass(frozen=True)
class RuleResult:
    """Result for one validation rule."""

    check: str
    severity: str
    affected_rows: int
    message: str

    @property
    def status(self) -> str:
        if self.severity == "Info":
            return "Info"
        if self.affected_rows == 0:
            return "Passed"
        return "Failed" if self.severity == "Error" else "Review"


@dataclass
class ValidationReport:
    """Complete validation output used by the UI and tests."""

    data: pd.DataFrame
    valid_mask: pd.Series
    rules: list[RuleResult]
    missing_columns: list[str]
    renamed_columns: dict[str, str]
    warning_mask: pd.Series

    @property
    def is_schema_valid(self) -> bool:
        return not self.missing_columns

    @property
    def valid_rows(self) -> pd.DataFrame:
        return self.data.loc[self.valid_mask].copy()

    @property
    def invalid_rows(self) -> pd.DataFrame:
        return self.data.loc[~self.valid_mask].copy()

    @property
    def invalid_row_count(self) -> int:
        return int((~self.valid_mask).sum())

    @property
    def warning_row_count(self) -> int:
        return int(self.warning_mask.sum())

    def rules_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Status": rule.status,
                    "Severity": rule.severity,
                    "Check": rule.check,
                    "Affected rows": rule.affected_rows,
                    "Details": rule.message,
                }
                for rule in self.rules
            ]
        )


def _normalized_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def canonicalize_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map common case and naming variants to the documented schema."""

    frame = data.copy()
    canonical_present = {str(column) for column in frame.columns}
    rename_map: dict[str, str] = {}

    for column in frame.columns:
        original = str(column)
        target = COLUMN_ALIASES.get(_normalized_header(original))
        if target and original != target and target not in canonical_present:
            rename_map[original] = target
            canonical_present.add(target)

    return frame.rename(columns=rename_map), rename_map


def _add_row_issue(
    row_issues: list[list[str]], mask: pd.Series, message: str
) -> None:
    for position in mask[mask].index:
        row_issues[int(position)].append(message)


def _known_values(
    known_categories: Mapping[str, Iterable[str]] | None, column: str
) -> set[str]:
    if not known_categories:
        return set()
    return {
        str(value).strip().casefold()
        for value in known_categories.get(column, [])
        if pd.notna(value)
    }


def validate_dataframe(
    data: pd.DataFrame,
    known_categories: Mapping[str, Iterable[str]] | None = None,
) -> ValidationReport:
    """Validate, normalize, and annotate a raw pricing dataframe.

    Errors make a row ineligible for scoring. Warnings remain scoreable and
    are deliberately surfaced to the user for review.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    frame, renamed = canonicalize_columns(data)
    frame = frame.reset_index(drop=True)
    row_count = len(frame)
    rules: list[RuleResult] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]

    if row_count == 0:
        rules.append(RuleResult("File contains rows", "Error", 1, "The uploaded file is empty."))

    rules.append(
        RuleResult(
            "Required columns",
            "Error",
            row_count if missing else 0,
            "Missing: " + ", ".join(missing) if missing else "All required columns are present.",
        )
    )

    valid_mask = pd.Series(True, index=frame.index, dtype=bool)
    warning_mask = pd.Series(False, index=frame.index, dtype=bool)
    row_issues: list[list[str]] = [[] for _ in range(row_count)]

    if missing or row_count == 0:
        valid_mask[:] = False
        if missing:
            issue = "Missing required column(s): " + ", ".join(missing)
            for messages in row_issues:
                messages.append(issue)
        frame["Validation_Status"] = "Invalid"
        frame["Validation_Issues"] = ["; ".join(messages) for messages in row_issues]
        return ValidationReport(frame, valid_mask, rules, missing, renamed, warning_mask)

    for column in ("PRICE_RETAIL", "PRICE_CURRENT"):
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid_numeric = converted.isna()
        nonpositive = converted.notna() & converted.le(0)
        frame[column] = converted

        label = column.replace("_", " ").title()
        rules.append(
            RuleResult(
                f"{label} is numeric",
                "Error",
                int(invalid_numeric.sum()),
                f"{label} must contain a number.",
            )
        )
        rules.append(
            RuleResult(
                f"{label} is positive",
                "Error",
                int(nonpositive.sum()),
                f"{label} must be greater than zero.",
            )
        )
        _add_row_issue(row_issues, invalid_numeric, f"{label} is missing or non-numeric")
        _add_row_issue(row_issues, nonpositive, f"{label} must be greater than zero")
        valid_mask &= ~(invalid_numeric | nonpositive)

    parsed_dates = pd.to_datetime(frame["RunDate"], errors="coerce")
    invalid_date = parsed_dates.isna()
    future_date = parsed_dates.notna() & parsed_dates.dt.date.gt(date.today())
    frame["RunDate"] = parsed_dates
    rules.extend(
        [
            RuleResult(
                "Run date is valid",
                "Error",
                int(invalid_date.sum()),
                "RunDate must be a parseable calendar date.",
            ),
            RuleResult(
                "Run date is not in the future",
                "Warning",
                int(future_date.sum()),
                "Future-dated rows are allowed but should be confirmed.",
            ),
        ]
    )
    _add_row_issue(row_issues, invalid_date, "RunDate is missing or invalid")
    _add_row_issue(row_issues, future_date, "RunDate is in the future")
    valid_mask &= ~invalid_date
    warning_mask |= future_date

    for column in ("DEPARTMENT", "CATEGORY"):
        values = frame[column].astype("string").str.strip()
        blank = values.isna() | values.eq("")
        frame[column] = values
        label = column.title()
        rules.append(
            RuleResult(
                f"{label} is populated",
                "Error",
                int(blank.sum()),
                f"{label} cannot be blank.",
            )
        )
        _add_row_issue(row_issues, blank, f"{label} is blank")
        valid_mask &= ~blank

        known = _known_values(known_categories, column)
        if known:
            unknown = values.notna() & ~values.str.casefold().isin(known)
            rules.append(
                RuleResult(
                    f"{label} was seen during training",
                    "Warning",
                    int(unknown.sum()),
                    "Unseen values can be scored but predictions are less reliable.",
                )
            )
            _add_row_issue(row_issues, unknown, f"{label} was not seen during training")
            warning_mask |= unknown

    ratio = frame["PRICE_CURRENT"] / frame["PRICE_RETAIL"].replace(0, pd.NA)
    unusual_ratio = ratio.notna() & ((ratio.lt(0.2)) | (ratio.gt(5.0)))
    rules.append(
        RuleResult(
            "Current-to-retail price ratio",
            "Warning",
            int(unusual_ratio.sum()),
            "Review rows where current price is below 20% or above 500% of retail price.",
        )
    )
    _add_row_issue(row_issues, unusual_ratio, "Unusual current-to-retail price ratio")
    warning_mask |= unusual_ratio

    duplicate_rows = frame.duplicated(keep=False)
    rules.append(
        RuleResult(
            "Duplicate records",
            "Warning",
            int(duplicate_rows.sum()),
            "Exact duplicate records should be reviewed before downstream use.",
        )
    )
    _add_row_issue(row_issues, duplicate_rows, "Exact duplicate record")
    warning_mask |= duplicate_rows

    if "SKU" in frame.columns:
        duplicate_sku = frame["SKU"].notna() & frame["SKU"].duplicated(keep=False)
        rules.append(
            RuleResult(
                "Duplicate SKU identifiers",
                "Warning",
                int(duplicate_sku.sum()),
                "Repeated SKUs may be valid history; confirm each row represents the intended period.",
            )
        )
        _add_row_issue(row_issues, duplicate_sku, "Duplicate SKU identifier")
        warning_mask |= duplicate_sku

    absent_optional = [column for column in OPTIONAL_COLUMNS if column not in frame.columns]
    rules.append(
        RuleResult(
            "Optional context columns",
            "Info",
            0,
            "Not supplied: " + ", ".join(absent_optional)
            if absent_optional
            else "All optional context columns are present.",
        )
    )

    frame["Validation_Status"] = valid_mask.map({True: "Valid", False: "Invalid"})
    frame["Validation_Issues"] = ["; ".join(messages) for messages in row_issues]

    return ValidationReport(frame, valid_mask, rules, missing, renamed, warning_mask)

