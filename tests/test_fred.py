from __future__ import annotations

import math

import pytest

from termo.data.fred import FRED_URL, DataValidationError, fetch_series_csv, parse_series_csv

GOOD = "observation_date,DGS1\n1977-02-15,5.39\n1977-02-16,5.40\n1977-02-21,\n1977-02-22,5.46\n"


def test_fetch_asks_for_one_series() -> None:
    seen: list[str] = []

    def fake_get(url: str) -> str:
        seen.append(url)
        return GOOD

    assert fetch_series_csv("DGS1", fake_get) == GOOD
    assert seen == [FRED_URL.format(series_id="DGS1")]


def test_parse_keeps_holidays_as_missing() -> None:
    series = parse_series_csv(GOOD, "DGS1")
    assert series.name == "DGS1"
    assert list(series.index.strftime("%Y-%m-%d")) == [
        "1977-02-15",
        "1977-02-16",
        "1977-02-21",
        "1977-02-22",
    ]
    assert series.iloc[0] == pytest.approx(5.39)
    assert math.isnan(series.iloc[2])


def test_parse_accepts_dot_as_missing() -> None:
    series = parse_series_csv("observation_date,DGS1\n1977-02-15,5.39\n1977-02-16,.\n", "DGS1")
    assert math.isnan(series.iloc[1])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("date,DGS1\n1977-02-15,5.39\n", "expected columns"),
        ("observation_date,DGS2\n1977-02-15,5.39\n", "expected columns"),
        ("observation_date,DGS1\n", "no rows"),
        ("observation_date,DGS1\n1977-02-16,5.4\n1977-02-15,5.3\n", "strictly increasing"),
        ("observation_date,DGS1\n1977-02-15,5.4\n1977-02-15,5.3\n", "strictly increasing"),
        ("observation_date,DGS1\nnot-a-date,5.4\n", "unparseable dates"),
        ("observation_date,DGS1\n1977-02-15,abc\n", "non-numeric"),
        ("observation_date,DGS1\n1977-02-15,\n", "no observed values"),
        ("observation_date,DGS1\n1977-02-15,539\n", "outside"),
        ("observation_date,DGS1\n1977-02-15,-0.5\n", "outside"),
    ],
)
def test_parse_rejects_malformed_files(text: str, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        parse_series_csv(text, "DGS1")
