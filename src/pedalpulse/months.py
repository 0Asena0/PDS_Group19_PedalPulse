"""Study months and the archive that holds each month."""

from datetime import date

from . import config


def study_months() -> list[str]:
    months = []
    year, month = config.STUDY_START.year, config.STUDY_START.month
    while (year, month) <= (config.STUDY_END.year, config.STUDY_END.month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def parse_months(spec: str | None) -> list[str]:
    """Accepts None (all study months), 'YYYY-MM', a comma list or a range 'YYYY-MM:YYYY-MM'."""
    every = study_months()
    if not spec:
        return every
    if ":" in spec:
        first, last = spec.split(":")
        selected = [m for m in every if first <= m <= last]
    else:
        selected = [m.strip() for m in spec.split(",")]
    unknown = [m for m in selected if m not in every]
    if unknown or not selected:
        raise ValueError(f"months outside the study period: {unknown or spec}")
    return selected


def month_key(month: str) -> str:
    return month.replace("-", "")


def archive_for_month(month: str) -> str:
    if month.startswith("2023"):
        return "2023-citibike-tripdata.zip"
    return f"{month_key(month)}-citibike-tripdata.zip"


def month_bounds(month: str) -> tuple[date, date]:
    year, mon = (int(part) for part in month.split("-"))
    start = date(year, mon, 1)
    end = date(year + 1, 1, 1) if mon == 12 else date(year, mon + 1, 1)
    return start, end
