"""Command line entry point: python -m pedalpulse <step> [--months SPEC] [--force]."""

import argparse
import time

from . import config
from .months import parse_months

STEPS = ("download", "load", "aggregate", "calendar", "stations", "cohort", "grid", "weather", "export", "report")


def run_step(step: str, months: list[str], force: bool) -> None:
    if step == "download":
        from .download import download

        download(months)
    elif step == "load":
        from .trips import load_months

        load_months(months, force)
    elif step == "aggregate":
        from .hourly import aggregate_months

        aggregate_months(months, force)
    elif step == "calendar":
        from .calendar_features import build_calendar

        build_calendar(force)
    elif step == "stations":
        from .stations import build_stations

        build_stations(force)
    elif step == "cohort":
        from .cohort import build_cohort

        build_cohort(force)
    elif step == "grid":
        from .grid import build_grid

        build_grid(force)
    elif step == "weather":
        from .weather import build_weather

        build_weather(force)
    elif step == "export":
        from .export import export_release

        export_release(force)
    elif step == "report":
        from .summary import write_summary

        write_summary()
    elif step == "verify":
        from .export import verify_release

        verify_release()


def main() -> None:
    parser = argparse.ArgumentParser(prog="pedalpulse")
    parser.add_argument("step", choices=(*STEPS, "all", "verify"))
    parser.add_argument("--months", help="YYYY-MM, comma list or YYYY-MM:YYYY-MM (download/load/aggregate)")
    parser.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    args = parser.parse_args()
    config.ensure_dirs()
    months = parse_months(args.months)
    steps = STEPS if args.step == "all" else (args.step,)
    for step in steps:
        started = time.time()
        print(f"== {step}")
        run_step(step, months, args.force)
        print(f"== {step} done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
