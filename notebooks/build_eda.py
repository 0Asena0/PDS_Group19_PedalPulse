"""Build notebooks/01_eda.ipynb from cell sources. Run: uv run python notebooks/build_eda.py"""

import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
TEXT = json.loads((HERE / "eda_text.json").read_text()) if (HERE / "eda_text.json").exists() else {}

SETUP = '''\
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pedalpulse import config

RELEASE = config.RELEASE_DIR
FIGURES = config.FIGURES_DIR
FIGURES.mkdir(parents=True, exist_ok=True)
EDA_END = pd.Timestamp(config.EDA_END)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")
con.execute("SET enable_progress_bar = false")
DEV = f"read_parquet('{(RELEASE / 'station_hours_dev.parquet').as_posix()}')"
EDA = f"(SELECT * FROM {DEV} WHERE date_local <= DATE '{config.EDA_END}')"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.show()

stations = pd.read_parquet(RELEASE / "stations.parquet")
cohort = stations[stations["is_cohort"]]
calendar = pd.read_parquet(RELEASE / "calendar.parquet")
daily = pd.read_parquet(RELEASE / "daily_system.parquet")
daily["date_local"] = pd.to_datetime(daily["date_local"])
daily = daily[daily["date_local"] <= EDA_END]
rows = con.execute(f"SELECT count(*), count(DISTINCT station_id), min(date_local), max(date_local) FROM {EDA}").fetchone()
print(f"EDA window: {rows[2]} to {rows[3]}, {rows[0]:,} station-hours, {rows[1]:,} cohort stations")
'''

CELLS = [
    ("daily_trips", '''\
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(daily["date_local"], daily["trips"] / 1000, lw=0.6, color="0.6", label="daily trips")
ax.plot(daily["date_local"], daily["trips"].rolling(28, center=True).mean() / 1000, lw=2, color="C0", label="28-day mean")
ax.set_ylabel("trips per day (thousands)")
ax.set_title("Citi Bike trips per day, all stations, 2023-01 to 2025-06")
ax.legend(loc="upper left")
save(fig, "01_daily_trips")
by_year = daily.groupby(daily["date_local"].dt.year)["trips"].agg(["sum", "mean", "max"])
summer = daily[daily["date_local"].dt.month.isin([6, 7, 8])].groupby(daily["date_local"].dt.year)["trips"].mean()
print(by_year.round(0)); print("mean per day in June to August (2025: June only):", summer.round(0).to_dict())
print(daily.nsmallest(3, "trips")[["date_local", "trips"]].to_string(index=False))
'''),
    ("coverage", '''\
monthly = con.execute(f"""
    SELECT date_trunc('month', date_local) AS month,
           count(DISTINCT station_id) AS stations_in_grid,
           avg((status = 'inactive')::INT) AS inactive_share,
           avg((pickups = 0)::INT) FILTER (WHERE status = 'observed') AS zero_share
    FROM {EDA} GROUP BY 1 ORDER BY 1
""").df()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(monthly["month"], monthly["stations_in_grid"], marker="o", ms=3)
ax[0].set_title("Cohort stations in the grid")
ax[0].set_ylabel("stations")
ax[1].plot(monthly["month"], monthly["zero_share"] * 100, marker="o", ms=3, label="zero pickups (observed hours)")
ax[1].plot(monthly["month"], monthly["inactive_share"] * 100, marker="o", ms=3, label="inactive hours")
ax[1].set_ylabel("% of station-hours")
ax[1].set_title("Zeros and inactive hours per month")
ax[1].legend(fontsize=8)
for a in ax:
    a.tick_params(axis="x", rotation=45)
save(fig, "02_coverage")
print(monthly.assign(zero_share=lambda d: d.zero_share.round(3), inactive_share=lambda d: d.inactive_share.round(4)).iloc[[0, 5, 11, 17, 23, 29]])
totals = con.execute(f"SELECT avg((status='inactive')::INT), avg((pickups=0)::INT) FILTER (WHERE status='observed') FROM {EDA}").fetchone()
print(f"inactive share {totals[0]:.4f}, zero share of observed hours {totals[1]:.3f}")
'''),
    ("count_distribution", '''\
dist = con.execute(f"SELECT pickups, count(*) AS n FROM {EDA} WHERE status = 'observed' GROUP BY 1 ORDER BY 1").df()
dist["share"] = dist["n"] / dist["n"].sum()
fig, ax = plt.subplots(figsize=(8, 3.4))
binned = dist.assign(bin=dist["pickups"].clip(upper=40)).groupby("bin")["share"].sum()
ax.bar(binned.index, binned.values, width=1.0, color="C0")
ax.set_yscale("log")
ax.set_xlabel("pickups per station-hour (40 = 40 or more)")
ax.set_ylabel("share of observed hours (log)")
ax.set_title("Distribution of hourly pickups")
save(fig, "03_count_distribution")
stats = con.execute(f"""SELECT avg(pickups), var_samp(pickups), quantile_cont(pickups, [0.5, 0.9, 0.99]), max(pickups)
                        FROM {EDA} WHERE status = 'observed'""").fetchone()
print(f"mean {stats[0]:.2f}, variance {stats[1]:.2f}, quantiles 50/90/99 {stats[2]}, max {stats[3]}")
print(dist.head(4))
'''),
    ("daily_profile", '''\
profile = con.execute(f"""
    SELECT s.hour_local, c.is_weekend OR c.is_federal_holiday AS day_off, avg(pickups) AS mean_pickups
    FROM {EDA} s JOIN read_parquet('{(RELEASE / 'calendar.parquet').as_posix()}') c USING (hour_utc)
    WHERE status = 'observed' GROUP BY ALL ORDER BY ALL
""").df()
fig, ax = plt.subplots(figsize=(8, 3.4))
for flag, label in ((False, "working day"), (True, "weekend or federal holiday")):
    part = profile[profile["day_off"] == flag]
    ax.plot(part["hour_local"], part["mean_pickups"], marker="o", ms=3, label=label)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("local hour")
ax.set_ylabel("mean pickups per station-hour")
ax.set_title("Daily profile")
ax.legend()
save(fig, "04_daily_profile")
print(profile.pivot(index="hour_local", columns="day_off", values="mean_pickups").round(2).loc[[3, 8, 12, 15, 17, 18, 22]])
'''),
    ("heatmap", '''\
heat = con.execute(f"""
    SELECT c.weekday, s.hour_local, avg(pickups) AS mean_pickups
    FROM {EDA} s JOIN read_parquet('{(RELEASE / 'calendar.parquet').as_posix()}') c USING (hour_utc)
    WHERE status = 'observed' GROUP BY ALL
""").df().pivot(index="weekday", columns="hour_local", values="mean_pickups")
fig, ax = plt.subplots(figsize=(10, 3.2))
image = ax.imshow(heat.values, aspect="auto", cmap="viridis")
ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_xticks(range(0, 24, 2), range(0, 24, 2))
ax.set_xlabel("local hour")
ax.grid(False)
fig.colorbar(image, ax=ax, label="mean pickups per station-hour")
ax.set_title("Weekday by hour")
save(fig, "05_heatmap_weekday_hour")
print(heat.round(2)[[8, 17, 18]])
'''),
    ("station_map", '''\
volume = con.execute(f"""
    SELECT station_id, sum(pickups) AS pickups, sum(pickups) / count(DISTINCT date_local) AS pickups_per_day
    FROM {EDA} WHERE status = 'observed' GROUP BY 1
""").df().merge(cohort[["station_id", "name", "lat", "lng", "borough"]], on="station_id")
fig, ax = plt.subplots(figsize=(6, 7))
points = ax.scatter(volume["lng"], volume["lat"], c=np.log10(volume["pickups_per_day"].clip(lower=1)), s=6, cmap="plasma")
ax.set_aspect(1 / np.cos(np.radians(40.73)))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
fig.colorbar(points, ax=ax, label="log10 pickups per day", shrink=0.7)
ax.set_title("Cohort stations by volume")
save(fig, "06_station_map")
print(volume["pickups_per_day"].describe().round(1))
print(volume.nlargest(5, "pickups_per_day")[["name", "borough", "pickups_per_day"]].round(0))
'''),
    ("boroughs", '''\
by_borough = volume.groupby("borough").agg(stations=("station_id", "size"), pickups=("pickups", "sum"),
                                           median_station=("pickups_per_day", "median")).sort_values("pickups")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].barh(by_borough.index, by_borough["stations"]); ax[0].set_title("Cohort stations")
ax[1].barh(by_borough.index, by_borough["pickups"] / 1e6); ax[1].set_title("Pickups 2023-01 to 2025-06 (millions)")
save(fig, "07_boroughs")
by_borough["share_pickups"] = by_borough["pickups"] / by_borough["pickups"].sum()
print(by_borough.round(3))
'''),
    ("demand_classes", '''\
volume["demand_class"] = pd.qcut(volume["pickups_per_day"], 3, labels=["low", "medium", "high"])
classes = con.execute(f"""SELECT station_id, avg((pickups = 0)::INT) AS zero_share FROM {EDA}
                          WHERE status = 'observed' GROUP BY 1""").df().merge(volume, on="station_id")
fig, ax = plt.subplots(figsize=(7, 3.4))
for label, part in classes.groupby("demand_class", observed=True):
    ax.hist(part["zero_share"] * 100, bins=40, range=(0, 100), alpha=0.6, label=f"{label} demand")
ax.set_xlabel("% of observed hours with zero pickups (per station)")
ax.set_ylabel("stations")
ax.set_title("Zero share by demand class (tertiles of pickups per day)")
ax.legend()
save(fig, "08_demand_classes")
print(classes.groupby("demand_class", observed=True).agg(stations=("station_id", "size"),
      max_pickups_per_day=("pickups_per_day", "max"), median_zero_share=("zero_share", "median")).round(3))
'''),
    ("weather", '''\
observed = pd.read_parquet(RELEASE / "weather_observed.parquet")
observed["date_local"] = observed["hour_utc"].dt.tz_convert(config.TIMEZONE).dt.tz_localize(None).dt.normalize()
weather_day = observed.groupby("date_local").agg(temperature=("temperature_2m", "mean"), precipitation=("precipitation", "sum"))
days = daily.merge(weather_day, left_on="date_local", right_index=True)
days["rain"] = pd.cut(days["precipitation"], [-0.1, 0.1, 5, 1000], labels=["dry (<0.1 mm)", "0.1-5 mm", ">5 mm"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].scatter(days["temperature"], days["trips"] / 1000, s=5, c=days["date_local"].dt.year, cmap="viridis")
ax[0].set_xlabel("daily mean temperature (deg C, ERA5)"); ax[0].set_ylabel("trips per day (thousands)")
ax[0].set_title("Trips and temperature")
groups = [days.loc[days["rain"] == label, "trips"] / 1000 for label in days["rain"].cat.categories]
ax[1].boxplot(groups, tick_labels=list(days["rain"].cat.categories))
ax[1].set_ylabel("trips per day (thousands)"); ax[1].set_title("Trips and daily precipitation")
save(fig, "09_weather")
print(days.groupby("rain", observed=True)["trips"].agg(["size", "median"]).round(0))
print(days.groupby(pd.cut(days["temperature"], [-20, 0, 10, 20, 30, 40]), observed=True)["trips"].median().round(0))
print("correlation trips/temperature:", round(days["trips"].corr(days["temperature"]), 2))
print(days.nsmallest(3, "trips")[["date_local", "trips", "temperature", "precipitation"]].to_string(index=False, float_format="%.1f"))
'''),
    ("holidays", '''\
day_types = calendar.drop_duplicates("date_local")[["date_local", "weekday", "is_weekend", "is_holiday", "holiday_name", "is_federal_holiday"]]
day_types["date_local"] = pd.to_datetime(day_types["date_local"])
days = days.merge(day_types, on="date_local")
days["day_type"] = np.where(days["is_federal_holiday"], "federal holiday",
                    np.where(days["is_holiday"], "NY state only", np.where(days["is_weekend"], "weekend", "working day")))
summary = days.groupby("day_type")["trips"].agg(["size", "median"])
ratio = days[days["is_holiday"]].copy()
ratio["kind"] = np.where(ratio["is_federal_holiday"], "federal", "NY state only")
ratio["same_month_working_median"] = ratio["date_local"].map(
    lambda d: days.loc[(days["day_type"] == "working day") & (days["date_local"].dt.to_period("M") == d.to_period("M")), "trips"].median())
ratio["ratio"] = ratio["trips"] / ratio["same_month_working_median"]
fig, ax = plt.subplots(figsize=(9, 8))
order = ratio.sort_values("ratio")
ax.barh(order["date_local"].dt.strftime("%Y-%m-%d") + " " + order["holiday_name"], order["ratio"],
        color=np.where(order["kind"] == "federal", "C0", "C1"))
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="C0"), plt.Rectangle((0, 0), 1, 1, color="C1")],
          labels=["federal", "NY state only"], loc="lower right")
ax.axvline(1, color="0.3", lw=1)
ax.set_xlabel("trips relative to the median working day of the same month")
ax.set_title("Holidays")
ax.tick_params(axis="y", labelsize=7.5)
save(fig, "10_holidays")
print(summary.round(0)); print(ratio.groupby("kind")["ratio"].median().round(2))
columns = ["date_local", "holiday_name", "kind", "ratio"]
print(order[columns].head(4).to_string(index=False, float_format="%.2f")); print(order[columns].tail(4).to_string(index=False, float_format="%.2f"))
'''),
    ("ebike_member", '''\
monthly_mix = daily.groupby(daily["date_local"].dt.to_period("M"))[["trips", "ebike_trips", "member_trips"]].sum()
monthly_mix["ebike_share"] = monthly_mix["ebike_trips"] / monthly_mix["trips"]
monthly_mix["member_share"] = monthly_mix["member_trips"] / monthly_mix["trips"]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(monthly_mix.index.to_timestamp(), monthly_mix["ebike_share"] * 100, marker="o", ms=3, label="e-bike share")
ax.plot(monthly_mix.index.to_timestamp(), monthly_mix["member_share"] * 100, marker="o", ms=3, label="member share")
ax.set_ylabel("% of trips"); ax.set_ylim(0, 100); ax.legend()
ax.set_title("E-bike and member share per month")
save(fig, "11_ebike_member")
print(monthly_mix[["ebike_share", "member_share"]].round(3).iloc[[0, 6, 12, 18, 24, -1]])
'''),
]

INTRO = """# PedalPulse: exploratory data analysis

Data: release `data-v1`, file `station_hours_dev.parquet` and the daily system totals, restricted to local dates up to 2025-06-30. The holdout file is not read. Counts are pickups per cohort station and UTC hour; hours with `status = inactive` are left out of the averages unless stated otherwise. Weather in this notebook is observed ERA5 weather and is only descriptive.
"""


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETUP)]
    for key, source in CELLS:
        if key in TEXT and "title" in TEXT[key]:
            cells.append(nbf.v4.new_markdown_cell(f"## {TEXT[key]['title']}"))
        cells.append(nbf.v4.new_code_cell(source))
        if key in TEXT:
            cells.append(nbf.v4.new_markdown_cell(TEXT[key]["text"]))
    notebook.cells = cells
    nbf.write(notebook, HERE / "01_eda.ipynb")


if __name__ == "__main__":
    build()
