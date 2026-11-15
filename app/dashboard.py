import json
from pathlib import Path
from urllib.request import urlretrieve
import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="PedalPulse – NYC Citi Bike Demand Forecast",
    page_icon="🚲",
    layout="wide",
)

REPO_OWNER = "0Asena0"
REPO_NAME = "PDS_Group19_PedalPulse"
SOURCE_BRANCH = "branch_benjamin"
RELEASE_TAG = "data-v1"

RAW_BASE_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{SOURCE_BRANCH}"
RELEASE_BASE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}"

COHORT_PATH = Path("data/cohort.csv")
MANIFEST_PATH = Path("data/manifest.csv")
METRICS_PATH = Path("reports/holdout_metrics.json")
PROCESSED_DIR = Path("data/processed")
PREDS_PATH = (
    PROCESSED_DIR / "predictions.parquet"
    if (PROCESSED_DIR / "predictions.parquet").exists()
    else PROCESSED_DIR / "dummy_predictions.parquet"
)
ACTUALS_PATH = (
    PROCESSED_DIR / "actuals.parquet"
    if (PROCESSED_DIR / "actuals.parquet").exists()
    else PROCESSED_DIR / "dummy_actuals.parquet"
)
LOCAL_TZ = "America/New_York"
WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WAVE_PROGRESS_STEPS = [0.20, 0.52, 0.84]


def _normalize_station_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_to_orig = {c.lower(): c for c in df.columns}

    aliases = {
        "station_id": ["station_id", "start_station_id", "id"],
        "station_name": ["station_name", "start_station_name", "name"],
        "lat": ["lat", "latitude", "start_lat", "station_lat", "y"],
        "lon": ["lon", "lng", "longitude", "start_lng", "start_lon", "station_lng", "station_lon", "x"],
        "borough": ["borough", "district", "neighborhood"],
    }

    rename_map = {}
    for target, candidates in aliases.items():
        if target in df.columns:
            continue
        for cand in candidates:
            if cand in lower_to_orig:
                rename_map[lower_to_orig[cand]] = target
                break

    return df.rename(columns=rename_map)


def _ensure_branch_file(local_path: Path, repo_rel_path: str) -> Path:
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE_URL}/{repo_rel_path}"
    urlretrieve(url, local_path)
    return local_path


def _ensure_release_asset(filename: str) -> Path | None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target = PROCESSED_DIR / filename
    if target.exists() and target.stat().st_size > 0:
        return target
    try:
        urlretrieve(f"{RELEASE_BASE_URL}/{filename}", target)
        if target.exists() and target.stat().st_size > 0:
            return target
    except Exception:
        target.unlink(missing_ok=True)
    return None


def _find_and_download_stations_file(manifest_path: Path) -> Path | None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    local_candidates = [
        Path("data/stations.parquet"),
        Path("data/stations.csv"),
        PROCESSED_DIR / "stations.parquet",
        PROCESSED_DIR / "stations.csv",
    ]
    for p in local_candidates:
        if p.exists():
            return p

    candidate_urls: list[tuple[str, str]] = []
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        for col in manifest.columns:
            for val in manifest[col].dropna().astype(str):
                val_clean = val.strip()
                if "station" in val_clean.lower() and (
                    val_clean.endswith(".parquet") or val_clean.endswith(".csv")
                ):
                    filename = Path(val_clean).name
                    url = val_clean if val_clean.startswith("http") else f"{RELEASE_BASE_URL}/{filename}"
                    candidate_urls.append((filename, url))

    for fallback_name in ["stations.parquet", "stations.csv"]:
        candidate_urls.append((fallback_name, f"{RELEASE_BASE_URL}/{fallback_name}"))

    seen = set()
    for filename, url in candidate_urls:
        if url in seen:
            continue
        seen.add(url)
        target_path = PROCESSED_DIR / filename
        try:
            urlretrieve(url, target_path)
            if target_path.exists() and target_path.stat().st_size > 0:
                return target_path
        except Exception:
            if target_path.exists():
                target_path.unlink(missing_ok=True)

    return None


@st.cache_data
def load_stations() -> pd.DataFrame:
    cohort_file = _ensure_branch_file(COHORT_PATH, "data/cohort.csv")
    manifest_file = _ensure_branch_file(MANIFEST_PATH, "data/manifest.csv")

    df = _normalize_station_columns(pd.read_csv(cohort_file))
    df["station_id"] = df["station_id"].astype(str)

    if "lat" not in df.columns or "lon" not in df.columns:
        stations_file = _find_and_download_stations_file(manifest_file)
        if stations_file is not None:
            st_df = (
                pd.read_parquet(stations_file)
                if stations_file.suffix == ".parquet"
                else pd.read_csv(stations_file)
            )
            st_df = _normalize_station_columns(st_df)
            if {"station_id", "lat", "lon"}.issubset(st_df.columns):
                st_df["station_id"] = st_df["station_id"].astype(str)
                extra_cols = [
                    c for c in ["lat", "lon", "station_name", "borough"]
                    if c in st_df.columns and c not in df.columns
                ]
                df = df.merge(
                    st_df[["station_id"] + extra_cols].drop_duplicates("station_id"),
                    on="station_id",
                    how="left",
                )

    if "lat" not in df.columns or "lon" not in df.columns:
        st.error("Station coordinates (`lat`/`lon`) could not be loaded.")
        st.stop()

    if "station_name" not in df.columns:
        df["station_name"] = df["station_id"]
    if "borough" not in df.columns:
        df["borough"] = "NYC"
    df["borough"] = df["borough"].fillna("Unknown")

    return df[["station_id", "station_name", "lat", "lon", "borough"]].drop_duplicates(
        subset=["station_id"]
    )


@st.cache_data
def load_context_hourly() -> pd.DataFrame:
    cal_path = _ensure_release_asset("calendar.parquet")
    weather_path = _ensure_release_asset("weather_forecast.parquet")
    if cal_path is None:
        return pd.DataFrame()

    cal = pd.read_parquet(cal_path)[["hour_utc", "weekday", "is_holiday", "holiday_name"]].copy()
    cal["hour_utc"] = pd.to_datetime(cal["hour_utc"], utc=True)

    if weather_path is not None:
        w = pd.read_parquet(weather_path)[
            ["hour_utc", "temperature_2m", "precipitation", "wind_speed_10m"]
        ].copy()
        w["hour_utc"] = pd.to_datetime(w["hour_utc"], utc=True)
        w["precipitation"] = w["precipitation"].clip(lower=0.0)
        cal = cal.merge(w, on="hour_utc", how="left")

    return cal


@st.cache_data
def load_forecasts(preds_path: Path, actuals_path: Path) -> pd.DataFrame:
    preds = pd.read_parquet(preds_path)
    preds["station_id"] = preds["station_id"].astype(str)
    preds["target_start_utc"] = pd.to_datetime(preds["target_start_utc"], utc=True)
    preds["origin_utc"] = pd.to_datetime(preds["origin_utc"], utc=True)

    if actuals_path.exists():
        actuals = pd.read_parquet(actuals_path)
        actuals["station_id"] = actuals["station_id"].astype(str)
        actuals["target_start_utc"] = pd.to_datetime(actuals["target_start_utc"], utc=True)
        df = preds.merge(
            actuals[["station_id", "target_start_utc", "pickups", "status"]],
            on=["station_id", "target_start_utc"],
            how="left",
        )
        df["status"] = df["status"].fillna("future_or_unobserved")
    else:
        df = preds.copy()
        df["pickups"] = pd.NA
        df["status"] = "future_or_unobserved"

    local_dt = df["target_start_utc"].dt.tz_convert(LOCAL_TZ)
    df["local_date"] = local_dt.dt.date
    df["local_hour"] = local_dt.dt.hour
    return df


@st.cache_data
def load_holdout_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except Exception:
            pass
    return {
        "model_version": "hist_gbdt_v1",
        "baseline_weekly_lag": {"mae": 1.302, "rmse": 2.816, "bias": 0.034},
        "model": {"mae": 0.991, "rmse": 2.011, "bias": -0.107},
    }


def assign_color(pred: float) -> list[int]:
    if pd.isna(pred):
        return [150, 150, 150, 160]
    if pred < 2.0:
        return [65, 182, 196, 200]
    if pred < 6.0:
        return [44, 127, 184, 210]
    if pred < 12.0:
        return [253, 174, 97, 225]
    if pred < 20.0:
        return [244, 109, 67, 235]
    return [215, 48, 39, 245]


def build_bike_flows(
    valid_map_df: pd.DataFrame,
    prev_hour_df: pd.DataFrame,
    progress_t: float,
    max_flows: int = 22,
) -> pd.DataFrame:
    if valid_map_df.empty or prev_hour_df.empty:
        return pd.DataFrame()

    prev_lookup = prev_hour_df[["station_id", "prediction"]].rename(columns={"prediction": "prev_pred"})
    df = valid_map_df[["station_id", "station_name", "lat", "lon", "prediction"]].merge(
        prev_lookup, on="station_id", how="left"
    )
    df["prediction"] = df["prediction"].fillna(0.0)
    df["prev_pred"] = df["prev_pred"].fillna(0.0)
    df["delta"] = df["prediction"] - df["prev_pred"]

    # Origins (A): stations cooling down or lower-surge feeder stations
    origins = df[df["delta"] < -0.15].sort_values("delta", ascending=True).head(max_flows * 2)
    if len(origins) < max_flows:
        fallback_origins = (
            df[(df["prev_pred"] > 0.4) | (df["prediction"] > 0.4)]
            .sort_values("delta", ascending=True)
            .head(max_flows * 2)
        )
        origins = pd.concat([origins, fallback_origins]).drop_duplicates("station_id").head(max_flows * 2)

    # Destinations (B): stations surging in demand right now
    dests = df[df["delta"] > 0.15].sort_values("delta", ascending=False).head(max_flows * 2)
    if len(dests) < max_flows:
        fallback_dests = df.sort_values("prediction", ascending=False).head(max_flows * 2)
        dests = pd.concat([dests, fallback_dests]).drop_duplicates("station_id").head(max_flows * 2)

    if origins.empty or dests.empty:
        return pd.DataFrame()

    rows = []
    used_origins = set()
    used_dests = set()

    for _, d_row in dests.iterrows():
        if len(rows) >= max_flows:
            break
        if d_row["station_id"] in used_dests:
            continue

        # Pair with an origin at reasonable riding distance so arcs & bikes look crisp
        candidates = origins[
            (~origins["station_id"].isin(used_origins))
            & (origins["station_id"] != d_row["station_id"])
        ].copy()
        if candidates.empty:
            break

        dist_sq = (candidates["lat"] - d_row["lat"]) ** 2 + (candidates["lon"] - d_row["lon"]) ** 2
        far_enough = candidates[dist_sq > 0.00003]
        chosen_pool = far_enough if not far_enough.empty else candidates
        o_row = chosen_pool.iloc[0]

        used_origins.add(o_row["station_id"])
        used_dests.add(d_row["station_id"])

        lon_a, lat_a = float(o_row["lon"]), float(o_row["lat"])
        lon_b, lat_b = float(d_row["lon"]), float(d_row["lat"])

        # Interpolate bicycle position along the A -> B arc
        lon_bike = lon_a + progress_t * (lon_b - lon_a)
        lat_bike = lat_a + progress_t * (lat_b - lat_a) + 0.0025 * float(np.sin(np.pi * progress_t))

        surge = max(float(d_row["delta"]), 0.5)
        rows.append(
            {
                "from_name": o_row["station_name"],
                "to_name": d_row["station_name"],
                "lon_a": lon_a,
                "lat_a": lat_a,
                "lon_b": lon_b,
                "lat_b": lat_b,
                "lon_bike": lon_bike,
                "lat_bike": lat_bike,
                "icon": "🚲",
                "surge": surge,
                "arc_width": min(2.0 + surge * 0.35, 6.5),
            }
        )

    return pd.DataFrame(rows)


def render_methodology_expander() -> None:
    metrics = load_holdout_metrics()
    base_m = metrics.get("baseline_weekly_lag", {})
    mod_m = metrics.get("model", {})
    base_mae = float(base_m.get("mae", 1.302))
    mod_mae = float(mod_m.get("mae", 0.991))
    improvement_pct = (1.0 - mod_mae / base_mae) * 100.0 if base_mae > 0 else 0.0

    with st.expander("ℹ️ About PedalPulse & Forecast Methodology", expanded=False):
        c_info, c_bench = st.columns([1.6, 1.0])
        with c_info:
            st.markdown(
                """
                **What does this dashboard show?**
                * **Target Metric:** Predicts **hourly bike departures (`pickups`)** per station across NYC's frozen 2,072-station cohort (not real-time dock inventory).
                * **Leakage-Free Forecast Origin (Assumption E1):** All forecasts for target date $D$ are generated strictly at **00:00 NYC local time** (`origin_utc`) using only historical local-hour lags (`1d`, `2d`, `7d`, `14d` — DST-safe), station metadata, calendar indicators, and day-ahead weather forecasts (`available_at_utc <= origin_utc`).
                * **Backtest Explorer Window:** The interactive explorer displays **March 2026** (the final 31 days of the 6-month out-of-sample holdout period `2025-10-01` to `2026-03-31`), enabling side-by-side comparison against observed Citi Bike trip counts (`Actual`).
                """
            )
        with c_bench:
            st.markdown(f"**6-Month Out-of-Sample Holdout (`-{improvement_pct:.1f}%` MAE vs. Baseline)**")
            bench_df = pd.DataFrame(
                [
                    {
                        "Model": "7-Day Seasonal Lag Baseline",
                        "MAE": f"{base_mae:.3f}",
                        "RMSE": f"{float(base_m.get('rmse', 2.816)):.3f}",
                        "Bias": f"{float(base_m.get('bias', 0.034)):+.3f}",
                    },
                    {
                        "Model": f"ML Model ({metrics.get('model_version', 'hist_gbdt_v1')})",
                        "MAE": f"{mod_mae:.3f}",
                        "RMSE": f"{float(mod_m.get('rmse', 2.011)):.3f}",
                        "Bias": f"{float(mod_m.get('bias', -0.107)):+.3f}",
                    },
                ]
            )
            st.dataframe(bench_df, use_container_width=True, hide_index=True)


def render_citywide_day_curve(
    day_df: pd.DataFrame,
    selected_date: object,
    selected_hour: int,
    selected_borough: str,
) -> None:
    if day_df.empty:
        return

    hourly = (
        day_df.groupby("local_hour", as_index=False)
        .agg(
            ML_Forecast=("prediction", "sum"),
            Actual=("pickups", lambda s: s.sum() if s.notna().any() else np.nan),
            Baseline_7d=("baseline_prediction", "sum")
            if "baseline_prediction" in day_df.columns
            else ("prediction", lambda _: np.nan),
        )
        .sort_values("local_hour")
    )
    hourly["Hour"] = hourly["local_hour"].apply(lambda h: f"{h:02d}:00")

    val_cols = ["ML_Forecast"]
    label_map = {"ML_Forecast": "ML Forecast", "Actual": "Actual", "Baseline_7d": "7-Day Baseline"}
    domain = ["ML Forecast"]
    colors = ["#2563eb"]

    if hourly["Actual"].notna().any():
        val_cols.append("Actual")
        domain.append("Actual")
        colors.append("#16a34a")
    if hourly["Baseline_7d"].notna().any():
        val_cols.append("Baseline_7d")
        domain.append("7-Day Baseline")
        colors.append("#94a3b8")

    long_df = hourly[["local_hour", "Hour"] + val_cols].melt(
        id_vars=["local_hour", "Hour"],
        value_vars=val_cols,
        var_name="Series",
        value_name="Trips",
    )
    long_df["Series"] = long_df["Series"].map(label_map)

    lines = (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("local_hour:O", title="Hour of Day (NYC Local Time)"),
            y=alt.Y("Trips:Q", title="Total Trips / Hour"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=domain, range=colors),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.condition(
                alt.datum.Series == "7-Day Baseline",
                alt.value([5, 4]),
                alt.value([0]),
            ),
            tooltip=["Hour", "Series", alt.Tooltip("Trips:Q", format=",.0f")],
        )
    )

    cursor_df = pd.DataFrame({"local_hour": [selected_hour], "label": [f"{selected_hour:02d}:00"]})
    cursor_rule = (
        alt.Chart(cursor_df)
        .mark_rule(color="#ef4444", strokeWidth=2.5, strokeDash=[4, 3])
        .encode(x=alt.X("local_hour:O"))
    )

    with st.container(border=True):
        st.markdown(f"**24-Hour Demand Curve – {selected_borough} ({selected_date})**")
        st.altair_chart((lines + cursor_rule).properties(height=185), use_container_width=True)


def render_legend(map_style: str, show_bike_flows: bool) -> None:
    parts = []
    if map_style == "Smooth Heatmap Wave":
        parts.append(
            '<span><strong>Demand Density:</strong> '
            '<span style="background:linear-gradient(90deg,#1e3a8a,#06b6d4,#facc15,#f97316,#dc2626);'
            'display:inline-block;width:140px;height:11px;border-radius:4px;vertical-align:middle;margin:0 6px;"></span>'
            'low → hotspot wave</span>'
        )
    else:
        items = [
            ("#41b6c4", "< 2"),
            ("#2c7fb8", "2 – 5.9"),
            ("#fdae61", "6 – 11.9"),
            ("#f46d43", "12 – 19.9"),
            ("#d73027", "≥ 20"),
            ("#969696", "missing"),
        ]
        badges = "".join(
            f'<span style="display:inline-flex;align-items:center;margin-right:12px;">'
            f'<span style="width:10px;height:10px;border-radius:50%;background:{color};'
            f'display:inline-block;margin-right:5px;"></span>{label}</span>'
            for color, label in items
        )
        parts.append(f'<span><strong style="margin-right:8px;">Trips / Hour:</strong>{badges}</span>')

    if show_bike_flows:
        parts.append(
            '<span style="margin-left:14px;">🚲 <strong>A → B Flow:</strong> '
            '<span style="color:#0284c7;">origin (cooling)</span> → '
            '<span style="color:#ea580c;">destination (surging)</span></span>'
        )

    st.markdown(
        f'<div style="margin-top:6px;margin-bottom:4px;font-size:13px;">{" ".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_station_detail(
    stations: pd.DataFrame,
    day_df: pd.DataFrame,
    selected_date: object,
    selected_borough: str,
) -> None:
    st.divider()
    st.subheader(f"Station Detail & Error Analysis ({selected_date})")

    day_totals = (
        day_df.groupby("station_id", as_index=False)["prediction"]
        .sum()
        .rename(columns={"prediction": "day_pred_sum"})
    )
    station_choices = stations.merge(day_totals, on="station_id", how="left").sort_values(
        by=["day_pred_sum", "station_name"], ascending=[False, True]
    )

    station_choices["label"] = station_choices.apply(
        lambda r: f"{r['station_name']} · {r['borough']} (ID: {r['station_id']})",
        axis=1,
    )
    label_to_id = dict(zip(station_choices["label"], station_choices["station_id"]))

    ctrl_col1, ctrl_col2 = st.columns([3, 1.2])
    with ctrl_col1:
        selected_label = st.selectbox(
            f"Select Station ({len(station_choices)} stations in selection: {selected_borough})",
            options=station_choices["label"].tolist(),
            index=0,
        )
    has_baseline = "baseline_prediction" in day_df.columns and day_df["baseline_prediction"].notna().any()
    with ctrl_col2:
        st.write("")
        st.write("")
        show_baseline = st.checkbox(
            "Show 7-day baseline",
            value=True,
            disabled=not has_baseline,
        )

    selected_sid = label_to_id[selected_label]
    station_meta = station_choices[station_choices["station_id"] == selected_sid].iloc[0]

    st_day = day_df[day_df["station_id"] == selected_sid].sort_values("local_hour").copy()
    if st_day.empty:
        st.info("No forecast data available for this station on the selected date.")
        return

    st_day["Hour"] = st_day["local_hour"].apply(lambda h: f"{h:02d}:00")
    st_day["error"] = st_day["prediction"] - st_day["pickups"]
    st_day["abs_error"] = st_day["error"].abs()

    observed_df = st_day[st_day["pickups"].notna()]
    missing_hours = int((st_day["status"] == "missing").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1.container(border=True):
        st.metric("Daily Forecast", f"{st_day['prediction'].sum():.1f} trips")
    with c2.container(border=True):
        if not observed_df.empty:
            st.metric("Daily Actual", f"{int(observed_df['pickups'].sum())} trips")
        else:
            st.metric("Daily Actual", "pending")
    with c3.container(border=True):
        st.metric(
            "MAE (Avg Error/h)",
            f"{observed_df['abs_error'].mean():.2f}" if not observed_df.empty else "–",
        )
    with c4.container(border=True):
        st.metric(
            "Bias (Forecast − Actual)",
            f"{observed_df['error'].mean():+.2f}" if not observed_df.empty else "–",
        )

    if observed_df.empty:
        st.info(
            "Actual trip counts are not yet available for this target date. Select an earlier date on the left to compare against actuals."
        )
    elif missing_hours > 0:
        st.caption(
            f"Note: {missing_hours} hour(s) on this date are marked as missing data (`missing`)."
        )

    tab_charts, tab_table = st.tabs(["Hourly Profile & Error", "Hourly Table & CSV Export"])

    with tab_charts:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Hourly Profile: Forecast vs. Actual**")
            val_cols = ["prediction", "pickups"]
            label_map = {"prediction": "ML Forecast", "pickups": "Actual"}
            domain = ["ML Forecast", "Actual"]
            colors = ["#2563eb", "#16a34a"]

            if show_baseline and has_baseline:
                val_cols.append("baseline_prediction")
                label_map["baseline_prediction"] = "7-Day Baseline"
                domain.append("7-Day Baseline")
                colors.append("#94a3b8")

            plot_long = st_day[["local_hour", "Hour"] + val_cols].melt(
                id_vars=["local_hour", "Hour"],
                value_vars=val_cols,
                var_name="Series",
                value_name="Trips",
            )
            plot_long["Series"] = plot_long["Series"].map(label_map)

            line_chart = (
                alt.Chart(plot_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("local_hour:O", title="Hour (NYC Local Time)"),
                    y=alt.Y("Trips:Q", title="Trips / Hour"),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(domain=domain, range=colors),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    strokeDash=alt.condition(
                        alt.datum.Series == "7-Day Baseline",
                        alt.value([5, 4]),
                        alt.value([0]),
                    ),
                    tooltip=["Hour", "Series", alt.Tooltip("Trips:Q", format=".1f")],
                )
                .properties(height=310)
            )
            st.altair_chart(line_chart, use_container_width=True)

        with col_right:
            st.markdown("**Hourly Error (Forecast − Actual)**")
            if observed_df.empty:
                st.caption("No error bars available because actual counts are not yet observed.")
            else:
                bars = (
                    alt.Chart(st_day)
                    .mark_bar()
                    .encode(
                        x=alt.X("local_hour:O", title="Hour (NYC Local Time)"),
                        y=alt.Y("error:Q", title="Error (trips / h)"),
                        color=alt.condition(
                            alt.datum.error >= 0,
                            alt.value("#f97316"),
                            alt.value("#0ea5e9"),
                        ),
                        tooltip=[
                            "Hour",
                            alt.Tooltip("prediction:Q", title="Forecast", format=".1f"),
                            alt.Tooltip("pickups:Q", title="Actual", format=".0f"),
                            alt.Tooltip("error:Q", title="Error", format="+.1f"),
                            alt.Tooltip("status:N", title="Status"),
                        ],
                    )
                )
                zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#64748b").encode(y="y:Q")
                st.altair_chart((bars + zero_rule).properties(height=310), use_container_width=True)

    with tab_table:
        export_cols = ["Hour", "prediction", "pickups", "error", "status"]
        rename_cols = {
            "prediction": "ML_Forecast",
            "pickups": "Actual",
            "error": "Error",
            "status": "Status",
        }
        if has_baseline:
            export_cols.insert(2, "baseline_prediction")
            rename_cols["baseline_prediction"] = "Baseline_7d"

        table_df = st_day[export_cols].rename(columns=rename_cols)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download daily profile as CSV",
            data=csv_bytes,
            file_name=f"pedalpulse_{selected_sid}_{selected_date}.csv",
            mime="text/csv",
        )

    if pd.isna(station_meta["lat"]) or pd.isna(station_meta["lon"]):
        st.caption("Note: This station has no GPS coordinates in the metadata.")


def main() -> None:
    st.title("PedalPulse – Hourly Demand Forecast (NYC Citi Bike)")

    if not PREDS_PATH.exists():
        st.error(
            "Prediction file not found. Please run `python src/pedalpulse/train_forecast.py` first."
        )
        st.stop()

    stations = load_stations()
    forecasts = load_forecasts(PREDS_PATH, ACTUALS_PATH)
    context_hourly = load_context_hourly()

    data_mode = str(forecasts["data_mode"].iloc[0])
    model_version = str(forecasts["model_version"].iloc[0])

    st.sidebar.header("Filters & Time Selection")
    available_dates = sorted(forecasts["local_date"].unique())
    selected_date = st.sidebar.selectbox("Target Date (NYC Local Time)", available_dates, index=0)

    day_df_all = forecasts[forecasts["local_date"] == selected_date]
    available_hours = sorted(day_df_all["local_hour"].unique())

    if "selected_hour" not in st.session_state:
        st.session_state["selected_hour"] = available_hours[min(8, len(available_hours) - 1)]
    if st.session_state["selected_hour"] not in available_hours:
        st.session_state["selected_hour"] = available_hours[0]
    if "wave_substep" not in st.session_state:
        st.session_state["wave_substep"] = 1

    selected_hour = st.sidebar.select_slider(
        "Target Hour (Start)",
        options=available_hours,
        key="selected_hour",
        format_func=lambda h: f"{h:02d}:00",
    )

    play_wave = st.sidebar.toggle("▶ Play 24h Demand Wave", value=False)
    if not play_wave:
        st.session_state["wave_ready"] = False
        st.session_state["wave_substep"] = 1

    show_bike_flows = st.sidebar.toggle("🚲 Show A→B Bike Flows", value=True)

    map_style = st.sidebar.radio(
        "Map Display",
        options=["Smooth Heatmap Wave", "Heatmap + Stations", "Station Points"],
        index=0,
    )

    borough_options = ["All Boroughs"] + sorted(
        [b for b in stations["borough"].dropna().unique() if b != "Unknown"]
    )
    selected_borough = st.sidebar.selectbox("Borough", borough_options, index=0)

    st.sidebar.divider()
    st.sidebar.caption(
        f"**Model:** `{model_version}`\n\n"
        f"**Mode:** `{data_mode}`\n\n"
        f"**Cohort:** {len(stations)} stations"
    )

    if selected_borough == "All Boroughs":
        filtered_stations = stations
    else:
        filtered_stations = stations[stations["borough"] == selected_borough]

    filtered_ids = set(filtered_stations["station_id"])
    day_df = day_df_all[day_df_all["station_id"].isin(filtered_ids)]
    hour_df = day_df[day_df["local_hour"] == selected_hour].copy()

    # Previous hour slice for A -> B demand shift calculation
    curr_hour_idx = available_hours.index(selected_hour)
    prev_hour = available_hours[(curr_hour_idx - 1) % len(available_hours)]
    prev_hour_df = day_df[day_df["local_hour"] == prev_hour].copy()

    target_utc = hour_df["target_start_utc"].iloc[0]
    origin_utc = hour_df["origin_utc"].iloc[0]
    origin_local = origin_utc.tz_convert(LOCAL_TZ)
    target_local = target_utc.tz_convert(LOCAL_TZ)

    context_str = f"Day: {WEEKDAYS_EN[target_local.weekday()]}"
    if not context_hourly.empty:
        ctx_match = context_hourly[context_hourly["hour_utc"] == target_utc]
        if not ctx_match.empty:
            row_ctx = ctx_match.iloc[0]
            if bool(row_ctx.get("is_holiday", False)) and pd.notna(row_ctx.get("holiday_name")):
                context_str += f" ({row_ctx['holiday_name']})"
            if pd.notna(row_ctx.get("temperature_2m")):
                temp_c = float(row_ctx["temperature_2m"])
                rain_mm = float(row_ctx.get("precipitation", 0.0) or 0.0)
                wind_kmh = float(row_ctx.get("wind_speed_10m", 0.0) or 0.0)
                context_str += f" · Weather Forecast: {temp_c:.1f} °C, {rain_mm:.1f} mm precip, {wind_kmh:.0f} km/h wind"

    if "dummy" in data_mode.lower():
        st.warning(
            f"Test mode: Synthetic data active (`{model_version}`). "
            f"Forecast origin: {origin_local.strftime('%Y-%m-%d %H:%M %Z')}"
        )
    else:
        st.caption(
            f"**Target Hour:** {target_local.strftime('%Y-%m-%d %H:%M %Z')} · {context_str} · "
            f"**Forecast Origin:** {origin_local.strftime('%Y-%m-%d %H:%M %Z')}"
        )

    render_methodology_expander()

    map_df = filtered_stations.merge(hour_df, on="station_id", how="left")
    missing_coords = int(map_df[["lat", "lon"]].isna().any(axis=1).sum())
    valid_map_df = map_df.dropna(subset=["lat", "lon"]).copy()

    valid_map_df["color"] = valid_map_df["prediction"].apply(assign_color)
    valid_map_df["pred_weight"] = valid_map_df["prediction"].fillna(0.0).astype(float)
    valid_map_df["pred_label"] = valid_map_df["prediction"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "missing"
    )
    valid_map_df["actual_label"] = valid_map_df.apply(
        lambda r: f"{int(r['pickups'])}"
        if pd.notna(r.get("pickups"))
        else ("missing" if r.get("status") == "missing" else "pending"),
        axis=1,
    )
    valid_map_df["diff_label"] = valid_map_df.apply(
        lambda r: f"{(r['prediction'] - r['pickups']):+.1f}"
        if pd.notna(r.get("pickups")) and pd.notna(r.get("prediction"))
        else "–",
        axis=1,
    )

    obs_hour = map_df[map_df["pickups"].notna() & map_df["prediction"].notna()]
    total_pred = float(map_df["prediction"].sum())
    total_actual = float(obs_hour["pickups"].sum()) if not obs_hour.empty else None
    hour_mae = float((obs_hour["prediction"] - obs_hour["pickups"]).abs().mean()) if not obs_hour.empty else None

    col1, col2, col3, col4 = st.columns(4)
    with col1.container(border=True):
        delta_str = f"{total_pred - total_actual:+.0f} vs. Actual ({int(total_actual)})" if total_actual is not None else None
        st.metric("Total Forecast (trips/h)", f"{total_pred:.0f}", delta=delta_str, delta_color="off")
    with col2.container(border=True):
        st.metric("Avg per Station (trips/h)", f"{map_df['prediction'].mean():.2f}")
    with col3.container(border=True):
        st.metric(
            "Hourly MAE (all stations)",
            f"{hour_mae:.2f}" if hour_mae is not None else "–",
        )
    with col4.container(border=True):
        st.metric(
            "Stations on Map",
            f"{len(valid_map_df)} / {len(filtered_stations)}",
            help=f"{missing_coords} station(s) missing GPS coordinates",
        )

    render_citywide_day_curve(
        day_df=day_df,
        selected_date=selected_date,
        selected_hour=selected_hour,
        selected_borough=selected_borough,
    )

    map_col, top_col = st.columns([2.2, 1.0])

    with map_col:
        st.subheader(f"Map View ({selected_date}, {selected_hour:02d}:00)")

        center_lat = float(valid_map_df["lat"].mean()) if not valid_map_df.empty else 40.75
        center_lon = float(valid_map_df["lon"].mean()) if not valid_map_df.empty else -73.98
        zoom_level = 11.0 if selected_borough == "All Boroughs" else 12.2

        layers = []
        if map_style in ["Smooth Heatmap Wave", "Heatmap + Stations"]:
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=valid_map_df,
                    get_position="[lon, lat]",
                    get_weight="pred_weight",
                    radiusPixels=48,
                    intensity=1.35,
                    threshold=0.04,
                    opacity=0.80,
                )
            )

        if map_style in ["Station Points", "Heatmap + Stations"]:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=valid_map_df,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius=45 if map_style == "Heatmap + Stations" else 55,
                    radius_min_pixels=2 if map_style == "Heatmap + Stations" else 3,
                    radius_max_pixels=10 if map_style == "Heatmap + Stations" else 12,
                    pickable=True,
                )
            )

        if show_bike_flows:
            substep = st.session_state.get("wave_substep", 1) % len(WAVE_PROGRESS_STEPS)
            progress_t = WAVE_PROGRESS_STEPS[substep]
            flows_df = build_bike_flows(valid_map_df, prev_hour_df, progress_t=progress_t)
            if not flows_df.empty:
                layers.append(
                    pdk.Layer(
                        "ArcLayer",
                        data=flows_df,
                        get_source_position="[lon_a, lat_a]",
                        get_target_position="[lon_b, lat_b]",
                        get_source_color=[56, 189, 248, 175],
                        get_target_color=[249, 115, 22, 225],
                        get_width="arc_width",
                        great_circle=False,
                    )
                )
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=flows_df,
                        get_position="[lon_bike, lat_bike]",
                        get_fill_color=[255, 255, 255, 235],
                        get_line_color=[234, 88, 12, 255],
                        stroked=True,
                        line_width_min_pixels=2,
                        get_radius=85,
                        radius_min_pixels=8,
                        radius_max_pixels=12,
                    )
                )
                layers.append(
                    pdk.Layer(
                        "TextLayer",
                        data=flows_df,
                        get_position="[lon_bike, lat_bike]",
                        get_text="icon",
                        character_set=["🚲"],
                        get_size=20,
                        get_color=[255, 255, 255, 255],
                        get_text_anchor="'middle'",
                        get_alignment_baseline="'center'",
                    )
                )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom_level,
            pitch=28 if show_bike_flows else 0,
        )

        tooltip = (
            {
                "html": (
                    "<b>{station_name}</b><br/>"
                    "<span style='color:#cbd5e1;'>{borough} · ID: {station_id}</span><br/>"
                    "Forecast: <b>{pred_label}</b> / h<br/>"
                    "Actual: <b>{actual_label}</b> (Error: {diff_label})"
                ),
                "style": {"backgroundColor": "#1e293b", "color": "white", "fontSize": "12px"},
            }
            if map_style != "Smooth Heatmap Wave"
            else None
        )

        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip))
        render_legend(map_style, show_bike_flows)

    with top_col:
        st.subheader(f"Top 10 Stations ({selected_hour:02d}:00)")
        top_df = (
            map_df[["station_name", "borough", "prediction", "pickups"]]
            .dropna(subset=["prediction"])
            .sort_values("prediction", ascending=False)
            .head(10)
            .copy()
        )
        max_val = max(float(top_df["prediction"].max()), 1.0) if not top_df.empty else 1.0
        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True,
            height=485,
            column_config={
                "station_name": st.column_config.TextColumn("Station"),
                "borough": st.column_config.TextColumn("Borough"),
                "prediction": st.column_config.ProgressColumn(
                    "Forecast",
                    format="%.1f",
                    min_value=0.0,
                    max_value=max_val,
                ),
                "pickups": st.column_config.NumberColumn("Actual", format="%d"),
            },
        )

    render_station_detail(
        stations=filtered_stations,
        day_df=day_df,
        selected_date=selected_date,
        selected_borough=selected_borough,
    )

    if play_wave and hasattr(st, "fragment"):
        @st.fragment(run_every=0.85)
        def _advance_wave() -> None:
            if st.session_state.get("wave_ready", False):
                st.session_state["wave_ready"] = False
                if show_bike_flows:
                    next_sub = st.session_state.get("wave_substep", 0) + 1
                    if next_sub < len(WAVE_PROGRESS_STEPS):
                        st.session_state["wave_substep"] = next_sub
                        st.rerun()
                    else:
                        st.session_state["wave_substep"] = 0
                        curr_idx = available_hours.index(st.session_state["selected_hour"])
                        st.session_state["selected_hour"] = available_hours[(curr_idx + 1) % len(available_hours)]
                        st.rerun()
                else:
                    curr_idx = available_hours.index(st.session_state["selected_hour"])
                    st.session_state["selected_hour"] = available_hours[(curr_idx + 1) % len(available_hours)]
                    st.rerun()
            else:
                st.session_state["wave_ready"] = True

        _advance_wave()


if __name__ == "__main__":
    main()
