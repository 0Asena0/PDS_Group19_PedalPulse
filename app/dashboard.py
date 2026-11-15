from pathlib import Path
from urllib.request import urlretrieve
import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="PedalPulse – NYC Citi Bike Prognose",
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
        st.error("Koordinaten (`lat`/`lon`) konnten nicht geladen werden.")
        st.stop()

    if "station_name" not in df.columns:
        df["station_name"] = df["station_id"]
    if "borough" not in df.columns:
        df["borough"] = "NYC"
    df["borough"] = df["borough"].fillna("Unbekannt")

    return df[["station_id", "station_name", "lat", "lon", "borough"]].drop_duplicates(
        subset=["station_id"]
    )


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


def render_legend() -> None:
    items = [
        ("#41b6c4", "< 2"),
        ("#2c7fb8", "2 – 5,9"),
        ("#fdae61", "6 – 11,9"),
        ("#f46d43", "12 – 19,9"),
        ("#d73027", "≥ 20"),
        ("#969696", "fehlend"),
    ]
    badges = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;font-size:13px;">'
        f'<span style="width:11px;height:11px;border-radius:50%;background:{color};'
        f'display:inline-block;margin-right:6px;"></span>{label}</span>'
        for color, label in items
    )
    st.markdown(
        f'<div style="margin-top:6px;margin-bottom:4px;">'
        f'<strong style="font-size:13px;margin-right:10px;">Ausleihen / Stunde:</strong>{badges}</div>',
        unsafe_allow_html=True,
    )


def render_station_detail(
    stations: pd.DataFrame,
    day_df: pd.DataFrame,
    selected_date: object,
    selected_borough: str,
) -> None:
    st.divider()
    st.subheader(f"Stationsdetail & Fehleranalyse ({selected_date})")

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
            f"Station auswählen ({len(station_choices)} Stationen in Auswahl: {selected_borough})",
            options=station_choices["label"].tolist(),
            index=0,
        )
    has_baseline = "baseline_prediction" in day_df.columns and day_df["baseline_prediction"].notna().any()
    with ctrl_col2:
        st.write("")
        st.write("")
        show_baseline = st.checkbox(
            "7-Tage-Baseline einblenden",
            value=True,
            disabled=not has_baseline,
        )

    selected_sid = label_to_id[selected_label]
    station_meta = station_choices[station_choices["station_id"] == selected_sid].iloc[0]

    st_day = day_df[day_df["station_id"] == selected_sid].sort_values("local_hour").copy()
    if st_day.empty:
        st.info("Für diese Station liegen am gewählten Tag keine Prognosedaten vor.")
        return

    st_day["Stunde"] = st_day["local_hour"].apply(lambda h: f"{h:02d}:00")
    st_day["error"] = st_day["prediction"] - st_day["pickups"]
    st_day["abs_error"] = st_day["error"].abs()

    observed_df = st_day[st_day["pickups"].notna()]
    missing_hours = int((st_day["status"] == "missing").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1.container(border=True):
        st.metric("Tagesprognose", f"{st_day['prediction'].sum():.1f} Fahrten")
    with c2.container(border=True):
        if not observed_df.empty:
            st.metric("Tages-Istwert", f"{int(observed_df['pickups'].sum())} Fahrten")
        else:
            st.metric("Tages-Istwert", "noch offen")
    with c3.container(border=True):
        st.metric(
            "MAE (Ø Fehler/h)",
            f"{observed_df['abs_error'].mean():.2f}" if not observed_df.empty else "–",
        )
    with c4.container(border=True):
        st.metric(
            "Bias (Prognose − Ist)",
            f"{observed_df['error'].mean():+.2f}" if not observed_df.empty else "–",
        )

    if observed_df.empty:
        st.info(
            "Für diesen Zieltag liegen noch keine Ist-Werte vor. Wähle links einen früheren Tag für den Ist-Vergleich."
        )
    elif missing_hours > 0:
        st.caption(
            f"Hinweis: {missing_hours} Stunde(n) sind an diesem Tag als Datenlücke (`missing`) markiert."
        )

    tab_charts, tab_table = st.tabs(["Verlauf & Abweichung", "Stundentabelle & CSV-Export"])

    with tab_charts:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Stündlicher Verlauf: Prognose vs. Ist-Wert**")
            val_cols = ["prediction", "pickups"]
            label_map = {"prediction": "ML-Prognose", "pickups": "Ist-Wert"}
            domain = ["ML-Prognose", "Ist-Wert"]
            colors = ["#2563eb", "#16a34a"]

            if show_baseline and has_baseline:
                val_cols.append("baseline_prediction")
                label_map["baseline_prediction"] = "7-Tage-Baseline"
                domain.append("7-Tage-Baseline")
                colors.append("#94a3b8")

            plot_long = st_day[["local_hour", "Stunde"] + val_cols].melt(
                id_vars=["local_hour", "Stunde"],
                value_vars=val_cols,
                var_name="Reihe",
                value_name="Ausleihen",
            )
            plot_long["Reihe"] = plot_long["Reihe"].map(label_map)

            line_chart = (
                alt.Chart(plot_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("local_hour:O", title="Stunde (Ortszeit NYC)"),
                    y=alt.Y("Ausleihen:Q", title="Ausleihen / Stunde"),
                    color=alt.Color(
                        "Reihe:N",
                        scale=alt.Scale(domain=domain, range=colors),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    strokeDash=alt.condition(
                        alt.datum.Reihe == "7-Tage-Baseline",
                        alt.value([5, 4]),
                        alt.value([0]),
                    ),
                    tooltip=["Stunde", "Reihe", alt.Tooltip("Ausleihen:Q", format=".1f")],
                )
                .properties(height=310)
            )
            st.altair_chart(line_chart, use_container_width=True)

        with col_right:
            st.markdown("**Stündliche Abweichung (Prognose − Ist-Wert)**")
            if observed_df.empty:
                st.caption("Keine Fehlerbalken verfügbar, da noch keine Ist-Werte vorliegen.")
            else:
                bars = (
                    alt.Chart(st_day)
                    .mark_bar()
                    .encode(
                        x=alt.X("local_hour:O", title="Stunde (Ortszeit NYC)"),
                        y=alt.Y("error:Q", title="Abweichung (Ausleihen / h)"),
                        color=alt.condition(
                            alt.datum.error >= 0,
                            alt.value("#f97316"),
                            alt.value("#0ea5e9"),
                        ),
                        tooltip=[
                            "Stunde",
                            alt.Tooltip("prediction:Q", title="Prognose", format=".1f"),
                            alt.Tooltip("pickups:Q", title="Ist-Wert", format=".0f"),
                            alt.Tooltip("error:Q", title="Fehler", format="+.1f"),
                            "status",
                        ],
                    )
                )
                zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#64748b").encode(y="y:Q")
                st.altair_chart((bars + zero_rule).properties(height=310), use_container_width=True)

    with tab_table:
        export_cols = ["Stunde", "prediction", "pickups", "error", "status"]
        rename_cols = {
            "prediction": "ML_Prognose",
            "pickups": "Ist_Wert",
            "error": "Abweichung",
            "status": "Status",
        }
        if has_baseline:
            export_cols.insert(2, "baseline_prediction")
            rename_cols["baseline_prediction"] = "Baseline_7d"

        table_df = st_day[export_cols].rename(columns=rename_cols)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Tagesverlauf als CSV herunterladen",
            data=csv_bytes,
            file_name=f"pedalpulse_{selected_sid}_{selected_date}.csv",
            mime="text/csv",
        )

    if pd.isna(station_meta["lat"]) or pd.isna(station_meta["lon"]):
        st.caption("Hinweis: Diese Station besitzt keine GPS-Koordinaten in den Metadaten.")


def main() -> None:
    st.title("PedalPulse – Stündliche Ausleihprognose (NYC Citi Bike)")

    if not PREDS_PATH.exists():
        st.error(
            "Prognosedatei fehlt. Bitte zuerst `python src/pedalpulse/train_forecast.py` ausführen."
        )
        st.stop()

    stations = load_stations()
    forecasts = load_forecasts(PREDS_PATH, ACTUALS_PATH)

    data_mode = str(forecasts["data_mode"].iloc[0])
    model_version = str(forecasts["model_version"].iloc[0])

    st.sidebar.header("Filter & Zeitauswahl")
    available_dates = sorted(forecasts["local_date"].unique())
    selected_date = st.sidebar.selectbox("Zieltag (Ortszeit NYC)", available_dates, index=0)

    day_df_all = forecasts[forecasts["local_date"] == selected_date]
    available_hours = sorted(day_df_all["local_hour"].unique())
    selected_hour = st.sidebar.select_slider(
        "Zielstunde (Beginn)",
        options=available_hours,
        value=available_hours[min(8, len(available_hours) - 1)],
        format_func=lambda h: f"{h:02d}:00 Uhr",
    )

    borough_options = ["Alle Stadtteile"] + sorted(
        [b for b in stations["borough"].dropna().unique() if b != "Unbekannt"]
    )
    selected_borough = st.sidebar.selectbox("Stadtteil (Borough)", borough_options, index=0)

    st.sidebar.divider()
    st.sidebar.caption(
        f"**Modell:** `{model_version}`\n\n"
        f"**Modus:** `{data_mode}`\n\n"
        f"**Kohorte:** {len(stations)} Stationen"
    )

    if selected_borough == "Alle Stadtteile":
        filtered_stations = stations
    else:
        filtered_stations = stations[stations["borough"] == selected_borough]

    filtered_ids = set(filtered_stations["station_id"])
    day_df = day_df_all[day_df_all["station_id"].isin(filtered_ids)]
    hour_df = day_df[day_df["local_hour"] == selected_hour].copy()

    origin_utc = hour_df["origin_utc"].iloc[0]
    origin_local = origin_utc.tz_convert(LOCAL_TZ)
    target_local = hour_df["target_start_utc"].iloc[0].tz_convert(LOCAL_TZ)

    if "dummy" in data_mode.lower():
        st.warning(
            f"Testmodus: Synthetische Daten aktiv (`{model_version}`). "
            f"Prognoseursprung: {origin_local.strftime('%Y-%m-%d %H:%M %Z')}"
        )
    else:
        st.caption(
            f"**Datenbasis:** Citi Bike System Data (`data-v1`) · "
            f"**Prognoseursprung:** {origin_local.strftime('%Y-%m-%d %H:%M %Z')} ({origin_utc.strftime('%H:%M UTC')}) · "
            f"**Zielstunde:** {target_local.strftime('%Y-%m-%d %H:%M %Z')} · "
            f"**Gebiet:** {selected_borough}"
        )

    map_df = filtered_stations.merge(hour_df, on="station_id", how="left")
    missing_coords = int(map_df[["lat", "lon"]].isna().any(axis=1).sum())
    valid_map_df = map_df.dropna(subset=["lat", "lon"]).copy()

    valid_map_df["color"] = valid_map_df["prediction"].apply(assign_color)
    valid_map_df["pred_label"] = valid_map_df["prediction"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "fehlend"
    )
    valid_map_df["actual_label"] = valid_map_df.apply(
        lambda r: f"{int(r['pickups'])}"
        if pd.notna(r.get("pickups"))
        else ("Lücke (missing)" if r.get("status") == "missing" else "offen"),
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
        delta_str = f"{total_pred - total_actual:+.0f} vs. Ist ({int(total_actual)})" if total_actual is not None else None
        st.metric("Prognose gesamt (Fahrten/h)", f"{total_pred:.0f}", delta=delta_str, delta_color="off")
    with col2.container(border=True):
        st.metric("Ø pro Station (Fahrten/h)", f"{map_df['prediction'].mean():.2f}")
    with col3.container(border=True):
        st.metric(
            "Stunden-MAE (alle Stationen)",
            f"{hour_mae:.2f}" if hour_mae is not None else "–",
        )
    with col4.container(border=True):
        st.metric(
            "Stationen auf Karte",
            f"{len(valid_map_df)} / {len(filtered_stations)}",
            help=f"{missing_coords} Station(en) ohne GPS-Koordinaten",
        )

    map_col, top_col = st.columns([2.2, 1.0])

    with map_col:
        st.subheader(f"Kartenansicht ({selected_date}, {selected_hour:02d}:00 Uhr)")

        center_lat = float(valid_map_df["lat"].mean()) if not valid_map_df.empty else 40.75
        center_lon = float(valid_map_df["lon"].mean()) if not valid_map_df.empty else -73.98
        zoom_level = 11.0 if selected_borough == "Alle Stadtteile" else 12.2

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=valid_map_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=55,
            radius_min_pixels=3,
            radius_max_pixels=12,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom_level,
            pitch=0,
        )

        tooltip = {
            "html": (
                "<b>{station_name}</b><br/>"
                "<span style='color:#cbd5e1;'>{borough} · ID: {station_id}</span><br/>"
                "Prognose: <b>{pred_label}</b> / h<br/>"
                "Ist-Wert: <b>{actual_label}</b> (Abw.: {diff_label})"
            ),
            "style": {"backgroundColor": "#1e293b", "color": "white", "fontSize": "12px"},
        }

        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
        render_legend()

    with top_col:
        st.subheader(f"Top 10 ({selected_hour:02d}:00 Uhr)")
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
                "borough": st.column_config.TextColumn("Bezirk"),
                "prediction": st.column_config.ProgressColumn(
                    "Prognose",
                    format="%.1f",
                    min_value=0.0,
                    max_value=max_val,
                ),
                "pickups": st.column_config.NumberColumn("Ist", format="%d"),
            },
        )

    render_station_detail(
        stations=filtered_stations,
        day_df=day_df,
        selected_date=selected_date,
        selected_borough=selected_borough,
    )


if __name__ == "__main__":
    main()
