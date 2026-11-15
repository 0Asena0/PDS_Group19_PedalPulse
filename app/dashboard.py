from pathlib import Path
from urllib.request import urlretrieve
import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="PedalPulse – Citi Bike Prognose",
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
PREDS_PATH = PROCESSED_DIR / "dummy_predictions.parquet"
ACTUALS_PATH = PROCESSED_DIR / "dummy_actuals.parquet"
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
        manifest_preview = (
            pd.read_csv(manifest_file).to_dict(orient="records")
            if manifest_file.exists()
            else "manifest.csv nicht gefunden"
        )
        st.error(
            "Koordinaten (`lat`/`lon`) konnten noch nicht geladen werden. "
            f"Inhalt von `data/manifest.csv`: `{manifest_preview}`"
        )
        st.stop()

    if "station_name" not in df.columns:
        df["station_name"] = df["station_id"]
    if "borough" not in df.columns:
        df["borough"] = "NYC"

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
        return [253, 174, 97, 220]
    if pred < 20.0:
        return [244, 109, 67, 230]
    return [215, 48, 39, 240]


def render_station_detail(
    stations: pd.DataFrame,
    day_df: pd.DataFrame,
    selected_date: object,
) -> None:
    st.divider()
    st.subheader(f"Stationsdetail: Tagesverlauf & Fehleranalyse ({selected_date})")

    # Stationen nach Tagesprognose sortieren, damit stark frequentierte Stationen oben stehen
    day_totals = (
        day_df.groupby("station_id", as_index=False)["prediction"]
        .sum()
        .rename(columns={"prediction": "day_pred_sum"})
    )
    station_choices = stations.merge(day_totals, on="station_id", how="left").sort_values(
        by=["day_pred_sum", "station_name"], ascending=[False, True]
    )

    station_choices["label"] = station_choices.apply(
        lambda r: f"{r['station_name']} (ID: {r['station_id']} · {r['borough']})",
        axis=1,
    )
    label_to_id = dict(zip(station_choices["label"], station_choices["station_id"]))

    selected_label = st.selectbox(
        "Station auswählen (oder Namen eintippen zum Suchen)",
        options=station_choices["label"].tolist(),
        index=0,
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
    c1.metric("Tagesprognose (Ausleihen)", f"{st_day['prediction'].sum():.1f}")
    if not observed_df.empty:
        c2.metric("Tages-Istwert (beobachtet)", f"{int(observed_df['pickups'].sum())}")
        c3.metric("MAE (Ø absoluter Fehler/h)", f"{observed_df['abs_error'].mean():.2f}")
        c4.metric("Bias (Ø Prognose − Ist)", f"{observed_df['error'].mean():+.2f}")
    else:
        c2.metric("Tages-Istwert", "noch offen")
        c3.metric("MAE", "–")
        c4.metric("Bias", "–")

    if observed_df.empty:
        st.info(
            "Für diesen Zieltag liegen noch keine Ist-Werte vor (Zukunftsprognose). "
            "Wähle links einen früheren Tag (z. B. `2025-07-14` oder `2025-07-15`), um den Ist-Vergleich und die Fehlerkurve zu prüfen."
        )
    elif missing_hours > 0:
        st.caption(
            f"Hinweis: Für diese Station sind am gewählten Tag {missing_hours} Stunde(n) als Datenlücke (`missing`) markiert und werden in der Fehlerberechnung ausgelassen."
        )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Stündlicher Verlauf: Prognose vs. Ist-Wert (Ausleihen / Stunde)**")
        plot_long = st_day[["local_hour", "Stunde", "prediction", "pickups"]].melt(
            id_vars=["local_hour", "Stunde"],
            value_vars=["prediction", "pickups"],
            var_name="Reihe",
            value_name="Ausleihen",
        )
        plot_long["Reihe"] = plot_long["Reihe"].map(
            {"prediction": "Prognose", "pickups": "Ist-Wert"}
        )

        line_chart = (
            alt.Chart(plot_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("local_hour:O", title="Stunde (Ortszeit NYC)"),
                y=alt.Y("Ausleihen:Q", title="Ausleihen / Stunde"),
                color=alt.Color(
                    "Reihe:N",
                    scale=alt.Scale(
                        domain=["Prognose", "Ist-Wert"],
                        range=["#2563eb", "#16a34a"],
                    ),
                ),
                tooltip=["Stunde", "Reihe", alt.Tooltip("Ausleihen:Q", format=".1f")],
            )
            .properties(height=300)
        )
        st.altair_chart(line_chart, use_container_width=True)

    with col_right:
        st.markdown("**Stündliche Abweichung (Prognose − Ist-Wert)**")
        if observed_df.empty:
            st.caption("Keine Fehlerbalken verfügbar, da noch keine Ist-Werte vorliegen.")
        else:
            err_chart = (
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
                .properties(height=300)
            )
            st.altair_chart(err_chart, use_container_width=True)

    with st.expander("Stundenwerte als Tabelle anzeigen"):
        table_df = st_day[["Stunde", "prediction", "pickups", "error", "status"]].rename(
            columns={
                "prediction": "Prognose",
                "pickups": "Ist-Wert",
                "error": "Abweichung (Prognose - Ist)",
                "status": "Status",
            }
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    if pd.isna(station_meta["lat"]) or pd.isna(station_meta["lon"]):
        st.caption("Hinweis: Diese Station gehört zu den Stationen ohne GPS-Koordinaten auf der Karte.")


def main() -> None:
    st.title("PedalPulse – Stündliche Ausleihprognose (NYC Citi Bike)")

    if not PREDS_PATH.exists():
        st.error(
            "Prognosedatei fehlt. Bitte zuerst `python src/pedalpulse/dummy_forecast.py` ausführen."
        )
        st.stop()

    stations = load_stations()
    forecasts = load_forecasts(PREDS_PATH, ACTUALS_PATH)

    data_mode = str(forecasts["data_mode"].iloc[0])
    model_version = str(forecasts["model_version"].iloc[0])

    if "dummy" in data_mode.lower():
        st.warning(
            f"Hinweis: Aktuell werden synthetische Testdaten angezeigt (Modus: `{data_mode}`, Modell: `{model_version}`)."
        )
    else:
        st.info(f"Historischer Backtest-Modus (`{data_mode}`) | Modell: `{model_version}`")

    st.sidebar.header("Zeitauswahl (Ortszeit NYC)")
    available_dates = sorted(forecasts["local_date"].unique())
    selected_date = st.sidebar.selectbox("Zieltag", available_dates, index=0)

    day_df = forecasts[forecasts["local_date"] == selected_date]
    available_hours = sorted(day_df["local_hour"].unique())
    selected_hour = st.sidebar.select_slider(
        "Zielstunde (Beginn)",
        options=available_hours,
        value=available_hours[min(8, len(available_hours) - 1)],
        format_func=lambda h: f"{h:02d}:00 Uhr",
    )

    hour_df = day_df[day_df["local_hour"] == selected_hour].copy()

    origin_utc = hour_df["origin_utc"].iloc[0]
    origin_local = origin_utc.tz_convert(LOCAL_TZ)
    target_utc = hour_df["target_start_utc"].iloc[0]
    target_local = target_utc.tz_convert(LOCAL_TZ)

    st.caption(
        f"**Quelle:** Citi Bike System Data (Kohorte: {len(stations)} Stationen) | "
        f"**Prognoseursprung:** {origin_local.strftime('%Y-%m-%d %H:%M %Z')} ({origin_utc.strftime('%H:%M UTC')}) | "
        f"**Zielstunde:** {target_local.strftime('%Y-%m-%d %H:%M %Z')} | "
        f"**Modus:** {data_mode}"
    )

    map_df = stations.merge(hour_df, on="station_id", how="left")
    missing_preds = int(map_df["prediction"].isna().sum())
    missing_coords = int(map_df[["lat", "lon"]].isna().any(axis=1).sum())
    valid_map_df = map_df.dropna(subset=["lat", "lon"]).copy()

    valid_map_df["color"] = valid_map_df["prediction"].apply(assign_color)
    valid_map_df["pred_label"] = valid_map_df["prediction"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "fehlend"
    )
    valid_map_df["actual_label"] = valid_map_df.apply(
        lambda r: f"{int(r['pickups'])}"
        if pd.notna(r.get("pickups"))
        else ("Lücke (missing)" if r.get("status") == "missing" else "nicht verfügbar"),
        axis=1,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prognose gesamt (Ausleihen/h)", f"{valid_map_df['prediction'].sum():.0f}")
    col2.metric("Ø pro Station (Ausleihen/h)", f"{valid_map_df['prediction'].mean():.2f}")
    col3.metric("Fehlende Prognosen", f"{missing_preds}")
    col4.metric("Ohne Koordinaten", f"{missing_coords}")

    st.subheader(f"Prognostizierte Ausleihen pro Station ({selected_date}, {selected_hour:02d}:00 Uhr)")

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
        latitude=float(valid_map_df["lat"].mean()),
        longitude=float(valid_map_df["lon"].mean()),
        zoom=11.2,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<b>{station_name}</b> (ID: {station_id})<br/>"
            "Prognose: <b>{pred_label}</b> Ausleihen/h<br/>"
            "Ist-Wert: <b>{actual_label}</b>"
        ),
        "style": {"backgroundColor": "#1f2937", "color": "white", "fontSize": "12px"},
    }

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

    st.markdown(
        "**Legende (Einheit: geschätzte Ausleihen / Stunde, feste Skala):** "
        "`< 2` (hellblau) · `2–5.9` (blau) · `6–11.9` (orange) · `12–19.9` (rot-orange) · `≥ 20` (dunkelrot) · `fehlend` (grau)"
    )

    render_station_detail(stations=stations, day_df=day_df, selected_date=selected_date)


if __name__ == "__main__":
    main()
