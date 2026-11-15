from pathlib import Path
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="PedalPulse – Citi Bike Prognose",
    layout="wide",
)

COHORT_PATH = Path("data/cohort.csv")
PREDS_PATH = Path("data/processed/dummy_predictions.parquet")
ACTUALS_PATH = Path("data/processed/dummy_actuals.parquet")
LOCAL_TZ = "America/New_York"


@st.cache_data
def load_stations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {
        "start_station_id": "station_id",
        "start_station_name": "station_name",
        "latitude": "lat",
        "longitude": "lon",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["station_id"] = df["station_id"].astype(str)

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
    # Feste Farbschritte über alle Stunden hinweg (RGBA)
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


def main() -> None:
    st.title("PedalPulse – Stündliche Ausleihprognose (NYC Citi Bike)")

    if not COHORT_PATH.exists() or not PREDS_PATH.exists():
        st.error(
            "Eingabedateien fehlen. Bitte zuerst `python src/pedalpulse/dummy_forecast.py` ausführen."
        )
        st.stop()

    stations = load_stations(COHORT_PATH)
    forecasts = load_forecasts(PREDS_PATH, ACTUALS_PATH)

    data_mode = str(forecasts["data_mode"].iloc[0])
    model_version = str(forecasts["model_version"].iloc[0])

    if "dummy" in data_mode.lower():
        st.warning(
            f"Hinweis: Aktuell werden synthetische Testdaten angezeigt (Modus: `{data_mode}`, Modell: `{model_version}`)."
        )
    else:
        st.info(f"Historischer Backtest-Modus (`{data_mode}`) | Modell: `{model_version}`")

    # Sidebar für Tag- und Stundenauswahl
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

    # Metadaten zum gewählten Prognosezeitpunkt
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

    # Mit Stationskoordinaten verbinden (Left Join auf Kohorte)
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


if __name__ == "__main__":
    main()
