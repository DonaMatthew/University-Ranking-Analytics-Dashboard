from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "timesData.csv"
INDICATORS = ["teaching", "research", "citations", "international", "income"]
DEFAULT_WEIGHTS = {
    "teaching": 25,
    "research": 25,
    "citations": 30,
    "international": 10,
    "income": 10,
}


def parse_world_rank(value: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    if text.startswith("="):
        text = text[1:]
    if "-" in text:
        left, right = text.split("-", 1)
        if left.strip().isdigit() and right.strip().isdigit():
            return (int(left) + int(right)) / 2
        return np.nan
    return float(text) if text.isdigit() else np.nan


def parse_percent(value: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("%", "").strip()
    return float(text) if text else np.nan


def parse_students(value: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    return float(text) if text else np.nan


def parse_female_ratio(value: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if ":" not in text:
        return np.nan
    left, right = [part.strip() for part in text.split(":", 1)]
    if not left.isdigit() or not right.isdigit():
        return np.nan
    female = float(left)
    male = float(right)
    total = female + male
    return (female / total) * 100 if total else np.nan


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["rank_numeric"] = df["world_rank"].apply(parse_world_rank)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["num_students"] = df["num_students"].apply(parse_students)
    df["international_students_pct"] = df["international_students"].apply(parse_percent)
    df["female_pct"] = df["female_male_ratio"].apply(parse_female_ratio)

    numeric_cols = INDICATORS + ["total_score", "student_staff_ratio"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].replace("-", np.nan), errors="coerce")

    return df.dropna(subset=["year"]).copy()


def compute_custom_model(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    modeled = df.copy()
    for indicator in INDICATORS:
        modeled[f"{indicator}_norm"] = modeled.groupby("year")[indicator].transform(
            lambda s: (s - s.min()) / (s.max() - s.min()) if s.notna().any() and s.max() > s.min() else np.nan
        )

    modeled["custom_score"] = 0.0
    for indicator, weight in weights.items():
        modeled["custom_score"] = modeled["custom_score"] + modeled[f"{indicator}_norm"].fillna(0) * weight

    modeled["custom_score"] = modeled["custom_score"] * 100
    modeled["custom_rank"] = modeled.groupby("year")["custom_score"].rank(method="min", ascending=False)
    modeled["rank_gap"] = modeled["custom_rank"] - modeled["rank_numeric"]
    return modeled


def ranking_trends_view(filtered: pd.DataFrame) -> None:
    st.subheader("Ranking trends")

    top_unis = (
        filtered.groupby("university_name")["rank_numeric"]
        .mean()
        .dropna()
        .sort_values()
        .head(20)
        .index.tolist()
    )
    selected_unis = st.multiselect(
        "Universities to compare over time",
        options=sorted(filtered["university_name"].unique()),
        default=top_unis[:5],
    )

    if selected_unis:
        trend = (
            filtered[filtered["university_name"].isin(selected_unis)]
            .groupby(["year", "university_name"], as_index=False)["rank_numeric"]
            .mean()
        )
        fig = px.line(
            trend,
            x="year",
            y="rank_numeric",
            color="university_name",
            markers=True,
            title="University rank trend (lower is better)",
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one university to plot trends.")

    country_trend = (
        filtered.groupby(["year", "country"], as_index=False)
        .agg(avg_rank=("rank_numeric", "mean"), universities=("university_name", "count"))
    )
    top_countries = (
        country_trend.groupby("country")["universities"].sum().sort_values(ascending=False).head(10).index
    )
    fig_country = px.line(
        country_trend[country_trend["country"].isin(top_countries)],
        x="year",
        y="avg_rank",
        color="country",
        markers=True,
        title="Country average rank trend (top 10 by representation)",
    )
    fig_country.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_country, use_container_width=True)


def country_comparison_view(filtered: pd.DataFrame) -> None:
    st.subheader("Country comparisons")
    year_options = sorted(filtered["year"].unique())
    if not year_options:
        st.info("No years available for country comparison.")
        return
    selected_year = st.selectbox(
        "Year for country comparison",
        year_options,
        index=len(year_options) - 1,
    )

    by_country = (
        filtered[filtered["year"] == selected_year]
        .groupby("country", as_index=False)
        .agg(
            avg_rank=("rank_numeric", "mean"),
            avg_total_score=("total_score", "mean"),
            avg_research=("research", "mean"),
            avg_citations=("citations", "mean"),
            universities=("university_name", "count"),
        )
        .dropna(subset=["avg_rank"])
    )
    by_country = by_country.sort_values("avg_rank").head(20)

    fig_rank = px.bar(
        by_country,
        x="country",
        y="avg_rank",
        color="universities",
        title=f"Average country rank in {int(selected_year)} (top 20)",
    )
    fig_rank.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_rank, use_container_width=True)

    fig_score = px.scatter(
        by_country,
        x="avg_total_score",
        y="avg_rank",
        size="universities",
        color="avg_citations",
        hover_name="country",
        title=f"Rank vs total score in {int(selected_year)}",
        labels={"avg_total_score": "Average total score", "avg_rank": "Average rank"},
    )
    fig_score.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_score, use_container_width=True)


def indicator_breakdown_view(filtered: pd.DataFrame) -> None:
    st.subheader("Indicator breakdowns")
    years = sorted(filtered["year"].unique())
    if not years:
        st.info("No years available for indicator breakdown.")
        return
    selected_year = st.selectbox(
        "Year for indicator breakdown",
        years,
        index=len(years) - 1,
        key="indicator_year",
    )
    year_df = filtered[filtered["year"] == selected_year]

    universities = year_df["university_name"].dropna().sort_values().unique().tolist()
    default_uni = "Harvard University" if "Harvard University" in universities else universities[0]
    selected_university = st.selectbox("University", universities, index=universities.index(default_uni))

    uni_row = year_df[year_df["university_name"] == selected_university].iloc[0]
    radar_values = [uni_row.get(ind, np.nan) for ind in INDICATORS]

    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=radar_values,
            theta=[i.capitalize() for i in INDICATORS],
            fill="toself",
            name=selected_university,
        )
    )
    radar.update_layout(
        title=f"Indicator profile for {selected_university} ({int(selected_year)})",
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
    )
    st.plotly_chart(radar, use_container_width=True)

    country_indicator = (
        year_df.groupby("country", as_index=False)[INDICATORS]
        .mean()
        .set_index("country")
        .dropna(how="all")
    )
    top_countries = year_df["country"].value_counts().head(15).index
    heat_df = country_indicator.loc[country_indicator.index.intersection(top_countries)]
    heat_df = heat_df.sort_values("research", ascending=False)
    fig_heat = px.imshow(
        heat_df,
        labels={"x": "Indicator", "y": "Country", "color": "Average score"},
        aspect="auto",
        title=f"Indicator heatmap by country ({int(selected_year)})",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_heat, use_container_width=True)


def custom_model_and_insights_view(filtered: pd.DataFrame, weights: dict[str, float]) -> None:
    st.subheader("Custom ranking model")
    st.caption(
        "Method: per-year min-max normalization of indicators, then weighted aggregation into a 0-100 custom score."
    )

    modeled = compute_custom_model(filtered, weights)
    years = sorted(modeled["year"].dropna().unique().tolist())
    if not years:
        st.info("No years available for custom model analysis.")
        return

    selected_year = st.selectbox(
        "Year for custom model results",
        years,
        index=len(years) - 1,
        key="custom_model_year",
    )
    top_n = st.slider("Top N universities to display", min_value=5, max_value=30, value=10, step=5)

    year_df = modeled[modeled["year"] == selected_year].copy()
    year_df = year_df.dropna(subset=["custom_rank"]).sort_values("custom_rank")

    top_table = year_df[
        ["university_name", "country", "custom_score", "custom_rank", "rank_numeric", "rank_gap"]
    ].head(top_n)
    top_table = top_table.rename(
        columns={
            "rank_numeric": "official_rank",
            "rank_gap": "custom_minus_official",
        }
    )
    st.dataframe(top_table, use_container_width=True, hide_index=True)

    compare_df = year_df.dropna(subset=["rank_numeric"])
    fig_compare = px.scatter(
        compare_df,
        x="rank_numeric",
        y="custom_rank",
        hover_name="university_name",
        color="country",
        title=f"Official vs custom rank ({int(selected_year)})",
        labels={"rank_numeric": "Official rank", "custom_rank": "Custom rank"},
    )
    fig_compare.update_xaxes(autorange="reversed")
    fig_compare.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader("Generated insights")

    first_year = int(min(years))
    last_year = int(max(years))
    first_df = modeled[modeled["year"] == first_year][["university_name", "custom_rank"]].rename(
        columns={"custom_rank": "rank_first"}
    )
    last_df = modeled[modeled["year"] == last_year][["university_name", "custom_rank"]].rename(
        columns={"custom_rank": "rank_last"}
    )
    movers = first_df.merge(last_df, on="university_name", how="inner")
    movers["rank_change"] = movers["rank_first"] - movers["rank_last"]
    movers = movers.dropna(subset=["rank_change"]).sort_values("rank_change", ascending=False)

    country_latest = (
        modeled[modeled["year"] == last_year]
        .groupby("country", as_index=False)
        .agg(avg_custom_score=("custom_score", "mean"), universities=("university_name", "count"))
        .sort_values("avg_custom_score", ascending=False)
    )

    corr_latest = modeled[modeled["year"] == last_year][INDICATORS + ["rank_numeric"]].corr(method="spearman")
    corr_series = corr_latest["rank_numeric"].drop(labels=["rank_numeric"]).sort_values()

    divergence = (
        modeled[modeled["year"] == last_year]
        .dropna(subset=["rank_gap"])
        .assign(abs_gap=lambda d: d["rank_gap"].abs())
        .sort_values("abs_gap", ascending=False)
    )

    insight_lines: list[str] = []
    if not movers.empty:
        top_mover = movers.iloc[0]
        insight_lines.append(
            f"- Biggest custom-rank improver ({first_year} to {last_year}): "
            f"**{top_mover['university_name']}** ({top_mover['rank_change']:.0f} places)."
        )
    if not country_latest.empty:
        leader = country_latest.iloc[0]
        insight_lines.append(
            f"- Highest average custom score in {last_year}: **{leader['country']}** "
            f"({leader['avg_custom_score']:.1f}) across {int(leader['universities'])} universities."
        )
    if not corr_series.empty:
        strongest = corr_series.index[0]
        strength = corr_series.iloc[0]
        insight_lines.append(
            f"- Strongest inverse relationship with official rank in {last_year}: "
            f"**{strongest}** (Spearman {strength:.2f})."
        )
    if not divergence.empty:
        most_divergent = divergence.iloc[0]
        direction = "better" if most_divergent["rank_gap"] < 0 else "worse"
        insight_lines.append(
            f"- Largest model disagreement in {last_year}: **{most_divergent['university_name']}** is ranked "
            f"{abs(most_divergent['rank_gap']):.0f} places {direction} by the custom model vs official rank."
        )

    if insight_lines:
        st.markdown("\n".join(insight_lines))
    else:
        st.info("Not enough data to generate insights for the selected filters.")


def main() -> None:
    st.set_page_config(page_title="University Ranking Analytics Dashboard", layout="wide")
    st.title("Global University Ranking Analytics Dashboard")
    st.caption("Source: Times Higher Education dataset (`data/timesData.csv`)")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        return

    df = load_data(DATA_PATH)
    st.sidebar.header("Global Filters")

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    selected_years = st.sidebar.slider(
        "Year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )

    countries = sorted(df["country"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect("Countries", options=countries, default=countries)
    st.sidebar.subheader("Custom model weights")
    raw_weights = {}
    for indicator in INDICATORS:
        raw_weights[indicator] = st.sidebar.slider(
            f"{indicator.capitalize()} weight",
            min_value=0,
            max_value=100,
            value=DEFAULT_WEIGHTS[indicator],
            step=5,
        )
    total_weight = sum(raw_weights.values())
    if total_weight == 0:
        normalized_weights = {indicator: 1 / len(INDICATORS) for indicator in INDICATORS}
        st.sidebar.warning("All weights are zero. Using equal weights.")
    else:
        normalized_weights = {k: v / total_weight for k, v in raw_weights.items()}

    filtered = df[
        (df["year"].between(selected_years[0], selected_years[1]))
        & (df["country"].isin(selected_countries))
    ]
    st.sidebar.metric("Rows in view", len(filtered))

    c1, c2, c3 = st.columns(3)
    c1.metric("Universities", int(filtered["university_name"].nunique()))
    c2.metric("Countries", int(filtered["country"].nunique()))
    avg_score = (
        f"{filtered['total_score'].mean():.1f}"
        if filtered["total_score"].notna().any()
        else "N/A"
    )
    c3.metric("Avg total score", avg_score)

    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    ranking_trends_view(filtered)
    country_comparison_view(filtered)
    indicator_breakdown_view(filtered)
    custom_model_and_insights_view(filtered, normalized_weights)


if __name__ == "__main__":
    main()
