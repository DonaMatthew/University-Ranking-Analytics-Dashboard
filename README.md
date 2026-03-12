# University Ranking Analytics Dashboard

Interactive Streamlit dashboard for analyzing global university ranking data from `data/timesData.csv`.

## Project Structure

```text
ranking-dashboard/
├── data/
│   └── timesData.csv
├── analysis/
│   └── ranking_analysis.py
├── dashboard/
│   └── ranking_dashboard.pbix
└── README.md
```

## Features

- Ranking trends over time (university-level and country-level)
- Country comparisons by average rank and score
- Indicator breakdowns with radar and heatmap visualizations
- Custom weighted ranking model (user-adjustable methodology)
- Auto-generated insights (top improver, country leader, key indicator, model disagreement)
- Sidebar filters for year range and country

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run analysis/ranking_analysis.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).
y
