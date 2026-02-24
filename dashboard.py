"""
Global Oncology Clinical Trials Dashboard
Answers: Where are trials happening?

Run with:  streamlit run dashboard.py
"""

import re

import streamlit as st
import pandas as pd
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Oncology Clinical Trials",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: #f0f4f8;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 10px 16px;
    }
    [data-testid="stMetricLabel"] { font-size:13px; font-weight:600; color:#475569; }
    [data-testid="stMetricValue"] { font-size:26px; font-weight:700; color:#1e3a5f; }
    h1  { color:#1e3a5f !important; }
    h3  { color:#334155; border-bottom:2px solid #e2e8f0; padding-bottom:5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PHASE_ORDER = [
    "Early Phase I", "Phase I", "Phase I/II",
    "Phase II", "Phase II/III", "Phase III",
    "Phase IV", "N/A", "Unknown",
]

PHASE_COLORS = {
    "Early Phase I": "#fde68a", "Phase I": "#fbbf24",  "Phase I/II": "#f97316",
    "Phase II":      "#ef4444", "Phase II/III": "#b91c1c", "Phase III": "#7c3aed",
    "Phase IV":      "#1d4ed8", "N/A": "#94a3b8",      "Unknown": "#e2e8f0",
}

FUNDER_COLORS = {
    "Industry":         "#2563eb",
    "Government / NIH": "#059669",
    "Academic / Other": "#d97706",
    "Unknown":          "#94a3b8",
}

# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────
_ZIP_RE = re.compile(r"^\d{3,}[\s\-]?\d*$")


def parse_locations(raw):
    """
    Extract (site_entry, country) tuples from a Locations field.

    Handles ClinicalTrials.gov export formats:
      - "Facility, City, State, Country; Facility, City, Country"
      - "City, Country; Country"
      - plain country list
    Returns [] when value is missing.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    results = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(",")]
        clean = [p for p in parts if p and not _ZIP_RE.match(p)]
        if clean:
            results.append((entry, clean[-1]))
    return results


def parse_phase(raw):
    """Normalise the Phases field to a standard label."""
    if pd.isna(raw) or str(raw).strip() in ("", "nan"):
        return "Unknown"
    s = str(raw).upper().replace("_", " ").replace("|", "/")

    if re.search(r"EARLY", s):
        return "Early Phase I"
    if re.search(r"PHASE\s*[1I].*PHASE\s*[2I][I]", s):
        return "Phase I/II"
    if re.search(r"PHASE\s*[2I][I].*PHASE\s*[3I]", s):
        return "Phase II/III"
    if re.search(r"PHASE\s*1|PHASE\s*I\b", s):
        return "Phase I"
    if re.search(r"PHASE\s*2|PHASE\s*II\b", s):
        return "Phase II"
    if re.search(r"PHASE\s*3|PHASE\s*III\b", s):
        return "Phase III"
    if re.search(r"PHASE\s*4|PHASE\s*IV\b", s):
        return "Phase IV"
    if re.search(r"^N\s*/?\s*A$|NOT APPLICABLE", s.strip()):
        return "N/A"
    return "Unknown"


def parse_funder(raw):
    """Map Funder Type to a broad category."""
    if pd.isna(raw):
        return "Unknown"
    s = str(raw).upper()
    if "INDUSTRY" in s:
        return "Industry"
    if s in {"NIH", "FED", "OTHER_GOV", "NETWORK"}:
        return "Government / NIH"
    if s == "OTHER":
        return "Academic / Other"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Data processing (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Processing data…")
def process(df):
    """
    Returns
    -------
    trial_df : one row per trial  (NCT Number deduplicated)
    site_df  : one row per (trial × location site)
    """
    df = df.copy()
    df["Phase_c"]    = df["Phases"].apply(parse_phase)
    df["Funder_c"]   = df["Funder Type"].apply(parse_funder)
    df["Enrollment"] = pd.to_numeric(df["Enrollment"], errors="coerce").fillna(0)

    rows = []
    for _, r in df.iterrows():
        locs = parse_locations(r.get("Locations", ""))
        if locs:
            for entry, country in locs:
                rows.append({
                    "NCT Number": r["NCT Number"],
                    "Phase":      r["Phase_c"],
                    "Funder":     r["Funder_c"],
                    "Enrollment": r["Enrollment"],
                    "Status":     r.get("Study Status", "Unknown"),
                    "Site_Entry": entry,
                    "Country":    country,
                })
        else:
            rows.append({
                "NCT Number": r["NCT Number"],
                "Phase":      r["Phase_c"],
                "Funder":     r["Funder_c"],
                "Enrollment": r["Enrollment"],
                "Status":     r.get("Study Status", "Unknown"),
                "Site_Entry": None,
                "Country":    "Unknown",
            })

    site_df = pd.DataFrame(rows)
    trial_df = site_df.drop_duplicates("NCT Number").copy()
    return trial_df, site_df


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — upload + filters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Oncology Trials")
    st.markdown("---")
    uploaded = st.file_uploader("Upload ClinicalTrials CSV", type="csv")
    st.markdown("---")
    st.markdown("### Filters")

if uploaded is None:
    st.title("Global Oncology Clinical Trials Dashboard")
    st.info(
        "Upload your **ClinicalTrials.gov CSV** in the sidebar to begin.\n\n"
        "Expected columns: `NCT Number`, `Study Status`, `Phases`, "
        "`Funder Type`, `Enrollment`, `Locations`"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Load & filter
# ─────────────────────────────────────────────────────────────────────────────
raw_df = pd.read_csv(uploaded)
trial_df, site_df = process(raw_df)

with st.sidebar:
    all_statuses = sorted(site_df["Status"].dropna().unique())
    sel_status   = st.multiselect("Study Status", all_statuses, default=all_statuses)

    avail_phases = [p for p in PHASE_ORDER if p in site_df["Phase"].unique()]
    sel_phases   = st.multiselect("Phase", avail_phases, default=avail_phases)

    all_funders = sorted(site_df["Funder"].dropna().unique())
    sel_funders = st.multiselect("Funder Type", all_funders, default=all_funders)

mask = (
    site_df["Status"].isin(sel_status) &
    site_df["Phase"].isin(sel_phases)  &
    site_df["Funder"].isin(sel_funders)
)
filt      = site_df[mask].copy()
filt_tri  = filt.drop_duplicates("NCT Number")
known     = filt[filt["Country"] != "Unknown"]

# ─────────────────────────────────────────────────────────────────────────────
# Aggregations
# ─────────────────────────────────────────────────────────────────────────────
trials_by_country = (
    known.groupby("Country")["NCT Number"].nunique()
    .reset_index(name="Trials").sort_values("Trials", ascending=False)
)

sites_by_country = (
    known[known["Site_Entry"].notna()].groupby("Country")["Site_Entry"].count()
    .reset_index(name="Sites").sort_values("Sites", ascending=False)
)

# Avoid double-counting enrollment across countries: sum per (NCT, Country) pair
enroll_by_country = (
    known.drop_duplicates(["NCT Number", "Country"])
    .groupby("Country")["Enrollment"].sum()
    .reset_index(name="Enrollment").sort_values("Enrollment", ascending=False)
)

phase_dist = (
    filt_tri.groupby("Phase").size().reset_index(name="Count")
)
phase_dist["Phase"] = pd.Categorical(phase_dist["Phase"], PHASE_ORDER, ordered=True)
phase_dist = phase_dist.sort_values("Phase")

funder_dist = filt_tri.groupby("Funder").size().reset_index(name="Count")

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
st.title("Global Oncology Clinical Trials Dashboard")
st.caption("Where are trials happening? — geographic distribution, phase breakdown, and enrollment capacity")
st.markdown("---")

# ── KPIs ─────────────────────────────────────────────────────────────────────
n_trials    = filt["NCT Number"].nunique()
n_countries = known["Country"].nunique()
n_sites     = int(known[known["Site_Entry"].notna()]["Site_Entry"].count())
n_enroll    = int(filt_tri["Enrollment"].sum())
pct_ind     = round(
    filt_tri[filt_tri["Funder"] == "Industry"].shape[0] / max(n_trials, 1) * 100, 1
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Trials",      f"{n_trials:,}")
k2.metric("Countries",         f"{n_countries:,}")
k3.metric("Trial Sites",       f"{n_sites:,}")
k4.metric("Total Enrollment",  f"{n_enroll:,}")
k5.metric("Industry Sponsored", f"{pct_ind}%")

st.markdown("---")

# ── World Maps ───────────────────────────────────────────────────────────────
st.markdown("### Geographic Distribution")

MAP_LAYOUT = dict(
    height=430,
    margin=dict(l=0, r=0, t=36, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    geo=dict(bgcolor="rgba(0,0,0,0)"),
)

t1, t2, t3 = st.tabs(["Trials per Country", "Sites per Country", "Enrollment Capacity"])

with t1:
    fig = px.choropleth(
        trials_by_country, locations="Country", locationmode="country names",
        color="Trials", color_continuous_scale="Blues",
        title="Oncology Trials by Country",
    )
    fig.update_layout(**MAP_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with t2:
    fig = px.choropleth(
        sites_by_country, locations="Country", locationmode="country names",
        color="Sites", color_continuous_scale="Greens",
        title="Trial Sites by Country",
    )
    fig.update_layout(**MAP_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with t3:
    fig = px.choropleth(
        enroll_by_country, locations="Country", locationmode="country names",
        color="Enrollment", color_continuous_scale="Oranges",
        title="Enrollment Capacity by Country",
    )
    fig.update_layout(**MAP_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Top Countries ─────────────────────────────────────────────────────────────
st.markdown("### Top Countries")
n_top = st.slider("Number of countries to display", 5, 30, 15)

BAR_LAYOUT = dict(coloraxis_showscale=False, height=490, showlegend=False)

c1, c2 = st.columns(2)

with c1:
    fig = px.bar(
        trials_by_country.head(n_top), x="Trials", y="Country", orientation="h",
        title=f"Top {n_top} — Trials", color="Trials", color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), **BAR_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.bar(
        sites_by_country.head(n_top), x="Sites", y="Country", orientation="h",
        title=f"Top {n_top} — Trial Sites", color="Sites", color_continuous_scale="Greens",
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), **BAR_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Phase & Funder ───────────────────────────────────────────────────────────
st.markdown("### Trial Characteristics")

c3, c4 = st.columns(2)

with c3:
    fig = px.pie(
        phase_dist, names="Phase", values="Count", hole=0.45,
        title="Phase Distribution",
        color="Phase", color_discrete_map=PHASE_COLORS,
        category_orders={"Phase": PHASE_ORDER},
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(height=430, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c4:
    fig = px.pie(
        funder_dist, names="Funder", values="Count", hole=0.45,
        title="Industry vs Non-Industry Sponsors",
        color="Funder", color_discrete_map=FUNDER_COLORS,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(height=430, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Enrollment + Phase-by-Country ─────────────────────────────────────────────
st.markdown("### Enrollment & Phase Breakdown by Country")

c5, c6 = st.columns(2)

with c5:
    fig = px.bar(
        enroll_by_country.head(n_top), x="Enrollment", y="Country", orientation="h",
        title=f"Top {n_top} — Enrollment Capacity",
        color="Enrollment", color_continuous_scale="Oranges",
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), **BAR_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with c6:
    top10 = trials_by_country.head(10)["Country"].tolist()
    phase_country = (
        known[known["Country"].isin(top10)]
        .drop_duplicates(["NCT Number", "Country"])
        .groupby(["Country", "Phase"]).size()
        .reset_index(name="Count")
    )
    phase_country["Phase"] = pd.Categorical(phase_country["Phase"], PHASE_ORDER, ordered=True)

    fig = px.bar(
        phase_country, x="Count", y="Country", orientation="h",
        color="Phase", barmode="stack",
        title="Phase Mix — Top 10 Countries",
        color_discrete_map=PHASE_COLORS,
        category_orders={"Phase": PHASE_ORDER},
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        height=490,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Summary Table ─────────────────────────────────────────────────────────────
st.markdown("### Country Summary Table")

summary = (
    trials_by_country
    .merge(sites_by_country,  on="Country", how="outer")
    .merge(enroll_by_country, on="Country", how="outer")
    .fillna(0)
    .astype({"Trials": int, "Sites": int, "Enrollment": int})
    .sort_values("Trials", ascending=False)
    .reset_index(drop=True)
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Country":    st.column_config.TextColumn("Country"),
        "Trials":     st.column_config.NumberColumn("Trials",     format="%d"),
        "Sites":      st.column_config.NumberColumn("Sites",      format="%d"),
        "Enrollment": st.column_config.NumberColumn("Enrollment", format="%d"),
    },
)

st.download_button(
    "Download Country Summary CSV",
    summary.to_csv(index=False),
    "oncology_trials_by_country.csv",
    "text/csv",
)

st.markdown("---")
st.caption("Data: ClinicalTrials.gov  |  Built with Streamlit & Plotly")
