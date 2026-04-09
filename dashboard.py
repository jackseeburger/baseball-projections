"""
⚾ Bayesian Baseball Projections Dashboard
Hierarchical Bayesian component models vs industry projection systems.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚾ Bayesian Baseball Projections",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).parent / "data" / "projections"
AGING_DIR = Path("/home/hermes/projects/baseball-assembly/data/projections")


@st.cache_data
def load_comparison():
    return pd.read_parquet(str(DATA_DIR / "comparison_2026.parquet"))


@st.cache_data
def load_our_model():
    return pd.read_parquet(str(DATA_DIR / "our_model_2026.parquet"))


@st.cache_data
def load_aging_curves():
    curves = {}
    for stat in ["k_rate", "bb_rate", "hr_rate", "iso", "babip"]:
        path = AGING_DIR / f"{stat}_aging_curve_2026.parquet"
        if path.exists():
            curves[stat] = pd.read_parquet(str(path))
    return curves


comparison = load_comparison()
our_model = load_our_model()
aging_curves = load_aging_curves()

# System display names
SYSTEMS = {
    "our": "🧠 Bayesian (Ours)",
    "stea": "📊 Steamer",
    "zips": "📈 ZiPS",
    "dept": "📋 Depth Charts",
}

STAT_LABELS = {
    "k_rate": "K%",
    "bb_rate": "BB%",
    "hr_rate": "HR/PA",
    "iso": "ISO",
    "babip": "BABIP",
    "avg": "AVG",
    "obp": "OBP",
    "slg": "SLG",
    "woba": "wOBA",
    "wrc_plus": "wRC+",
    "off": "Off (Runs)",
}

# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════
st.sidebar.title("⚾ Projection Explorer")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 Player Lookup", "📊 System Comparison", "📈 Aging Curves", "🏆 Leaderboards"],
)

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

# Team filter
teams = sorted(comparison["team"].dropna().unique())
selected_teams = st.sidebar.multiselect("Team", teams, default=[])

# Position filter (from stand for now)
stands = sorted(comparison["stand"].dropna().unique())
selected_stand = st.sidebar.multiselect("Bats", stands, default=[])

# Age range
age_range = st.sidebar.slider(
    "Age Range",
    int(comparison["age"].min()),
    int(comparison["age"].max()),
    (20, 40),
)

# Apply filters
filtered = comparison.copy()
if selected_teams:
    filtered = filtered[filtered["team"].isin(selected_teams)]
if selected_stand:
    filtered = filtered[filtered["stand"].isin(selected_stand)]
filtered = filtered[
    (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])
]


# ══════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════
def format_rate(val, stat):
    """Format a stat value for display."""
    if pd.isna(val):
        return "—"
    if stat in ("wrc_plus",):
        return f"{val:.0f}"
    if stat in ("off", "wraa"):
        return f"{val:.1f}"
    if stat in ("hr",):
        return f"{val:.0f}"
    return f"{val:.3f}"


def get_comparison_for_player(row, stat):
    """Get all system values for a stat for one player."""
    results = {}
    for prefix, label in SYSTEMS.items():
        col = f"{prefix}_{stat}"
        if col in row.index and pd.notna(row[col]):
            results[label] = row[col]
    return results


def correlation_matrix(df, stat):
    """Compute pairwise correlations between systems for a stat."""
    cols = {label: f"{prefix}_{stat}" for prefix, label in SYSTEMS.items()}
    valid_cols = {k: v for k, v in cols.items() if v in df.columns}
    data = df[list(valid_cols.values())].rename(columns={v: k for k, v in valid_cols.items()})
    return data.corr()


# ══════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("⚾ 2026 Bayesian Baseball Projections")
    st.markdown("""
    **Hierarchical Bayesian component models** built with PyMC — five independent models
    (K%, BB%, HR rate, ISO, BABIP) with HSGP aging curves, assembled into full offensive projections.
    
    Compare our model against **Steamer**, **ZiPS**, and **FanGraphs Depth Charts**.
    """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    n_matched = len(filtered)
    with col1:
        st.metric("Players Projected", len(our_model))
    with col2:
        st.metric("Matched vs FanGraphs", n_matched)
    with col3:
        # Correlation with Steamer wOBA
        common = filtered[["our_woba", "stea_woba"]].dropna()
        corr = common["our_woba"].corr(common["stea_woba"]) if len(common) > 10 else 0
        st.metric("wOBA Corr vs Steamer", f"{corr:.3f}")
    with col4:
        common = filtered[["our_woba", "zips_woba"]].dropna()
        corr = common["our_woba"].corr(common["zips_woba"]) if len(common) > 10 else 0
        st.metric("wOBA Corr vs ZiPS", f"{corr:.3f}")

    st.markdown("---")

    # System-level comparison table
    st.subheader("📊 System Averages (Qualified Hitters)")

    summary_data = []
    for stat, label in STAT_LABELS.items():
        row = {"Stat": label}
        for prefix, sys_label in SYSTEMS.items():
            col = f"{prefix}_{stat}"
            if col in filtered.columns:
                vals = filtered[col].dropna()
                if len(vals) > 0:
                    row[sys_label] = f"{vals.mean():.3f}" if stat not in ("wrc_plus", "off") else f"{vals.mean():.1f}"
        summary_data.append(row)

    st.dataframe(pd.DataFrame(summary_data).set_index("Stat"), use_container_width=True)

    # Scatter: Our wOBA vs Steamer wOBA
    st.subheader("🔬 Our wOBA vs Steamer wOBA")
    scatter_data = filtered[["name", "team", "our_woba", "stea_woba", "age"]].dropna()
    if len(scatter_data) > 0:
        fig = px.scatter(
            scatter_data,
            x="stea_woba",
            y="our_woba",
            hover_name="name",
            hover_data={"team": True, "age": True, "our_woba": ":.3f", "stea_woba": ":.3f"},
            color="age",
            color_continuous_scale="viridis",
            labels={"stea_woba": "Steamer wOBA", "our_woba": "Our Model wOBA", "age": "Age"},
        )
        # Add y=x line
        min_val = min(scatter_data["stea_woba"].min(), scatter_data["our_woba"].min()) - 0.01
        max_val = max(scatter_data["stea_woba"].max(), scatter_data["our_woba"].max()) + 0.01
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(dash="dash", color="gray", width=1),
            showlegend=False,
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Biggest disagreements
    st.subheader("🤔 Biggest Disagreements vs Steamer")
    disagree = filtered[["name", "team", "age", "our_woba", "stea_woba"]].dropna().copy()
    disagree["diff"] = disagree["our_woba"] - disagree["stea_woba"]
    disagree["abs_diff"] = disagree["diff"].abs()
    disagree = disagree.sort_values("abs_diff", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**We're more bullish:**")
        bullish = disagree[disagree["diff"] > 0].head(10)
        for _, r in bullish.iterrows():
            st.markdown(f"- **{r['name']}** ({r['team']}, {int(r['age'])}): Ours {r['our_woba']:.3f} vs Steamer {r['stea_woba']:.3f} (+{r['diff']:.3f})")

    with col2:
        st.markdown("**We're more bearish:**")
        bearish = disagree[disagree["diff"] < 0].head(10)
        for _, r in bearish.iterrows():
            st.markdown(f"- **{r['name']}** ({r['team']}, {int(r['age'])}): Ours {r['our_woba']:.3f} vs Steamer {r['stea_woba']:.3f} ({r['diff']:.3f})")


elif page == "🔍 Player Lookup":
    st.title("🔍 Player Projection Card")

    # Player search
    player_names = sorted(filtered["name"].dropna().unique())
    selected_player = st.selectbox("Search Player", player_names, index=0)

    if selected_player:
        player = filtered[filtered["name"] == selected_player].iloc[0]

        # Header
        st.markdown(f"## {player['name']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Team", player["team"])
        with col2:
            st.metric("Age", int(player["age"]))
        with col3:
            st.metric("Bats", player["stand"])

        st.markdown("---")

        # Component rates comparison
        st.subheader("📊 Component Rate Comparison")

        comp_stats = ["k_rate", "bb_rate", "hr_rate", "iso", "babip"]
        comp_data = []
        for stat in comp_stats:
            for prefix, label in SYSTEMS.items():
                col = f"{prefix}_{stat}"
                if col in player.index and pd.notna(player[col]):
                    comp_data.append({
                        "Stat": STAT_LABELS[stat],
                        "System": label,
                        "Value": player[col],
                    })

        if comp_data:
            comp_df = pd.DataFrame(comp_data)
            fig = px.bar(
                comp_df,
                x="Stat",
                y="Value",
                color="System",
                barmode="group",
                color_discrete_map={
                    "🧠 Bayesian (Ours)": "#1f77b4",
                    "📊 Steamer": "#ff7f0e",
                    "📈 ZiPS": "#2ca02c",
                    "📋 Depth Charts": "#d62728",
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Aggregate stats table
        st.subheader("📈 Aggregate Projections")

        agg_stats = ["avg", "obp", "slg", "woba", "wrc_plus", "off"]
        agg_rows = []
        for prefix, label in SYSTEMS.items():
            row = {"System": label}
            for stat in agg_stats:
                col = f"{prefix}_{stat}"
                if col in player.index and pd.notna(player[col]):
                    row[STAT_LABELS[stat]] = format_rate(player[col], stat)
                else:
                    row[STAT_LABELS[stat]] = "—"
            # HR count
            hr_col = f"{prefix}_hr"
            if hr_col in player.index and pd.notna(player[hr_col]):
                row["HR"] = f"{player[hr_col]:.0f}"
            else:
                row["HR"] = "—"
            agg_rows.append(row)

        st.dataframe(pd.DataFrame(agg_rows).set_index("System"), use_container_width=True)

        # Uncertainty visualization (our model only)
        st.subheader("🎯 Projection Uncertainty (Our Model)")
        uncert_data = []
        for stat in comp_stats:
            val = player[f"our_{stat}"]
            std_col = f"our_{stat}_std"
            if std_col in player.index and pd.notna(player[std_col]):
                std = player[std_col]
                uncert_data.append({
                    "Stat": STAT_LABELS[stat],
                    "Projected": val,
                    "Lower (1σ)": val - std,
                    "Upper (1σ)": val + std,
                    "Lower (2σ)": val - 2*std,
                    "Upper (2σ)": val + 2*std,
                })

        if uncert_data:
            udf = pd.DataFrame(uncert_data)
            fig = go.Figure()
            # 2-sigma band
            fig.add_trace(go.Bar(
                x=udf["Stat"], y=udf["Upper (2σ)"] - udf["Lower (2σ)"],
                base=udf["Lower (2σ)"],
                marker_color="rgba(31, 119, 180, 0.15)",
                name="95% interval",
            ))
            # 1-sigma band
            fig.add_trace(go.Bar(
                x=udf["Stat"], y=udf["Upper (1σ)"] - udf["Lower (1σ)"],
                base=udf["Lower (1σ)"],
                marker_color="rgba(31, 119, 180, 0.4)",
                name="68% interval",
            ))
            # Point estimate
            fig.add_trace(go.Scatter(
                x=udf["Stat"], y=udf["Projected"],
                mode="markers",
                marker=dict(size=12, color="#1f77b4", line=dict(width=2, color="white")),
                name="Projection",
            ))
            fig.update_layout(height=350, barmode="overlay", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)


elif page == "📊 System Comparison":
    st.title("📊 Model Comparison Deep Dive")

    stat_choice = st.selectbox(
        "Select Stat to Compare",
        list(STAT_LABELS.keys()),
        format_func=lambda x: STAT_LABELS[x],
    )

    # Correlation matrix
    st.subheader(f"Pairwise Correlations — {STAT_LABELS[stat_choice]}")
    corr_mat = correlation_matrix(filtered, stat_choice)
    fig = px.imshow(
        corr_mat,
        text_auto=".3f",
        color_continuous_scale="RdBu_r",
        zmin=0.5, zmax=1.0,
        aspect="auto",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Distribution comparison
    st.subheader(f"Distribution — {STAT_LABELS[stat_choice]}")
    dist_data = []
    for prefix, label in SYSTEMS.items():
        col = f"{prefix}_{stat_choice}"
        if col in filtered.columns:
            vals = filtered[col].dropna()
            for v in vals:
                dist_data.append({"System": label, "Value": v})

    if dist_data:
        dist_df = pd.DataFrame(dist_data)
        fig = px.histogram(
            dist_df, x="Value", color="System",
            barmode="overlay", opacity=0.6, nbins=40,
            color_discrete_map={
                "🧠 Bayesian (Ours)": "#1f77b4",
                "📊 Steamer": "#ff7f0e",
                "📈 ZiPS": "#2ca02c",
                "📋 Depth Charts": "#d62728",
            },
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: Our model vs each system
    st.subheader(f"Scatter — Our Model vs Others ({STAT_LABELS[stat_choice]})")
    other_systems = {k: v for k, v in SYSTEMS.items() if k != "our"}

    cols = st.columns(len(other_systems))
    for i, (prefix, label) in enumerate(other_systems.items()):
        with cols[i]:
            our_col = f"our_{stat_choice}"
            other_col = f"{prefix}_{stat_choice}"
            if other_col in filtered.columns:
                scat = filtered[["name", our_col, other_col]].dropna()
                if len(scat) > 0:
                    corr = scat[our_col].corr(scat[other_col])
                    mae = (scat[our_col] - scat[other_col]).abs().mean()
                    fig = px.scatter(
                        scat, x=other_col, y=our_col,
                        hover_name="name",
                        title=f"vs {label}<br>r={corr:.3f}, MAE={mae:.3f}",
                    )
                    mn = min(scat[our_col].min(), scat[other_col].min()) - 0.005
                    mx = max(scat[our_col].max(), scat[other_col].max()) + 0.005
                    fig.add_trace(go.Scatter(
                        x=[mn, mx], y=[mn, mx],
                        mode="lines", line=dict(dash="dash", color="gray"),
                        showlegend=False,
                    ))
                    fig.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    # Residual analysis
    st.subheader(f"Residual Analysis — Ours vs Steamer ({STAT_LABELS[stat_choice]})")
    our_col = f"our_{stat_choice}"
    stea_col = f"stea_{stat_choice}"
    if stea_col in filtered.columns:
        resid = filtered[["name", "team", "age", our_col, stea_col]].dropna().copy()
        resid["residual"] = resid[our_col] - resid[stea_col]
        
        fig = px.scatter(
            resid, x="age", y="residual",
            hover_name="name",
            hover_data={"team": True, our_col: ":.3f", stea_col: ":.3f"},
            color="residual",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            labels={"age": "Player Age", "residual": "Our Model − Steamer"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


elif page == "📈 Aging Curves":
    st.title("📈 HSGP Aging Curves")
    st.markdown("""
    Each component model learns its own aging curve using a **Hilbert Space Gaussian Process (HSGP)**
    with a Matérn-5/2 kernel. These capture non-linear, asymmetric aging — different stats peak at different ages.
    """)

    if not aging_curves:
        st.warning("No aging curve data found. Run the models to generate aging curves.")
    else:
        stat_names = {
            "k_rate": "K% (Strikeout Rate)",
            "bb_rate": "BB% (Walk Rate)",
            "hr_rate": "HR Rate (per PA)",
            "iso": "ISO (Isolated Power)",
            "babip": "BABIP",
        }

        # Combined aging curves
        st.subheader("All Components — Normalized Aging Effects")
        fig = go.Figure()
        colors = {"k_rate": "#e74c3c", "bb_rate": "#3498db", "hr_rate": "#e67e22",
                  "iso": "#9b59b6", "babip": "#2ecc71"}

        for stat, df in aging_curves.items():
            if "age" in df.columns and "aging_effect" in df.columns:
                curve = df.groupby("age")["aging_effect"].mean().reset_index()
                # Normalize to peak = 0
                curve["normalized"] = curve["aging_effect"] - curve["aging_effect"].max()
                fig.add_trace(go.Scatter(
                    x=curve["age"], y=curve["normalized"],
                    mode="lines+markers",
                    name=stat_names.get(stat, stat),
                    line=dict(color=colors.get(stat, "#333"), width=2),
                    marker=dict(size=4),
                ))

        fig.update_layout(
            height=500,
            xaxis_title="Age",
            yaxis_title="Aging Effect (relative to peak)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig.add_vline(x=27, line_dash="dot", line_color="gray", annotation_text="Age 27")
        st.plotly_chart(fig, use_container_width=True)

        # Individual curves with uncertainty
        st.subheader("Individual Aging Curves")
        selected_stat = st.selectbox(
            "Select Component",
            list(aging_curves.keys()),
            format_func=lambda x: stat_names.get(x, x),
        )

        if selected_stat in aging_curves:
            df = aging_curves[selected_stat]
            if "age" in df.columns:
                fig = go.Figure()

                if "aging_effect" in df.columns:
                    curve = df.groupby("age").agg(
                        mean=("aging_effect", "mean"),
                        std=("aging_effect", "std"),
                    ).reset_index()

                    if "std" in curve.columns and curve["std"].notna().any():
                        fig.add_trace(go.Scatter(
                            x=pd.concat([curve["age"], curve["age"][::-1]]),
                            y=pd.concat([curve["mean"] + 2*curve["std"],
                                        (curve["mean"] - 2*curve["std"])[::-1]]),
                            fill="toself",
                            fillcolor="rgba(31, 119, 180, 0.1)",
                            line=dict(width=0),
                            name="95% CI",
                        ))
                        fig.add_trace(go.Scatter(
                            x=pd.concat([curve["age"], curve["age"][::-1]]),
                            y=pd.concat([curve["mean"] + curve["std"],
                                        (curve["mean"] - curve["std"])[::-1]]),
                            fill="toself",
                            fillcolor="rgba(31, 119, 180, 0.2)",
                            line=dict(width=0),
                            name="68% CI",
                        ))

                    fig.add_trace(go.Scatter(
                        x=curve["age"], y=curve["mean"],
                        mode="lines+markers",
                        name="Mean Effect",
                        line=dict(color="#1f77b4", width=3),
                    ))

                    # Find peak
                    peak_age = curve.loc[curve["mean"].idxmax(), "age"]
                    fig.add_vline(x=peak_age, line_dash="dash", line_color="red",
                                annotation_text=f"Peak: {peak_age:.0f}")

                fig.update_layout(
                    height=450,
                    title=f"{stat_names[selected_stat]} — HSGP Aging Curve",
                    xaxis_title="Age",
                    yaxis_title="Aging Effect (logit/log scale)",
                )
                st.plotly_chart(fig, use_container_width=True)


elif page == "🏆 Leaderboards":
    st.title("🏆 2026 Projection Leaderboards")

    # Stat selector
    lb_stat = st.selectbox(
        "Rank By",
        ["woba", "wrc_plus", "off", "avg", "obp", "slg", "iso", "hr_rate", "k_rate", "bb_rate", "babip"],
        format_func=lambda x: STAT_LABELS.get(x, x),
    )

    # System selector
    lb_system = st.selectbox(
        "System",
        list(SYSTEMS.keys()),
        format_func=lambda x: SYSTEMS[x],
    )

    n_show = st.slider("Show Top N", 10, 100, 25)

    col_name = f"{lb_system}_{lb_stat}"
    ascending = lb_stat == "k_rate"  # Lower K% is better

    if col_name in filtered.columns:
        lb = filtered[["name", "team", "age", "stand"]].copy()

        # Add the selected stat for all systems
        for prefix, label in SYSTEMS.items():
            c = f"{prefix}_{lb_stat}"
            if c in filtered.columns:
                lb[label] = filtered[c].apply(lambda v: format_rate(v, lb_stat) if pd.notna(v) else "—")
                lb[f"_sort_{prefix}"] = filtered[c]

        sort_col = f"_sort_{lb_system}"
        if sort_col in lb.columns:
            lb = lb.dropna(subset=[sort_col])
            lb = lb.sort_values(sort_col, ascending=ascending).head(n_show)
            
            # Drop sort columns
            lb = lb[[c for c in lb.columns if not c.startswith("_sort_")]]
            lb = lb.rename(columns={"name": "Player", "team": "Team", "age": "Age", "stand": "Bats"})
            lb["Age"] = lb["Age"].astype(int)
            lb = lb.reset_index(drop=True)
            lb.index = lb.index + 1  # 1-indexed rank

            st.dataframe(lb, use_container_width=True, height=min(n_show * 35 + 38, 900))
    else:
        st.warning(f"No data for {SYSTEMS[lb_system]} — {STAT_LABELS[lb_stat]}")

    # Multi-system leaderboard comparison
    st.markdown("---")
    st.subheader("📊 Rank Comparison Across Systems")

    rank_stat = st.selectbox(
        "Compare Rankings For",
        ["woba", "wrc_plus", "off"],
        format_func=lambda x: STAT_LABELS.get(x, x),
        key="rank_stat",
    )

    rank_data = filtered[["name", "team"]].copy()
    for prefix, label in SYSTEMS.items():
        col = f"{prefix}_{rank_stat}"
        if col in filtered.columns:
            rank_data[f"{label} Rank"] = filtered[col].rank(ascending=False, method="min")
            rank_data[f"{label} Value"] = filtered[col]

    # Show players with biggest rank differences
    our_rank_col = f"{SYSTEMS['our']} Rank"
    stea_rank_col = f"{SYSTEMS['stea']} Rank"
    if our_rank_col in rank_data.columns and stea_rank_col in rank_data.columns:
        rank_data["Rank Δ"] = rank_data[stea_rank_col] - rank_data[our_rank_col]
        rank_data = rank_data.dropna(subset=[our_rank_col, stea_rank_col])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🚀 We Rank Higher (vs Steamer)**")
            higher = rank_data.sort_values("Rank Δ", ascending=False).head(10)
            for _, r in higher.iterrows():
                st.markdown(
                    f"- **{r['name']}**: Our #{int(r[our_rank_col])} vs Steamer #{int(r[stea_rank_col])} "
                    f"(+{int(r['Rank Δ'])} spots)"
                )
        with col2:
            st.markdown("**📉 We Rank Lower (vs Steamer)**")
            lower = rank_data.sort_values("Rank Δ", ascending=True).head(10)
            for _, r in lower.iterrows():
                st.markdown(
                    f"- **{r['name']}**: Our #{int(r[our_rank_col])} vs Steamer #{int(r[stea_rank_col])} "
                    f"({int(r['Rank Δ'])} spots)"
                )


# ══════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Architecture**
- 5 hierarchical Bayesian models (PyMC)
- HSGP aging curves (Matérn-5/2)
- Binomial aggregation
- Trained on Modal cloud compute
- Tracked with Weights & Biases

**Data Sources**
- Statcast (2015-2025)
- FanGraphs projections API
""")
