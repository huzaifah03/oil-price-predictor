import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crude Oil Price Predictor",
    page_icon="🛢️",
    layout="centered"
)

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    h1 { color: #f0a500 !important; }
    h2, h3 { color: #c9a84c !important; }
    .metric-card {
        background: #1c1f2e;
        border: 1px solid #2e3150;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-label { font-size: 13px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #f0a500; }
    .metric-delta { font-size: 13px; margin-top: 4px; }
    .info-box {
        background: #1a1d2e;
        border-left: 3px solid #f0a500;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 14px;
        color: #aaa;
        margin-bottom: 16px;
    }
    .warn-box {
        background: #1a1d2e;
        border-left: 3px solid #e05c5c;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 14px;
        color: #aaa;
        margin-bottom: 16px;
    }
    .neutral-box {
        background: #1a1d2e;
        border-left: 3px solid #555;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 14px;
        color: #aaa;
        margin-bottom: 16px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #c9a84c;
        margin-top: 24px;
        margin-bottom: 8px;
        border-bottom: 1px solid #2e3150;
        padding-bottom: 6px;
    }
    .stButton>button {
        background-color: #f0a500;
        color: #0f1117;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-size: 16px;
        width: 100%;
    }
    .stButton>button:hover { background-color: #d4920a; }
    .winner-card {
        background: linear-gradient(135deg, #1c2a1c, #1c2a18);
        border: 1px solid #3a6b3a;
        border-radius: 10px;
        padding: 16px 22px;
        margin-top: 16px;
    }
    .impact-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        margin-left: 8px;
    }
    .context-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 12px;
    }
    .context-item {
        background: #1c1f2e;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #aaa;
    }
    .context-item strong { color: #f0a500; }
</style>
""", unsafe_allow_html=True)

# ─── Region × Event Impact Matrix ──────────────────────────────────────────────
# Values are fractional price impact (0.10 = +10%, -0.08 = -8%)
# None means "not applicable" for that combination

IMPACT_MATRIX = {
    "Middle East": {
        "War / Armed Conflict":               0.25,
        "Sanctions on Producer":              0.18,
        "OPEC Production Cut":                0.12,
        "OPEC Production Increase":          -0.09,
        "Recession / Economic Slowdown":     -0.10,
        "Natural Disaster (supply disruption)": 0.15,
        "Energy Policy Shift":               -0.04,
    },
    "Eastern Europe / Russia": {
        "War / Armed Conflict":               0.15,
        "Sanctions on Producer":              0.20,
        "OPEC Production Cut":                None,
        "OPEC Production Increase":           None,
        "Recession / Economic Slowdown":     -0.08,
        "Natural Disaster (supply disruption)": 0.05,
        "Energy Policy Shift":               -0.06,
    },
    "West Africa": {
        "War / Armed Conflict":               0.08,
        "Sanctions on Producer":              0.05,
        "OPEC Production Cut":                0.10,
        "OPEC Production Increase":          -0.07,
        "Recession / Economic Slowdown":     -0.06,
        "Natural Disaster (supply disruption)": 0.08,
        "Energy Policy Shift":               -0.03,
    },
    "Latin America": {
        "War / Armed Conflict":               0.06,
        "Sanctions on Producer":              0.07,
        "OPEC Production Cut":                0.09,
        "OPEC Production Increase":          -0.07,
        "Recession / Economic Slowdown":     -0.07,
        "Natural Disaster (supply disruption)": 0.06,
        "Energy Policy Shift":               -0.04,
    },
    "South Asia": {
        "War / Armed Conflict":               0.01,
        "Sanctions on Producer":              0.02,
        "OPEC Production Cut":                None,
        "OPEC Production Increase":           None,
        "Recession / Economic Slowdown":     -0.09,
        "Natural Disaster (supply disruption)": 0.01,
        "Energy Policy Shift":               -0.05,
    },
    "East Asia / China": {
        "War / Armed Conflict":               0.02,
        "Sanctions on Producer":              0.03,
        "OPEC Production Cut":                None,
        "OPEC Production Increase":           None,
        "Recession / Economic Slowdown":     -0.14,
        "Natural Disaster (supply disruption)": 0.02,
        "Energy Policy Shift":               -0.08,
    },
    "North America": {
        "War / Armed Conflict":               0.03,
        "Sanctions on Producer":              0.04,
        "OPEC Production Cut":                None,
        "OPEC Production Increase":           None,
        "Recession / Economic Slowdown":     -0.12,
        "Natural Disaster (supply disruption)": 0.07,
        "Energy Policy Shift":               -0.10,
    },
}

# Rationale shown to user for each region × event
RATIONALE = {
    ("Middle East", "War / Armed Conflict"):
        "Middle East supplies ~31% of global oil. War risks Strait of Hormuz closure (~20% of global supply) → extreme price spike.",
    ("Middle East", "Sanctions on Producer"):
        "Sanctions on Iran, Iraq, or Saudi Arabia directly cut exportable supply from the world's largest reserve region.",
    ("Middle East", "OPEC Production Cut"):
        "Most OPEC core members are Middle Eastern. A cut here has the highest cartel impact on global supply.",
    ("Middle East", "OPEC Production Increase"):
        "Saudi-led production surge historically crashes prices (e.g. 2014, 2020).",
    ("Middle East", "Recession / Economic Slowdown"):
        "Demand-side pressure, but Middle East is a producer — impact partially offset by supply constraint incentives.",
    ("Middle East", "Natural Disaster (supply disruption)"):
        "Infrastructure damage to Gulf terminals or pipelines can instantly remove millions of barrels/day.",
    ("Middle East", "Energy Policy Shift"):
        "Regional renewable push has limited near-term supply effect given cost structures.",

    ("Eastern Europe / Russia", "War / Armed Conflict"):
        "Russia is the #2 global oil exporter. Armed conflict disrupts pipeline flows to Europe and Asia.",
    ("Eastern Europe / Russia", "Sanctions on Producer"):
        "Sanctions on Russia (as seen in 2022) remove a major supplier, forcing expensive rerouting and price spikes.",
    ("Eastern Europe / Russia", "Recession / Economic Slowdown"):
        "Reduced demand from Europe, a major consumer, softens prices.",
    ("Eastern Europe / Russia", "Natural Disaster (supply disruption)"):
        "Pipeline or terminal damage in Russia/Kazakhstan affects significant export volumes.",
    ("Eastern Europe / Russia", "Energy Policy Shift"):
        "European energy transition accelerates away from Russian oil — bearish long-term.",

    ("West Africa", "War / Armed Conflict"):
        "Nigeria and Angola are major OPEC producers. Conflict disrupts Niger Delta operations.",
    ("West Africa", "Sanctions on Producer"):
        "Rare, but sanctions on Nigeria/Angola would remove ~4 million bbl/day from markets.",
    ("West Africa", "OPEC Production Cut"):
        "West African OPEC members follow quota cuts — moderate supply impact.",
    ("West Africa", "OPEC Production Increase"):
        "Increased West African output adds supply but is secondary to Gulf producers.",
    ("West Africa", "Recession / Economic Slowdown"):
        "Lower global demand reduces appetite for West African grades.",
    ("West Africa", "Natural Disaster (supply disruption)"):
        "Gulf of Guinea terminal or offshore rig damage affects exports.",
    ("West Africa", "Energy Policy Shift"):
        "Minimal near-term impact — region still heavily dependent on oil revenue.",

    ("Latin America", "War / Armed Conflict"):
        "Venezuela and Mexico produce oil but aren't global chokepoints — localized impact.",
    ("Latin America", "Sanctions on Producer"):
        "US sanctions on Venezuela have historically tightened heavy crude supply.",
    ("Latin America", "OPEC Production Cut"):
        "Venezuela is an OPEC member; Ecuador and others follow quotas.",
    ("Latin America", "OPEC Production Increase"):
        "Added supply from Brazil/Venezuela is significant but secondary.",
    ("Latin America", "Recession / Economic Slowdown"):
        "Regional recession reduces demand and can trigger production instability.",
    ("Latin America", "Natural Disaster (supply disruption)"):
        "Hurricane or earthquake damage to refineries (e.g. Gulf of Mexico) briefly tightens supply.",
    ("Latin America", "Energy Policy Shift"):
        "Brazil's pre-salt investment shifts signal long-term supply growth.",

    ("South Asia", "War / Armed Conflict"):
        "India, Pakistan, Bangladesh are consumers, not producers. No major reserves or shipping chokepoints affected. Minimal price impact.",
    ("South Asia", "Sanctions on Producer"):
        "South Asia doesn't export oil meaningfully — sanctions here affect demand, not supply.",
    ("South Asia", "Recession / Economic Slowdown"):
        "India is the 3rd largest oil consumer globally — a South Asian recession meaningfully reduces demand.",
    ("South Asia", "Natural Disaster (supply disruption)"):
        "No major production infrastructure — only refinery or port disruption possible.",
    ("South Asia", "Energy Policy Shift"):
        "India's aggressive solar push reduces long-term demand forecasts.",

    ("East Asia / China", "War / Armed Conflict"):
        "East Asia is the world's largest oil-consuming region but not a producer. War disrupts demand, not supply.",
    ("East Asia / China", "Sanctions on Producer"):
        "Sanctions on China would suppress the world's #1 oil importer — reducing global demand sharply.",
    ("East Asia / China", "Recession / Economic Slowdown"):
        "China alone accounts for ~15% of global oil demand. A Chinese recession is the most bearish demand-side shock possible.",
    ("East Asia / China", "Natural Disaster (supply disruption)"):
        "Port or refinery disruption in China/Japan briefly tightens refined product supply.",
    ("East Asia / China", "Energy Policy Shift"):
        "China's EV adoption and renewable push is the single biggest long-term demand destruction factor.",

    ("North America", "War / Armed Conflict"):
        "US/Canada are major producers but a domestic conflict is extremely unlikely. Minimal global supply effect.",
    ("North America", "Sanctions on Producer"):
        "Rare scenario — US self-sanctions unlikely. Canada sanctions would tighten heavy crude.",
    ("North America", "Recession / Economic Slowdown"):
        "US recession cuts the world's #1 or #2 oil consumer — significant demand destruction.",
    ("North America", "Natural Disaster (supply disruption)"):
        "Gulf of Mexico hurricanes can shut ~1.5 million bbl/day temporarily — meaningful short-term spike.",
    ("North America", "Energy Policy Shift"):
        "US green policy (drilling bans, IRA) suppresses domestic production and long-term investment.",
}

# ─── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("CL=F", period="10y", interval="1wk", progress=False, auto_adjust=True)
    df = df[["Close"]].dropna()
    df.columns = ["Price"]
    df.index = pd.to_datetime(df.index)
    return df

# ─── Feature Engineering ───────────────────────────────────────────────────────
def build_features(df, event_factor):
    d = df.copy()
    d["lag1"]    = d["Price"].shift(1)
    d["lag2"]    = d["Price"].shift(2)
    d["lag4"]    = d["Price"].shift(4)
    d["lag8"]    = d["Price"].shift(8)
    d["ma4"]     = d["Price"].rolling(4).mean()
    d["ma12"]    = d["Price"].rolling(12).mean()
    d["std4"]    = d["Price"].rolling(4).std()
    d["pct_chg"] = d["Price"].pct_change()
    d["event"]   = event_factor
    d["target"]  = d["Price"].shift(-1)
    d.dropna(inplace=True)
    return d

# ─── Train & Predict ───────────────────────────────────────────────────────────
def train_and_predict(df, event_factor):
    data = build_features(df, event_factor)
    feature_cols = ["lag1","lag2","lag4","lag8","ma4","ma12","std4","pct_chg","event"]
    X = data[feature_cols].values
    y = data["target"].values
    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    latest_features = scaler.transform(X[-1].reshape(1, -1))

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        "XGBoost":           XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                          random_state=42, verbosity=0),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred_test = model.predict(X_test_s)
        mae  = mean_absolute_error(y_test, y_pred_test)
        r2   = r2_score(y_test, y_pred_test)
        pred = float(model.predict(latest_features)[0])
        results.append({"Model": name, "Predicted Price": pred, "MAE (test)": mae, "R² (test)": r2})
    return results, float(df["Price"].iloc[-1])

# ─── Helpers ───────────────────────────────────────────────────────────────────
def impact_color(factor):
    if factor is None:    return "#555"
    if factor > 0.10:     return "#e05c5c"
    if factor > 0.04:     return "#e09a3a"
    if factor > 0:        return "#d4c44a"
    if factor > -0.05:    return "#7ecb7e"
    return "#4caf50"

def impact_label(factor):
    if factor is None:         return "N/A"
    pct = factor * 100
    sign = "+" if pct > 0 else ""
    if abs(pct) >= 15:         severity = "Extreme"
    elif abs(pct) >= 10:       severity = "High"
    elif abs(pct) >= 5:        severity = "Moderate"
    else:                      severity = "Low"
    return f"{severity} ({sign}{pct:.0f}%)"

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("🛢️ Crude Oil Price Predictor")
st.markdown(
    '<div class="info-box">Predicts <strong>next week\'s WTI crude oil price</strong> using 10 years of weekly data. '
    'Select a geopolitical event <em>and its region</em> — impact factors are calibrated to actual supply/demand geography.</div>',
    unsafe_allow_html=True
)

with st.spinner("Fetching latest WTI price data…"):
    try:
        df = load_data()
        data_ok = True
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        data_ok = False

if data_ok:
    current_price = float(df["Price"].iloc[-1])
    last_date     = df.index[-1].strftime("%b %d, %Y")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current WTI Price</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="metric-delta" style="color:#888">as of {last_date}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        week_change = float(df["Price"].iloc[-1] - df["Price"].iloc[-2])
        color = "#4caf50" if week_change >= 0 else "#f44336"
        arrow = "▲" if week_change >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Week-on-Week Change</div>
            <div class="metric-value" style="color:{color}">{arrow} ${abs(week_change):.2f}</div>
            <div class="metric-delta" style="color:{color}">{week_change/float(df['Price'].iloc[-2])*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        yr_high = float(df["Price"].tail(52).max())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">52-Week High</div>
            <div class="metric-value">${yr_high:.2f}</div>
            <div class="metric-delta" style="color:#888">past 12 months</div>
        </div>""", unsafe_allow_html=True)

    # ── Event Selection ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Geopolitical Event</div>', unsafe_allow_html=True)

    col_e, col_r = st.columns(2)
    with col_e:
        all_events = list(list(IMPACT_MATRIX.values())[0].keys())
        selected_event = st.selectbox("Event type", all_events)
    with col_r:
        all_regions = list(IMPACT_MATRIX.keys())
        selected_region = st.selectbox("Region", all_regions)

    # Get impact factor
    event_factor = IMPACT_MATRIX[selected_region][selected_event]

    # Show impact badge + rationale
    label = impact_label(event_factor)
    color = impact_color(event_factor)

    if event_factor is None:
        st.markdown(
            f'<div class="neutral-box">⚪ <strong>{selected_event}</strong> in <strong>{selected_region}</strong> — '
            f'<span style="color:#888">Not applicable</span> (this region has no relevant production or supply role for this event type). '
            f'A baseline factor of 0 will be used.</div>',
            unsafe_allow_html=True
        )
        event_factor = 0.0
    else:
        box_class = "warn-box" if event_factor > 0.10 else "info-box"
        rationale = RATIONALE.get((selected_region, selected_event), "")
        sign = "+" if event_factor > 0 else ""
        pct  = event_factor * 100
        st.markdown(
            f'<div class="{box_class}">'
            f'<strong>{selected_event} · {selected_region}</strong> &nbsp;'
            f'<span style="background:{color};color:#111;padding:2px 9px;border-radius:10px;font-size:12px;font-weight:700">'
            f'{label}</span><br><br>'
            f'<span style="color:#ccc">{rationale}</span><br><br>'
            f'<span style="color:#888">Impact factor applied to models: '
            f'<strong style="color:{color}">{sign}{pct:.0f}%</strong></span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Quick Reference Matrix ─────────────────────────────────────────────────
    with st.expander("📊 View full region × event impact matrix"):
        matrix_rows = []
        for region in IMPACT_MATRIX:
            row = {"Region": region}
            for event in all_events:
                f = IMPACT_MATRIX[region][event]
                if f is None:
                    row[event[:18]] = "N/A"
                else:
                    sign = "+" if f > 0 else ""
                    row[event[:18]] = f"{sign}{f*100:.0f}%"
            matrix_rows.append(row)
        matrix_df = pd.DataFrame(matrix_rows).set_index("Region")
        st.dataframe(matrix_df, use_container_width=True)

    # ── Run Prediction ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"> </div>', unsafe_allow_html=True)
    if st.button("🔮 Run Prediction"):
        with st.spinner("Training models and generating predictions…"):
            results, last_price = train_and_predict(df, event_factor)

        st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

        rows = []
        for r in results:
            delta     = r["Predicted Price"] - last_price
            delta_pct = delta / last_price * 100
            rows.append({
                "Model":             r["Model"],
                "Predicted ($/bbl)": f"${r['Predicted Price']:.2f}",
                "Change vs Now":     f"{'▲' if delta>=0 else '▼'} ${abs(delta):.2f} ({delta_pct:+.1f}%)",
                "MAE":               f"${r['MAE (test)']:.2f}",
                "R²":                f"{r['R² (test)']:.3f}",
            })

        table_df = pd.DataFrame(rows)
        st.dataframe(table_df.set_index("Model"), use_container_width=True)

        best       = max(results, key=lambda x: x["R² (test)"])
        best_pred  = best["Predicted Price"]
        best_delta = best_pred - last_price

        st.markdown(f"""
        <div class="winner-card">
            <div style="font-size:13px;color:#888;margin-bottom:4px">🏆 Best performing model (highest R²)</div>
            <div style="font-size:20px;font-weight:700;color:#7ecb7e">{best['Model']}</div>
            <div style="margin-top:10px;font-size:15px;color:#ccc">
                Next week's predicted price:
                <span style="font-size:22px;font-weight:700;color:#f0a500">${best_pred:.2f}</span>
                &nbsp;
                <span style="color:{'#4caf50' if best_delta>=0 else '#f44336'}">
                    {'▲' if best_delta>=0 else '▼'} ${abs(best_delta):.2f}
                </span>
            </div>
            <div style="font-size:12px;color:#666;margin-top:8px">
                R² = {best['R² (test)']:.3f} · MAE = ${best['MAE (test)']:.2f} on held-out test set
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📈 Historical WTI Price (2 Years)</div>', unsafe_allow_html=True)
        st.line_chart(df.tail(104)["Price"], color="#f0a500", use_container_width=True)

        st.markdown(
            '<div class="info-box" style="margin-top:16px">⚠️ <strong>Disclaimer:</strong> These predictions are for '
            'educational purposes only. Do not use this for financial decisions.</div>',
            unsafe_allow_html=True
        )
