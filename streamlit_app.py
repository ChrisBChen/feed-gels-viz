import re
from io import StringIO
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# App config
# ============================================================

APP_TITLE = "Feed Gels Dashboard"
APP_ICON = "🥤"

PLOTLY_TEMPLATE = "plotly_white"  # consistent, clean

# Raw columns -> friendly UI labels
LABELS = {
    "display_name": "Product (Title — Variant)",
    "title": "Product Title",
    "variant_title": "Variant",
    "vendor": "Brand / Vendor",
    "product_type": "Product Type",
    "price": "Raw Price",
    "price_usd": "Price (USD)",
    "compare_at_price": "Compare-at Price",
    "servings_est": "Servings (estimated)",
    "price_per_serving_usd": "Price per Serving (USD)",
    "meta.pim.calories": "Calories",
    "meta.pim.carbohydrates": "Carbs (g)",
    "meta.pim.sugar": "Sugar (g)",
    "meta.pim.caffeine": "Caffeine (mg)",
    "carbs_per_usd": "Carbs per $",
    "calories_per_usd": "Calories per $",
    "meta.okendo.summaryData.reviewAverageValue": "Avg Review",
    "meta.okendo.summaryData.reviewCount": "Review Count",
    "meta.pim.in_stock": "In Stock",
    "meta.pim.vegan": "Vegan",
    "meta.custom.discontinued": "Discontinued",
    "product_image": "Image URL",
    "handle": "Handle",
}

# Formatting helpers for display
FMT = {
    "price_usd": "${:,.2f}",
    "price_per_serving_usd": "${:,.2f}",
    "carbs_per_usd": "{:,.1f}",
    "calories_per_usd": "{:,.0f}",
    "meta.okendo.summaryData.reviewAverageValue": "{:,.2f}",
    "meta.okendo.summaryData.reviewCount": "{:,.0f}",
    "meta.pim.carbohydrates": "{:,.0f}",
    "meta.pim.calories": "{:,.0f}",
    "meta.pim.caffeine": "{:,.0f}",
    "meta.pim.sugar": "{:,.0f}",
    "servings_est": "{:,.0f}",
}

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1uw2V057vEJuZGLYs6zXiJrEJolRiDZvET3jnaMzsryw/edit?gid=753516515#gid=753516515"
)


# ============================================================
# Helpers: loading + cleaning
# ============================================================

def extract_sheet_id(url: str) -> str | None:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None


def extract_gid(url: str) -> str:
    try:
        qs = parse_qs(urlparse(url).query)
        gid = qs.get("gid", ["0"])[0]
        return gid if gid else "0"
    except Exception:
        return "0"


def google_sheet_to_csv_url(sheet_url: str) -> str:
    sheet_id = extract_sheet_id(sheet_url)
    if not sheet_id:
        raise ValueError("Could not find a Google Sheet ID in the URL.")
    gid = extract_gid(sheet_url)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def sniff_delimiter(text: str) -> str:
    if "\t" in text and text.count("\t") > text.count(","):
        return "\t"
    return ","


def normalize_price_value(x):
    if pd.isna(x):
        return np.nan
    try:
        v = float(x)
    except Exception:
        return np.nan
    if v >= 1000 and abs(v - round(v)) < 1e-9:
        return v / 100.0
    return v


def parse_bool(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return True
    if s in {"false", "f", "no", "n", "0"}:
        return False
    return np.nan


def parse_servings_from_variant(variant_title: str) -> float:
    if not isinstance(variant_title, str) or not variant_title.strip():
        return np.nan

    vt = variant_title.strip()

    if re.search(r"\bsingle\s*serving\b", vt, flags=re.IGNORECASE):
        return 1.0

    m = re.search(r"\bbox\s*of\s*(\d+)\b", vt, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.search(r"\b(\d+)\s*(?:gels?|servings?)\b", vt, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.search(r"\b(\d+)\s*pack\b", vt, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    nums = re.findall(r"\b(\d{1,4})\b", vt)
    for n in nums:
        val = int(n)
        if 1 <= val <= 200:
            return float(val)

    return np.nan


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_cols(df: pd.DataFrame, cols: list[str], fill="") -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df


def clean_feed_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    bool_cols = [
        "meta.pim.dairy_free",
        "meta.pim.gluten_free",
        "meta.pim.in_stock",
        "meta.pim.non_gmo",
        "meta.pim.nut_free",
        "meta.pim.organic_ingredients",
        "meta.pim.peanut_free",
        "meta.pim.soy_free",
        "meta.pim.sugar_free",
        "meta.pim.vegan",
        "meta.pim.vegetarian",
        "meta.custom.discontinued",
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].map(parse_bool)

    numeric_cols = [
        "price",
        "compare_at_price",
        "inventory_quantity",
        "meta.okendo.summaryData.reviewAverageValue",
        "meta.okendo.summaryData.reviewCount",
        "meta.pim.added_sugars",
        "meta.pim.caffeine",
        "meta.pim.calories",
        "meta.pim.carbohydrates",
        "meta.pim.protein",
        "meta.pim.saturated_fat",
        "meta.pim.sodium",
        "meta.pim.sugar",
        "meta.pim.total_fat",
        "meta.pim.trans_fat",
    ]
    df = coerce_numeric(df, [c for c in numeric_cols if c in df.columns])

    df["price_usd"] = df["price"].apply(normalize_price_value) if "price" in df.columns else np.nan
    df["servings_est"] = df["variant_title"].apply(parse_servings_from_variant) if "variant_title" in df.columns else np.nan
    df["price_per_serving_usd"] = df["price_usd"] / df["servings_est"]

    if "meta.pim.carbohydrates" in df.columns:
        df["carbs_per_usd"] = df["meta.pim.carbohydrates"] / df["price_per_serving_usd"]
    else:
        df["carbs_per_usd"] = np.nan

    if "meta.pim.calories" in df.columns:
        df["calories_per_usd"] = df["meta.pim.calories"] / df["price_per_serving_usd"]
    else:
        df["calories_per_usd"] = np.nan

    title = df["title"] if "title" in df.columns else ""
    variant = df["variant_title"] if "variant_title" in df.columns else ""
    df["display_name"] = (
        title.astype(str).str.strip() + " — " + variant.astype(str).str.strip()
    ).str.strip(" —")

    df = ensure_cols(df, ["vendor", "product_type"], fill="Unknown")
    df = ensure_cols(df, ["title", "variant_title", "display_name"], fill="")

    return df


@st.cache_data(show_spinner=False)
def load_from_google_sheet(sheet_url: str) -> pd.DataFrame:
    csv_url = google_sheet_to_csv_url(sheet_url)
    return pd.read_csv(csv_url)


@st.cache_data(show_spinner=False)
def load_from_upload(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")

    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(StringIO(text), sep=sep)
            if df.shape[1] >= 5:
                return df
        except Exception:
            pass

    return pd.read_csv(StringIO(text), sep=None, engine="python")


@st.cache_data(show_spinner=False)
def load_from_paste(pasted_text: str) -> pd.DataFrame:
    sep = sniff_delimiter(pasted_text)
    return pd.read_csv(StringIO(pasted_text), sep=sep)


def friendly(name: str) -> str:
    return LABELS.get(name, name)


def chart_base(fig, title: str):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        title_font=dict(size=16),
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text="",
    )
    return fig


def safe_metric_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("Load your Feed gel dataset and explore price, nutrition, and reviews. Designed for clean, client-ready output.")

# Sidebar: data loading + filters (in a form for better UX)
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Google Sheets", "Upload file", "Paste text"], index=0)

    sheet_url = ""
    uploaded = None
    pasted = ""

    if source == "Google Sheets":
        sheet_url = st.text_input(
            "Google Sheets URL",
            value=DEFAULT_SHEET_URL,
            help="Sheet must be viewable without auth, for CSV export to work.",
        )
    elif source == "Upload file":
        uploaded = st.file_uploader("Upload CSV/TSV", type=["csv", "tsv", "txt"])
    else:
        pasted = st.text_area("Paste TSV/CSV", height=160, placeholder="Paste header + rows here (tabs or commas).")

    st.divider()
    st.header("Filters")

    # We only show real filters after data loads, but we keep the submit button here.
    apply_btn = st.button("Apply / Refresh", use_container_width=True)


# Load data with a clean status box (instead of raw exception popups)
df_raw = None
load_error = None

with st.status("Loading data…", expanded=False) as status:
    try:
        if source == "Google Sheets" and sheet_url.strip():
            df_raw = load_from_google_sheet(sheet_url.strip())
        elif source == "Upload file" and uploaded is not None:
            df_raw = load_from_upload(uploaded)
        elif source == "Paste text" and pasted.strip():
            df_raw = load_from_paste(pasted.strip())
        else:
            df_raw = None

        if df_raw is None:
            status.update(label="Waiting for data source input", state="complete")
        else:
            status.update(label="Data loaded", state="complete")
    except Exception as e:
        load_error = str(e)
        status.update(label="Load failed", state="error")

if load_error:
    st.error("Could not load data. Check the source settings and try again.")
    with st.expander("Show error details"):
        st.code(load_error)
    st.stop()

if df_raw is None:
    st.info("Use the sidebar to load data (Google Sheets, upload, or paste).")
    st.stop()

df = clean_feed_df(df_raw)

# ------------------------------------------------------------
# Sidebar filters (now that df exists)
# ------------------------------------------------------------
with st.sidebar:
    with st.form("filters_form", border=False):
        vendors = sorted([v for v in df["vendor"].dropna().unique().tolist() if str(v).strip()])
        selected_vendors = st.multiselect("Brand / Vendor", vendors, default=vendors)

        def slider_for_col(label, col):
            if col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                return None
            mn, mx = float(s.min()), float(s.max())
            # protect against identical mn/mx
            if mn == mx:
                return (mn, mx)
            return st.slider(label, mn, mx, (mn, mx))

        price_rng = slider_for_col("Price per serving (USD)", "price_per_serving_usd")
        carbs_rng = slider_for_col("Carbs (g)", "meta.pim.carbohydrates")
        caffeine_rng = slider_for_col("Caffeine (mg)", "meta.pim.caffeine")

        with st.expander("Diet / availability", expanded=False):
            vegan_only = st.checkbox("Vegan only", value=False, disabled=("meta.pim.vegan" not in df.columns))
            in_stock_only = st.checkbox("In stock only", value=False, disabled=("meta.pim.in_stock" not in df.columns))
            exclude_discontinued = st.checkbox(
                "Exclude discontinued",
                value=True,
                disabled=("meta.custom.discontinued" not in df.columns),
            )

        submitted = st.form_submit_button("Apply filters", use_container_width=True)


# Apply filters
f = df.copy()

if selected_vendors:
    f = f[f["vendor"].isin(selected_vendors)]

if price_rng and "price_per_serving_usd" in f.columns:
    f = f[f["price_per_serving_usd"].between(price_rng[0], price_rng[1], inclusive="both")]

if carbs_rng and "meta.pim.carbohydrates" in f.columns:
    f = f[f["meta.pim.carbohydrates"].between(carbs_rng[0], carbs_rng[1], inclusive="both")]

if caffeine_rng and "meta.pim.caffeine" in f.columns:
    f = f[f["meta.pim.caffeine"].between(caffeine_rng[0], caffeine_rng[1], inclusive="both")]

if "meta.pim.vegan" in f.columns:
    if vegan_only:
        f = f[f["meta.pim.vegan"] == True]

if "meta.pim.in_stock" in f.columns:
    if in_stock_only:
        f = f[f["meta.pim.in_stock"] == True]

if "meta.custom.discontinued" in f.columns:
    if "exclude_discontinued" in locals() and exclude_discontinued:
        f = f[(f["meta.custom.discontinued"] != True) | (f["meta.custom.discontinued"].isna())]


# ------------------------------------------------------------
# Top KPIs
# ------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Rows", f"{len(f):,}")
k2.metric("Unique products", f"{f['title'].nunique():,}" if "title" in f.columns else "—")
k3.metric("Unique vendors", f"{f['vendor'].nunique():,}")

if "meta.okendo.summaryData.reviewCount" in f.columns:
    total_reviews = pd.to_numeric(f["meta.okendo.summaryData.reviewCount"], errors="coerce").fillna(0).sum()
    k4.metric("Total reviews (sum)", safe_metric_int(total_reviews))
else:
    k4.metric("Total reviews (sum)", "—")

with st.expander("Data quality checks", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing servings_est", safe_metric_int(f["servings_est"].isna().sum()))
    c2.metric("Missing price_usd", safe_metric_int(f["price_usd"].isna().sum()))
    c3.metric("Missing price/serving", safe_metric_int(f["price_per_serving_usd"].isna().sum()))
    st.caption("If servings_est is missing, price-per-serving and value metrics will also be missing for those rows.")


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_overview, tab_value, tab_nutrition, tab_table = st.tabs(["Overview", "Value", "Nutrition", "Table / Export"])


with tab_overview:
    st.subheader("Overview")

    c1, c2 = st.columns(2)

    with c1:
        if "price_per_serving_usd" in f.columns and f["price_per_serving_usd"].notna().any():
            data = f.dropna(subset=["price_per_serving_usd"]).copy()
            fig = px.histogram(
                data,
                x="price_per_serving_usd",
                nbins=40,
                labels={k: friendly(k) for k in data.columns},
                hover_data=["display_name", "vendor", "servings_est", "price_usd"],
            )
            fig.update_traces(hovertemplate=None)  # use plotly default, cleaner
            fig = chart_base(fig, "Price per serving distribution")
            fig.update_xaxes(tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No usable price per serving values under current filters.")

    with c2:
        vc = f["vendor"].value_counts().reset_index()
        vc.columns = ["vendor", "count"]
        fig = px.bar(vc.head(20), x="vendor", y="count", labels={"vendor": "Vendor", "count": "Variants"})
        fig = chart_base(fig, "Top vendors by number of variants (filtered)")
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    if all(c in f.columns for c in ["price_per_serving_usd", "meta.okendo.summaryData.reviewAverageValue"]):
        s = f.dropna(subset=["price_per_serving_usd", "meta.okendo.summaryData.reviewAverageValue"]).copy()
        if not s.empty:
            fig = px.scatter(
                s,
                x="price_per_serving_usd",
                y="meta.okendo.summaryData.reviewAverageValue",
                color="vendor",
                labels={k: friendly(k) for k in s.columns},
                hover_data=["display_name", "servings_est", "price_usd"],
            )
            fig = chart_base(fig, "Avg review vs price per serving")
            fig.update_xaxes(tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)


with tab_value:
    st.subheader("Value: carbs / calories per dollar")

    c1, c2 = st.columns(2)

    with c1:
        if all(c in f.columns for c in ["carbs_per_usd", "vendor"]) and f["carbs_per_usd"].notna().any():
            tmp = f.dropna(subset=["carbs_per_usd"]).copy()
            vendor_median = (
                tmp.groupby("vendor", as_index=False)["carbs_per_usd"]
                .median()
                .sort_values("carbs_per_usd", ascending=False)
                .head(20)
            )
            fig = px.bar(
                vendor_median,
                x="vendor",
                y="carbs_per_usd",
                labels={"vendor": "Vendor", "carbs_per_usd": "Carbs per $ (median)"},
            )
            fig = chart_base(fig, "Top vendors by median carbs per $ (filtered)")
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need carbs + price-per-serving to compute carbs per $.")

    with c2:
        if all(c in f.columns for c in ["calories_per_usd", "vendor"]) and f["calories_per_usd"].notna().any():
            tmp = f.dropna(subset=["calories_per_usd"]).copy()
            vendor_median = (
                tmp.groupby("vendor", as_index=False)["calories_per_usd"]
                .median()
                .sort_values("calories_per_usd", ascending=False)
                .head(20)
            )
            fig = px.bar(
                vendor_median,
                x="vendor",
                y="calories_per_usd",
                labels={"vendor": "Vendor", "calories_per_usd": "Calories per $ (median)"},
            )
            fig = chart_base(fig, "Top vendors by median calories per $ (filtered)")
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need calories + price-per-serving to compute calories per $.")

    st.markdown("#### Best value variants (carbs per $)")
    if "carbs_per_usd" in f.columns and f["carbs_per_usd"].notna().any():
        top = (
            f.dropna(subset=["carbs_per_usd", "price_per_serving_usd"])
            .sort_values("carbs_per_usd", ascending=False)
            .head(25)
        )
        show_cols = [c for c in [
            "display_name",
            "vendor",
            "price_usd",
            "servings_est",
            "price_per_serving_usd",
            "meta.pim.carbohydrates",
            "meta.pim.calories",
            "meta.pim.caffeine",
            "carbs_per_usd",
        ] if c in top.columns]

        st.dataframe(
            top[show_cols].rename(columns=friendly),
            use_container_width=True,
            hide_index=True,
            column_config={
                friendly(c): st.column_config.NumberColumn(format=FMT.get(c, None))
                for c in show_cols
                if c in FMT
            },
        )
    else:
        st.info("No carbs per $ values available under current filters.")


with tab_nutrition:
    st.subheader("Nutrition Explorer")

    needed = ["meta.pim.carbohydrates", "meta.pim.calories", "price_per_serving_usd"]
    if all(c in f.columns for c in needed):
        s = f.dropna(subset=needed).copy()

        if "meta.pim.caffeine" in s.columns:
            s["caffeine_bucket"] = pd.cut(
                pd.to_numeric(s["meta.pim.caffeine"], errors="coerce"),
                bins=[-0.1, 0.1, 25, 50, 100, 5000],
                labels=["0", "0–25", "25–50", "50–100", "100+"],
                include_lowest=True,
            )
        else:
            s["caffeine_bucket"] = "Unknown"

        if not s.empty:
            fig = px.scatter(
                s,
                x="meta.pim.carbohydrates",
                y="meta.pim.calories",
                size="price_per_serving_usd",
                color="caffeine_bucket",
                labels={
                    "meta.pim.carbohydrates": "Carbs (g)",
                    "meta.pim.calories": "Calories",
                    "price_per_serving_usd": "Price per serving (USD)",
                    "caffeine_bucket": "Caffeine bucket",
                },
                hover_data=["display_name", "vendor", "price_per_serving_usd", "meta.pim.caffeine"],
            )
            fig = chart_base(fig, "Carbs vs calories (bubble size = price per serving)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rows available after filtering for nutrition scatter.")
    else:
        st.info("Need carbs, calories, and price-per-serving to show the nutrition scatter.")

    c1, c2 = st.columns(2)

    with c1:
        if all(c in f.columns for c in ["meta.pim.sugar", "meta.pim.carbohydrates"]) and f["meta.pim.sugar"].notna().any():
            s = f.dropna(subset=["meta.pim.sugar", "meta.pim.carbohydrates"]).copy()
            fig = px.scatter(
                s,
                x="meta.pim.carbohydrates",
                y="meta.pim.sugar",
                color="vendor",
                labels={"meta.pim.carbohydrates": "Carbs (g)", "meta.pim.sugar": "Sugar (g)", "vendor": "Vendor"},
                hover_data=["display_name"],
            )
            fig = chart_base(fig, "Sugar vs carbohydrates")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sugar values available under current filters.")

    with c2:
        if "meta.pim.caffeine" in f.columns and f["meta.pim.caffeine"].notna().any():
            fig = px.histogram(
                f.dropna(subset=["meta.pim.caffeine"]),
                x="meta.pim.caffeine",
                nbins=30,
                labels={"meta.pim.caffeine": "Caffeine (mg)"},
            )
            fig = chart_base(fig, "Caffeine distribution (mg)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No caffeine values available under current filters.")


with tab_table:
    st.subheader("Browse + export")

    default_cols = [c for c in [
        "display_name",
        "vendor",
        "product_type",
        "price_usd",
        "servings_est",
        "price_per_serving_usd",
        "meta.pim.calories",
        "meta.pim.carbohydrates",
        "meta.pim.caffeine",
        "meta.okendo.summaryData.reviewAverageValue",
        "meta.okendo.summaryData.reviewCount",
        "product_image",
        "handle",
    ] if c in f.columns]

    # Show friendly names in the picker, but keep underlying column keys
    cols_picker = {friendly(c): c for c in f.columns}
    picked_friendly = st.multiselect(
        "Columns",
        options=list(cols_picker.keys()),
        default=[friendly(c) for c in default_cols],
    )
    cols = [cols_picker[x] for x in picked_friendly]

    st.dataframe(
        f[cols].rename(columns=friendly),
        use_container_width=True,
        height=520,
        hide_index=True,
        column_config={
            friendly(c): st.column_config.NumberColumn(format=FMT.get(c, None))
            for c in cols
            if c in FMT
        },
    )

    csv = f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="feed_gels_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "Notes: servings_est is inferred from variant_title, and may be blank for unusual formats. "
    "Prices are normalized via a heuristic to handle mixed cents/dollars formats."
)
