import re
from io import StringIO
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------
# Helpers: loading + cleaning
# ----------------------------

def extract_sheet_id(url: str) -> str | None:
    """
    Accepts URLs like:
    https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit?usp=sharing
    """
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None


def extract_gid(url: str) -> str:
    """
    Gets gid=... from querystring if present; default '0'.
    """
    try:
        qs = parse_qs(urlparse(url).query)
        gid = qs.get("gid", ["0"])[0]
        print(qs)
        print(gid)
        return gid if gid else "0"
    except Exception:
        return "0"


def google_sheet_to_csv_url(sheet_url: str) -> str:
    """
    Builds a CSV export URL for a public Google Sheet.
    """
    sheet_id = extract_sheet_id(sheet_url)
    if not sheet_id:
        raise ValueError("Could not find a Google Sheet ID in the URL.")
    gid = extract_gid(sheet_url)
    print(sheet_id)
    print(gid)
    # Works when the sheet is viewable without auth
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def sniff_delimiter(text: str) -> str:
    """
    Very small delimiter sniffer for pasted/raw text.
    """
    # Prefer tab if it looks like TSV (common in copy-paste)
    if "\t" in text and text.count("\t") > text.count(","):
        return "\t"
    return ","


def normalize_price_value(x):
    """
    Convert mixed price formats:
    - Some rows look like dollars (12.49, 45, 52.8)
    - Some rows look like cents as integers (17900, 4500, 1600)
    Heuristic: if numeric >= 1000 AND looks integer-ish => divide by 100.
    """
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
    """
    Estimate gels/servings in the variant based on strings like:
      "Single Serving"
      "4 Gels"
      "Box of 6"
      "Box of 30"
      "5 Pack"
      "6 Pack"
      "Gel Combo 12 Pack"
      "Variety 8 Pack"
    If unknown, returns NaN.
    """
    if not isinstance(variant_title, str) or not variant_title.strip():
        return np.nan

    vt = variant_title.strip()

    # "Single Serving"
    if re.search(r"\bsingle\s*serving\b", vt, flags=re.IGNORECASE):
        return 1.0

    # "Box of 12"
    m = re.search(r"\bbox\s*of\s*(\d+)\b", vt, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    # "4 Gels" / "30 Gels"
    m = re.search(r"\b(\d+)\s*(?:gels?|servings?)\b", vt, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    # "6 Pack" / "5 Pack" / "Variety 8 Pack"
    m = re.search(r"\b(\d+)\s*pack\b", vt, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    # "Gel Combo 12 Pack"
    m = re.search(r"\b(\d+)\s*pack\b", vt, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Fallback: any standalone number, but avoid grabbing years etc.
    # (Only accept 1..1000 range)
    nums = re.findall(r"\b(\d{1,4})\b", vt)
    for n in nums:
        val = int(n)
        if 1 <= val <= 200:  # gels/packs probably under 200
            return float(val)

    return np.nan


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_feed_df(df: pd.DataFrame) -> pd.DataFrame:
    # Trim whitespace column names
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize booleans (common columns in your sample)
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

    # Key numeric columns
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

    # Normalize price formats
    if "price" in df.columns:
        df["price_usd"] = df["price"].apply(normalize_price_value)
    else:
        df["price_usd"] = np.nan

    # Servings per variant (estimate)
    if "variant_title" in df.columns:
        df["servings_est"] = df["variant_title"].apply(parse_servings_from_variant)
    else:
        df["servings_est"] = np.nan

    # Price per gel/serving
    df["price_per_serving_usd"] = df["price_usd"] / df["servings_est"]

    # Carbs per $ and calories per $
    if "meta.pim.carbohydrates" in df.columns:
        df["carbs_per_usd"] = df["meta.pim.carbohydrates"] / df["price_per_serving_usd"]
    else:
        df["carbs_per_usd"] = np.nan

    if "meta.pim.calories" in df.columns:
        df["calories_per_usd"] = df["meta.pim.calories"] / df["price_per_serving_usd"]
    else:
        df["calories_per_usd"] = np.nan

    # A nice display name
    title = df["title"] if "title" in df.columns else ""
    variant = df["variant_title"] if "variant_title" in df.columns else ""
    df["display_name"] = (title.astype(str).str.strip() + " — " + variant.astype(str).str.strip()).str.strip(" —")

    # Vendor fallback
    if "vendor" not in df.columns:
        df["vendor"] = "Unknown"

    # Product type fallback
    if "product_type" not in df.columns:
        df["product_type"] = "Unknown"

    return df


@st.cache_data(show_spinner=False)
def load_from_google_sheet(sheet_url: str) -> pd.DataFrame:
    csv_url = google_sheet_to_csv_url(sheet_url)
    df = pd.read_csv(csv_url)
    return df


@st.cache_data(show_spinner=False)
def load_from_upload(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    # Try utf-8, fallback latin-1
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")

    # Try CSV first, then TSV
    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(StringIO(text), sep=sep)
            if df.shape[1] >= 5:
                return df
        except Exception:
            pass

    # Last resort: pandas infer
    return pd.read_csv(StringIO(text), sep=None, engine="python")


@st.cache_data(show_spinner=False)
def load_from_paste(pasted_text: str) -> pd.DataFrame:
    sep = sniff_delimiter(pasted_text)
    df = pd.read_csv(StringIO(pasted_text), sep=sep)
    return df


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Feed Gels Dashboard", layout="wide")

st.title("🥤 Feed Gels Dashboard")
st.caption("Load your Feed gel dataset (Google Sheets / upload / paste) and explore price + nutrition + reviews.")

with st.sidebar:
    st.header("1) Load data")

    source = st.radio(
        "Choose a source",
        ["Google Sheets", "Upload file", "Paste text"],
        index=0
    )

    sheet_url = ""
    uploaded = None
    pasted = ""

    if source == "Google Sheets":
        sheet_url = st.text_input(
            "Google Sheets URL",
            value="https://docs.google.com/spreadsheets/d/1uw2V057vEJuZGLYs6zXiJrEJolRiDZvET3jnaMzsryw/edit?gid=753516515#gid=753516515",
            help="Sheet must be viewable by anyone with the link (no login) for CSV export to work."
        )
        st.caption("Tip: if it fails, try File → Share → set 'Anyone with the link' → Viewer.")

    elif source == "Upload file":
        uploaded = st.file_uploader("Upload CSV or TSV", type=["csv", "tsv", "txt"])

    else:
        pasted = st.text_area(
            "Paste TSV/CSV here",
            height=180,
            placeholder="Paste your header + rows here (tabs or commas)."
        )

    st.divider()
    st.header("2) Filters (after load)")
    st.caption("Filters appear after data loads.")


# Load data
df_raw = None
error = None

try:
    if source == "Google Sheets":
        if sheet_url.strip():
            df_raw = load_from_google_sheet(sheet_url.strip())
    elif source == "Upload file":
        if uploaded is not None:
            df_raw = load_from_upload(uploaded)
    else:
        if pasted.strip():
            df_raw = load_from_paste(pasted.strip())
except Exception as e:
    error = str(e)

if error:
    st.error(f"Could not load data: {error}")

if df_raw is None:
    st.info("Load data using the sidebar to get started.")
    st.stop()

df = clean_feed_df(df_raw)

# Ensure key columns exist even if missing
for col in ["vendor", "title", "variant_title", "product_type", "display_name"]:
    if col not in df.columns:
        df[col] = ""

# Sidebar filters (now that df exists)
with st.sidebar:
    vendors = sorted([v for v in df["vendor"].dropna().unique().tolist() if str(v).strip() != ""])
    selected_vendors = st.multiselect("Vendor", vendors, default=vendors)

    # Numeric filter helpers
    def slider_for_col(label, col, default_min=None, default_max=None):
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return None
        mn, mx = float(s.min()), float(s.max())
        if default_min is None:
            default_min = mn
        if default_max is None:
            default_max = mx
        return st.slider(label, mn, mx, (default_min, default_max))

    price_rng = slider_for_col("Price per serving ($)", "price_per_serving_usd")
    carbs_rng = slider_for_col("Carbs per serving (g)", "meta.pim.carbohydrates")
    caffeine_rng = slider_for_col("Caffeine (mg)", "meta.pim.caffeine")

    vegan_only = st.checkbox("Vegan only (if column exists)", value=False)
    in_stock_only = st.checkbox("In stock only (if column exists)", value=False)
    exclude_discontinued = st.checkbox("Exclude discontinued (if column exists)", value=True)

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

if vegan_only and "meta.pim.vegan" in f.columns:
    f = f[f["meta.pim.vegan"] == True]

if in_stock_only and "meta.pim.in_stock" in f.columns:
    f = f[f["meta.pim.in_stock"] == True]

if exclude_discontinued and "meta.custom.discontinued" in f.columns:
    # keep rows that are not discontinued OR unknown
    f = f[(f["meta.custom.discontinued"] != True) | (f["meta.custom.discontinued"].isna())]


# ----------------------------
# Summary KPIs
# ----------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Rows", f"{len(f):,}")
kpi2.metric("Unique products", f"{f['title'].nunique():,}" if "title" in f.columns else "—")
kpi3.metric("Unique vendors", f"{f['vendor'].nunique():,}")
if "meta.okendo.summaryData.reviewCount" in f.columns:
    kpi4.metric("Total reviews (sum)", f"{int(pd.to_numeric(f['meta.okendo.summaryData.reviewCount'], errors='coerce').fillna(0).sum()):,}")
else:
    kpi4.metric("Total reviews (sum)", "—")


# ----------------------------
# Tabs for exploration
# ----------------------------
tab_overview, tab_value, tab_nutrition, tab_table = st.tabs(
    ["Overview", "Value", "Nutrition", "Table / Export"]
)

with tab_overview:
    st.subheader("Overview")

    c1, c2 = st.columns(2)

    with c1:
        # Price per serving distribution
        if "price_per_serving_usd" in f.columns and f["price_per_serving_usd"].notna().any():
            fig = px.histogram(
                f.dropna(subset=["price_per_serving_usd"]),
                x="price_per_serving_usd",
                nbins=40,
                title="Price per serving distribution",
                hover_data=["display_name", "vendor"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No usable price_per_serving_usd values to plot.")

    with c2:
        # Vendor counts
        vc = f["vendor"].value_counts().reset_index()
        vc.columns = ["vendor", "count"]
        fig = px.bar(
            vc.head(20),
            x="vendor",
            y="count",
            title="Top vendors by number of variants (filtered)",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    # Reviews vs price scatter
    if all(c in f.columns for c in ["price_per_serving_usd", "meta.okendo.summaryData.reviewAverageValue"]):
        s = f.dropna(subset=["price_per_serving_usd", "meta.okendo.summaryData.reviewAverageValue"])
        if not s.empty:
            fig = px.scatter(
                s,
                x="price_per_serving_usd",
                y="meta.okendo.summaryData.reviewAverageValue",
                color="vendor",
                hover_data=["display_name"],
                title="Review average vs price per serving"
            )
            st.plotly_chart(fig, use_container_width=True)


with tab_value:
    st.subheader("Value: carbs / calories per dollar")

    c1, c2 = st.columns(2)

    with c1:
        if all(c in f.columns for c in ["carbs_per_usd", "vendor"]) and f["carbs_per_usd"].notna().any():
            tmp = f.dropna(subset=["carbs_per_usd"]).copy()
            # Vendor medians are often more stable than mean
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
                title="Top vendors by median carbs per $ (filtered)",
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need carbohydrates + price info to compute carbs_per_usd.")

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
                title="Top vendors by median calories per $ (filtered)",
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need calories + price info to compute calories_per_usd.")

    # Product-level leaderboard
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
            "carbs_per_usd"
        ] if c in top.columns]
        st.dataframe(top[show_cols], use_container_width=True)
    else:
        st.info("No carbs_per_usd values available under current filters.")


with tab_nutrition:
    st.subheader("Nutrition Explorer")

    # Scatter: carbs vs calories, size by price per serving, color by caffeine bucket
    needed = ["meta.pim.carbohydrates", "meta.pim.calories", "price_per_serving_usd"]
    if all(c in f.columns for c in needed):
        s = f.dropna(subset=needed).copy()

        if "meta.pim.caffeine" in s.columns:
            s["caffeine_bucket"] = pd.cut(
                pd.to_numeric(s["meta.pim.caffeine"], errors="coerce"),
                bins=[-0.1, 0.1, 25, 50, 100, 5000],
                labels=["0", "0–25", "25–50", "50–100", "100+"],
                include_lowest=True
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
                hover_data=["display_name", "vendor", "price_per_serving_usd"],
                title="Carbs vs calories (bubble size = price per serving)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rows available for scatter plot after filtering.")
    else:
        st.info("Need carbohydrates, calories, and price_per_serving to show the nutrition scatter.")

    c1, c2 = st.columns(2)

    with c1:
        # Sugar vs carbs
        if all(c in f.columns for c in ["meta.pim.sugar", "meta.pim.carbohydrates"]) and f["meta.pim.sugar"].notna().any():
            s = f.dropna(subset=["meta.pim.sugar", "meta.pim.carbohydrates"]).copy()
            fig = px.scatter(
                s,
                x="meta.pim.carbohydrates",
                y="meta.pim.sugar",
                color="vendor",
                hover_data=["display_name"],
                title="Sugar vs carbohydrates"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sugar values available to plot.")

    with c2:
        # Caffeine distribution
        if "meta.pim.caffeine" in f.columns and f["meta.pim.caffeine"].notna().any():
            fig = px.histogram(
                f.dropna(subset=["meta.pim.caffeine"]),
                x="meta.pim.caffeine",
                nbins=30,
                title="Caffeine distribution (mg)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No caffeine values available to plot.")


with tab_table:
    st.subheader("Browse + export")

    # Choose columns to show
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

    cols = st.multiselect(
        "Columns to display",
        options=f.columns.tolist(),
        default=default_cols
    )

    st.dataframe(f[cols], use_container_width=True, height=520)

    # Download filtered
    csv = f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="feed_gels_filtered.csv",
        mime="text/csv"
    )

st.caption(
    "Notes: servings_est is inferred from variant_title and may be NaN for unusual formats. "
    "Prices are normalized using a heuristic to handle mixed cents/dollars formats."
)
