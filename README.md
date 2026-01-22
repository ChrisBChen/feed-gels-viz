# Feed Gels Dashboard (Streamlit) 🥤

This is an interactive Streamlit dashboard for exploring a Feed gels product dataset. It supports loading data from Google Sheets, file uploads, or pasted CSV/TSV text, then calculates normalized pricing, inferred servings per variant, and value metrics like carbs per dollar and calories per dollar.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project uses Python 3.10+ and the following modules:

* [Streamlit](https://streamlit.io/)
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [Plotly Express](https://plotly.com/python/plotly-express/)

Install these modules as necessary using Python's [pip installer](https://pypi.org/project/pip/):

```
pip install streamlit pandas numpy plotly
```

### Installing

1. Clone the repo.

```
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

2. (Recommended) Create and activate a virtual environment.

#### macOS / Linux

```
python -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies.

```
pip install -r requirements.txt
```

> If you do not have a `requirements.txt` yet, you can install directly via:
>
> ```
> pip install streamlit pandas numpy plotly
> ```

## Running the Dashboard

From the root folder of the repo, run:

```
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`).

## Using the Dashboard

### Data sources

The dashboard supports three data input methods:

1. **Google Sheets**

   * Paste a Google Sheets URL in the sidebar.
   * The sheet must be viewable without authentication (e.g., “Anyone with the link” → Viewer).
   * The app converts the URL into a CSV export endpoint automatically.

2. **Upload file**

   * Upload `.csv`, `.tsv`, or `.txt` files.
   * The app attempts CSV first, then TSV, then falls back to pandas inference.

3. **Paste text**

   * Paste TSV or CSV directly.
   * The app sniffs whether the delimiter is tab or comma.

### What the app computes

After loading, the app cleans and enriches the dataset:

* `price_usd`

  * Normalizes mixed price formats (dollars vs cents) with a heuristic.
* `servings_est`

  * Attempts to infer servings from `variant_title` (e.g., “Box of 6”, “4 Gels”, “Single Serving”, “6 Pack”).
* `price_per_serving_usd`

  * `price_usd / servings_est`
* `carbs_per_usd`

  * `carbohydrates / price_per_serving_usd` (if carbs exist)
* `calories_per_usd`

  * `calories / price_per_serving_usd` (if calories exist)
* `display_name`

  * A readable label for charts and tables: `title — variant_title`

### Filters

Filters are applied in the sidebar after data loads:

* Vendor multi-select
* Price per serving range
* Carbs range (if present)
* Caffeine range (if present)
* Optional booleans if present (vegan, in stock, discontinued)

### Tabs

* **Overview**

  * Price per serving distribution
  * Vendor counts
  * Reviews vs price (if review columns exist)
* **Value**

  * Vendor medians for carbs/$ and calories/$
  * “Best value variants” table
* **Nutrition**

  * Carbs vs calories bubble plot (bubble size = price per serving)
  * Sugar vs carbs (if sugar exists)
  * Caffeine distribution (if caffeine exists)
* **Table / Export**

  * Browse filtered rows
  * Choose columns to display
  * Download filtered CSV

## Data Format

The app is designed for feed-like product exports and will work best with columns like:

* `title`
* `variant_title`
* `vendor`
* `product_type`
* `price`

Optional columns supported:

* Nutrition:

  * `meta.pim.calories`
  * `meta.pim.carbohydrates`
  * `meta.pim.sugar`
  * `meta.pim.caffeine`
* Reviews:

  * `meta.okendo.summaryData.reviewAverageValue`
  * `meta.okendo.summaryData.reviewCount`
* Flags:

  * `meta.pim.in_stock`
  * `meta.pim.vegan`
  * `meta.custom.discontinued`

If optional columns are missing, the dashboard will hide or skip visualizations that require them.

## Authors

* **Chris Chen** - [GitHub](https://github.com/ChrisBChen)

## Acknowledgments

* Built with [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/python/)
