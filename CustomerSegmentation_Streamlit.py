from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
st.title("Customer Segmentation")
st.caption("Explore customer purchase behavior with Recency, Frequency, and Monetary (RFM) features.")

DATA_DIR = Path(__file__).parent / "data"
COLUMNS = ["Customer_id", "day", "Quantity", "Sales"]
RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


def read_customer_data(file, filename):
    """Read a CDNOW-style whitespace file or a CSV with the expected four columns."""
    if filename.lower().endswith(".csv"):
        raw = pd.read_csv(file)
        normalized = {str(column).strip().lower(): column for column in raw.columns}
        aliases = {
            "customer_id": ("customer_id", "customerid", "customer"),
            "day": ("day", "date", "order_date"),
            "quantity": ("quantity", "qty"),
            "sales": ("sales", "amount", "spend", "monetary"),
        }
        targets = ("customer_id", "day", "quantity", "sales")
        matched = {
            target: next((normalized[name] for name in aliases[target] if name in normalized), None)
            for target in targets
        }
        if all(matched.values()):
            raw = raw.rename(
                columns={matched[target]: COLUMNS[index] for index, target in enumerate(targets)}
            )[COLUMNS]
        elif len(raw.columns) == 4:
            # Header names are not recognized; follow the documented positional schema.
            raw.columns = COLUMNS
        else:
            raise ValueError(
                "CSV must contain customer ID, date, quantity, and sales columns, "
                "or exactly four columns in that order."
            )
    else:
        raw = pd.read_csv(
            file, sep=r"\s+", header=None, names=COLUMNS, encoding="latin-1"
        )

    if raw.empty:
        raise ValueError("The selected file has no data rows.")
    raw["Customer_id"] = raw["Customer_id"].astype("string").str.strip()
    raw["day"] = pd.to_datetime(
        raw["day"].astype("string").str.replace(r"\.0$", "", regex=True),
        errors="coerce",
    )
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["Sales"] = pd.to_numeric(raw["Sales"], errors="coerce")
    raw = raw.dropna(subset=COLUMNS)
    raw = raw[raw["Customer_id"] != ""]
    if raw.empty:
        raise ValueError("No usable rows remain. Check the customer IDs, dates, quantities, and sales.")
    return raw


@st.cache_data(show_spinner=False)
def load_sample(path):
    return read_customer_data(path, path.name)


with st.sidebar:
    st.header("Data")
    data_source = st.radio("Choose a source", ["Sample file", "Upload a file"])
    data = None
    if data_source == "Sample file":
        sample_files = sorted(DATA_DIR.glob("*.txt")) + sorted(DATA_DIR.glob("*.csv"))
        if not sample_files:
            st.info("Add a .txt or .csv sample to the data/ folder, or choose Upload a file.")
        else:
            sample_path = st.selectbox("Sample dataset", sample_files, format_func=lambda p: p.name)
            try:
                data = load_sample(sample_path)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not read the sample file: {exc}")
    else:
        uploaded = st.file_uploader("Choose CDNOW .txt or .csv data", type=["txt", "csv"])
        if uploaded is not None:
            try:
                data = read_customer_data(uploaded, uploaded.name)
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"Could not read the uploaded file: {exc}")

if data is None:
    st.info("Choose a valid sample dataset or upload a file to get started.")
    st.stop()

# Include every input column so even same-size datasets invalidate an old model.
fingerprint = int(pd.util.hash_pandas_object(data, index=True).sum())
if st.session_state.get("data_fingerprint") != fingerprint:
    st.session_state["data_fingerprint"] = fingerprint
    st.session_state.pop("segmentation", None)

snapshot_date = data["day"].max() + pd.Timedelta(days=1)
rfm = data.groupby("Customer_id").agg(
    Recency=("day", lambda dates: (snapshot_date - dates.max()).days),
    Frequency=("Customer_id", "size"),
    Monetary=("Sales", "sum"),
)
rfm = rfm.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=RFM_COLUMNS)

m1, m2, m3 = st.columns(3)
m1.metric("Valid transactions", f"{len(data):,}")
m2.metric("Customers", f"{len(rfm):,}")
m3.metric("Date range", f"{data['day'].min():%Y-%m-%d} – {data['day'].max():%Y-%m-%d}")
with st.expander("Preview and data quality"):
    st.caption("Invalid or incomplete rows are removed during loading. The preview shows the cleaned data.")
    st.dataframe(data.head(10), use_container_width=True)
    st.dataframe(rfm.describe().round(2), use_container_width=True)

st.subheader("Customer segments")
if len(rfm) < 3:
    st.warning("At least three customers are needed to compare and fit customer segments.")
else:
    max_k = min(10, len(rfm) - 1)
    scaled = StandardScaler().fit_transform(rfm[RFM_COLUMNS])
    scored = []
    for k in range(2, max_k + 1):
        candidate = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = candidate.fit_predict(scaled)
        if 1 < len(set(labels)) < len(rfm):
            scored.append((k, silhouette_score(scaled, labels)))

    if not scored:
        st.warning("The selected data does not support a meaningful multi-cluster solution.")
    else:
        score_table = pd.DataFrame(scored, columns=["Clusters", "Silhouette score"])
        recommended_k = int(score_table.loc[score_table["Silhouette score"].idxmax(), "Clusters"])
        default_index = next(i for i, (k, _) in enumerate(scored) if k == recommended_k)
        selected_k = st.selectbox(
            "Number of clusters",
            options=[k for k, _ in scored],
            index=default_index,
            help="Suggested value: the tested cluster count with the highest silhouette score.",
        )
        if st.button("Build segments", type="primary"):
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("kmeans", KMeans(n_clusters=selected_k, random_state=42, n_init=10)),
                ]
            )
            labels = pipeline.fit_predict(rfm[RFM_COLUMNS])
            result = rfm.copy()
            result["Segment"] = labels.astype(str)
            st.session_state["segmentation"] = {
                "pipeline": pipeline,
                "segments": result,
                "cluster_count": selected_k,
            }

        st.caption(f"Suggested cluster count: {recommended_k} (best of the tested silhouette scores).")
        st.dataframe(score_table.round(3), use_container_width=True)

segmentation = st.session_state.get("segmentation")
if segmentation:
    segments = segmentation["segments"]
    st.success(
        f"Built {segmentation['cluster_count']} segments. IDs are model labels; profile them before assigning marketing names."
    )
    profile = segments.groupby("Segment").agg(
        Customers=("Recency", "size"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
    ).round(2)
    profile["Customer share (%)"] = (profile["Customers"] / len(segments) * 100).round(1)
    left, right = st.columns(2)
    left.dataframe(profile, use_container_width=True)
    right.plotly_chart(
        px.scatter(
            segments.reset_index(),
            x="Recency",
            y="Monetary",
            size="Frequency",
            color="Segment",
            hover_data=["Customer_id"],
            title="Customer RFM profile",
        ),
        use_container_width=True,
    )
    st.download_button(
        "Download customer segments",
        segments.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv",
    )

    st.subheader("Assign a customer to a segment")
    st.caption("Enter RFM values calculated with the same definitions and snapshot date as the training data.")
    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Recency (days)", min_value=0, value=int(rfm["Recency"].median()))
    frequency = c2.number_input(
        "Frequency (transactions)", min_value=1, value=max(1, int(rfm["Frequency"].median()))
    )
    monetary = c3.number_input("Monetary (total sales)", value=float(rfm["Monetary"].median()))
    if st.button("Assign segment"):
        customer = pd.DataFrame([[recency, frequency, monetary]], columns=RFM_COLUMNS)
        segment_id = segmentation["pipeline"].predict(customer)[0]
        st.info(f"Assigned to Segment {segment_id}.")
