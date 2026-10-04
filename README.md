# Customer Segmentation & Marketing Analytics

An interactive RFM customer segmentation project built with Python, scikit-learn, and Streamlit. It turns transaction-level customer data into recency, frequency, and monetary features, then lets users explore and assign customers to fitted segments.

## What the app does

- Loads a sample dataset from `data/` or accepts a CDNOW-style text file or CSV.
- Validates required values, parses dates and numeric fields, and drops invalid rows.
- Calculates customer RFM features:
  - **Recency:** days since the customer's most recent purchase, measured against one day after the latest date in the dataset.
  - **Frequency:** number of transaction rows for the customer.
  - **Monetary:** sum of the `Sales` field for the customer.
- Standardizes all three features before K-Means clustering, so monetary scale does not overwhelm the other features.
- Compares feasible cluster counts with silhouette scores and suggests the best-scoring tested value.
- Profiles and downloads segments, and predicts a segment for a new customer using the same fitted scaler and model.

Segment IDs are model labels, not predefined business categories. Review each segment's RFM profile before naming it or using it to drive a campaign.

## Repository structure

- `CustomerSegmentationAnalysis.ipynb`: Notebook project material.
- `CustomerSegmentation_Streamlit.py`: Streamlit application and RFM workflow.
- `data/`: Optional sample datasets.
- `requirements.txt`: Python dependencies.

## Input data

### CDNOW text format

The app accepts whitespace-delimited files with four fields and no header, in this order:

```text
Customer_id day Quantity Sales
```

Dates such as `19970101` are accepted. For example:

```text
1 19970101 2 29.50
1 19970115 1 14.75
2 19970203 3 42.00
```

### CSV format

A CSV can either have exactly four columns in the order `Customer_id, day, Quantity, Sales`, or use headers matching customer ID, date, quantity, and sales (for example, `customer_id,date,quantity,sales`).

## Run locally

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
streamlit run CustomerSegmentation_Streamlit.py
```

Choose a sample dataset from `data/` or upload a file in the sidebar. At least three distinct customers are needed to compare cluster counts. New-customer RFM values must use the same definitions and date snapshot as the data used to build the segments.
