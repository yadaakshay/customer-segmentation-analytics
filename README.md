# Customer Segmentation & Marketing Analytics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-yellow)

This repository contains an end-to-end Unsupervised Machine Learning pipeline designed to segment customer profiles for targeted marketing strategies. By identifying distinct purchasing behaviors and customer demographics, this project enables data-driven decision-making for customer retention, acquisition, and tailored marketing campaigns.

## Business Impact
In consulting and retail analytics, understanding the customer base is critical for optimizing marketing spend. This model automates the discovery of high-value customer segments (e.g., VIPs, At-Risk, Loyal Customers) using advanced clustering techniques, directly translating raw transaction data into actionable marketing intelligence.

## Technical Approach
* **Exploratory Data Analysis (EDA):** Analyzed consumer demographics, income distributions, and spending scores to identify baseline trends and outliers.
* **Feature Engineering & Preprocessing:** Standardized high-dimensionality continuous variables and applied **Principal Component Analysis (PCA)** for dimensionality reduction.
* **Unsupervised Learning:** Engineered clustering algorithms including **K-Means Clustering**.
* **Model Evaluation:** Utilized the **Elbow Method** and **Silhouette Score** optimization to determine the statistically ideal number of distinct customer segments.
* **Deployment:** Deployed the segmentation model into a highly interactive, business-facing web application using **Streamlit**, allowing stakeholders to visualize clusters in real-time.

## Repository Structure
* `CustomerSegmentationAnalysis.ipynb`: Comprehensive Jupyter Notebook containing EDA, PCA, and K-Means training.
* `CustomerSegmentation_Streamlit.py`: Core application script for the Streamlit web dashboard.
* `data/`: Contains the raw and processed customer demographic/transaction datasets.
* `requirements.txt`: Python dependencies required to run the environment.

## Running the Application
To run the Streamlit dashboard locally:
```bash
pip install -r requirements.txt
streamlit run CustomerSegmentation_Streamlit.py
```
