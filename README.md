# InAnalytics

![Dashboard Screenshot](Images/CLV_Dashboard.png)


InAnalytics is a comprehensive insurance analytics dashboard built with Streamlit, Altair, and scikit-learn. It offers end-to-end insights into customer lifetime value (CLV), segmentation, policy trends, and more—all wrapped in a visually appealing interface with a modern gradient background.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Instructions](#usage-instructions)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview
InAnalytics is designed for insurance companies looking to transform their raw data into actionable insights. It provides:

- **Real-time CLV Predictions**: Using trained machine learning models (Random Forest, Gradient Boosting, Linear Regression).
- **Segmentation**: Group customers based on RFM metrics (Recency, Frequency, Monetary).
- **Policy & Claims Analysis**: Visualize key trends in policies, claims, and anomalies.
- **Easy Deployment**: A single Streamlit app that can be hosted on Heroku, AWS, or Streamlit Community Cloud.

By combining interactive charts, tables, and metric cards, InAnalytics helps business stakeholders quickly gauge performance and identify strategic opportunities.

## Key Features
### **Home Dashboard**
- Displays high-level metrics (Total Revenue, Active Policies, etc.).
- Monthly revenue trends in a line chart.
- Quick data preview table.

### **ML Models**
- Compares model performance metrics (RMSE, R²) across multiple algorithms.
- Visualizes model performance with bar charts.

### **CLV Prediction**
- Allows on-the-fly CLV predictions by inputting key customer features (e.g., Age, Premium, Frequency).
- Supports multiple model selection (Linear Regression, Random Forest, Gradient Boosting).

### **Segmentation**
- Loads customer segments (K-Means on RFM) to show distribution, average RFM metrics, and scatter plots.
- Identifies high-value or at-risk customer clusters.

### **Data Analysis**
- Pie, bar, and line charts for policy distribution, claim trends, and anomaly detection.
- Highlights unusual or suspicious claims above the 95th percentile.

### **Reports & Settings**
- Placeholder tabs for future development (e.g., custom reporting, user preferences).

### **Gradient Theming**
- A light blue gradient background for a modern data-centric look.
- White metric cards for clarity.

## Project Structure
A typical layout might look like this:
```
InAnalytics/
├── data/
│   ├── insurance_analysis.csv      <-- Synthetic data for Data Analysis
│   ├── customer_segments.csv       <-- Created by customer_segmentation.py
│   └── insurance_customers.csv     <-- Optional data for CLV training
├── models/
│   ├── model_linearregression.pkl
│   ├── model_randomforest.pkl
│   ├── model_gradientboosting.pkl
│   └── model_metrics.csv           <-- Created by model_training.py
├── scripts/
│   ├── model_training.py           <-- Trains models, saves metrics & .pkl
│   ├── customer_segmentation.py    <-- Creates customer_segments.csv
│   └── generate_insurance_analysis.py
├── dashboard/
│   ├── app.py                      <-- Main Streamlit dashboard
│   └── assets/
│       ├── inanalytics_banner.png  <-- Banner image for README
│       └── logo.png                <-- Optional brand logo
├── images/                         <-- Folder for storing screenshots
│   ├── dashboard_screenshot.png    <-- Dashboard screenshot
│   ├── ml_models_screenshot.png    <-- ML models screenshot
├── requirements.txt
└── README.md                       <-- This file
```

## Installation & Setup
### **Clone the Repository**
```bash
git clone https://github.com/YourUsername/InAnalytics.git
cd InAnalytics
```
### **(Optional) Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scriptsctivate      # On Windows
```
### **Install Dependencies**
```bash
pip install -r requirements.txt
```
### **Generate/Load Data**
- Run `model_training.py` to train models and produce `model_metrics.csv` + `.pkl` files in `models/`.
- Run `customer_segmentation.py` to generate `customer_segments.csv`.
- Run `generate_insurance_analysis.py` if you need synthetic data for the Data Analysis tab.

## Usage Instructions
### **Navigate to the Dashboard Folder**
```bash
cd dashboard
```
### **Launch Streamlit**
```bash
streamlit run app.py
```
The app typically opens at [http://localhost:8501](http://localhost:8501).

Explore the **Home, ML Models, CLV Prediction, Segmentation, Data Analysis, Reports, and Settings** tabs.

## Screenshots
### **Home Page**
- High-level metrics, monthly revenue chart, and sample data preview.

### **ML Models**
- Comparing model performance (RMSE, R²) across multiple algorithms.

### **CLV Prediction**
- Real-time CLV predictions by selecting a model and inputting features.

### **Segmentation**
- Customer clusters with RFM metrics and scatter plots for deeper insights.

## Future Enhancements
- **Automated Retraining & Deployment**: Set up CI/CD pipelines to retrain models and redeploy the app automatically.
- **Role-Based Access**: Implement user authentication and authorization to secure sensitive data.
- **Advanced Visualizations**: Add more charts (box plots, heatmaps) for claims, risk, or fraud detection.
- **Integration with External Services**: Connect to real-time data sources (AWS S3, Azure Blob Storage, etc.).

## Contributing
- Fork the repository and create a new branch for your feature or bug fix.
- Submit a pull request with clear documentation and a test plan.
- Discuss changes in the PR to refine and merge them.

We welcome contributions from the community—whether bug reports, feature requests, or code improvements.

## License
This project is licensed under the MIT License. Feel free to use and adapt InAnalytics to suit your needs.

---
Thank you for checking out **InAnalytics**! If you have any questions, issues, or suggestions, please open an issue or contact us directly.
