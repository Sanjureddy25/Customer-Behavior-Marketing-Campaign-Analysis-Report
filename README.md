# Customer Behavior & Marketing Campaign Analysis

## 📌 Project Overview
This project performs a comprehensive analysis of customer behavior and marketing campaign performance using Python. It simulates a retail environment with generated datasets to derive actionable business insights, focusing on customer segmentation, retention, and marketing ROI.

## 🚀 Key Features
- **Customer Behavior Analysis**:
  - **Segmentation**: Identification of High-Value vs. Low-Value customers (Pareto Analysis).
  - **Retention**: Calculation of Retention Rate and Churn Rate (>90 days inactive).
  - **Frequency**: Analysis of purchase cycles and weekly order trends.
- **Marketing Performance Evaluation**:
  - **Campaign Metrics**: Calculation of CTR, Conversion Rate, CPL, ROI, and CAC.
  - **Optimization**: Identification of profitable vs. wasteful campaigns.
- **Data Pipeline**:
  - Automated data generation (Mock Data).
  - Robust data cleaning (handling duplicates, nulls, and outliers).
  - Visualization generation using Matplotlib/Seaborn.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Libraries**:
  - `pandas`: Data manipulation and analysis.
  - `numpy`: Numerical operations.
  - `matplotlib` & `seaborn`: Data visualization.

## 📂 Project Structure
```
├── analysis.py           # Main script for data processing and analysis
├── data_generator.py     # Script to generate realistic mock datasets
├── Final_Report.md       # Detailed report of findings and insights
├── explanation.md        # Simplified project summary for non-technical stakeholders
├── output/               # Directory containing generated charts and plots
│   ├── revenue_by_customer.png
│   ├── retention_heatmap.png
│   ├── ctr_comparison.png
│   └── ...
└── README.md             # Project documentation
```

## ⚙️ Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/customer-marketing-analysis.git
    cd customer-marketing-analysis
    ```

2.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn
    ```

3.  **Generate Data**:
    Run the generator to create `customer_data.csv` and `campaign_data.csv`.
    ```bash
    python data_generator.py
    ```

4.  **Run Analysis**:
    Execute the analysis script to calculate metrics and generate visualizations.
    ```bash
    python analysis.py
    ```
    *Results will be printed to the console and charts saved to the `output/` folder.*

## 📊 Key Insights
- **High-Value Dominance**: The top customer segment generates **2.3x more revenue** than the lower segment.
- **Churn Risk**: A significant portion of users become inactive after 90 days, highlighting a need for re-engagement campaigns.
- **Marketing Efficiency**: The average campaign **ROI is ~77%**, but specific campaigns show negative returns and should be optimized.

## 📝 License
This project is open-source and available under the MIT License.
