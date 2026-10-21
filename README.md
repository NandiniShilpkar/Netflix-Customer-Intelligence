# Netflix Customer Intelligence & AI Analytics Copilot

A customer analytics and machine learning platform built to understand customer behavior, identify customers at risk of churn, and turn subscription, viewing, payment, and support data into useful business insights.

The project combines PostgreSQL, Python, machine learning, and Streamlit into an end-to-end analytics workflow.

## About the Project

This project brings together customer, subscription, viewing, payment, support, and churn data to create a customer intelligence workflow. It includes data validation, historical feature engineering, churn prediction, customer segmentation, model evaluation, and an interactive Streamlit dashboard.

## What the Project Does

### Dashboard
Provides a high-level view of the customer base and key customer metrics.

### Business KPIs
Provides business-oriented metrics across subscriptions, payments, viewing activity, and customer behavior.

### Churn Intelligence
Includes churn-risk scoring, customer ranking, feature importance, historical churn analysis, and individual customer explanations using SHAP.

The current dashboard uses a Random Forest model selected through chronological validation.

### Customer Segmentation
Groups customers according to behavioral and engagement patterns to support customer analysis and retention strategies.

## Data

The project works with multiple related datasets:

- Customers
- Subscriptions
- Subscription plans
- Viewing activity
- Payments
- Support tickets
- Customer feedback
- Content
- Churn labels

Approximate dataset sizes:

| Table | Rows |
|---|---:|
| Customers | 8,000 |
| Subscriptions | 8,875 |
| Viewing activity | 66,422 |
| Payments | 110,735 |
| Support tickets | 6,120 |
| Customer feedback | 5,046 |
| Content | 500 |
| Churn labels | 8,000 |
| Subscription plans | 3 |

## Data Pipeline



## Churn Prediction

The churn model uses customer information available at historical snapshots to predict whether a customer will churn during the following 30-day period.

The features include customer demographics, subscription tenure, viewing behavior, payment activity, failed payments, support interactions, satisfaction, and resolution information.

The model is also used to rank customers so that a retention team can prioritize higher-risk customers.

## Model Validation

Because churn prediction is time-dependent, the historical models were evaluated using chronological validation rather than relying only on a random train-test split.

The selected Random Forest model uses:

- 300 trees
- 
- Balanced class weights
- Random state = 42

### Rolling Validation Results

| Validation period | ROC-AUC | Average Precision | F1 |
|---|---:|---:|---:|
| Feb → Mar | 0.610 | 0.050 | 0.101 |
| Feb + Mar → Apr | 0.598 | 0.058 | 0.110 |
| Feb + Mar + Apr → May | 0.630 | 0.075 | 0.130 |

These are historical experimental results and should not be interpreted as production-level model performance.

## Customer Ranking

For retention use cases, ranking customers can be more useful than selecting a single probability threshold.

On the historical May evaluation period, the Random Forest captured more churners within the top 10% and top 20% of ranked customers than the Logistic Regression baseline.

## Explainability

The project uses Random Forest feature importance for global model interpretation and SHAP TreeExplainer for customer-level explanations.

This helps answer why an individual customer receives a particular churn-risk score.

## Technology Stack

- Python
- PostgreSQL
- SQLAlchemy
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Plotly
- Streamlit
- GitHub Codespaces

## Project Structure



## Running the Project

### 1. Clone the repository



### 2. Create a virtual environment



Activate it:



### 3. Install dependencies



### 4. Configure environment variables

Create a  file in the project root using  as a reference.



Do not commit your  file or expose database credentials publicly.

### 5. Run the application



## Project Limitations

This project is intended as an analytics and machine learning portfolio project. The historical churn labels and follow-up windows have limitations, so the current model should not be treated as a production churn system.

Production use would require additional validation, calibration, monitoring, drift detection, stronger label verification, and automated retraining.

## Future Improvements

- Natural-language analytics interface
- Automated retention recommendations
- Customer lifetime value prediction
- Cohort and retention analysis
- Model monitoring
- Automated data refresh and retraining
- Cloud deployment

## Why I Built This Project

I wanted to build something that goes beyond training a machine learning model on a dataset. The focus is the complete workflow from business problem and data preparation through database design, feature engineering, machine learning, evaluation, explainability, and business reporting.

## Author

**Nandini Shilpkar**

Machine Learning / Data Science

GitHub: [NandiniShilpkar](https://github.com/NandiniShilpkar)
