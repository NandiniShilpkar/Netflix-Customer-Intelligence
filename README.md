# Netflix Customer Intelligence & AI Analytics Copilot

<p align="center">

**An End-to-End AI-Powered Customer Intelligence & Retention Analytics Platform**

Predict churn • Understand customers • Quantify revenue risk • Recommend retention actions • Ask business questions using AI

<br>

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_AI-black)](https://ollama.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/)

</p>

---

## 🚀 Live Project / Repository

### 🔗 GitHub Repository

**[Netflix Customer Intelligence & AI Analytics Copilot](https://github.com/NandiniShilpkar/Netflix-Customer-Intelligence)**

> Portfolio project demonstrating an end-to-end customer intelligence workflow using data engineering, PostgreSQL, machine learning, business analytics, Customer 360, retention optimization, and local AI.

---

# 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Project Objectives](#-project-objectives)
- [What This Project Does](#-what-this-project-does)
- [Key Business Questions](#-key-business-questions)
- [Architecture](#-architecture)
- [Data Architecture](#-data-architecture)
- [Dataset Overview](#-dataset-overview)
- [Data Engineering](#-data-engineering)
- [Data Quality](#-data-quality)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning](#-machine-learning)
- [Model Validation](#-model-validation)
- [Customer Decision Engine](#-customer-decision-engine)
- [Retention Optimization](#-retention-optimization)
- [Customer 360](#-customer-360)
- [AI Analytics Copilot](#-ai-analytics-copilot)
- [Local AI](#-local-ai)
- [Executive Analytics](#-executive-analytics)
- [Production AI Monitoring](#-production-ai-monitoring)
- [Key Business Results](#-key-business-results)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Environment Configuration](#-environment-configuration)
- [Database Setup](#-database-setup)
- [Run the Application](#-run-the-application)
- [AI Copilot Setup](#-ai-copilot-setup)
- [Example Questions](#-example-questions)
- [Screenshots](#-screenshots)
- [Documentation](#-documentation)
- [Security](#-security)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Portfolio Value](#-portfolio-value)
- [Version](#-version)
- [Disclaimer](#-disclaimer)

---

# 🎯 Project Overview

**Netflix Customer Intelligence & AI Analytics Copilot** is an end-to-end customer intelligence platform designed to transform customer data into actionable business decisions.

The project combines:

- Data Engineering
- PostgreSQL
- SQL
- Feature Engineering
- Machine Learning
- Churn Prediction
- Customer Segmentation
- Customer 360
- Revenue-at-Risk Analysis
- Retention Optimization
- Executive Analytics
- Local Large Language Models
- AI Analytics Copilot
- Production AI Monitoring
- Streamlit

The goal is not simply to predict churn.

The goal is to answer:

> **Who is likely to churn?**

> **How valuable is the customer?**

> **How much revenue is at risk?**

> **What action should the business take?**

---

# 💼 Business Problem

Subscription businesses need to continuously understand customer behavior and identify customers who may leave.

A traditional dashboard may show:

- Customer activity
- Subscription status
- Revenue
- Viewing behavior
- Support tickets
- Feedback

But a business needs more than descriptive analytics.

It needs to know:

1. Which customers are at risk?
2. Which customers should be prioritized?
3. How much revenue is exposed?
4. Why is a customer considered risky?
5. What retention action should be taken?
6. Which customers require immediate attention?

This project addresses those questions through an integrated customer intelligence workflow.

---

# 🎯 Project Objectives

### 1. Build a reliable data foundation

Clean, validate, and integrate customer datasets.

### 2. Predict customer churn

Build a machine-learning model that estimates the probability of churn within the next 30 days.

### 3. Convert predictions into business decisions

Transform churn probabilities into actionable customer priorities.

### 4. Quantify revenue exposure

Estimate monthly and annualized revenue at risk.

### 5. Build Customer 360

Create a complete customer-level profile combining behavioral, subscription, support, feedback, risk, and value information.

### 6. Add AI-powered analytics

Allow users to ask business questions using natural language.

### 7. Monitor production readiness

Track data quality, scoring coverage, model health, and local AI availability.

---

# 🔍 What This Project Does

The complete workflow is:

```text
Raw Customer Data
       ↓
Data Cleaning
       ↓
Data Validation
       ↓
PostgreSQL
       ↓
Feature Engineering
       ↓
Churn Prediction
       ↓
Customer Risk Scoring
       ↓
Customer Value Analysis
       ↓
Decision Engine
       ↓
Retention Optimization
       ↓
Customer 360
       ↓
Executive Analytics
       ↓
AI Analytics Copilot
       ↓
Production Monitoring
```

---

# ❓ Key Business Questions

The platform is designed to answer questions such as:

### Customer Risk

- Who has the highest churn risk?
- Which customers should be saved immediately?
- How many customers are high risk?

### Revenue

- How much monthly revenue is at risk?
- What is the annualized revenue exposure?
- Which high-risk customers have the highest value?

### Retention

- Who should receive proactive engagement?
- Who requires customer experience recovery?
- Who requires immediate retention intervention?

### Customer 360

- What is the complete profile of a customer?
- What is the customer's churn risk?
- What is their health score?
- What is their value score?
- How much revenue is at risk?

### AI Analytics

- Can a business user ask these questions in natural language?
- Can the AI explain analytical results without inventing numbers?

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │     Raw Datasets    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Cleaning &     │
                         │ Validation          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     PostgreSQL      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Churn Prediction  │
                         │   Random Forest     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Customer Risk Score │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
             ┌───────────┐  ┌──────────────┐  ┌──────────────┐
             │ Customer  │  │ Decision     │  │ Revenue Risk │
             │ 360       │  │ Engine       │  │ Analysis     │
             └─────┬─────┘  └──────┬───────┘  └──────┬───────┘
                   │               │                 │
                   └───────────────┼─────────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │ Retention           │
                         │ Optimization        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Executive Analytics │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ AI Analytics        │
                         │ Copilot             │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Streamlit       │
                         │     Dashboard       │
                         └─────────────────────┘
```

---

# 🗄️ Data Architecture

The project uses nine primary datasets.

| Dataset | Purpose |
|---|---|
| `customers` | Customer demographic and registration information |
| `subscriptions` | Subscription lifecycle, plan, status, cancellation and price |
| `subscription_plans` | Plan pricing and product characteristics |
| `viewing_activity` | Viewing sessions, duration, completion and device information |
| `content` | Content catalogue |
| `payments` | Customer payment history |
| `support_tickets` | Customer support interactions |
| `customer_feedback` | Customer ratings, feedback and sentiment |
| `churn_labels` | Historical churn outcome labels |

---

# 📊 Dataset Size

| Table | Rows |
|---|---:|
| customers | 8,000 |
| subscriptions | 8,875 |
| subscription_plans | 3 |
| viewing_activity | 66,422 |
| content | 500 |
| payments | 110,735 |
| support_tickets | 6,120 |
| customer_feedback | 5,046 |
| churn_labels | 8,000 |

---

# 🧹 Data Engineering

The data pipeline includes:

- Data inspection
- Data cleaning
- Duplicate detection
- Missing-value analysis
- Relationship validation
- Feature preparation
- Data quality reporting

### Cleaning Result

Customer duplicate validation identified:

```text
Rows before:              8,048
Rows after:               8,000
Exact duplicates removed:    48
```

### Relationship Validation

All configured table relationships were validated with:

```text
Unmatched relationships: 0
```

---

# 📈 Data Quality

The project treats data quality as a first-class component of the system.

The pipeline monitors:

- Missing values
- Duplicate records
- Relationship integrity
- Feature completeness
- Scoring coverage
- Model readiness

The final monitoring layer reports approximately:

```text
Readiness Score: 83%
Scoring Coverage: 100%
Duplicates: 0
```

> A readiness score of 83% does not mean that 17% of customers are missing. It reflects the completeness/readiness of the configured analytical features.

---

# 🧠 Feature Engineering

Features are derived from multiple customer dimensions.

### Customer Features

- Age
- Country
- City
- Acquisition channel
- Customer segment
- Preferred language

### Subscription Features

- Plan
- Monthly price
- Subscription status
- Auto-renewal
- Cancellation information
- Subscription duration

### Viewing Features

- Watch minutes
- Viewing sessions
- Completion percentage
- Recent viewing activity
- Device behavior

### Support Features

- Number of support tickets
- Unresolved tickets
- Resolution time
- Customer satisfaction

### Feedback Features

- Feedback count
- Average rating
- Sentiment information

### Revenue Features

- Monthly revenue
- Revenue at risk
- Annualized revenue at risk

---

# 🤖 Machine Learning

## Churn Prediction

The active churn model is a:

**Random Forest Classifier**

### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Number of Trees | 300 |
| Minimum Samples per Leaf | 20 |
| Class Weight | Balanced |
| Random State | 42 |
| Target | `churn_next_30_days` |
| Training Rows | 32,709 |

### Model Artifact

```text
models/historical_random_forest_leaf20.joblib
```

---

# 📊 Model Validation

The model was evaluated using rolling historical validation.

| Validation Window | ROC-AUC | Average Precision | F1 |
|---|---:|---:|---:|
| Feb → Mar | 0.610 | 0.050 | 0.101 |
| Feb + Mar → Apr | 0.598 | 0.058 | 0.110 |
| Feb + Mar + Apr → May | 0.630 | 0.075 | 0.130 |

The rolling validation approach is designed to better reflect the time-dependent nature of churn.

### Important Interpretation

The model demonstrates useful ranking capability but has moderate predictive performance.

Therefore:

> **The churn score should be treated as a prioritization signal rather than a guaranteed prediction.**

---

# 🎯 Customer Decision Engine

The Decision Engine converts analytical scores into actionable customer priorities.

Current scored cohort:

**6,260 customers**

| Decision | Customers |
|---|---:|
| 🟢 Monitor | 3,216 |
| 🟡 Engage | 1,833 |
| ⚪ Low Priority | 1,154 |
| 🔴 Save Now | 57 |

### Revenue Exposure

```text
Monthly Revenue at Risk:
$4,517.72

Annualized Revenue at Risk:
$54,212.64
```

This converts machine-learning predictions into business-oriented decision intelligence.

---

# 🔄 Retention Optimization

The Retention Optimization module identifies customers who should receive retention interventions.

### Current Target

**1,890 customers**

| Retention Strategy | Customers |
|---|---:|
| Proactive Engagement | 1,611 |
| Customer Experience Recovery | 222 |
| Immediate Retention | 55 |
| High-Value Rescue | 2 |

### Recovery Scenario

A 30% annual recovery scenario represents:

**$16,263.79**

> This is a scenario estimate and not a guaranteed financial outcome.

---

# 👤 Customer 360

Customer 360 provides a unified customer-level view.

It combines:

- Customer information
- Subscription
- Churn risk
- Health score
- Value score
- Priority score
- Recommended action
- Revenue at risk
- Viewing activity
- Support activity
- Feedback

---

## Example: CUST000685

| Metric | Value |
|---|---:|
| Plan | Premium |
| Churn Risk | 72.87% |
| Health Score | 53.35 |
| Value Score | 52.51 |
| Priority Score | 56.90 |
| Decision | Save Now |
| Monthly Revenue at Risk | $22.99 |
| Annualized Revenue at Risk | $275.88 |
| Viewing Sessions | 7 |
| Watch Minutes | 314 |

### Example AI Question

```text
Show me customer 360 for CUST000685
```

The application identifies the customer ID and routes the request to the Customer 360 analytical layer.

---

# 🧑‍💻 AI Analytics Copilot

The AI Analytics Copilot adds a natural-language interface on top of the analytics platform.

Instead of navigating through multiple dashboards, a business user can ask:

```text
Which customers have the highest churn risk?
```

or:

```text
How much revenue is at risk?
```

or:

```text
Show me customer 360 for CUST000685
```

The system routes supported questions to deterministic analytical functions and uses local AI to generate business-friendly explanations.

---

# 🧩 AI Copilot Architecture

```text
User Question
      ↓
Intent Detection
      ↓
Customer ID / Business Intent Detection
      ↓
Analytics Function
      ↓
Structured Analytics Context
      ↓
Local LLM
      ↓
Business Explanation
      ↓
Streamlit UI
```

The system is designed so that the LLM does not become the source of truth for numerical business metrics.

The analytical layer remains responsible for the actual numbers.

---

# 🧠 Local AI

The project uses **Ollama** for local AI.

### Model

```text
llama3.2:3b
```

### Local Endpoint

```text
http://127.0.0.1:11434/api/generate
```

### Why Local AI?

Using a local model provides:

- No paid API requirement
- Local execution
- Better control over analytics context
- Reduced dependency on external LLM APIs
- A practical demonstration of local AI integration

The prompt instructs the model to:

- Use only supplied analytics
- Avoid inventing numbers
- Avoid inventing customer IDs
- Keep responses concise

---

# 📊 Executive Analytics

The Executive Analytics module provides management-level KPIs.

| KPI | Result |
|---|---:|
| Customers Scored | 6,260 |
| High-Risk Customers | 709 |
| Save Now | 57 |
| Engage | 1,833 |
| Retention Target | 1,890 |
| Monthly Revenue at Risk | $4,517.72 |
| Annual Revenue at Risk | $54,212.64 |
| 30% Recovery Scenario | $16,263.79 |

This layer converts detailed customer analytics into executive-level business information.

---

# 🩺 Production AI Monitoring

The monitoring layer provides operational visibility.

| Metric | Result |
|---|---:|
| Customers Scored | 6,260 |
| Low Risk | 2,237 |
| Medium Risk | 3,314 |
| High Risk | 709 |
| Scoring Coverage | 100% |
| Missing Feature Values | 10,545 |
| Duplicates | 0 |
| Readiness | 83% |
| Churn Model | Healthy |
| Local AI | Healthy |

This demonstrates a production-oriented mindset where:

> **Data quality + model health + scoring coverage + AI availability**

are monitored separately.

---

# 📸 Screenshots

Project screenshots are stored in:

```text
assets/screenshots/
```

Recommended screenshots include:

### Executive Analytics

```text
assets/screenshots/01-executive-analytics.png
```

### Customer Intelligence

```text
assets/screenshots/02-customer-intelligence.png
```

### Customer 360

```text
assets/screenshots/03-customer-360.png
```

### Churn Intelligence

```text
assets/screenshots/04-churn-intelligence.png
```

### Retention Optimization

```text
assets/screenshots/05-retention-optimization.png
```

### AI Analytics Copilot

```text
assets/screenshots/06-ai-copilot.png
```

### Production Monitoring

```text
assets/screenshots/07-production-monitoring.png
```

> Screenshots can be added to this README once captured from the final Streamlit application.

---

# 📈 Key Business Results

## Customer Intelligence

```text
6,260
Customers Scored
```

## High Risk

```text
709
High-Risk Customers
```

## Immediate Retention

```text
57
Save Now Customers
```

## Retention Target

```text
1,890
Customers
```

## Monthly Revenue at Risk

```text
$4,517.72
```

## Annualized Revenue at Risk

```text
$54,212.64
```

## Recovery Scenario

```text
$16,263.79
```

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Database | PostgreSQL |
| Database Access | SQLAlchemy, Psycopg |
| Machine Learning | Scikit-learn |
| Visualization | Plotly |
| Application | Streamlit |
| Local AI | Ollama |
| LLM | Llama 3.2 3B |
| Version Control | Git |
| Repository | GitHub |
| Development | GitHub Codespaces |

---

# 📁 Project Structure

```text
Netflix-Customer-Intelligence/
│
├── app.py
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── .env.example
├── .gitignore
│
├── assets/
│   └── screenshots/
│       └── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── docs/
│   ├── Netflix_Database_Schema.docx
│   ├── Netflix_Dataset_Dictionary.docx
│   ├── Netflix_SQL_Business_Questions.docx
│   ├── PRD_Netflix_Customer_Intelligence_Copilot.docx
│   ├── PROJECT_REPORT.md
│   ├── PROJECT_SHOWCASE.md
│   ├── PROJECT_STATUS.md
│   └── RELEASE_CHECKLIST.md
│
├── models/
│   └── historical_random_forest_leaf20.joblib
│
├── src/
│   ├── data/
│   ├── database/
│   ├── dashboard/
│   ├── features/
│   ├── models/
│   └── services/
│
└── sql/
    └── schema.sql
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/NandiniShilpkar/Netflix-Customer-Intelligence.git
cd Netflix-Customer-Intelligence
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS / GitHub Codespaces

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Verify:

```bash
pip check
```

Expected:

```text
No broken requirements found.
```

---

# 🔐 Environment Configuration

Create a local `.env` file.

```bash
cp .env.example .env
```

Configure your PostgreSQL connection:

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require
```

Do **not** commit `.env`.

The project uses `.gitignore` to prevent credentials from being committed.

---

# 🗄️ Database Setup

The application uses PostgreSQL.

The database schema is documented in:

```text
docs/Netflix_Database_Schema.docx
```

SQL schema:

```text
sql/schema.sql
```

The application uses:

```text
SQLAlchemy
Psycopg
PostgreSQL
```

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The application will open through the Streamlit/GitHub Codespaces forwarded port.

---

# 🤖 AI Copilot Setup

The project uses Ollama for local AI.

Install Ollama and download the model:

```bash
ollama pull llama3.2:3b
```

Start Ollama:

```bash
ollama serve
```

The application expects:

```text
http://127.0.0.1:11434/api/generate
```

The configured model is:

```text
llama3.2:3b
```

---

# 💬 Example AI Questions

Try questions such as:

```text
Show me customer 360 for CUST000685
```

```text
Give me the profile of CUST000685
```

```text
What is the churn risk of CUST000685?
```

```text
What is the health score of CUST000685?
```

```text
Which customers have the highest churn risk?
```

```text
How much revenue is at risk?
```

```text
Which customers should receive retention actions?
```

---

# 📚 Documentation

Additional project documentation is available in the `docs/` directory.

### Project Report

[Project Report](docs/PROJECT_REPORT.md)

### Project Showcase

[Project Showcase](docs/PROJECT_SHOWCASE.md)

### Project Status

[Project Status](docs/PROJECT_STATUS.md)

### Release Checklist

[Release Checklist](docs/RELEASE_CHECKLIST.md)

### Database Schema

[Netflix Database Schema](docs/Netflix_Database_Schema.docx)

### Dataset Dictionary

[Netflix Dataset Dictionary](docs/Netflix_Dataset_Dictionary.docx)

### SQL Business Questions

[Netflix SQL Business Questions](docs/Netflix_SQL_Business_Questions.docx)

### Product Requirements

[Netflix Customer Intelligence Copilot PRD](docs/PRD_Netflix_Customer_Intelligence_Copilot.docx)

---

# 🔒 Security

Security practices implemented in the project include:

- `.env` excluded from Git
- No tracked live credentials
- `.env.example` contains placeholders
- Dependency validation
- Python syntax validation
- Git diff validation
- Model artifact validation
- Release checks before publishing

Never commit:

```text
.env
API keys
database passwords
secrets.toml
private credentials
```

---

# ⚠️ Limitations

This project has several important limitations.

### Machine Learning

The churn model has moderate predictive performance.

It should be considered a prioritization tool rather than a guaranteed prediction system.

### Historical Data

Historical validation does not guarantee future performance.

### Revenue Recovery

The 30% recovery calculation is a scenario estimate.

It does not represent guaranteed revenue recovery.

### Retention Testing

The recommended retention actions have not yet been validated through live A/B testing.

### Local AI

LLM response quality depends on the installed local model.

### Dataset

This is a portfolio/project dataset and does not represent confidential Netflix production data.

---

# 🚀 Future Enhancements

Potential future improvements include:

### Machine Learning

- SHAP explainability
- Probability calibration
- Hyperparameter optimization
- Automated retraining
- Model drift detection
- Feature drift monitoring

### Analytics

- Cohort analysis
- Lifetime Value prediction
- Customer journey analysis
- Recommendation engine
- Advanced segmentation

### AI

- Retrieval-Augmented Generation
- Business documentation RAG
- Tool-calling architecture
- Multi-agent analytics
- Conversational dashboard navigation

### Production

- Docker deployment
- Cloud deployment
- CI/CD
- Model registry
- Feature store
- Real-time scoring
- Role-based access control

### Experimentation

- Retention A/B testing
- Treatment-effect modeling
- Incrementality analysis
- Campaign optimization

---

# 💼 Portfolio Value

This project demonstrates skills relevant to:

### Data Scientist

- Exploratory analysis
- Feature engineering
- Machine learning
- Model validation
- Business analytics

### Machine Learning Engineer

- ML pipeline
- Model artifact management
- Scoring
- Monitoring
- Production-oriented architecture

### AI/ML Engineer

- LLM integration
- Local AI
- Prompt engineering
- Analytics routing
- AI application development

### Data Analyst

- KPI development
- Customer segmentation
- Revenue analysis
- Executive dashboards
- Business decision support

---

# 🧠 What Makes This Project Different?

This is not only a churn prediction project.

It connects:

```text
DATA
 ↓
INSIGHT
 ↓
PREDICTION
 ↓
CUSTOMER RISK
 ↓
BUSINESS PRIORITY
 ↓
RETENTION ACTION
 ↓
REVENUE IMPACT
 ↓
AI EXPLANATION
 ↓
MONITORING
```

The core philosophy is:

> **Predict → Prioritize → Explain → Recommend → Monitor**

---

# 📌 Project Highlights

### End-to-End

The project covers the complete analytics lifecycle from raw data to business decisions.

### Business-Oriented

The machine-learning output is translated into customer priorities and revenue exposure.

### Customer-Centric

Customer 360 provides a unified view of individual customers.

### AI-Enabled

Natural-language analytics are supported through a local LLM.

### Production-Minded

Data quality, model health, scoring coverage, and AI availability are explicitly monitored.

---

# 🏆 Final Project Summary

**Netflix Customer Intelligence & AI Analytics Copilot** demonstrates how modern data science and AI can be combined to build a practical customer intelligence platform.

The system can:

- Clean customer data
- Validate data relationships
- Engineer behavioral features
- Predict churn
- Score customer risk
- Calculate revenue exposure
- Prioritize customers
- Recommend retention actions
- Generate Customer 360 profiles
- Provide executive analytics
- Answer business questions using AI
- Monitor data and AI readiness

---

# 🔗 Project Links

### GitHub Repository

**[View the complete project on GitHub](https://github.com/NandiniShilpkar/Netflix-Customer-Intelligence)**

### Project Documentation

**[Project Report]**

**[Project Showcase](https://netflix-customer-intelligence-tsmnzjnygbsxbgtbfbvcnd.streamlit.app/)**

---

# 📦 Version

```text
v1.0.0
```

Release:

**Netflix Customer Intelligence & AI Analytics Copilot v1.0.0**

---

# ⚠️ Disclaimer

This is an independent portfolio project created for educational and professional demonstration purposes.

It uses a project dataset and is **not affiliated with, endorsed by, or based on confidential production systems of Netflix**.

The financial figures, churn predictions, recovery scenarios, and customer profiles shown in the application are derived from the project dataset and should not be interpreted as actual Netflix business information.

---

# 👩‍💻 Author

**Nandini Shilpkar**

B.Tech — Computer Science & Engineering

Interested in:

- Data Science
- Machine Learning
- Artificial Intelligence
- Applied AI
- Business Analytics
- Customer Intelligence

---

<p align="center">

### ⭐ If you find this project interesting, consider starring the repository.

**Built with Python • PostgreSQL • Scikit-learn • Streamlit • Ollama • AI**

</p>
