# 📉 Telco Customer Churn Analysis & Prediction
**Data Science Internship Project — Saiket Systems (2026)**
Intern: Syed Nadimul Haque  |  Ref: SKS/A2/C110478  |  Date: March 2026

## 📌 Project Overview
Customer churn — when a customer stops using a service — is one of the most costly problems for telecom companies. Acquiring a new customer costs 5–7× more than retaining an existing one. This project builds a complete end-to-end data science pipeline to **identify which customers are likely to churn** before they leave, using real-world telecom data.

The pipeline covers everything from raw data cleaning and exploratory analysis to customer segmentation and machine learning model comparison — providing both data-driven insights and actionable business recommendations.

## 🎯 Problem Statement
How can customer data be analysed to identify at-risk customers, uncover the key drivers of churn, and build predictive models that help a telecom company intervene before customers leave?

## 📊 Dataset
| Attribute | Detail |
|---|---|
| Source | IBM Telco Customer Churn Dataset |
| Total Records | 7,043 customers |
| Features | 21 columns — demographics, services, billing, contract info |
| Target Variable | Churn (Yes = churned, No = stayed) |
| Churned Customers | 1,869 (~26.54%) |
| Retained Customers | 5,174 (~73.46%) |
| Class Imbalance Ratio | ~1 Churned : 2.8 Retained |

Key features include: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, InternetService, OnlineSecurity, TechSupport, and demographic fields like gender, Partner, Dependents.

## ✅ Tasks Completed
| # | Task | Description | Status |
|---|---|---|---|
| 1 | Data Preparation | Loading, cleaning, encoding, train/test split | ✅ Complete |
| 2 | Exploratory Data Analysis | Churn rate, demographics, tenure, contracts, payment | ✅ Complete |
| 3 | Customer Segmentation | Tenure groups, charge groups, high-value at-risk customers | ✅ Complete |
| 4 | Churn Prediction Model | Logistic Regression, Decision Tree, Random Forest + evaluation | ✅ Complete |

*Note: 4 out of 6 tasks completed as per internship requirements.*

## 🛠️ Technologies & Libraries
| Category | Libraries |
|---|---|
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn (Logistic Regression, Decision Tree, Random Forest) |
| Class Balancing | imbalanced-learn (SMOTE) |
| Feature Scaling | StandardScaler |
| Evaluation | accuracy_score, roc_auc_score, classification_report, roc_curve |

## 📁 Repository Structure
telco-churn-analysis/
│
├── SAIKET_INTERNSHIP.ipynb          # Full end-to-end pipeline (all 4 tasks)
├── Telco_Customer_Churn_Dataset.csv # Dataset (7,043 customer records)
└── README.md                        # Project documentation

## 🔍 Methodology

### Task 1 — Data Preparation
**Goal:** Get the raw data into a clean, model-ready format.

Steps performed:
- Loaded 7,043 rows × 21 columns from CSV
- Discovered TotalCharges stored as object (text) due to blank entries — converted to numeric using `pd.to_numeric(..., errors='coerce')`
- Found 11 missing values in TotalCharges (new customers with 0 tenure) — filled with median to preserve all rows
- Dropped customerID — a unique identifier with zero predictive value
- Encoded target: Churn → Yes=1, No=0 → confirmed 26.54% churn rate
- Saved a readable copy (df_eda) before encoding for use in charts
- Applied one-hot encoding (`pd.get_dummies`) to all categorical columns → expanded from 20 to 30 features
- Performed stratified 80/20 train-test split (stratified to preserve class ratio)

**Result:** Train: 5,634 rows | Test: 1,409 rows — both sets with identical churn proportions.

### Task 2 — Exploratory Data Analysis (EDA)
**Goal:** Understand customer behaviour, identify patterns, and visualise churn drivers.

**Chart 1 — Overall Churn Rate**
- Pie chart + bar chart showing the churn split
- 26.54% of customers churned (1,869 out of 7,043)
- Established this as the baseline the models must improve upon

**Chart 2 — Churn by Demographics**

| Factor | Finding |
|---|---|
| Gender | Almost no difference — male vs female churn rates nearly identical |
| Partner | Customers without a partner churn significantly more |
| Dependents | Customers without dependents churn at a much higher rate |

*Insight: Single, independent customers are the most at-risk demographic.*

**Chart 3 — Tenure vs Churn**

| Group | Avg Tenure |
|---|---|
| No Churn | 37.6 months |
| Churned | 18.0 months |

*Insight: Churned customers had half the tenure of loyal ones. The first 12 months are the most critical window for retention.*

**Chart 4 — Contract Type & Payment Method**

| Contract Type | Churn Rate |
|---|---|
| Month-to-month | ~43% |
| One year | ~11% |
| Two year | ~3% |

| Payment Method | Churn Rate |
|---|---|
| Electronic check | ~45% (highest) |
| Mailed check | ~19% |
| Bank transfer (auto) | ~17% |
| Credit card (auto) | ~15% |

*Insight: Month-to-month contracts and electronic check payments are the two strongest churn indicators in the entire dataset.*

### Task 3 — Customer Segmentation
**Goal:** Divide customers into meaningful groups and identify those at highest risk.

Customers were binned into groups using `pd.cut()`:

| Dimension | Groups |
|---|---|
| Tenure | 0–1 Yr (2,175) · 1–2 Yrs (1,024) · 2–4 Yrs (1,594) · 4–6 Yrs (2,239) |
| Monthly Charges | Low ≤$35 · Medium $35–65 · High $65–95 · Very High >$95 |
| Contract Type | Month-to-month · One year · Two year |

**Key Segment Findings**
- 0–1 year tenure customers had the highest churn rate across all segments
- Very High charge customers on month-to-month contracts showed extreme churn risk
- Two-year contract customers had near-zero churn regardless of charge level

**High-Value At-Risk Customers**
Filtered customers meeting all three criteria: High/Very High charges + Month-to-month contract + Churned

| Metric | Value |
|---|---|
| Count | 1,186 customers |
| Avg Monthly Charges | ~£68 |
| Avg Tenure | 16.2 months |

These 1,186 customers represent the highest-revenue loss segment — paying the most, least committed, and leaving early — the priority target for any retention strategy.

**Churn Rate Heatmap**
A `pivot_table` + `sns.heatmap` mapped churn rate at every intersection of tenure group × charge group. The darkest cells (highest churn) consistently appeared at 0–1 Yr tenure + High/Very High charges — confirming the critical risk window.

### Task 4 — Churn Prediction Model
**Goal:** Train and compare machine learning models to predict which customers will churn.

**Pre-processing for ML**
- Feature scaling — StandardScaler normalised all features so MonthlyCharges doesn't dominate binary 0/1 columns
- SMOTE applied to training data only to balance the class imbalance

| | Before SMOTE | After SMOTE |
|---|---|---|
| Class 0 (No Churn) | 4,139 | 4,139 |
| Class 1 (Churned) | 1,495 | 4,139 |

SMOTE generates synthetic churner samples so the model learns equally from both classes. Applied to training data only — never the test set.

**Models Trained**

| Model | Configuration |
|---|---|
| Logistic Regression | class_weight='balanced', max_iter=1000 |
| Decision Tree | class_weight='balanced', max_depth=5 |
| Random Forest | class_weight='balanced', n_estimators=100 |

**Model Results**

| Model | Accuracy | ROC-AUC | Churn Recall |
|---|---|---|---|
| Logistic Regression | ~79% | ~0.84 | ~0.80 |
| Decision Tree | ~76% | ~0.76 | ~0.75 |
| **Random Forest ✅** | **~82%** | **~0.87** | **~0.78** |

**Winner: Random Forest** — highest AUC (0.87) and accuracy. An AUC of 0.87 means the model correctly ranks a churner above a non-churner 87% of the time.

## 🧠 Key Findings
| # | Finding |
|---|---|
| 1 | Month-to-month contracts are the single strongest predictor of churn (~43% rate vs ~3% for two-year) |
| 2 | New customers (0–12 months) are at highest risk — the critical retention window |
| 3 | Electronic check users churn at ~45% — nearly 3× the rate of auto-pay users |
| 4 | Gender has almost no effect on churn — demographic segmentation by gender adds minimal value |
| 5 | 1,186 high-value customers (avg ~£68/month) left within ~16 months on flexible contracts |
| 6 | SMOTE balanced training from 4,139 vs 1,495 to 4,139 vs 4,139 — significantly improving model recall |
| 7 | Random Forest outperformed all models with AUC 0.87 and ~82% accuracy |

## 💡 Business Recommendations
Based on the analysis and model findings:

- **Prioritise months 1–12** — deploy proactive outreach, welcome calls, and loyalty perks for all new customers in this window
- **Offer contract upgrade incentives** to month-to-month customers — moving them to a 1-year contract cuts churn risk from 43% → 11%
- **Migrate electronic check users to automatic payment** via targeted bill-pay discount campaigns
- **Deploy the Random Forest model as a live early warning system** — flag customers with predicted churn probability >60% for immediate retention action
- **Focus retention budget on the 1,186 high-value at-risk segment** — retaining a meaningful share of them delivers a retention strategy projected to save approximately **£303K in annual revenue**

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/SyedNadim123/telco-churn-analysis.git
cd telco-churn-analysis

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter

# 3. Open the notebook
jupyter notebook SAIKET_INTERNSHIP.ipynb
```
Make sure `Telco_Customer_Churn_Dataset.csv` is in the same folder as the notebook before running.

## 📚 Internship Context
| Field | Detail |
|---|---|
| Company | Saiket Systems — ISO 9001:2015 Certified |
| Domain | Data Science |
| Role | Data Science Intern |
| Reference No. | SKS/A2/C110478 |
| Date | March 2026 |

## 👤 Author
**Syed Nadimul Haque**
MSc Data Science | ML & AI Engineer | Software Engineer
