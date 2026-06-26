# Telco Customer Churn Analysis & Prediction

**Data Science Internship Project — Saiket Systems (March 2026)**
Syed Nadimul Haque | Ref: SKS/A2/C110478

---

## What this project does

Customer churn — when someone stops using a service — is one of the most expensive problems in telecom. Acquiring a new customer costs 5–7 times more than keeping an existing one. This project builds a complete data science pipeline that identifies which customers are likely to leave before they actually do.

It covers everything from raw data cleaning and exploratory analysis through to customer segmentation and machine learning model comparison, with the goal of producing both data-driven insights and actionable business recommendations.

**Core question:** How can we analyse customer data to spot at-risk users, understand what drives churn, and build models that let a telecom company intervene in time?

---

## Dataset

The dataset is the IBM Telco Customer Churn Dataset — 7,043 customer records across 21 columns covering demographics, service subscriptions, billing, and contract information. The target variable is `Churn` (Yes/No). About 26.5% of customers churned, giving a class imbalance ratio of roughly 1:2.8.

Key features include tenure, monthly and total charges, contract type, payment method, internet service, online security, tech support, and demographic fields like gender, partner status, and dependents.

---

## What I did

### 1. Data Preparation

Started with 7,043 rows and 21 columns. `TotalCharges` was stored as text because of blank entries — converted it to numeric and filled 11 missing values (all from zero-tenure new customers) with the median. Dropped `customerID` since it has no predictive value. Encoded the target variable, applied one-hot encoding to all categorical columns (expanding from 20 to 30 features), and performed a stratified 80/20 train-test split to preserve the class ratio.

### 2. Exploratory Data Analysis

Explored the data through a series of visualisations to understand churn patterns:

- **Overall churn rate:** 26.54% — this is the baseline any model needs to beat.
- **Demographics:** Gender made almost no difference to churn. Customers without a partner or dependents churned significantly more — single, independent customers are the most at-risk group.
- **Tenure:** Churned customers averaged 18 months of tenure compared to 37.6 months for loyal ones. The first 12 months are clearly the critical retention window.
- **Contract type:** Month-to-month contracts had a ~43% churn rate, one-year contracts ~11%, and two-year contracts just ~3%.
- **Payment method:** Electronic check users churned at ~45% — nearly three times the rate of auto-pay users (~15–17%).

### 3. Customer Segmentation

Binned customers by tenure (0–1 yr, 1–2 yrs, 2–4 yrs, 4–6 yrs), monthly charge level (Low/Medium/High/Very High), and contract type. A churn rate heatmap across tenure and charge groups confirmed the pattern: the darkest cells (highest churn) consistently appeared at 0–1 year tenure with high or very high charges.

Filtered for high-value at-risk customers — those with high monthly charges, month-to-month contracts, who had already churned. That gave 1,186 customers averaging ~£68/month and just 16.2 months of tenure. These represent the highest revenue-loss segment: paying the most, least committed, and leaving early.

### 4. Churn Prediction Models

Applied StandardScaler for feature normalisation and used SMOTE on the training set only to balance the classes (from 4,139 vs 1,495 to 4,139 vs 4,139). Trained three models:

| Model | Accuracy | ROC-AUC | Churn Recall |
|---|---|---|---|
| Logistic Regression | ~79% | ~0.84 | ~0.80 |
| Decision Tree | ~76% | ~0.76 | ~0.75 |
| Random Forest | ~82% | ~0.87 | ~0.78 |

**Random Forest came out on top** — an AUC of 0.87 means it correctly ranks a churner above a non-churner 87% of the time.

---

## Key findings

1. Month-to-month contracts are the single strongest churn predictor (~43% vs ~3% for two-year).
2. New customers in their first 12 months are at highest risk.
3. Electronic check users churn at ~45% — nearly 3x the rate of auto-pay users.
4. Gender has virtually no effect on churn.
5. 1,186 high-value customers (avg ~£68/month) left within ~16 months on flexible contracts.
6. SMOTE significantly improved model recall by balancing training data.
7. Random Forest outperformed all models with AUC 0.87 and ~82% accuracy.

---

## Business recommendations

- **Focus on months 1–12.** Deploy proactive outreach, welcome calls, and loyalty perks for all new customers during this window.
- **Incentivise contract upgrades.** Moving month-to-month customers to a one-year contract cuts churn risk from 43% to 11%.
- **Migrate electronic check users to auto-pay** through targeted bill-pay discount campaigns.
- **Deploy the Random Forest model as a live early warning system.** Flag customers with predicted churn probability above 60% for immediate retention action.
- **Prioritise the 1,186 high-value at-risk segment.** Retaining a meaningful share of them could save approximately £303K in annual revenue.

---

## How to run

```bash
git clone https://github.com/SyedNadim123/telco-churn-analysis.git
cd telco-churn-analysis

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter

jupyter notebook SAIKET_INTERNSHIP.ipynb
```

Make sure `Telco_Customer_Churn_Dataset.csv` is in the same folder as the notebook.

---

## Tech stack

**Data processing:** pandas, numpy
**Visualisation:** matplotlib, seaborn
**Machine learning:** scikit-learn (Logistic Regression, Decision Tree, Random Forest)
**Class balancing:** imbalanced-learn (SMOTE)
**Evaluation:** accuracy, ROC-AUC, classification report, ROC curves

---

## Internship context

This was completed as part of a Data Science internship at Saiket Systems (ISO 9001:2015 certified), covering 4 of 6 prescribed tasks.

---

**Author:** Syed Nadimul Haque MSc Data Science | ML & AI Engineer
