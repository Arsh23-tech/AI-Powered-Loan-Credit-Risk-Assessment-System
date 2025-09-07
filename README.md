# AI-Powered-Loan-Credit-Risk-Assessment-System

## 📖 Project Overview

The AI-Powered Credit Risk Assessment System is a machine learning–based dashboard that predicts the risk of loan default for credit applicants.
It combines LightGBM modeling, SHAP explainability, and an interactive Streamlit frontend to generate risk scores, classify applicants into Low / Medium / High risk levels, and provide AI-generated insights for decision-making.

This project demonstrates how financial institutions can leverage AI for credit risk management while ensuring transparency and interpretability.

## 🚀 Features

🧠 Machine Learning Model (LightGBM) trained on applicant data.

📊 Risk Scoring System (0–1000 scale) with thresholds for Low, Medium, and High risk.

🔎 Explainable AI with SHAP: Shows key factors influencing individual predictions.

📝 AI-Generated Risk Reports: Pre-written summaries tailored to each risk band (Low / Medium / High).

🎛 Interactive Dashboard built with Streamlit for real-time applicant evaluation.

📂 Modular Project Structure (src/, models/, data/, notebooks/) for maintainability.

## 🧩 How It Works

1. Input Applicant Information: Age, credit limit, bill amount, education, marital status, etc.

2. Model Prediction: LightGBM predicts the probability of default.

3. Risk Score Generation: Converts prediction to a score (0–1000).

4. Risk Band Classification:

   * Low Risk → Stable applicant, likely to repay.

   * Medium Risk → Moderate probability of default, requires closer review.

   * High Risk → High chance of default, lending is risky.

5. Explainability with SHAP: Shows top features influencing risk.

6. AI Report: Displays pre-written insights tailored to the applicant’s risk band.

## 📊 Example Outputs

* Risk Score: 514/1000
* Risk Level: Medium
* Top Factors: Credit Limit ↓, Age ↑, Bill Amount ↑
* AI Report:
     The applicant demonstrates moderate repayment capacity, supported by a stable credit limit and consistent payment history.
     However, high outstanding bills and credit utilization close to the limit raise concerns about short-term liquidity pressure.
     While the risk of default is not immediate, the profile suggests potential stress under adverse financial conditions.
     Recommendation: Conduct a closer review of recent payment patterns and request additional supporting documents (e.g., income proof, assets) before loan approval.
  
## 📈 Future Improvements

* Integration with real-time applicant databases.

* Use of OpenAI API or LangChain for fully dynamic AI-generated risk reports.

* Deployment on AWS / Azure for production.

* Expanded feature engineering and additional models for comparison.

## 👨‍💻 Tech Stack

* Python

* LightGBM (modeling)

* SHAP (explainability)

* Streamlit (frontend)

* Pandas, NumPy, Scikit-learn (data handling)

## 📬 Contact

Author: Arsh Chandrakar

📧 Email: arshchand012@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/arsh-chandrakar/
