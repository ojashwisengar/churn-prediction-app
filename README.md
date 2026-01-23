**Customer Churn Prediction (XGBoost)**

**Business Problem** **-**
Customer churn leads to revenue loss. The goal of this project is to identify customers who are likely to leave so that retention teams can take action early. Missing a churner is more costly than a false alarm, so we prioritize high recall.

**Solution** **-**
I built a machine learning classification pipeline using:
1. Logistic Regression
2. Random Forest
3. XGBoost (final model)

After threshold tuning and hyperparameter optimization, the final XGBoost model achieves:
1. Recall (churn): ~82%
2. Precision (churn): ~46%
3. ROC-AUC: ~0.81
This significantly reduces missed churners compared to baseline models.

**Features Used**
1. tenure
2. MonthlyCharges
3. TotalCharges
4. Contract type
5. Internet services
6. Payment method
7. Security & support services
and more…

**Tech Stack**
1. Python
2. Pandas, NumPy
3. Scikit-learn
4. XGBoost
5. Streamlit (deployment)

**Web App**

Try the live app here:

https://churn-prediction-app-ojashwi.streamlit.app/

**How to Run Locally**

pip install -r requirements.txt
streamlit run app.py

Author

Ojashwi Sengar

Aspiring Data Scientist / ML Engineer
