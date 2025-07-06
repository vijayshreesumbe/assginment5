🏠 Predicting House Sale Prices with Machine Learning
This project is based on the Kaggle competition House Prices: Advanced Regression Techniques, where the goal is to predict house sale prices based on various property features.

📦 Dataset
train.csv: 1460 rows × 81 columns (includes SalePrice)

test.csv: 1459 rows × 80 columns (no SalePrice)

Download from the Kaggle competition page

📊 Steps Followed
✅ 1. Data Loading
Read train.csv and test.csv using pandas

✅ 2. Data Cleaning
Handled missing values:

Filled categorical columns with 'None'

Filled numerical columns with 0 or group-wise medians

Applied mode/median for other features

✅ 3. Feature Engineering
Created new features to improve model performance:

TotalSF = Total livable area (Basement + 1st + 2nd floors)

TotalBath = Total number of bathrooms (weighted)

HouseAge = Age of house at time of sale

SinceRemod = Time since last remodel

IsRemodeled = 1 if remodeled, else 0

HasPool, Has2ndFlr = Binary indicators

✅ 4. Encoding Categorical Variables
One-Hot Encoding for nominal features

Label Encoding for ordinal quality features (e.g., Ex, Gd, TA)

✅ 5. Target Transformation
Log-transformed the target variable SalePrice using np.log1p() to reduce skewness

✅ 6. Model Building
Trained two models:

LinearRegression

RandomForestRegressor (n_estimators=100)

✅ 7. Model Evaluation
Used RMSE and R² Score on a validation split (20%)

Visualized feature importances for Random Forest

✅ 8. Prediction & Submission
Applied preprocessing to test set

Predicted SalePrice, applied np.expm1() to reverse log transform

Created submission.csv for Kaggle upload

🧠 Libraries Used
bash
Copy
Edit
pandas
numpy
scikit-learn
seaborn
matplotlib
🏁 To Run the Project
Download the data from Kaggle

Run the notebook or Python script in this order:

Data loading

Cleaning

Feature engineering

Encoding

Model training

Prediction

Save submission.csv

📝 Future Improvements
Try XGBoost or LightGBM for better accuracy

Perform feature selection or dimensionality reduction

Hyperparameter tuning (e.g., with GridSearchCV)
