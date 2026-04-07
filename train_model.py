import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from  sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#first step we load the data

df = pd.read_csv("data/raw/ds_salaries.csv")
pd.set_option("display.max_columns", 50)
print("First 5 rows:")
print(df.head())
df.info()
print("\nShape before cleaning:", df.shape)

#second step we clean the data
# Remove rows with missing values in the 'salary_in_usd' column
df = df.dropna(subset=['salary_in_usd'])
df = df.drop_duplicates()
print("\nShape after dropping duplicates:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nColumns:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
# in third step we choose the target "salary_in_usd" because it is the variable we want to predict, and the features are all the other columns that can help us make that prediction.
target_column = "salary_in_usd"
# we put all features that affect the salary in a list  we remove uncessary columns , x is for input and y is for target output
feature_columns = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size"
]

df = df[feature_columns + [target_column]]

X = df[feature_columns]
y = df[target_column]
# we preprocess the data by crtureseating a pipeline that handles both numerical and categorical features. For numerical features, we impute missing values using the median. For categorical features, we impute missing values using the most frequent value and then apply one-hot encoding to convert them into a format suitable for machine learning algorithms.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_features= ["work_year", "remote_ratio"]
categorical_features = [ "experience_level", "employment_type", "job_title", "employee_residence", "company_location", "company_size"]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
#we use pipeline because same preprosessing steps will be aapplied automatically ,clean ,less bugs 
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    ))
])

# now we split our data into training and testing sets to evaluate the performance of our model on unseen data. We use 80% of the data for training and 20% for testing.
#we work with different data to stimulaee real world scenario where we have to predict on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#evaluate to know how good our model is
# 10) Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

joblib.dump(model, "salary_prediction_model.pkl")
print("\nModel saved as salary_prediction_model.pkl")