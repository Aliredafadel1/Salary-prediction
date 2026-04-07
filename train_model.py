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
# we preprocess the data by creating a pipeline that handles both numerical and categorical features. For numerical features, we impute missing values using the median. For categorical features, we impute missing values using the most frequent value and then apply one-hot encoding to convert them into a format suitable for machine learning algorithms.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)




