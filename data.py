import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/Aliredafadel1/salary-prediction/main/ds_salaries.csv"
df = pd.read_csv(url)
print(df.head())
pd.set_option('display.max_columns', 50)