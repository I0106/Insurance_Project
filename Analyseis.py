import pandas as pd

# read the dataset into a dataframe
df = pd.read_csv('insurance.csv')
print(df.head(10))

# the max, min and default value of age

max_value = df['age'].max()
min_value = df['age'].min()
default_value = df['age'].mean()

# the max, min and default value of bmi

max_value = df['bmi'].max()
min_value = df['bmi'].min()
default_value = df['bmi'].mean()

# the max, min and default value of children

max_value = df['children'].max()
min_value = df['children'].min()
default_value = df['children'].mean()


# shape of df
df.shape

# info
df.info()

# summary statistics
df.describe()

# unique
df.nunique()

# plot summary statistics
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df)
plt.show()

df.head(3)

# label encoding of sex
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])


df.head()

# label encoding of sex
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['smoker'] = le.fit_transform(df['smoker'])

df.head()


# label encoding of sex
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['region'] = le.fit_transform(df['region'])

df.head(10)

# train a linear regression model that will predict charges
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# model creation
model = LinearRegression()

# identify features & label
X = df.drop('charges', axis=1)
y = df['charges']

print(X.dtypes)

# an example from X
X.head(1)

# an example from y
y.head(1)

# splitting in a 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train.head()

# model training
model.fit(X_train, y_train)

# model prediction
y_pred = model.predict(X_test)
# model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# a single testing instance
new_data = X_test.iloc[0]
new_data

# predict charges for new_data
model.predict([new_data])

print(df.iloc[764])

# Save our model
import joblib
joblib.dump(model, "insurance.pkl")

# donwload the file
import os
file = os.path.abspath("insurance.pkl")
