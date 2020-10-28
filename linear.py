from sklearn import linear_model
import pandas as pd
import data as dt

data_train = pd.read_csv('insurance.csv')
data_train_labels = pd.read_csv('label.csv')
header = list(data_train.columns)

# data prep
X_train_data = dt.to2D_List(data_train, header)
X_train_labels = data_train_labels['charges'].tolist()

# print(X_train_data)
# liner regretion
reg = linear_model.LinearRegression()
reg.fit(X_train_data, X_train_labels)

pred = reg.predict([[30, 1, 31.07, 0, 0, 3]])
print(pred)
