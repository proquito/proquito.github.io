import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import numpy as np



Microsoft = pd.read_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\Microsoft.csv", index_col=[], header=0, parse_dates=['date'])



Microsoft['date'] = pd.to_datetime(Microsoft['date'], format="%d/%m/%y")
Microsoft['month'] =  Microsoft['date'].dt.month
Microsoft['day'] =  Microsoft['date'].dt.day
Microsoft['day_name'] =  Microsoft['date'].dt.day_name()
Microsoft['day_name']=Microsoft['day_name'].replace({'Monday': 1, 'Tuesday' : 2, 'Wednesday' : 3,'Thursday': 4, 'Friday':5}, regex=True)

target = "close"
y=Microsoft[target]
x = Microsoft.iloc[:, [2,3,4,5,6,7,8]]


xs= scale(x)
x_train = x[0:54]
x_test = x[55:]

y_train=y[0:54]
y_test = y[55:]


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, n_jobs=-1, oob_score=True)
model=rf.fit(x_train, y_train)
predictions = rf.predict(x_test)
print(rf.score(predictions,y_test))
print(rf.oob_score_)
from sklearn.metrics import f1_score
from sklearn import metrics

from xgboost import XGBRegressor
models = XGBRegressor()
models.fit(x_train, y_train)
prediction = models.predict(x_test)


errors = abs(predictions - y_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.DataFrame()
df['actual'] = y_test
df['prediction'] = predictions
df['diff'] = np.abs(df['actual'] - df['prediction'])
df['%'] = (predictions / y_test) * 100

print(df['diff'].mean())


print(mean_absolute_error(y_test, predictions))
print(r2_score(df['actual'], df['prediction'] ))

from sklearn.metrics import mean_absolute_error

mean_absolute_error(df['actual'], df['prediction'])


mape = 100 * (errors / y_test)


# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

plt.plot(predictions)
plt.plot(prediction)
plt.plot(y_test)

#Microsoft.set_index('date', inplace=True)

from sklearn.model_selection import train_test_split
train_data = scale(train)
test_data = scale(test)

