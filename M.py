import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=None)
#
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)


xs= scale(x)
x_train = xs[0:44]
x_test = xs[45:]

y_train=y[0:44]
y_test = y[45:]

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoLarsCV

from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor()


#rf = LassoLarsCV(normalize=False)
rf = RidgeCV()
model=rf.fit(x_train,y_train)
predictions = rf.predict(x_test)
print(rf.score(x_train,y_train))

# print(rf.score(predictions,y_test))
# from xgboost import XGBRegressor
# models = XGBRegressor()
# models.fit(x_train, y_train)
# predictions = models.predict(x_test)

errors = abs(predictions - y_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.DataFrame()
df['actual'] = y_test
df['prediction'] = predictions
df['diff'] = np.abs(df['actual'] - df['prediction'])
df['%'] = (predictions / y_test) *100

from sklearn.metrics import mean_squared_error


print(pd.Series(rf.coef_, index = x.columns))
print(df['diff'].mean())
print(mean_squared_error(df['actual'], df['prediction']))
print(r2_score(df['actual'], df['prediction'] ))



#movement= (y[44]-df['actual'][45],df['actual'][45]-df['actual'][46],df['actual'][46]-df['actual'][47],df['actual'][47]-df['actual'][48],df['actual'][48]-df['actual'][49],df['actual'][49]-df['actual'][50],df['actual'][50]-df['actual'][51],df['actual'][51]-df['actual'][52])


actuals_move = pd.DataFrame()
actuals_move['Actuals'] =df['actual'].pct_change() * 100
actuals_move.index = range(20)


actuals_move.loc[actuals_move['Actuals'] < 0, 'Up or Down?'] = 0
actuals_move.loc[actuals_move['Actuals'] > 0, 'Up or Down?'] = 1


predictions_move =  pd.DataFrame()
predictions_move['prediction'] = df['prediction'].pct_change() * 100
predictions_move.index = range(20)

predictions_move.loc[predictions_move['prediction'] <0, 'Up or Down?'] = 0
predictions_move.loc[predictions_move['prediction'] > 0, 'Up or Down?'] = 1
8


plt.plot(predictions_move['Up or Down?'])
plt.plot(actuals_move['Up or Down?'])


y_test.index = range(20)
plt.plot(predictions)

plt.plot(y_test)