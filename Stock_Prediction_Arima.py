import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from pasta.augment import inline
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
from dateutil.parser import parse
from sklearn import  metrics
from sklearn.metrics import mean_squared_error
import numpy as np



prices = pd.read_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\NYSE\prices-split-adjusted.csv", index_col=[], header=0)
fundamentals = pd.read_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\NYSE\fundamentals.csv", index_col=[], header=0)



prices['date']=prices['date'].replace({'00:00:00': ''}, regex=True)
prices['date'] = pd.to_datetime(prices['date'], format="%Y-%m-%d")
prices['year'] =  prices['date'].dt.year
prices['month'] =  prices['date'].dt.month
prices['quarter'] = prices['date'].dt.quarter
prices['day'] =  prices['date'].dt.day
prices['day_name'] =  prices['date'].dt.day_name()

prices['day_name']=prices['day_name'].replace({'Monday': 1, 'Tuesday' : 2, 'Wednesday' : 3,'Thursday': 4, 'Friday':5}, regex=True)

prices_df = pd.get_dummies(data=prices, columns=['symbol'])

#del prices_df['date']

#datetime.strptime(prices['date'])


df = prices.iloc[:250, :11]
df['date']=df['date'].replace({'00:00:00': ''}, regex=True)
#del df['date']
df['daily_returns'] = df['close'].pct_change()
df['daily_returns'].dropna(inplace=True)
df['daily_returns']=df['daily_returns'].replace({'nan': 0}, regex=True)
df['quarter'] = df['date'].dt.quarter

df['symbol']=df['symbol'].replace({'WLTW': 1}, regex=True)



del df['symbol']
del df['open']
del df['low']
del df['high']
del df['volume']
del df['daily_returns']
del df['quarter']
del df['month']
df['date']

train = df[0:225]
test = df[225:]

from sklearn.preprocessing import scale

train_data = scale(train)
test_data = scale(test)


model= ARIMA(train_data,order=(9,2,1))
model_fit = model.fit()

print(model_fit.aic)

prediction = model_fit.forecast(steps = 25)[0]
print(prediction)

plt.plot(test,color='red')
plt.plot(prediction)
RidgeCV(SelectFromModel(input_matrix, max_features=0.7000000000000001, n_estimators=100, threshold=0.15000000000000002))

plt.plot(prediction)
plt.plot(test_data)

from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(train, model=’multiplicative’)
fig = result.plot()
plot_mpl(fig)

 import itertools
 p=d=q = range(0,5)
 pdq=list(itertools.product(p,d,q))
 pdq


 import warnings
 warnings.filterwarnings('ignore')
 for parameters in pdq:
    try:
         model = ARIMA(train, order=parameters)
         model_fit = model.fit()
          best = (parameters,model_fit.aic)

     except:
         continue
dr = df[['daily_returns']]
dr.rolling(12).mean().plot(figsize=(10,10), linewidth=5, fontsize=10)
plt.xlabel('daily', fontsize=10);


from sklearn.metrics import f1_score

print("F1 Score:", metrics.f1_score(test, prediction, average="weighted"))

plt.title('Stocks')
plt.xlabel('Year')
plt.ylabel('Close')
plt.show()
df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('date', fontsize=20);
