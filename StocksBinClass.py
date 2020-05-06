import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np




MicrosoftB = pd.read_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\Microsoft.csv", index_col=[], header=0, parse_dates=['date'])



MicrosoftB['date'] = pd.to_datetime(MicrosoftB['date'], format="%d/%m/%y")
MicrosoftB['month'] =  MicrosoftB['date'].dt.month
MicrosoftB['day'] =  MicrosoftB['date'].dt.day
MicrosoftB['day_name'] =  MicrosoftB['date'].dt.day_name()
MicrosoftB['day_name']=MicrosoftB['day_name'].replace({'Monday': 1, 'Tuesday' : 2, 'Wednesday' : 3,'Thursday': 4, 'Friday':5}, regex=True)

MicrosoftB['Class'] = MicrosoftB['close'].pct_change() * 100
MicrosoftB.loc[MicrosoftB['Class'] < 0, 'target'] = 0
MicrosoftB.loc[MicrosoftB['Class'] > 0, 'target'] = 1

MicrosoftB.dropna(inplace=True)

Y=MicrosoftB['target']

X = MicrosoftB.iloc[:, [2,3,4,5,6,7,8]]

Xs= scale(X)
X_train = X[0:44]
X_test = X[45:]

Y_train=Y[0:44]
Y_test = Y[45:]



from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, Y_train)
predictionB = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, predictionB))
print("F1 Score:", metrics.f1_score(Y_test, predictionB, average="weighted"))
print("Confusion_matrics:", confusion_matrix(Y_test, predictionB))


Y_test.index = range(18)
plt.plot(predictionB)

plt.plot(Y_test)
