import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.svm import LinearSVC



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

#model = LinearSVC(C=20.0, dual=True, loss="hinge", penalty="l2", tol=0.0001)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()

# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()

# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(max_depth=10, max_features=None, min_samples_leaf=15)

#
# from sklearn.neighbors import  KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=5)
#
# from sklearn.linear_model import RidgeClassifier
# model = RidgeClassifier()
#
# from sklearn.linear_model import SGDClassifier
#  model = SGDClassifier()


#model = SVC(kernel='linear')
 # from sklearn.linear_model import LogisticRegression
 # model = LogisticRegression()


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline, make_union
# from sklearn.svm import LinearSVC
# from tpot.builtins import OneHotEncoder, StackingEstimator
#
# # exported_pipeline = make_pipeline(
# #     OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
# #     StackingEstimator(estimator=LogisticRegression(C=25.0, dual=False, penalty="l1")),
# #     LinearSVC(C=20.0, dual=True, loss="hinge", penalty="l2", tol=0.0001))
# #
# #
# # exported_pipeline.fit(X_train, Y_train)
# # results = exported_pipeline.predict(X_test)

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