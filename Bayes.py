
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\wealthNB.csv", index_col=[], header=0)

target = " Income"
y = data[target]
x = data.iloc[:, [0,2,4,5,6,9,10,11,12]]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)



#model = GaussianNB()
#model.fit(x_train, y_train)


data = pd.DataFrame(data=data)



from xgboost import XGBClassifier, plot_importance
model = XGBClassifier()
model.fit(x_train, y_train)
from xgboost import plot_importance
plot_importance(model)

prediction = model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("F1 Score:", metrics.f1_score(y_test, prediction, average="weighted"))
print("Confusion_matrics:", confusion_matrix(y_test, prediction))

#plt.show()