import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import metrics



data = pd.read_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\wealthNB.csv", index_col=[], header=0)
target = " Income"
y = data[target]
x = data.iloc[:, [0,2,4,5,6,9,10,11,12]]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.25, min_samples_leaf=6, min_samples_split=15, n_estimators=100)
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("F1 Score:", metrics.f1_score(y_test, prediction, average="weighted"))
print("Confusion_matrics:", confusion_matrix(y_test, prediction))
