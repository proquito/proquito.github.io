
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
import pandas as pd


data = pd.read_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\wealthNB.csv", index_col=[], header=0)

print(data.head(5))
print(list(data.columns))

print(data.info())
data[data.WorkClass != '?']

print(list(data.columns))

features = "Age"#, "WorkClass"#, ' fnlwgt', ' Education', ' Education-num', ' Marital-status', ' Occupation', ' Relationship', ' Race', ' Sex', ' Capital-gain', ' Capital-loss', ' Hours', ' Country'
target = " Income"

data_2 = data[['Age', ' WorkClass', ' fnlwgt', ' Education-num', ' Marital-status', ' Occupation', ' Relationship', ' Race', ' Sex', ' Hours', ' Country']]


y = data[target]


x = data.iloc[:, [0,1,3,5,6,8,9,10,12,13]]

xs = scale(x)

print(y)

#print(x)

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

xs_train, xs_test, y_train, y_test = train_test_split(xs, y, test_size = 0.1, random_state=0)

model = LogisticRegression()
model.fit(xs_train, y_train)

prediction = model.predict(xs_test)


print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("F1 Score:", metrics.f1_score(y_test, prediction, average="weighted"))
print("Confusion_matrics:", confusion_matrix(y_test, prediction))
