import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile

train = pd.read_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\training.csv", index_col=[], header=0)
test = pd.read_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\testing.csv", index_col=[], header=0)


train['TransactionId']=train['TransactionId'].replace({'TransactionId_': ''}, regex=True)
train['BatchId']=train['BatchId'].replace({'BatchId_': ''}, regex=True)
train['AccountId']=train['AccountId'].replace({'AccountId_': ''}, regex=True)
train['SubscriptionId']=train['SubscriptionId'].replace({'SubscriptionId_': ''}, regex=True)
train['CustomerId']=train['CustomerId'].replace({'CustomerId_': ''}, regex=True)
train['ProviderId']=train['ProviderId'].replace({'ProviderId_': ''}, regex=True)
train['ProductId']=train['ProductId'].replace({'ProductId_': ''}, regex=True)
train['ChannelId']=train['ChannelId'].replace({'ChannelId_': ''}, regex=True)
train['CurrencyCode']=train['CurrencyCode'].replace({'UGX': 1}, regex=True)
train['TransactionStartTime']=train['TransactionStartTime'].replace({'T': ' ', 'Z':''}, regex=True)


test['TransactionId']=test['TransactionId'].replace({'TransactionId_': ''}, regex=True)
test['BatchId']=test['BatchId'].replace({'BatchId_': ''}, regex=True)
test['AccountId']=test['AccountId'].replace({'AccountId_': ''}, regex=True)
test['SubscriptionId']=test['SubscriptionId'].replace({'SubscriptionId_': ''}, regex=True)
test['CustomerId']=test['CustomerId'].replace({'CustomerId_': ''}, regex=True)
test['ProviderId']=test['ProviderId'].replace({'ProviderId_': ''}, regex=True)
test['ProductId']=test['ProductId'].replace({'ProductId_': ''}, regex=True)
test['ChannelId']=test['ChannelId'].replace({'ChannelId_': ''}, regex=True)
test['CurrencyCode']=test['CurrencyCode'].replace({'UGX': 1}, regex=True)
test['TransactionStartTime']=test['TransactionStartTime'].replace({'T': ' ', 'Z':''}, regex=True)


#Conversion of columns

train['TransactionId'] = train['TransactionId'].astype(int)
train['BatchId']=train['BatchId'].astype(int)
train['AccountId']=train['AccountId'].astype(int)
train['SubscriptionId']=train['SubscriptionId'].astype(int)
train['CustomerId']=train['CustomerId'].astype(int)
train['ProviderId']=train['ProviderId'].astype(int)
train['ProductId']=train['ProductId'].astype(int)
train['ChannelId']=train['ChannelId'].astype(int)


train['TransactionStartTime'] = pd.to_datetime(train['TransactionStartTime'], format="%Y-%m-%d %H:%M:%S")
train['TransactionStartTime'] = pd.to_datetime(train['TransactionStartTime'], format="%Y-%m-%d %H:%M:%S")
train['year'] =  train['TransactionStartTime'].dt.year
train['month'] =  train['TransactionStartTime'].dt.month
train['day'] =  train['TransactionStartTime'].dt.day
train['day_name'] =  train['TransactionStartTime'].dt.day_name()
train['hour'] =  train['TransactionStartTime'].dt.hour
train['minute'] =  train['TransactionStartTime'].dt.minute
train['second'] =  train['TransactionStartTime'].dt.second

train['day_name']=train['day_name'].replace({'Monday': 1, 'Tuesday' : 2, 'Wednesday' : 3,'Thursday': 4, 'Friday':5,'Saturday':6, 'Sunday':7}, regex=True)

test['TransactionId'] = test['TransactionId'].astype(int)
test['BatchId']=test['BatchId'].astype(int)
test['AccountId']=test['AccountId'].astype(int)
test['SubscriptionId']=test['SubscriptionId'].astype(int)
test['CustomerId']=test['CustomerId'].astype(int)
test['ProviderId']=test['ProviderId'].astype(int)
test['ProductId']=test['ProductId'].astype(int)
test['ChannelId']=test['ChannelId'].astype(int)

test['TransactionStartTime'] = pd.to_datetime(test['TransactionStartTime'], format="%Y-%m-%d %H:%M:%S")
test['year'] =  test['TransactionStartTime'].dt.year
test['month'] =  test['TransactionStartTime'].dt.month
test['day'] =  test['TransactionStartTime'].dt.day
test['day_name'] =  test['TransactionStartTime'].dt.day_name()
test['hour'] =  test['TransactionStartTime'].dt.hour
test['minute'] =  test['TransactionStartTime'].dt.minute
test['second'] =  test['TransactionStartTime'].dt.second

test['day_name']=test['day_name'].replace({'Monday': 1, 'Tuesday' : 2, 'Wednesday' : 3,'Thursday': 4, 'Friday':5,'Saturday':6, 'Sunday':7}, regex=True)


print(train.head())
train.info()



train_df = pd.get_dummies(data=train, columns=['ProductCategory'])
test_df = pd.get_dummies(data=test, columns=['ProductCategory'])


train_merged = pd.merge(train, train_df)
train_merged['ProductCategory_retail']=0
del train_merged['TransactionStartTime']
del train_merged['ProductCategory']
del train_merged['second']

#scaler = StandardScaler()
#train_df = scaler.fit_transform(train_df)

test_merged = pd.merge(test, test_df)
test_merged['ProductCategory_other']=0
test_merged['FraudResult']=0
del test_merged['TransactionStartTime']
del test_merged['ProductCategory']
del test_merged['second']


x = train_merged[['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
       'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ChannelId',
       'Amount', 'Value', 'PricingStrategy', 'year', 'month',
       'day', 'day_name', 'hour', 'minute', 'ProductCategory_airtime',
       'ProductCategory_data_bundles', 'ProductCategory_financial_services',
       'ProductCategory_movies', 'ProductCategory_other',
       'ProductCategory_ticket', 'ProductCategory_transport',
       'ProductCategory_tv', 'ProductCategory_utility_bill',
       'ProductCategory_retail']]

x_tests=test_merged[['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
       'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ChannelId',
       'Amount', 'Value', 'PricingStrategy', 'year', 'month',
       'day', 'day_name', 'hour', 'minute', 'ProductCategory_airtime',
       'ProductCategory_data_bundles', 'ProductCategory_financial_services',
       'ProductCategory_movies', 'ProductCategory_other',
       'ProductCategory_ticket', 'ProductCategory_transport',
       'ProductCategory_tv', 'ProductCategory_utility_bill',
       'ProductCategory_retail']]


target = train_merged['FraudResult']
y=target

#y_test=test_merged['FraudResult']

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.fit_transform(x_test)

#from xgboost import XGBClassifier, plot_importance
model = XGBClassifier()
model.fit(x_train, y_train)
from xgboost import plot_importance
plot_importance(model)

#model = RandomForestClassifier(SelectPercentile (LogisticRegression(C=0.0001, dual=False, penalty=11), percentile=5), bootstrap=False, criterion='entropy', max_features=0.6500000000000001, min_samples_leaf=1, min_samples_split=8, n_estimators=2)
#model.fit(x_train,target)

prediction = model.predict(x_test)

#print("F1 Score:", metrics.f1_score(y_test, prediction, average="weighted"))
print("Confusion_matrics:", confusion_matrix(y_test, prediction))

df = pd.DataFrame(prediction)
#df.to_csv(r"C:\Users\Wilson.Ramapuputla\Desktop\ML\Ressyyyy.csv",index=False)

