#libaries used
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from pandas import concat
from  sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR


#Reading data
dftest = read_csv('datatest.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
dftraining = read_csv('datatraining.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
dftest2 = read_csv('datatest2.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
dataset = concat([dftest, dftraining, dftest2])
values = dataset.values
#data split into training and test
X=values[:, :-1]
y =values[:, -1]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=1)
train_Time=dataset.iloc[:len(dataset)-12]
test_Time=dataset.iloc[len(dataset)-12:]
#ARIMA model testing
arima = ARIMA(train_Time['Occupancy'], order=(2,1,0))
arima_fit = arima.fit()
print(arima_fit.summary())
#AR model testing
model = AutoReg(train_Time['Occupancy'], lags=1)
model_fit = model.fit()
print(model_fit.summary())
#VAR model testing
model = VAR(endog=train_Time)
model_fit = model.fit(4)
print(model_fit.summary())
#linear regression testing
linear_regression= LinearRegression()
linear_regression.fit(train_x, train_y)
predictionlinear = linear_regression.predict(test_x)
print(r2_score(test_y,predictionlinear))
#Logistic regression testing
Logregression = LogisticRegression()
Logregression.fit(train_x, train_y)
prediction = Logregression.predict(test_x)
print(classification_report(test_y, prediction,))
#logistic regression feature selection
print(dataset.columns)
featureslogistic=[0,1,2,3,4,5]
for features in featureslogistic:
   X, y = values[:, features].reshape((len(values), 1)), values[:, -1]
   train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=1)
   model = LogisticRegression()
   model.fit(train_x, train_y)
   yhat = model.predict(test_x)
   score = accuracy_score(test_y, yhat)
   print(score)






