import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import datetime

def parser(X):
    return datetime.strptime(X,'%Y-%m')

var=pd.read_csv('C://Users/Gopi/Desktop/Time Series/sales-cars.csv',index_col=0,parse_dates=[0])

print(var.index)
print(var.head())

var.plot()
print(type(var))
var.plot(style='k.')

var_ma=var.rolling(window=100).mean()
var_ma.plot(figsize=(6,4))
plt.show()

#Intigrated of Order 1, denoted by 'd'(difference), one of the parameter in ARIMA model
final=var.diff(periods=1)
final=final[1:]
print(final.head())

print(var.shift(1).head())
print(final.shift(1).head())

# Implementing Base Line Model

var_base=pd.concat([var,var.shift(1)],axis=1)
print(var_base.head())

var_base.columns=['a','b']
print(var_base.columns)

var_base.dropna(inplace=True)

from sklearn.metrics import mean_squared_error

var_error=mean_squared_error(var_base.a,var_base.b)
print(var_error)

import numpy as np
print(np.sqrt(var_error))

# Train Test Division
X=var.values
train=X[0:27]
test=X[27:]
predections=[]

# Auto Regression Model(AR)

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

model_ar = AR(train)
model_ar_fit = model_ar.fit()

predections1=model_ar_fit.predict(start=27,end=36)

model = AR(train)
model_fit = model.fit()

predections1=model_ar_fit.predict(start=27,end=36)

plt.plot(test)
plt.plot(predections1,color='green')

# ARIMA MODEL

# p,d,q
#p = periods taken for AR model
#d = Integrated order, difference
#q=periods in moving average model

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_pacf(var)

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(var)

X=var.values
train=X[0:27]
test=X[27:]
predections=[]

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(train,order=(3,1,2))
model_fit=model.fit()
predict=model_fit.forecast(steps=9)[0]
print(predict)
print(test)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(test,predict)))

plt.plot(test)
plt.plot(predict,color='red')

import warnings
warnings.filterwarnings("ignore")

p_values=range(0,2)
d_values=range(0,1)
q_values=range(0,2)

for p in p_values:
    for d in d_values:
        for q in q_values:
            order= (p,d,q)
            train,test=train,test
            predections= list()
            for i in range(len(test)):
                try:
                    model = ARIMA(train,order)
                    model_fit = model.fit(disp=0)
                    pred_y = model_fit.forecast()[0]
                    predections.append(pred_y)
                    error=mean_squared_error(test,predections)
                    print('ARIMA%s RMSE = %.2f'%(order,error))
                except:
                    continue