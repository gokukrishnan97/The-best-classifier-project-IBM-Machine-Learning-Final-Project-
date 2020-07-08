import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import csv


df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")

#take a look at the data set
print (df.head())

# summarize the data
print (df.describe())

cdf= df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print (cdf.head(9))

'''
viz= cdf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()'''
'''
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()'''
'''
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()'''

#Train data resdistribution
msk= np.random.rand(len(df)) < 0.8
train= cdf [msk]
test= cdf [~msk]
'''
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()'''

#Modelling
from sklearn import  linear_model
regr= linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x,train_y)

#coeffecients
print ('Coeffecients:', regr.coef_)
print ('Intercept:', regr.intercept_)
'''
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]* train_x+ regr.intercept_[0], '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()'''

from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat= regr.predict(test_x)

print ("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat-test_y)))
print ("Residual Sum of squares: %2f" % np.mean(test_y_hat-test_y) ** 2)
print ("R2-score: %2f" % r2_score(test_y_hat, test_y))