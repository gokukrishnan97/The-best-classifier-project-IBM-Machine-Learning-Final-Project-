import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")

#take a look at the data set
print (df.head())

# summarize the data
print (df.describe())

cdf= df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print (cdf.head(9))

'''
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()'''

#Train data resdistribution
msk= np.random.rand(len(df)) < 0.8
train= cdf [msk]
test= cdf [~msk]

from sklearn.preprocessing import PolynomialFeatures
from sklearn import  linear_model
train_x =np.asanyarray(train[['ENGINESIZE']])
train_y =np.asanyarray(train[['CO2EMISSIONS']])
test_x =np.asanyarray(test[['ENGINESIZE']])
test_y =np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly= poly.fit_transform(train_x)
print (train_x_poly)

clf= linear_model.LinearRegression()
train_y=clf.fit(train_x_poly,train_y)
#The coeffecients
print('Coeffecients:', clf.coef_)
print('Intercept:', clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0,0.1)
yy= clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]* np.power(XX, 2)
pl.plot(XX, yy, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

from sklearn.metrics import r2_score

test_x_poly= poly.fit_transform(test_x)
test_y_= clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )