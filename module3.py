# Module 3: Linear regression

# New imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Code after this


bikes_df = pd.read_csv('data/bikes_subsampled.csv')
temperature = bikes_df['temperature'].values
bikes_count = bikes_df['count'].values

plt.scatter(temperature, bikes_count, color='k')
plt.xlabel("temperature")
plt.ylabel("bikes hired")
# plt.tight_layout()
# plt.show()

a = 28
temperature_predict = np.expand_dims(a=np.linspace(-5,40,100),axis=1)
bikes_count_predict = a*temperature_predict
# plt.scatter(temperature, bikes_count, color='k')
plt.plot(temperature_predict,bikes_count_predict,linewidth=2)
plt.show()

### next ... a prediction

linear_regression = LinearRegression()
temperature_ = np.expand_dims(temperature,1)
linear_regression.fit(temperature_, bikes_count)

print 'Bikes hired at 5 defrees Celsius:', linear_regression.predict(5.)[0]
# value that minimizes the RMS error
print 'Optimal slope:' , linear_regression.coef_[0]
print 'Optimal intercept:', linear_regression.intercept_