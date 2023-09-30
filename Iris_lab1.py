import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
data=pd.read_csv('Iris.csv')

data.columns
data.info()
data.describe()
sns.heatmap(data.corr(),annot=True)

encoder=LabelEncoder()
data['Species']=encoder.fit_transform(data['Species'])

x=data.drop(['Species'],axis=1)
y=data['Species']

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.25,
                                               random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.coef_
regressor.intercept_
y_pred=regressor.predict(x_test)

np.sqrt(metrics.mean_squared_error(y_test,y_pred))
metrics.mean_absolute_error(y_test,y_pred)
metrics.r2_score(y_test,y_pred)
