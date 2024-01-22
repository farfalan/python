import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import norm

mu, sigma = 15, 1
x = np.linspace(0,1,1000)
y = np.linspace(mu,mu,1000)
y2 = y + np.random.normal(0,sigma,len(x))
y_class  = np.where(mu-y2 < 0,False,True)

X_train,X_test, y_train,y_test = train_test_split(y2.reshape(-1,1),y_class,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train.reshape(-1,1),y_train)

y_pred = model.predict(y2.reshape(-1,1))

fig, ax = plt.subplots()
y3 = np.linspace(0,1,len(X_test[y_test == True]))
y4 = np.linspace(0,1,len(X_test[y_test == False]))

ax.scatter(y3,X_test[y_test == True],c='blue', marker='o')
ax.scatter(y4,X_test[y_test == False],c='red', marker='x')
ax.plot(x,y)
plt.show()

