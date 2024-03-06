import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn import svm


heart_disease = fetch_ucirepo(id=45)
X = np.array(heart_disease.data.features)
Y = np.array(heart_disease.data.targets)


#Imput Data
imputer = SimpleImputer(missing_values = np.nan, strategy="most_frequent")
imputer = imputer.fit(X)
X= imputer.transform(X)

#only most 3 correlated Features with the Target = cp;ca;thal
myNewMatrix = X[:,[2,11,12]]

#make class 0 and 1 to 0 and 3 to 4 to have 0 or 4 as targets.              
for i in range(len(Y.ravel())):
    if Y[i]==1:
        Y[i] = 0
    elif Y[i]==2:
        Y[i]=0
    elif Y[i] == 3:
        Y[i]= 4   

#reduce to 2 Diemensions 
pca = PCA(n_components=2)
pca.fit(myNewMatrix)
X_pca = pca.transform(myNewMatrix)

                
for i in range(len(Y.ravel())):
    if Y[i]==1:
        Y[i] = 0
    elif Y[i]==2:
        Y[i]=0
    elif Y[i] == 3:
        Y[i]= 4   

X_train,X_test,y_train,Y_Test = train_test_split(X_pca,Y,test_size= 0.1,random_state= 42)


# SVM linear fit
svm_clf = LinearSVC(C=10)
svm_clf.fit(X_train, y_train)

#accuracy
y_pred = svm_clf.predict(X_test)
print("y_test:",Y_Test.ravel())
print("y_pred:",y_pred)
accuracy = accuracy_score(Y_Test,y_pred)
print("accuracy:",{accuracy})

fig, ax = plt.subplots()

# Get the coefficients (normal vector)
w = svm_clf.coef_[0]
intercept = svm_clf.intercept_

# Compute the slope for 2D visualization
slope = w[0] / w[1]
colors = {0:'g',1:'orange',2:'darkorange',3:'r',4:'darkred'}

#plot (x_pca,Y)
for i in range(-1,len(X)-1):

    noise = np.random.uniform(0,0.5,1)
    X01Noised = X_pca[i+1:i+2,0:1] + noise
    
    noise = np.random.uniform(0,0.5,1)
    X23Noised = X_pca[i+1:i+2,1:2] + noise 
    
    plt.plot(X01Noised,X23Noised,c=colors[Y[i+1:i+2].item()], marker ='o')
    
# Plot the decision boundary
x_points = np.linspace(-3, 3)
for i  in range(1):
    w= svm_clf.coef_[[i],[0,1]]
    b= svm_clf.intercept_[i]
    y_points = (w[0] / w[1]) * x_points + b / w[1] 
    plt.plot(x_points,y_points)
        
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
