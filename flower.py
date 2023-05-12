from sklearn.datasets import load_iris  #Load dataset
from sklearn.model_selection import train_test_split    #Use for split X,Y train and test
from sklearn.neighbors import KNeighborsClassifier  # Determine Model KNN
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay  # View confusion matrix Var&Cov
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt

def data_Info(X,Y,X_train,X_test):
    print(f'X shape={X.shape} and Y={Y.shape}')
    print('----------------------------------')
    print(f'X Data number is= {len(X)} \n Y [25:75:3] Data are= {Y[25:75:3]}')
    print('----------------------------------')
    print(f' X_Train shape={X_train.shape} \t X_Test={X_test.shape}')
    print('----------------------------------')


DF=load_iris()
Y=DF.target
X=DF.data

X_train,X_test,Y_train,Y_test= train_test_split(X,Y) # split data default 70%Train 30%Test

Scaler = MinMaxScaler()
X_train = Scaler.fit_transform(X_train)
X_test  = Scaler.transform(X_test)

print(np.cov(X.T))  # check Cov matrix
print(np.linalg.matrix_rank(np.cov(X.T))) #check Rank

model = KNeighborsClassifier(n_neighbors=3) #determin K number=3
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

print('Ypred:')
print(Y_pred)
print('Ytest:')
print(Y_test)
print(confusion_matrix(Y_test,Y_pred))  # view var & Cov
print(classification_report(Y_test,Y_pred)) # view accuracy avrage ..
cm = confusion_matrix(Y_test, Y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()


