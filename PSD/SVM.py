import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

path = 'traindats.csv'
data = pd.read_csv(path)
data.head()

from sklearn.preprocessing import LabelEncoder
lab_encod = LabelEncoder()
data['Classification_encod'] = lab_encod.fit_transform(data['Classification'])
data = data.drop(['Classification'],axis = 'columns')
data.head()

x = data.iloc[:,1:]
x = x.drop(['Classification_encod'],axis = 'columns')
y = data.iloc[:,-1]
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x,columns = ['TotalArea','TailArea'])
x = x.values
y = y.values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,
                                                 random_state = 42)
print("Training data shape",x_train.shape)
print("Test data shape",x_test.shape)

#Classification using SVM

from sklearn.preprocessing import StandardScaler
from sklearn import svm

linear_svm = svm.SVC(kernel = 'linear')
rbf_svm = svm.SVC(kernel = 'rbf')
poly_svm = svm.SVC(kernel = 'poly')


linear_svm.fit(x_train,y_train)
rbf_svm.fit(x_train,y_train)
poly_svm.fit(x_train,y_train)

y_pred_linear = linear_svm.predict(x_test)
y_pred_rbf = rbf_svm.predict(x_test)
y_pred_poly = poly_svm.predict(x_test)

def accuracy_fn(y_pred,y_test):
    accuracy = 0
    for i in range(len(y_pred)):
        if y_pred[i]==y_test[i]:
            accuracy  = accuracy + 1
    
    return accuracy/len(y_pred)

accuracy_linear = accuracy_fn(y_pred_linear,y_test)
accuracy_rbf = accuracy_fn(y_pred_rbf,y_test)
accuracy_poly = accuracy_fn(y_pred_poly,y_test)


print("Linear Kernel accuracy: ",accuracy_linear)
print("Rbf kernel accuracy: ",accuracy_rbf)
print("Polynomial kernel accuracy: ",accuracy_poly)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

confusion_mat = confusion_matrix(y_test,y_pred_linear)
print(confusion_mat)

classification_repo = classification_report(y_test,y_pred_linear)
print(classification_repo)

accuracy_svm = max(max(accuracy_linear,accuracy_rbf),accuracy_poly)
print("Accuracy using SVM",accuracy_svm)














