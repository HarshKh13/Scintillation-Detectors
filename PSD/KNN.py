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

#Classification using KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(x_train,y_train)
accuracy_score = knn.score(x_test,y_test)
print("Accuracy",accuracy_score)


#Finding the best k value
nearest_neighbours = np.arange(1,20)
train_accuracy = np.empty(len(nearest_neighbours))
test_accuracy = np.empty(len(nearest_neighbours))

for i,k in enumerate(nearest_neighbours):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    
    temp_train_acc = knn.score(x_train,y_train)
    temp_test_acc = knn.score(x_test,y_test)
    train_accuracy[i] = temp_train_acc
    test_accuracy[i] = temp_test_acc
    

print(train_accuracy)
print(test_accuracy)

plt.plot(nearest_neighbours,train_accuracy,label = 'Training data accuracy')
plt.plot(nearest_neighbours,test_accuracy,label = 'Test data accuracy')
plt.legend()
plt.xlabel('num_neighbours')
plt.ylabel('Accuracy')
plt.title('Find best number of nearest neighbours')
plt.show()


best_nearest_neighbors = 3
knn_best = KNeighborsClassifier(n_neighbors = best_nearest_neighbors)
knn_best.fit(x_train,y_train)
accuracy_knn = knn_best.score(x_test,y_test)
y_pred = knn_best.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)

classification_repo = classification_report(y_test,y_pred)
print(classification_repo)

print("Accuracy using K Nearest Neighbors",accuracy_knn)












