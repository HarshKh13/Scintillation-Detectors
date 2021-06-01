import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 
data = pd.read_csv(path)
data.head()

from sklearn.preprocessing import LabelEncoder
lab_encod = LabelEncoder()
data['Classification_encod'] = lab_encod.fit_transform(data['Classification'])
data.drop[['Classification']]
data.head()

data.describe()

x1 = data['TotalArea']
x2 = data['TailArea']
y = data['Classification_encod']

#plot of classification against total area
plt.scatter(x1,y,color = 'red')
plt.xlabel('Total Area')
plt.ylabel('Classification')
plt.title('Classification pllot against total area')
plt.show()

#plot of classification against tail area
plt.scatter(x2,y,color = 'red')
plt.xlabel('Tail Area')
plt.ylabel('Classification')
plt.title('Classification plot against tail area')
plt.show()

plt.figure()
plt.scatter(data['TotalArea'][data.Classification_encod==1],
            data['TailArea'][data.Classification_encod==1],
            marker = 'D',
            color = 'red',
            label = '')
plt.scatter(data['TotalArea'][data.Classification_encod==0],
            data['TailArea'][data.Classification_encod==0],
            marker = 'o',
            color = 'blue',
            label = '')
plt.xlabel('TotalArea')
plt.ylabel('TailArea')
plt.title('Entire plot')
plt.legend(loc = 'lower right')
plt.show()

########################################################################################
########################################################################################
#Classification  using SVM
import statsmodels.api as sm
x = data[['TotalArea','TailArea']]
y = data['Classification']
x = sm.add_constant(x)
logit_model = sm.Logit(y,x)
result = logit_model.fit()
print(result.summary())



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
y = y.values.reshape(1,-1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,
                                                 random_state = 42)
print("Training data shape",x_train.shape)
print("Test data shape",x_test.shape)


log_reg = LogisticRegression()
log_reg.fit(x_train,y_train.ravel())
y_pred = log_reg.predict(x_test)
print(y_pred[:10])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred,y_test)
print("Accuracy",accuracy_score)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))

y_prob_train = log_reg.predict_proba(x_train)[:,1]
y_prob_train.reshape(1,-1)
print(y_prob_train[:20])

y_prob_test = log_reg.predict_proba(x_test)[:,1]
y_prob_test.reshape(1,-1)
print(y_prob_test[:20])

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
fpr,tpr,thresholds = roc_curve(y_train,y_prob_train)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,color='blue',label = 'ROC curve (area = %0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color = 'red')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("ROC curve for training data")
plt.legend(loc = 'lower right')
plt.show()

fpr,tpr,thresholds = roc_curve(y_test,y_prob_test)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,color='blue',label = 'ROC curve (area = %0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color = 'red')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("ROC curve for training data")
plt.legend(loc = 'lower right')
plt.show()

fpr,tpr,thresholds = roc_curve(y_test,y_prob_test)
indices = np.arange(len(fpr))
df_roc = pd.DataFrame({'fpr':pd.Series(fpr,indices),'tpr':pd.Series(tpr,indices),
                       '1-fpr':pd.Series(1-fpr,indices),
                       'tf':pd.Series(tpr-(1-fpr),indices),
                       'thresholds':pd.Series(thresholds,indices)})
df_roc.head()

df_roc.iloc[(df_roc.tf-0).abs().argsort()[:1]]

fig,ax = plt.subplots()
plt.plot(df_roc['tpr'],color = 'blue')
plt.plot(df_roc['1-fpr'],color = 'red')
plt.xlabel('1-False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

#Classification using optimal threshold value
from sklearn.preprocessing import binarize
opt_thresh = 
y_pred = binarize(y_prob_test.reshape(1,-1), opt_thresh)[0]
print(y_pred[:20])

confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)

classification_repo = classification_report(y_test,y_pred)
print(classification_repo)

accuracy_logit = accuracy_score(y_pred,y_test)
print("Accuracy Score using Logistic Regression",accuracy_logit)

####################################################################################
####################################################################################
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

best_nearest_neighbors = 
knn_best = KNeighborsClassifier(n_neighbors = best_nearest_neighbors)
knn_best.fit(x_train,y_train)
accuracy_knn = knn_best.score(x_test,y_test)
y_pred = knn_best.predict(x_test)

confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)

classification_repo = classification_report(y_test,y_pred)
print(classification_repo)

print("Accuracy using K Nearest Neighbors",accuracy_knn)


#########################################################################################
#########################################################################################
#Classification using SVM

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

linear_svm = svm.SVC(kernel = 'linear')
rbf_svm = svm.SVC(kernel = 'rbf')
poly_svm = svm.SVC(kernel = 'poly')

linear_svm.fit(x_train,y_train)
rbf_svm.fit(x_train,y_train)
poly_svm.fit(x_train,y_train)

y_pred_linear = linear_svm.predict(x_test)
y_pred_rbf = rbf_svm.predict(x_test)
y_pred_poly = poly_svm.fit(x_test)

accuracy_linear = accuracy_score(y_pred_linear,y_test)
accuracy_rbf = accuracy_score(y_pred_rbf,y_test)
accuracy_poly = accuracy_score(y_pred_poly,y_test)

confusion_mat = confusion_matrix(y_test,y_pred_linear)
print(confusion_mat)

classification_repo = classification_report(y_test,y_pred_linear)
print(classification_repo)

accuracy_svm = max(max(accuracy_linear,accuracy_rbf),accuracy_poly)
print("Accuracy using SVM",accuracy_svm)

####################################################################################
####################################################################################
#Classification using Artificial Neural Networks
def build_model1(input_node):
    inputs = layers.Input(shape = input_node, name = "input",dtype=np.float32)
    x = tf.keras.layers.Dense(128,activation='relu',name='hidden1')(inputs)
    x = layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2,activation='softmax',name='output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_model2(input_node):
    inputs = layers.Input(shape = input_node, name = "input",dtype=np.float32)
    x = tf.keras.layers.Dense(256,activation='relu',name='hidden1')(inputs)
    x = layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2,activation='softmax',name='output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model3(input_node):
    inputs = layers.Input(shape = input_node, name = "input",dtype=np.float32)
    x = tf.keras.layers.Dense(128,activation='relu',name='hidden1')(inputs)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,activation='relu',name='hidden2')(x)
    x = layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2,activation='softmax',name='output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_model4(input_node):
    inputs = layers.Input(shape = input_node, name = "input",dtype=np.float32)
    x = tf.keras.layers.Dense(128,activation='relu',name='hidden1')(inputs)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256,activation='relu',name='hidden2')(x)
    x = layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2,activation='softmax',name='output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model5(input_node):
    inputs = layers.Input(shape = input_node, name = "input",dtype=np.float32)
    x = tf.keras.layers.Dense(128,activation='relu',name='hidden1')(inputs)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(512,activation='relu',name='hidden2')(x)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256,activation='relu',name='hidden3')(x)
    x = layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2,activation='softmax',name='output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_model6(input_node):
    inputs = layers.Input(shape = input_node, name = "input",dtype=np.float32)
    x = tf.keras.layers.Dense(128,activation='relu',name='hidden1')(inputs)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256,activation='relu',name='hidden2')(x)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256,activation='relu',name='hidden3')(x)
    x = layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2,activation='softmax',name='output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_model7(input_node):
    inputs = layers.Input(shape = input_node, name = "input",dtype=np.float32)
    x = tf.keras.layers.Dense(128,activation='relu',name='hidden1')(inputs)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256,activation='relu',name='hidden2')(x)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256,activation='relu',name='hidden3')(x)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,activation='relu',name='hidden4')(x)
    x = layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2,activation='softmax',name='output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model



def build_model8(input_node):
    inputs = layers.Input(shape = input_node, name = "input",dtype=np.float32)
    x = tf.keras.layers.Dense(128,activation='relu',name='hidden1')(inputs)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256,activation='relu',name='hidden2')(x)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(512,activation='relu',name='hidden3')(x)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256,activation='relu',name='hidden4')(x)
    x = layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2,activation='softmax',name='output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


accuracy_list = []

num_epochs = 20
batch_size = 264

model1 = build_model1(num_feature)
model1.summary()
model1.fit(x_temp_train,y_train_ann,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model1.predict(x_temp_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_score(y_pred_temp,y_test_ann)
accuracy_list.append(["model1",temp_accuracy])

###########################################################################################################

model2 = build_model2(num_feature)
model2.summary()
model2.fit(x_temp_train,y_train_ann,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model2.predict(x_temp_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_score(y_pred_temp,y_test_ann)
accuracy_list.append(["model2",temp_accuracy])

#############################################################################################################

model3 = build_model3(num_feature)
model3.summary()
model3.fit(x_temp_train,y_train_ann,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model3.predict(x_temp_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_score(y_pred_temp,y_test_ann)
accuracy_list.append(["model3",temp_accuracy])

#############################################################################################################

model4 = build_model4(num_feature)
model4.summary()
model4.fit(x_temp_train,y_train_ann,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model4.predict(x_temp_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_score(y_pred_temp,y_test_ann)
accuracy_list.append(["model4",temp_accuracy])

#################################################################################################################

model5 = build_model5(num_feature)
model5.summary()
model5.fit(x_temp_train,y_train_ann,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model5.predict(x_temp_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_score(y_pred_temp,y_test_ann)
accuracy_list.append(["model5",temp_accuracy])

###################################################################################################################

model5 = build_model5(num_feature)
model5.summary()
model5.fit(x_temp_train,y_train_ann,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model5.predict(x_temp_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_score(y_pred_temp,y_test_ann)
accuracy_list.append(["model5",temp_accuracy])

###################################################################################################################

model6 = build_model6(num_feature)
model6.summary()
model6.fit(x_temp_train,y_train_ann,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model6.predict(x_temp_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_score(y_pred_temp,y_test_ann)
accuracy_list.append(["model6",temp_accuracy])

#############################################################################################################

model7 = build_model7(num_feature)
model7.summary()
model7.fit(x_temp_train,y_train_ann,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model7.predict(x_temp_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_score(y_pred_temp,y_test_ann)
accuracy_list.append(["model7",temp_accuracy])

#############################################################################################################

model8 = build_model8(num_feature)
model8.summary()
model8.fit(x_temp_train,y_train_ann,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model8.predict(x_temp_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_score(y_pred_temp,y_test_ann)
accuracy_list.append(["model8",temp_accuracy])

model_names = []
accuracy_of_model = []
for model_name, accuracy in accuracy_list:
    print("MODEL NAME: ",model_name," ","Accuracy: ",accuracy)
    model_names.append(model_name)
    accuracy_of_model.append(accuracy)
    
confusion_mat = confusion_matrix(y_test,y_pred_linear)
print(confusion_mat)

classification_repo = classification_report(y_test,y_pred_linear)
print(classification_repo)

accuracy_ann = np.max(accuracy_of_model)
print("Accuracy using Artificial Neural Networks",accuracy_ann)



