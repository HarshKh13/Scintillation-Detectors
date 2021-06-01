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

def accuracy_fn(y_pred,y_test):
    accuracy = 0
    for i in range(len(y_pred)):
        if y_pred[i]==y_test[i]:
            accuracy  = accuracy + 1
    
    return accuracy/len(y_pred)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

num_epochs = 10
batch_size = 264

num_feature = 2
model1 = build_model1(num_feature)
model1.summary()
model1.fit(x_train,y_train,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model1.predict(x_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_fn(y_pred_temp,y_test)
accuracy_list.append(["model1",temp_accuracy])

###########################################################################################################

model2 = build_model2(num_feature)
model2.summary()
model2.fit(x_train,y_train,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model2.predict(x_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_fn(y_pred_temp,y_test)
accuracy_list.append(["model2",temp_accuracy])

#############################################################################################################

model3 = build_model3(num_feature)
model3.summary()
model3.fit(x_train,y_train,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model3.predict(x_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_fn(y_pred_temp,y_test)
accuracy_list.append(["model3",temp_accuracy])

#############################################################################################################

model4 = build_model4(num_feature)
model4.summary()
model4.fit(x_train,y_train,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model4.predict(x_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_fn(y_pred_temp,y_test)
accuracy_list.append(["model4",temp_accuracy])

#################################################################################################################

model5 = build_model5(num_feature)
model5.summary()
model5.fit(x_train,y_train,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model5.predict(x_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_fn(y_pred_temp,y_test)
accuracy_list.append(["model5",temp_accuracy])


###################################################################################################################

model6 = build_model6(num_feature)
model6.summary()
model6.fit(x_train,y_train,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model6.predict(x_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_fn(y_pred_temp,y_test)
accuracy_list.append(["model6",temp_accuracy])

#############################################################################################################

model7 = build_model7(num_feature)
model7.summary()
model7.fit(x_train,y_train,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model7.predict(x_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_fn(y_pred_temp,y_test)
accuracy_list.append(["model7",temp_accuracy])

#############################################################################################################

model8 = build_model8(num_feature)
model8.summary()
model8.fit(x_train,y_train,batch_size = batch_size,
         epochs = num_epochs)

y_pred_hot = model8.predict(x_test)
y_pred_temp = np.array(y_pred_hot.argmax(axis=1))
temp_accuracy = accuracy_fn(y_pred_temp,y_test)
accuracy_list.append(["model8",temp_accuracy])



model_names = []
accuracy_of_model = []
for model_name, accuracy in accuracy_list:
    print("MODEL NAME: ",model_name," ","Accuracy: ",accuracy)
    model_names.append(model_name)
    accuracy_of_model.append(accuracy)
    
    
plt.scatter(model_names,accuracy_of_model)

best_ann_model = model8
y_pred = model7.predict(x_test)
y_pred = np.array(y_pred_hot.argmax(axis=1))
accuracy_ann = accuracy_fn(y_pred,y_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)

classification_repo = classification_report(y_test,y_pred)
print(classification_repo)

accuracy_ann = np.max(accuracy_of_model)
print("Accuracy using Artificial Neural Networks",accuracy_ann)














