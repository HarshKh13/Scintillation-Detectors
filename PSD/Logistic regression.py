import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'traindats.csv'
data = pd.read_csv(path)
data.head()

from sklearn.preprocessing import LabelEncoder
lab_encod = LabelEncoder()
data['Classification_encod'] = lab_encod.fit_transform(data['Classification'])
data = data.drop(['Classification'],axis = 'columns')
data.head()

data.describe()

x1 = data['TotalArea']
x2 = data['TailArea']
y = data['Classification_encod']

#plot of classification against total area
plt.scatter(x1,y,color = 'blue')
plt.xlabel('Total Area')
plt.ylabel('Classification')
plt.title('Classification plot against total area')
plt.show()

#plot of classification against tail area
plt.scatter(x2,y,color = 'blue')
plt.xlabel('Tail Area')
plt.ylabel('Classification')
plt.title('Classification plot against tail area')
plt.show()

#Plot with total area on x-axis and tail area on y-axis
f = plt.figure()
f.set_figwidth(10)
f.set_figheight(8)
plt.scatter(data['TotalArea'][data.Classification_encod==1],
            data['TailArea'][data.Classification_encod==1],
            marker = 'D',
            color = 'red',
            label = 'Neutron')
plt.scatter(data['TotalArea'][data.Classification_encod==0],
            data['TailArea'][data.Classification_encod==0],
            marker = 'o',
            color = 'blue',
            label = 'Gamma')
plt.xlabel('TotalArea')
plt.ylabel('TailArea')
plt.title('Entire plot')
plt.legend(loc = 'lower right')
plt.show()

#Classification  using Logistic Regression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
x = data.iloc[:,1:]
x = x.drop(['Classification_encod'],axis = 'columns')
y = data.iloc[:,-1]
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x,columns = ['TotalArea','TailArea'])
x = x.values
y = y.values

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,
                                                 random_state = 42)
print("Training data shape",x_train.shape)
print("Test data shape",x_test.shape)


log_reg = LogisticRegression()
log_reg.fit(x_train,y_train.ravel())
y_pred = log_reg.predict(x_test)
print(y_pred[:10])

print(type(y_pred))
print(type(y_test))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred,y_test)
print("Accuracy",accuracy)

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
plt.xlim([-0.05,1.0])
plt.ylim([0.0,1.05])
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
plt.xlim([-0.05,1.0])
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
opt_thresh = 0.31621
y_pred = binarize(y_prob_test.reshape(1,-1), opt_thresh)[0]
print(y_pred[:20])

confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)

classification_repo = classification_report(y_test,y_pred)
print(classification_repo)

accuracy_logit = accuracy_score(y_pred,y_test)
print("Accuracy Score using Logistic Regression",accuracy_logit)

















