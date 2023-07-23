from sklearn.datasets import load_iris
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedKFold ,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,make_scorer,auc
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC 
import pickle 
import os

iris=load_iris() 

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df["target"]=iris.target

# Spliting
X=df.drop(["target"],axis=1)
y=df["target"]
#y=y.astype('category')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=99)

# Standard Scaler 
sd=StandardScaler()
stdz=sd.fit(X_train)
X_train_sd=pd.DataFrame(stdz.transform(X_train),columns=X_train.columns)
X_test_sd=pd.DataFrame(stdz.transform(X_test),columns=X_test.columns)

# Modelling
svc_=SVC(random_state=99,probability=True,decision_function_shape='ovr')
kernel = ['linear', 'poly', 'rbf']
tolerance = [1e-3, 1e-4, 1e-5, 1e-6]
C = [1, 1.5, 2, 2.5, 3]
grid = dict(kernel=kernel, tol=tolerance, C=C)
cv=StratifiedKFold(n_splits = 10, random_state=99,shuffle=True)

gridSearch=GridSearchCV(estimator=svc_,param_grid=grid,n_jobs=-1,cv=cv,scoring="roc_auc_ovr", refit=True)
searchResults = gridSearch.fit(X_train, y_train)
 
print(accuracy_score(searchResults.predict(X_test),y_test))   
print(roc_auc_score(y_test,searchResults.predict_proba(X_test),multi_class="ovr"))

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(gridSearch, open(filename, 'wb'))
print(f"Model saved in the path:{os.getcwd()}")
