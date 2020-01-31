from flask import Flask
import json
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns  
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
api = Api(app=app)

#création des différents dataframe
features = pd.read_csv("features.txt",sep='\t',header=None)

x_train = pd.read_csv("Train/X_train.txt",delimiter=' ',sep='\t',header=None)
y_train = pd.read_csv("Train/y_train.txt",delimiter=' ',sep='\t',header=None)

x_train.columns = list(features[0])
y_train.columns = ["activity"]

x_test = pd.read_csv("Test/X_test.txt",delimiter=' ',sep='\t',header=None)
y_test = pd.read_csv("Test/y_test.txt",delimiter=' ',sep='\t',header=None)

y_test.columns=["activity"]
x_test.columns=list(features[0])

labels = list(pd.read_csv("activity_labels.txt",sep='\t',header=None)[0])
activity_labels = []
for label in labels:
    l = label.split()
    activity_labels.append(l[1])



@api.route("/predict")
def predict():
    n_components = request.args.get('n_components', '')

    pca = PCA(n_components=n_components)
    pca = pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    lr = LogisticRegression()
    lr = lr.fit(x_train_pca,y_train)

    y_predit_test = lr.predict(x_test_pca)
    acc = accuracy_score(y_predit_test,y_test)

    return json.dumps({
        "accuracy" : acc
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8000', debug=True)