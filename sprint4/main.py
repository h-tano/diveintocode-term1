import svm
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    train_df = pd.read_csv("sprint4/wdbc.data",header=None)
    y = train_df[1]
    X = train_df.drop([0,1],axis=1)
    scaler = preprocessing.StandardScaler()
    X_std = scaler.fit_transform(X)
    y = list(map(lambda l: 1 if l == 'M' else -1, y))
    X_train , X_test , y_train , y_test =\
     train_test_split(X_std, y, test_size=0.2, random_state=0)
    # ラベルは1, -1

    # スケーリング

    clf = svm.SVM()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Confusion Matrix")
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
