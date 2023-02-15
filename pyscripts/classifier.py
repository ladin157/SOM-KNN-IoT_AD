import numpy as np
from sklearn import svm, linear_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils.visualization import plot_confusion_matrix, plot_roc_curve_auc


def dt_classify():
    pass


def rf_classify():
    pass


def svm_classify():
    pass


def classify(X_train, y_train, X_test, y_test, algo='DT'):
    algo = algo.upper()
    if algo.__eq__('KNN'):
        clf = KNeighborsClassifier(n_neighbors=3, p=2)
    elif algo.__eq__('SVM'):  # SVM kernel RBF
        clf = svm.SVC(kernel='rbf', degree=3, C=1, decision_function_shape='ovo')
    # if algo.__eq__('GMM'):
    #     clf = dict((covar_type, GMM(n_components=n_classes,
    #                                 covariance_type=covar_type, init_params='wc', n_iter=20))
    #                for covar_type in ['spherical', 'diag', 'tied', 'full'])
    elif algo.__eq__('SOFTMAX'):  # Softmax Regression
        clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

    elif algo.__eq__('RF'):
        clf = RandomForestClassifier(max_depth=2, random_state=0)
    elif algo.__eq__('DT'):
        clf = DecisionTreeClassifier(random_state=0)
    elif algo.__eq__('XGB'):
        clf = XGBClassifier(objective='binary:logistic', random_state=42)
    else:
        clf = KNeighborsClassifier(n_neighbors=3, p=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_test = [i if i == 1 else 2 for i in y_test]
    y_pred = [i if i == 1 else 2 for i in y_pred]
    print(classification_report(y_test, y_pred, digits=3))
    # pretty_plot_confusion_matrix(y_test=y_test, predictions=y_pred)
    # print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm=cm)
    # plot roc curve and calculate auc_score
    # chi su dung cho tu hai lop tro len
    if len(np.unique(y_test)) >= 2 and len(np.unique(y_pred)) >= 2:
        fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=2)
        auc_score = roc_auc_score(y_test, y_pred)
        print("AUC score: ", auc_score)
        # roc_auc = auc(fpr, tpr)
        # print(roc_auc)
        plot_roc_curve_auc(fpr=fpr, tpr=tpr, roc_auc=auc_score)
