from cProfile import label
import copy
import math

import matplotlib.pyplot as plt
import numpy as np
from hyperopt import hp, Trials, fmin, tpe, rand, atpe, anneal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.sandbox.regression.gmm import GMM
from sklearn import svm, linear_model

from utils import minisom
from utils.som_utils import som_fn
from utils.visualization import quantization_errors_visualization, outliers_visualization, plot_confusion_matrix, \
    plot_roc_curve_auc


def _som_classify(som, winmap, data):  # , X_train, y_train):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    # winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


def _som_classify_different_algo(som, winmap, data, algo='KNN'):
    """
    Classify using KNN-based algorithm
    """
    # results = list()

    # positions = list()
    wmap = copy.deepcopy(winmap)
    # get label for position
    for position in wmap:
        wmap[position] = wmap[position].most_common(1)[0][0]
    # get the positions
    data_x = wmap.keys()
    # get the labels
    data_y = wmap.values()
    # convert list of tuple to two dimensional array for data_x
    data_x = [[x[0], x[1]] for x in data_x]
    # convert list of tupple to one dimensional array for data_y
    data_y = [y for y in data_y]
    n_classes = set(data_y).__len__()
    if algo.__eq__('KNN'):
        clf = KNeighborsClassifier(n_neighbors=3, p=2)
    elif algo.__eq__('SVM'): # SVM kernel RBF
        clf = svm.SVC(kernel='rbf', degree=3, C=1, decision_function_shape='ovo')
    # if algo.__eq__('GMM'):
    #     clf = dict((covar_type, GMM(n_components=n_classes,
    #                                 covariance_type=covar_type, init_params='wc', n_iter=20))
    #                for covar_type in ['spherical', 'diag', 'tied', 'full'])
    elif algo.__eq__('SOFTMAX'): # Softmax Regression
        clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

    elif algo.__eq__('RF'):
        clf = RandomForestClassifier(max_depth=2, random_state=0)
    else:
        clf = KNeighborsClassifier(n_neighbors=3, p=2)
    # fit the KNN
    clf.fit(X=data_x, y=data_y)
    # now get SOM output from data
    # X_test = list()
    y_pred = list()
    for x in data:
        win_pos = som.winner(x)
        # sua lai ti doan nay
        if win_pos in winmap:
            label = wmap[win_pos]
            # print(label)
            y_pred.append(label)
        else:
            label = clf.predict([[win_pos[0], win_pos[1]]])
            # print(label)
            y_pred.append(label[0])
        # X_test.append(win_pos)
    # X_test = [[x[0], x[1]] for x in X_test]
    # y_pred = clf.predict(X=X_test)
    return y_pred


def _som_clustering(som, data, cluster_index):
    plt.figure(figsize=(20, 20))

    # plotting the clusters using the first 2 dimentions of the data
    for c in np.unique(cluster_index):
        plt.scatter(data[cluster_index == c, 0],
                    data[cluster_index == c, 1], label='cluster=' + str(c), alpha=.7)

    # plotting centroids
    for centroid in som.get_weights():
        plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                    s=80, linewidths=35, color='k', label='centroid')
    plt.legend()
    plt.show()


def som_classification(som, winmap, X_test, y_test, using_algo=False, algo='KNN'):
    '''
    Phan loai cho bengin va cac lop khac.
    '''
    # xu ly cho truong hop 1, tinh lai confusion matrix, benign la lop 1, con lai la lop 2
    # y_train = [i if i == 1 else 2 for i in y_train]
    # chuyen doi label cho y_test
    y_test = [i if i == 1 else 2 for i in y_test]
    if using_algo:
        y_pred = _som_classify_different_algo(som=som, winmap=winmap, data=X_test, algo=algo)
    else:
        # , X_train=X_train, y_train=y_train)
        y_pred = _som_classify(som=som, winmap=winmap, data=X_test)
    # chuyen doi label cho y_pred
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


def train_som(X_train, y_train, algo='tpe', som_x=None, som_y=None, sigma=6, learning_rate=2.0, verbose=False,
              show_progressbar=False, max_evals=500):
    print("---------------------------------Train SOM-------------------------------------")
    print("Number of feature: ", X_train.shape[1])
    # TRAINING AND TUNING PARAMS FOR SOM
    # get parameters
    # the params haven't passes --> optimize to get best values
    if som_x is None and som_y is None:
        print("The default values of som_x and som_y are None")
        print(
            "Hyper-parameters optimization process. The algorithm used is {}.".format(algo))
        if algo.__eq__('tpe'):
            algo = tpe.suggest
        elif algo.__eq__('rand'):
            algo = rand.suggest
        elif algo.__eq__('atpe'):
            algo = atpe.suggest
        elif algo.__eq__('anneal'):
            algo = anneal.suggest
        else:
            print("Default algorithm is tpe")
            algo = tpe.suggest
        space = {
            'sigma': hp.uniform('sigma', 5, 10),
            'learning_rate': hp.uniform('learning_rate', 0.05, 5),
            'x': hp.uniform('x', 20, 50),
            'data_benign': X_train
        }
        trials = Trials()
        # max_evals can be set to 1200, but for speed, we set to 100
        best = fmin(fn=som_fn,
                    space=space,
                    algo=algo,
                    max_evals=max_evals,
                    trials=trials,
                    verbose=verbose,
                    show_progressbar=show_progressbar)
        print('Best: {}'.format(best))
        som_x = math.ceil(best['x'])
        som_y = math.ceil(best['x'])
        sigma = best['sigma']
        learning_rate = best['learning_rate']
    som_turned = minisom.MiniSom(x=som_x,
                                 y=som_y,
                                 input_len=X_train.shape[1],
                                 sigma=sigma,
                                 learning_rate=learning_rate)
    print("---------SOM has been turned!-----------")
    print("Starting SOM Weights init")
    som_turned.pca_weights_init(X_train)
    print("Perform SOM (turned) train random")
    som_turned.train_random(X_train, 1000, verbose=0)
    winmap = som_turned.labels_map(X_train, y_train)
    y_benign = [y for y in y_train if y == 1]
    # doan nay can xem lai outliers_percentage = 1 - (len(y_benign) / len(y_train))
    outliers_percentage = len(y_benign) / len(y_train)
    print(outliers_percentage)
    return som_turned, winmap, outliers_percentage


def test_som(som, winmap, X_test, y_test, using_algo=False, algo='KNN', outliers_percentage=None):
    #     # kiem tra thu voi du lieu huan luyen
    #     quantization_errors = np.linalg.norm(som_turned.quantization(X_train_normalized) - X_train_normalized, axis=1)
    # kiem tra thu voi du lieu test

    # print("----------------------------------------------------------------------")
    # print("Compute quantization errors and error threshold")
    # quantization_errors = np.linalg.norm(som_turned.quantization(X_test) - X_test, axis=1)
    # print(quantization_errors)
    # error_treshold = np.percentile(quantization_errors,
    #                                min(100 * (1 - outliers_percentage) + 5, 100))
    # is_outlier = quantization_errors > error_treshold

    # print("Visualize quantization error")
    # quantization_errors_visualization(quantization_errors=quantization_errors, error_treshold=error_treshold)

    # print("Outliers visualization")
    # #     outliers_visualization(data=X_train_normalized, is_outlier=is_outlier, indexes=[0, 1])
    # #     outliers_visualization(data=X_train_normalized, is_outlier=is_outlier, indexes=[1, 2])
    # outliers_visualization(data=X_test, is_outlier=is_outlier, indexes=[0, 1])
    # outliers_visualization(data=X_test, is_outlier=is_outlier, indexes=[1, 2])

    print("----------------------------------------------------------------------")
    print("SOM classification")
    # som_classification(som=som_turned, X_train=X_train_normalized, y_train=y_train, X_test=X_test_normalized, y_test=y_test)
    som_classification(som=som, winmap=winmap, X_test=X_test,
                       y_test=y_test, using_algo=using_algo, algo=algo)
    print("-----------Testing SOM done!-------------")
