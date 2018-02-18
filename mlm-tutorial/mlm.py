#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import scipy
import numpy
import matplotlib
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
import sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class MLMTutorial:

    def __init__(self):

        # dataset is loaded every time
        self.load_dataset()

        # Evaluating some algorithms.
        # Making some predictions.


    # 1. Loading the dataset.
    def load_dataset(self):
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        self.dataset = pandas.read_csv('iris.data', names=names)


    # 2. Summarizing the dataset.
    def summarize_dataset(self):

        # 2.1 Dimensions of the dataset.
        print("Dimensions: %s instances, %s attributes" % self.dataset.shape)

        # 2.2 Peek at the data itself.
        print(self.dataset.head(5))

        # 2.3 Statistical summary of all attributes.
        print(self.dataset.describe())

        # 2.4 Breakdown of the data by the class variable.
        print(self.dataset.groupby('class').size())

    # 3. Visualizing the dataset.

    # 3.1 Univariate Plots
    def plot_whisker_box(self):
        self.dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        plt.show()

    def plot_histo(self):
        self.dataset.hist()
        plt.show()

    # 3.2 Multivariate Plots
    def plot_scatter(self):
        scatter_matrix(self.dataset)
        plt.show()

    # Create test dataset; compare models; plot the results
    def test_and_compare(self):

        self.split_dataset()

        # settings for the test
        seed = 7
        scoring = 'accuracy'

        # loading models and check run them
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        # plot the results
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

    def knn_predict(self):

        self.split_dataset()

        # Make predictions on validation dataset
        knn = KNeighborsClassifier()
        knn.fit(self.X_train, self.Y_train)
        predictions = knn.predict(self.X_validation)
        print(accuracy_score(self.Y_validation, predictions))
        print(confusion_matrix(self.Y_validation, predictions))
        print(classification_report(self.Y_validation, predictions))

    def split_dataset(self):

        # 80% - 20% split of the dataset
        array = self.dataset.values
        X = array[:,0:4]
        Y = array[:,4]
        validation_size = 0.20
        seed = 7
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = \
            model_selection.train_test_split(X, Y, test_size=validation_size,
                                             random_state=seed)

    def get_versions(self):
        print('Python: {}'.format(sys.version))
        print('scipy: {}'.format(scipy.__version__))
        print('numpy: {}'.format(numpy.__version__))
        print('matplotlib: {}'.format(matplotlib.__version__))
        print('pandas: {}'.format(pandas.__version__))
        print('sklearn: {}'.format(sklearn.__version__))

    def default(self):
        pass

def main():

    try:
        cmd = sys.argv[1]
    except IndexError as e:
        cmd = 'default'

    mlm = MLMTutorial()

    if cmd == 'versions':
        mlm.get_versions()
    elif cmd == 'summarize':
        mlm.summarize_dataset()
    elif cmd == 'plot_wb':
        mlm.plot_whisker_box()
    elif cmd == 'plot_histo':
        mlm.plot_histo()
    elif cmd == 'plot_scatter':
        mlm.plot_scatter()
    elif cmd == 'test_compare':
        mlm.test_and_compare()
    elif cmd == 'knn_predict':
        mlm.knn_predict()
    else:
        mlm.default()

if __name__ == '__main__':
    main()
