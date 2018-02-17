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
    else:
        mlm.default()

if __name__ == '__main__':
    main()
