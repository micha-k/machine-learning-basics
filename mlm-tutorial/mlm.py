#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn


def get_versions():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))

def default():
    pass

def main():

    try:
        cmd = sys.argv[1]
    except IndexError as e:
        cmd = 'default'
    
    if cmd == 'versions':
        get_versions()
    else:
        default()

if __name__ == '__main__':
    main()
