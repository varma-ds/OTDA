import os
import numpy as np
import pandas as pd
import pickle

def readdata(file, names=None):
    print(file)
    if os.path.isfile(file):
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".csv":
            return pd.read_csv(file, names=names)
        elif file_extension == ".xlsx":
            return pd.read_excel(file, names=names)
    else:
        raise Exception('Invalid feedback path!. Hint: Use forward slashes e.g. use C:/Users instead C:\\Users')


def save_objects(obj_list,file):
    with open(file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(obj_list, f)


def load_objects(file):
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        return(pickle.load(f))# -*- coding: utf-8 -*-

