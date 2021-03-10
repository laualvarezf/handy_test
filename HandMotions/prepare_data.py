import pandas as pd
from glob import glob
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import pickle

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels

def loading_one_subject(subject_number):
    y_raw= []
    raw = []
    data_path= os.path.join(os.path.abspath(os.path.dirname(__file__)),'..','raw_date', 'train')
    path = os.path.join(data_path, 'subj%d_series*_data.csv')
    fnames =  sorted(glob(path % (subject_number)))
    for fname in fnames:
        data,labels=prepare_data_train(fname)
        raw.append(data)
        y_raw.append(labels)
    X = pd.concat(raw).reset_index()
    X.drop(columns="index", inplace=True)
    y = pd.concat(y_raw).reset_index()
    y.drop(columns="index", inplace=True)
    return X, y

def convert_df_to_numpy(df):
    return np.asarray(df.astype(float))

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(X, wavelet='db2', level=3):
    coeff = pywt.wavedec(X, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(X)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

def preprocess_data(X, y):
    X= convert_df_to_numpy(X)
    y=convert_df_to_numpy(y)
    X=wavelet_denoising(X)
    return X, y

def custom_train_test_split(X, y):
    splitrate=-X.shape[0]//5*2
    xval=X[splitrate:splitrate//2]
    yval=y[splitrate:splitrate//2]
    xtest=X[splitrate//2:]
    ytest=y[splitrate//2:]
    xtrain=X[:splitrate]
    ytrain=y[:splitrate]
    return xtrain, xval, xtest, ytrain, yval, ytest

# X1, y1= loading_one_subject(1)
# print(type(X1))
# print(type(y1))

# X_preprocessed, y_preprocessed = preprocess_data(X1, y1)
# print('Preprocessed subject one data')
# print(type(X_preprocessed))
# print(type(y_preprocessed))
# print(len(X_preprocessed))
# print(len(y_preprocessed))

# print('Train test split data')
# xtrain1, xval1, xtest1, ytrain1, yval1, ytest1 = custom_train_test_split(X_preprocessed, y_preprocessed)
# print(type(xtrain1), type(xval1), type(xtest1), type(ytrain1), type(yval1), type(ytest1))
# print(len(xtrain1), len(xval1), len(xtest1), len(ytrain1), len(yval1), len(ytest1))

def prepare_and_save_data(number_of_subjects):
    for i in range(1, (number_of_subjects+1)):
        print('starting the preprocessing)')
        X, y= loading_one_subject(i)
        X_preprocessed, y_preprocessed = preprocess_data(X, y)
        print('finished preprocessing, starting train test split')
        xtrain, xval, xtest, ytrain, yval, ytest = custom_train_test_split(X_preprocessed, y_preprocessed)
        print('loading xval into a file')
        data_path= os.path.join(os.path.abspath(os.path.dirname(__file__)),'data')
        path_test = os.path.join(data_path, f'xtest{i}.pkl')
        print(data_path)
        print(path_test)
        with open(path_test, "wb") as file:
            pickle.dump(xtest, file)
        print('done')
    return 'Saved correctly'

# prepare_and_save_data(12)
