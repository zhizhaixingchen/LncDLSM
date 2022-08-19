import numpy as np
from util.preprocessing import generate_data
from sklearn.preprocessing import OneHotEncoder

def load_data(train_data, test_data):
    _, [train_fft, train_5_mer, train_4_mer, train_3_mer] = generate_data(train_data)
    train_fft = np.array(train_fft)
    train_5_mer = np.array(train_5_mer)
    train_4_mer = np.array(train_4_mer)
    train_3_mer = np.array(train_3_mer)
    train_data = [train_fft, train_5_mer, train_4_mer, train_3_mer]

    _, [test_fft, test_5_mer, test_4_mer, test_3_mer] = generate_data(test_data)
    test_fft = np.array(test_fft)
    test_5_mer = np.array(test_5_mer)
    test_4_mer = np.array(test_4_mer)
    test_3_mer = np.array(test_3_mer)
    test_data = [test_fft, test_5_mer, test_4_mer, test_3_mer]
    return train_data, test_data


def label2dense(label_file):
    arr = np.load(label_file)
    arr = arr.reshape(-1, 1)
    arr = OneHotEncoder().fit_transform(arr).todense()
    return np.array(arr)
