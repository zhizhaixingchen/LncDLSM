import argparse

from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, f1_score
import tensorflow as tf

from Model import LncDLSM_model
from util.preprocessing import generate_data
import tensorflow.python.keras.backend as K
import numpy as np
import os
from tensorflow.python.keras.models import load_model
from data_handle import label2dense

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

def focal_loss(y_true, y_pred, epsilon=0.1):
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    y_true = K.clip(y_true, epsilon, 1 - epsilon)
    return K.mean(K.abs(y_true - y_pred) * K.binary_crossentropy(y_true, y_pred), axis=-1)


def get_result_online(test_data, model_file):
    model = load_model(model_file, custom_objects={"focal_loss": focal_loss})
    preds = model.predict(test_data)
    return preds


def evaluate(y_trues, y_prob, save_file):
    auc = roc_auc_score(y_trues, y_prob)
    y_true = np.argmax(y_trues, axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    sen = recall_score(y_true, y_pred)
    spe = recall_score(1 - y_true, 1 - y_pred)
    acc = accuracy_score(y_true, y_pred)
    f_1 = f1_score(y_true, y_pred)
    print('Sensitivity:{:.4f}\nSpecificity:{:.4f}\nAccuracy:{:.4f}\nF1-score:{:.4f}\nAUC:{:.4f}'
          .format(sen, spe, acc, f_1, auc))
    with open(save_file, 'w') as sf:
        sf.write('Sensitivity:{:.4f}\nSpecificity:{:.4f}\nAccuracy:{:.4f}\nF1-score:{:.4f}\nAUC:{:.4f}'
                 .format(sen, spe, acc, f_1, auc))


if __name__ == '__main__':
    """
    python evaluate.py -ff ../data/human/test_data.fa -lf ../data/human/test_label.npy -mf ../model/model_human_base.hdf5 -sf ../output/test_result.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-ff", dest="fasta_file", help="fasta file")
    parser.add_argument("-lf", dest="label_file", help="label file")
    parser.add_argument("-mf", dest="model_file", help="model file")
    parser.add_argument("-sf", dest="save_file", help="save file")
    conf = parser.parse_args()
    # test_data = []
    _, test_data = generate_data(conf.fasta_file)
    label = label2dense(conf.label_file)
    preds = get_result_online(test_data, conf.model_file)
    evaluate(label, preds, conf.save_file)
