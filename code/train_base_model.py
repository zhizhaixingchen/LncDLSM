from data_handle import load_data, label2dense
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow import optimizers
from Model import LncDLSM_model
import tensorflow.python.keras.backend as K
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def focal_loss(y_true, y_pred, epsilon=0.1):
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    y_true = K.clip(y_true, epsilon, 1 - epsilon)
    return K.mean(K.abs(y_true - y_pred) * K.binary_crossentropy(y_true, y_pred), axis=-1)


def train(train_data, test_data, train_label, test_label, model_file, choose_model):
    shape_fft = train_data[0].shape[1:]  # (1500,4,2)
    shape_1 = train_data[1].shape[1:]  # 1024
    shape_2 = train_data[2].shape[1:]  # 256
    shape_3 = train_data[3].shape[1:]  # 64

    batch_size = 32
    epoch = 40
    optimizer = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, schedule_decay=0.002)
    loss = focal_loss
    metric = ['accuracy']

    fft_model, mer_model, agg_model = LncDLSM_model(shape_fft=shape_fft, shape_1=shape_1, shape_2=shape_2,
                                                    shape_3=shape_3)
    if choose_model == 'fft':
        final_model = fft_model
    elif choose_model == 'mer':
        final_model = mer_model
    else:
        final_model = agg_model

    checkpointer = ModelCheckpoint(monitor='accuracy', filepath=model_file, save_best_only=True, mode='max')
    final_model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    final_model.summary()
    final_model.fit(x=train_data,
                    y=train_label,
                    batch_size=batch_size, epochs=epoch,
                    validation_data=(test_data, test_label),
                    callbacks=[checkpointer])


if __name__ == '__main__':
    """
        python train_base_model.py -trd ../data/human/train_data.fa -trl ../data/human/train_label.npy
        -ted ../data/human/test_data.fa -tel ../data/human/test_label.npy
        -mf ../model/model_human_base.hdf5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-trd", dest="train_data", help="train data")
    parser.add_argument("-trl", dest="train_label", help="train label")
    parser.add_argument("-ted", dest="test_data", help="train data")
    parser.add_argument("-tel", dest="test_label", help="train label")
    parser.add_argument("-mf", dest="model_file", help="model save file")
    parser.add_argument("-cm", dest="choose_model", help="fft mer or fft+mer", default='fft+mer')
    conf = parser.parse_args()
    train_data, test_data = load_data(conf.train_data, conf.test_data)
    train_label = label2dense(conf.train_label)
    test_label = label2dense(conf.test_label)
    train(train_data, test_data, train_label, test_label, conf.model_file, conf.choose_model)
