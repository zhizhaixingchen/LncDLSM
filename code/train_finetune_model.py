import argparse
import os
from data_handle import load_data, label2dense
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.models import load_model

# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'


def focal_loss(y_true, y_pred, epsilon=0.1):
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    y_true = K.clip(y_true, epsilon, 1 - epsilon)
    return K.mean(K.abs(y_true - y_pred) * K.binary_crossentropy(y_true, y_pred), axis=-1)


def train(train_data, test_data, train_label, test_label, model_file, base_model_file):
    batch_size = 32
    epoch = 40
    metric = ['accuracy']
    model = load_model(base_model_file, custom_objects={'focal_loss': focal_loss})
    checkpointer = ModelCheckpoint(monitor='accuracy', filepath=model_file,
                                   save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.75, patience=1, min_lr=0.000001, verbose=0, mode='max')
    earlystop = EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=6, verbose=0, mode='max')
    model.compile(optimizer="adam", loss=focal_loss, metrics=metric)
    model.summary()
    model.fit(x=train_data,
              y=train_label,
              batch_size=batch_size, epochs=epoch,
              validation_data=(test_data, test_label),
              callbacks=[checkpointer, reduce_lr, earlystop])
if __name__ == '__main__':
    """
        python train_finetune_model.py -trd ../data/cow/train_data.fa -trl ../data/cow/train_label.fa
            -ted ../data/cow/test_data.fa -tel ../data/cow/test_label.fa
            -mf ../model/model_cow_finetune.hdf5
            -bmf ../model/model_human_base.hdf5
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-trd", dest="train_data", help="train data")
    parser.add_argument("-trl", dest="train_label", help="train label")
    parser.add_argument("-ted", dest="test_data", help="train data")
    parser.add_argument("-tel", dest="test_label", help="train label")
    parser.add_argument("-mf", dest="model_file", help="model save file")
    parser.add_argument("-bmf", dest="base_model_file", help="base model file")
    conf = parser.parse_args()
    train_data, test_data = load_data(conf.train_data, conf.test_data)
    train_label = label2dense(conf.train_label)
    test_label = label2dense(conf.test_label)
    train(train_data, test_data, train_label, test_label, conf.model_file, conf.base_model_file)
