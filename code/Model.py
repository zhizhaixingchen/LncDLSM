import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Lambda, MaxPooling1D, Conv1D, Flatten, concatenate, GaussianDropout, \
    Dense

def squeeze(x):
    return K.squeeze(x, axis=-2)


def CNN(x, size):
    x = Conv2D(filters=16, kernel_size=(size, 4), activation='elu', kernel_initializer='glorot_normal',
               bias_initializer='glorot_normal')(x)
    x = Lambda(squeeze)(x)
    x = MaxPooling1D(pool_size=size)(x)
    x = Conv1D(filters=4, kernel_size=1, activation='elu', kernel_initializer='glorot_normal',
               bias_initializer='glorot_normal')(x)
    x = MaxPooling1D(pool_size=32 // size)(x)
    x = Flatten()(x)
    return x


# FFT Model
def FFT_model(input_fft):
    fft_1 = CNN(input_fft, 4)
    fft_2 = CNN(input_fft, 8)
    fft_3 = CNN(input_fft, 16)
    fft = concatenate([fft_1, fft_2, fft_3])
    GaussianDropout(0.1)
    fft = Dense(64, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(fft)
    fft = Dense(8, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(fft)
    return fft


# k-mer Model
def k_mer_model(input_1, input_2, input_3):
    GaussianDropout(0.1)
    x = Dense(256, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(input_1)
    x = Dense(64, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(x)
    GaussianDropout(0.1)
    y = Dense(64, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(input_2)
    merge_1 = concatenate([x, y])
    merge_1 = Dense(64, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(
        merge_1)
    merge_1 = Dense(16, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(
        merge_1)
    GaussianDropout(0.1)
    z = Dense(16, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(input_3)
    merge_2 = concatenate([merge_1, z])
    merge_2 = Dense(16, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(
        merge_2)
    merge_2 = Dense(8, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(merge_2)
    return merge_2


# model main
def LncDLSM_model(shape_fft, shape_1, shape_2, shape_3):
    # FFT
    input_fft = Input(shape=shape_fft)
    fft = FFT_model(input_fft=input_fft)
    # K-mer
    input_1 = Input(shape=shape_1)  # 1024
    input_2 = Input(shape=shape_2)  # 256
    input_3 = Input(shape=shape_3)  # 64
    """
    5mer:1024-256-64
    4mer:256-64
    concat:128-64-16
    3mer:64-16
    concat:32-16-8
    """
    merge_2 = k_mer_model(input_1, input_2, input_3)
    """
    final merge concat:16->8
    """
    merge = concatenate([fft, merge_2], name='advanced_characterization')
    merge = Dense(8, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal',
                  name="merge_init")(merge)

    # output
    # fft output
    fft_output = Dense(2, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='glorot_normal',
                       name='fft_output')(fft)
    # mer output
    mer_output = Dense(2, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='glorot_normal',
                       name='mer_output')(merge_2)
    # fft + mer output
    agg_output = Dense(2, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='glorot_normal',
                       name='main_output')(merge)

    fft_model = Model(inputs=[input_fft, input_1, input_2, input_3], outputs=[fft_output])
    mer_model = Model(inputs=[input_fft, input_1, input_2, input_3], outputs=[mer_output])
    agg_model = Model(inputs=[input_fft, input_1, input_2, input_3], outputs=[agg_output])

    return fft_model, mer_model, agg_model