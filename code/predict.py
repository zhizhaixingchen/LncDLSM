import argparse
from util.preprocessing import generate_data
from tensorflow.python.keras import backend as K
import pandas as pd
from tensorflow.python.keras.models import load_model
import math
def focal_loss(y_true, y_pred, epsilon=0.1):
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    y_true = K.clip(y_true, epsilon, 1 - epsilon)
    return K.mean(K.abs(y_true - y_pred) * K.binary_crossentropy(y_true, y_pred), axis=-1)

def get_result(model_file, save_file, save_table_file, test_data, seq_id):
    model = load_model(model_file, custom_objects={"focal_loss": focal_loss})
    # model.summary()
    probability = model.predict(test_data)
    test_pd = pd.DataFrame(data=[seq_id, list(map(lambda x:math.floor(max(x)*1000)/1000,probability))],index=None).T
    test_pd.columns = ['seq_id', 'probability']
    test_pd['result'] = pd.DataFrame(probability).T.apply(lambda x: "lncRNA" if x[0] < 0.5 else "mRNA")

    new_columns = ['seq_id','result','probability']
    test_pd = test_pd[new_columns]
    test_pd.to_csv(save_table_file,index=False)
    with open(file=save_file,mode='w') as file:
        test_pd.apply(lambda row: file.write(f">{row['seq_id']}\n{row['result']}\n{row['probability']}\n\n"), axis=1)




if __name__ == '__main__':
    """
        python predict.py -ff ../data/human/test_data.fa -s human -sf ../output/test_result.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-ff", dest="fasta_file", help="fasta file",default="../data/human/test_data.fa")
    parser.add_argument("-s", dest="species", help="species",default="human")
    parser.add_argument("-sf", dest="save_file", help="save txt file",default="../output/output.txt")
    parser.add_argument("-stf", dest="save_table_file", help="save table file",default="../output/output_table.csv")
    conf = parser.parse_args()

    if conf.species == 'human':
        model_file = "../model2/model_human_base.hdf5"
    else:
        model_file = f"../model2/model_{conf.species}_finetune.hdf5"

    seq_id, test_data = generate_data(conf.fasta_file)
    get_result(model_file, conf.save_file, conf.save_table_file, test_data, seq_id)
