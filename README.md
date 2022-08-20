# LncDLSM

## overview

The implementation of  *LncDLSM: Identification of Long Non-coding RNAs with Deep Learning-based Sequence Model*.

LncDLSM is a tool for computing the coding potential of  RNA sequences. For your convenience, an on-line website is provided as follow.

http://39.106.16.168/lncDLSM/

## Usage

### Data Preparation

1. Please download the datasets "lncDLSM_data.rar" from https://github.com/zhizhaixingchen/LncDLSM/releases/tag/lncDLSM_data.
2. Unzip "lncDLSM_data.rar" and put "data" folder in the path 'LncDLSM'.

### Training

- training base model

  ```python
  python train_base_model.py
  -trd <train data file>
  -trl <train label file>
  -ted <test data file>
  -tel <test label file>
  -mf <model save file>
  [-cm <[fft|mer|fft+mer]>]
  ```

- training fine-turn model

  ```python
  python train_finetune_model.py
  -trd <train data file>
  -trl <train label file>
  -ted <test data file>
  -tel <test label file>
  -mf <model save file>
  -bmf <base model file>
  ```

### Evaluation

```
python evaluate.py
-ff <fasta file>
-lf <label file>
-mf <model file>
-sf <save file>
```

### prediction

```python
python predict.py
-ff <fasta file>
-s <[human|rat|mouse|cow|pig]>
-sf <save text file>
-stf <save csv file>
```

