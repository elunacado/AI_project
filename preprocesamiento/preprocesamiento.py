import pandas as pd
from preprocesamiento.normalizacionDelDataset import *
from preprocesamiento.construirValidation import dividir_dataset

training = '../dataset/train.data'


OBJETIVO_PARA_ENTRENAR = {
    0: 100000,  # Nothing in hand
    1: 100000,  # One pair
    2: 30000,  # Two pairs
    3: 30000,  # Three of a kind
    4: 10000,  # Straight
    5: 10000,  # Flush
    6: 10000,  # Full house
    7: 1000,  # Four of a kind
    8: 100,  # Straight flush
    9: 100   # Royal flush
}


df_trainning = pd.read_csv(training, header=None)
df_trainning = quitarNull(df_trainning)
df_oversampleado = oversamplearDataset(df_trainning, OBJETIVO_PARA_ENTRENAR)
df_final = undersamplearDataset(df_oversampleado, OBJETIVO_PARA_ENTRENAR)
df_final, scaler = normalizarDataset(df_final)


evaluacionDelBalanceoDelDataset(df_final)

df_final.to_csv('preprocessed_train.data', index=False, header=None)
