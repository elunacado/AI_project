import pandas as pd
import joblib
from normalizacionDelDataset import *


training = '../dataset/train.data'
validation = '../dataset/validation.data'
test = '../dataset/test/test.data'

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


df_trainning = pd.read_csv(training, header=None, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS'])
df_validation = pd.read_csv(validation, header=None, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS'])
df_test = pd.read_csv(test, header=None, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS'])

df_trainning = quitarNull(df_trainning)
df_validation = quitarNull(df_validation)
df_test = quitarNull(df_test)


#Oversampling y undersampling exclusivos para el dataset de entrenamiento.
df_trainning = oversampling(df_trainning, OBJETIVO_PARA_ENTRENAR)
df_trainning = undersampling(df_trainning, OBJETIVO_PARA_ENTRENAR)


df_trainning_normalizado, scaler, encoder = normalizarDataset(df_trainning)
joblib.dump(scaler, "../modelo/mlp_raw/scaler.pkl")
joblib.dump(encoder, "../modelo/mlp_raw/encoder.pkl")
df_trainning_normalizado.to_csv('preprocesamiento_trainning.data', index=False, header=False)

df_validation_normalizado, scaler, encoder = normalizarDataset(df_validation)
df_validation_normalizado.to_csv('preprocesamiento_validation.data', index=False, header=False)

df_test_normalizado, scaler, encoder = normalizarDataset(df_test)
df_test_normalizado.to_csv('preprocesamiento_test.data', index=False, header=False)