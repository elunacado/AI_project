import pandas as pd
import joblib
from normalizacionDelDataset import *


training = '../dataset/train.data'
validation = '../dataset/validation.data'
test = '../dataset/test/test.data'

OBJETIVO_PARA_ENTRENAR = {
    0: 10000,  # Nothing in hand
    1: 10000,  # One pair
    2: 10000,  # Two pairs
    3: 10000,  # Three of a kind
    4: 10000,  # Straight
    5: 1000,  # Flush
    6: 1000,  # Full house
    7: 1000,  # Four of a kind
    8: 1000,  # Straight flush
    9: 1000   # Royal flush
}


df_trainning = pd.read_csv(training, header=None, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS'])
df_validation = pd.read_csv(validation, header=None, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS'])
df_test = pd.read_csv(test, header=None, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS'])

df_trainning = quitarNull(df_trainning)
df_validation = quitarNull(df_validation)
df_test = quitarNull(df_test)


#Oversampling y undersampling exclusivos para el dataset de entrenamiento.
#df_trainning = oversampling(df_trainning, OBJETIVO_PARA_ENTRENAR)
df_trainning = undersampling(df_trainning, OBJETIVO_PARA_ENTRENAR)


df_trainning_normalizado, scaler, encoder = normalizarDataset(df_trainning)
joblib.dump(scaler, "../modelo/mlp_raw/scaler.pkl")
joblib.dump(encoder, "../modelo/mlp_raw/encoder.pkl")
joblib.dump(scaler, "../modelo/mlp_curado/scaler.pkl")
joblib.dump(encoder, "../modelo/mlp_curado/encoder.pkl")
joblib.dump(scaler, "../modelo/mlp_optimizado/scaler.pkl")
joblib.dump(encoder, "../modelo/mlp_optimizado/encoder.pkl")



df_trainning_normalizado.to_csv('../preprocesamiento/preprocesamiento_trainning.data', index=False, header=False)

df_validation_normalizado, scaler, encoder = normalizarDataset(df_validation)
df_validation_normalizado.to_csv('../preprocesamiento/preprocesamiento_validation.data', index=False, header=False)

df_test_normalizado, scaler, encoder = normalizarDataset(df_test)
df_test_normalizado.to_csv('../preprocesamiento/preprocesamiento_test.data', index=False, header=False)