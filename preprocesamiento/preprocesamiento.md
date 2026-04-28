## Ejecución

El archivo principal de preprocesamiento es:

```bash
    python preprocesamiento.py 
```

El preprocesamiento consta de lo siguiente:

```python
import pandas as pd
from normalizacionDelDataset import *
from construirValidation import dividir_dataset
```

Llamamos a los datasets y les asignamos variables para su uso a posteriori
```python
    training = '../dataset/train.data'
    validation = '../dataset/validation.data'
    test = '../dataset/test/poker-hand-testing.data'
```

* Importacion de pandas
* Importar el codigo para la denormalizacion
* Importar el codigo para la separacion del trainning de validacion

```python
OBJETIVO_PARA_ENTRENAR = {
    0: 10000,  # Nothing in hand
    1: 10000,  # One pair
    2: 10000,  # Two pairs
    3: 10000,  # Three of a kind
    4: 10000,  # Straight
    5: 1000,  # Flush
    6: 1000,  # Full house
}

```
Aqui dividariamos el dataset de training para generar el validation.data que procesaremos luego, sin embargo como ya realizamos esta accion comentaremos la linea 
```pyhton
    #dividir_dataset(training)

```

Posteriormente, se cargan los datos de entrenamiento, validation y test, de esta manera a

```python
    df_trainning = pd.read_csv(training, header=None, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS'])
    df_validation = pd.read_csv(validation, header=None, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS'])
    df_test = pd.read_csv(test, header=None, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS'])

  
```

Retiramos los valores de vacios de los datasets
```python
    df_trainning = quitarNull(df_trainning)
    df_validation = quitarNull(df_validation)
    df_test = quitarNull(df_test)
```

Realizamos la tecnica de oversampling y undersampling, seguir leyendo para mas info de estas tecnicas de preprocesamiento 
```python
    df_trainning = undersampling(df_trainning, OBJETIVO_PARA_ENTRENAR)

```
Exportamos el scaler y el encoder a sus respectivas carpetas ,con el objetivo de utilizarlos para noramlizar las queries de algun usuario
```python
    df_trainning_normalizado, scaler, encoder = normalizarDataset(df_trainning)
    joblib.dump(scaler, "../modelo/mlp_raw/scaler.pkl")
    joblib.dump(encoder, "../modelo/mlp_raw/encoder.pkl")
    joblib.dump(scaler, "../modelo/mlp_curado/scaler.pkl")
    joblib.dump(encoder, "../modelo/mlp_curado/encoder.pkl")
    joblib.dump(scaler, "../modelo/mlp_optimizado/scaler.pkl")
    joblib.dump(encoder, "../modelo/mlp_optimizado/encoder.pkl")

```

## Preprocesamiento del dataset

En esta etapa se aplican transformaciones a los conjuntos de entrenamiento, validación y prueba con el objetivo de preparar los datos para el modelo de aprendizaje automático.

Las técnicas utilizadas son:

- **One-Hot Encoding** para las variables categóricas (S1–S5)
- **Min-Max Scaling** para las variables numéricas (C1–C5)

Se utilizan funciones separadas debido a que el método `fit()` de `sklearn` **solo debe aplicarse al conjunto de entrenamiento**, ya que es el que aprende los parámetros del preprocesamiento (categorías, mínimos y máximos).  
Los conjuntos de validación y prueba únicamente aplican `transform()`, reutilizando dichos parámetros para evitar *data leakage*.

```python
df_trainning_normalizado, scaler, encoder = normalizarDatasetTrainning(df_trainning)
df_trainning_normalizado.to_csv('preprocesamiento_trainning.data', index=False, header=False)

df_validation_normalizado = normalizarDatasetValidationTest(df_validation, scaler, encoder)
df_validation_normalizado.to_csv('preprocesamiento_validation.data', index=False, header=False)

df_test_normalizado = normalizarDatasetValidationTest(df_test, scaler, encoder)
df_test_normalizado.to_csv('preprocesamiento_test.data', index=False, header=False)
```


## Tecnicas de preprocesamiento utilizadas

### Undersampling
El undersampling consiste en reducir el tamaño de la o las clases mayoritarias del dataset mediante el recorte aleatorio del dataset de prueba, con el objetivo de balancear el dataset.

Finalmente, se evalúa el balance del dataset para verificar la nueva distribución de clases y se guarda el conjunto preprocesado en un archivo .data.

```python
    def undersampling(df, objetivo):
        """
        Undersamplear: Reducir la cantidad de muestras de la clase mayoritaria
        para igualar a la clase minoritaria.
        Se usara para las clases 0 a 2, (Nothing in hand, One pair, Two pairs)
        que son las clases mas comunes en el dataset.
        """
        # Seleccionamos la columna final como y
        # y luego el resto como x
        nombre_columna = df.columns[-1]
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        conteos = y.value_counts().sort_index()

        # Solo aplicar undersampling a clases que necesitan bajar
        estrategia_under = {
            clase: meta
            for clase, meta in objetivo.items()
            if conteos.get(clase, 0) > meta
        }

        under = RandomUnderSampler(sampling_strategy=estrategia_under, random_state=42)
        X_bal, y_bal = under.fit_resample(X, y)

        # Crear un nuevo DataFrame con los datos balanceados
        df_balanceado = pd.DataFrame(X_bal, columns=df.columns[:-1])
        df_balanceado[nombre_columna] = y_bal

        print("=== UNDERSAMPLING ===")
        print("Clases reducidas:")
        for clase, meta in estrategia_under.items():
            print(f"  Clase {clase}: {conteos.get(clase, 0):,} → {meta:,} muestras")
        print(f"\nTotal antes:  {len(df):,}")
        print(f"Total después: {len(df_balanceado):,}")

        return df_balanceado
```

### Min-Max Scaler
En el Min-Max Scaler alteramos los valores para colocarlos dentro del rango de 0 a 1, haciendo que todos los valores de la característica se escalen proporcionalmente entre su valor mínimo y máximo.

### One Hot Encoding
El One Hot Encoding es una técnica utilizada para transformar variables categóricas en variables numéricas, creando una nueva columna binaria por cada categoría posible.

```python
    def normalizarDataset(df):
        nombre_columna = df.columns[-1]
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        suit_cols = ['S1', 'S2', 'S3', 'S4', 'S5']
        rank_cols = ['C1', 'C2', 'C3', 'C4', 'C5']

        encoder = OneHotEncoder(sparse_output=False)
        suit_encoded = encoder.fit_transform(x[suit_cols])
        suit_encoded_cols = encoder.get_feature_names_out(suit_cols)
        df_suits = pd.DataFrame(suit_encoded, columns=suit_encoded_cols)

        scaler = MinMaxScaler()
        rank_scaled = scaler.fit_transform(x[rank_cols])
        df_ranks = pd.DataFrame(rank_scaled, columns=rank_cols)

        df_normalizado = pd.concat([df_suits, df_ranks], axis=1)
        df_normalizado[nombre_columna] = y.values

        print("=== NORMALIZADO! ===")
        return df_normalizado, scaler, encoder
```


## Biografia
### Poker Dataset
https://walintonc.github.io/papers/ml_pokerhand.pdf


### Undersampling
https://towardsdatascience.com/a-good-machine-learning-classifiers-accuracy-metric-for-the-poker-hand-dataset-44cc3456b66d/ 
https://www.aluracursos.com/blog/como-lidiar-con-el-desbalanceo-de-datos

### Min-Max Scaler
https://medium.com/@iamkamleshrangi/how-min-max-scaler-works-9fbebb9347da

### One Hot Encoding
https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/ 