## Dataset

Se utilizó el siguiente dataset:

**Poker Hand Classification**  
https://www.kaggle.com/datasets/dysphorfia/poker-hand-classification

---

## Objetivo de la entrega

Para esta entrega se solicitó:

- Seleccionar un dataset  
- Realizar la separación de los datos  
- Aplicar un proceso de preprocesamiento  
- Preparar el dataset para su uso en un modelo de aprendizaje automático  

---

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
training = '../dataset/train.data'
OBJETIVO_PARA_ENTRENAR = {
    0: 100000,  # High Card
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
#Oversampling y undersampling exclusivos para el dataset de entrenamiento.
df_trainning = oversampling(df_trainning, OBJETIVO_PARA_ENTRENAR)
df_trainning = undersampling(df_trainning, OBJETIVO_PARA_ENTRENAR)
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
### Oversampling
Aumentar la cantidad de instancias de la clase minoritaria mediante la duplicacion de instancias aleatorias para brindarles una mayor presencia en el dataset

### Undersampling
El undersampling consiste en reducir el tamaño de la o las clases mayoritarias del dataset mediante el recorte aleatorio del dataset de prueba, con el objetivo de balancear el dataset.

Finalmente, se evalúa el balance del dataset para verificar la nueva distribución de clases y se guarda el conjunto preprocesado en un archivo .data.

### Min-Max Scaler
En el Min-Max Scaler alteramos los valores para colocarlos dentro del rango de 0 a 1, haciendo que todos los valores de la característica se escalen proporcionalmente entre su valor mínimo y máximo.

### One Hot Encoding
El One Hot Encoding es una técnica utilizada para transformar variables categóricas en variables numéricas, creando una nueva columna binaria por cada categoría posible.

## Biografia
### Poker Dataset use
https://walintonc.github.io/papers/ml_pokerhand.pdf


### Oversamling and Undersampling
https://towardsdatascience.com/a-good-machine-learning-classifiers-accuracy-metric-for-the-poker-hand-dataset-44cc3456b66d/ 
https://www.aluracursos.com/blog/como-lidiar-con-el-desbalanceo-de-datos

### Min-Max Scaler
https://medium.com/@iamkamleshrangi/how-min-max-scaler-works-9fbebb9347da

### One Hot Encoding
https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/ 