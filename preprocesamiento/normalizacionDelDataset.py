import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler,  OneHotEncoder

def quitarNull(df):
    df = df.dropna()          
    return df

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

def normalizarQuery(df, scaler, encoder):
     # Separar features y etiqueta
    nombre_columna = df.columns[-1]
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Columnas
    suit_cols = ['S1', 'S2', 'S3', 'S4', 'S5']
    rank_cols = ['C1', 'C2', 'C3', 'C4', 'C5']

    # 🔹 One-Hot Encoding (SIN fit)
    suit_encoded = encoder.transform(x[suit_cols])
    suit_encoded_cols = encoder.get_feature_names_out(suit_cols)
    df_suits = pd.DataFrame(suit_encoded, columns=suit_encoded_cols, index=df.index)

    # 🔹 Escalado (SIN fit)
    rank_scaled = scaler.transform(x[rank_cols])
    df_ranks = pd.DataFrame(rank_scaled, columns=rank_cols, index=df.index)

    # 🔹 Concatenar
    df_final = pd.concat([df_suits, df_ranks], axis=1)

    # 🔹 Agregar etiqueta
    df_final[nombre_columna] = y.values

    return df_final

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