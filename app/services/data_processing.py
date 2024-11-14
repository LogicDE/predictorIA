import pandas as pd

def normalizar_datos(datos: pd.DataFrame):
    columnas_fecha = ["FEC_NOT", "FEC_CON", "FEC_HOS", "FEC_DEF", "FECHA_NTO"]
    for col in columnas_fecha:
        datos[col] = pd.to_datetime(datos[col], errors='coerce')

    datos = datos.drop_duplicates()
    datos = datos.dropna(subset=["CONSECUTIVE", "COD_EVE", "EDAD"])

    datos_json = datos.to_dict(orient="records")
    return datos_json
