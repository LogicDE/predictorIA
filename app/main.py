from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO
from typing import List

from app.services.database import get_database
from app.services.data_processing import normalizar_datos

app = FastAPI()

# Especificar las columnas requeridas y sus posibles variaciones
COLUMNAS_REQUERIDAS = {
    "fec_not": ["fec_not", "FEC_NOT"],
    "semana": ["semana", "SEMANA"],
    "año": ["año", "AÑO"],
    "edad_": ["edad_", "edad", "EDAD_"],
    "uni_med_": ["uni_med_", "uni_med", "UNI_MED_"],
    "sexo_": ["sexo_", "sexo", "SEXO"],
    "cod_dpto_o": ["cod_dpto_o", "COD_DPTO_O"],
    "cod_mun_o": ["cod_mun_o", "COD_MUN_O"],
    "tip_ss_": ["tip_ss_", "tip_ss", "ESTADO_P"]  # Incluye ESTADO_P como nombre alternativo
}

@app.post("/cargar-datos/", response_description="Carga de datos desde archivos Excel o CSV a MongoDB")
async def cargar_datos(archivos: List[UploadFile] = File(...)):
    collection = get_database()  # Accedemos directamente a la colección
    
    datos_completos = []  # Lista para acumular los datos procesados de cada archivo

    for archivo in archivos:
        if not archivo.filename.endswith((".xlsx", ".xls", ".csv", ".xlsm")):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos de Excel o CSV.")
        
        try:
            # Leer el archivo según su tipo
            contenido = await archivo.read()
            if archivo.filename.endswith((".xlsx", ".xls", ".xlsm")):
                datos = pd.read_excel(BytesIO(contenido))
            elif archivo.filename.endswith(".csv"):
                datos = pd.read_csv(BytesIO(contenido))

            # Convertir nombres de columnas a minúsculas para facilitar la coincidencia
            datos.columns = [col.lower() for col in datos.columns]

            # Crear un diccionario de coincidencias de columnas
            columnas_mapeadas = {}
            for col_requerida, nombres_posibles in COLUMNAS_REQUERIDAS.items():
                for nombre in nombres_posibles:
                    if nombre.lower() in datos.columns:
                        columnas_mapeadas[col_requerida] = nombre.lower()
                        break

            # Verificar si todas las columnas requeridas están presentes
            columnas_faltantes = set(COLUMNAS_REQUERIDAS) - set(columnas_mapeadas)
            if columnas_faltantes:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Columnas faltantes en el archivo {archivo.filename}: {', '.join(columnas_faltantes)}"
                )

            # Renombrar las columnas mapeadas en `datos` para que usen los nombres estandarizados
            datos.rename(columns=columnas_mapeadas, inplace=True)

            # Filtrar las columnas necesarias con nombres estandarizados
            datos_filtrados = datos[list(COLUMNAS_REQUERIDAS.keys())]

            # Convertir los datos a un formato JSON para insertarlos en MongoDB
            datos_json = normalizar_datos(datos_filtrados)
            datos_completos.extend(datos_json)  # Acumular datos de todos los archivos procesados

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al procesar el archivo {archivo.filename}: {str(e)}")

    # Intentar insertar todos los datos acumulados en MongoDB
    try:
        result = collection.insert_many(datos_completos)
        if result.inserted_ids:
            return {"mensaje": "Datos cargados exitosamente en MongoDB", "ids_insertados": str(result.inserted_ids)}
        else:
            raise HTTPException(status_code=500, detail="No se insertaron datos en MongoDB.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al insertar datos en MongoDB: {str(e)}")
