from fastapi import APIRouter, HTTPException
from app.services.procesar_archivo_service import ProcesarArchivoService
from app.models.regresion_model import RegresionModel
from app.models.red_neuronal_model import RedNeuronalModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
from fastapi.responses import FileResponse, HTMLResponse
import plotly.graph_objects as go



router = APIRouter()

# Crear un directorio para almacenar las imágenes de los gráficos
if not os.path.exists("static/images"):
    os.makedirs("static/images")

@router.post("/entrenar-modelos/")
async def entrenar_modelos():
    try:
        # Cargar los datos desde MongoDB
        df = ProcesarArchivoService.cargar_datos_mongo()
        print("Columnas del DataFrame:", df.columns)

        # Crear la columna 'objetivo'
        df['objetivo'] = df['edad_']

        # Asegurarse de que la columna 'objetivo' está presente
        if 'objetivo' not in df.columns:
            raise HTTPException(status_code=400, detail="Columna 'objetivo' no encontrada en los datos")

        # Imputar valores faltantes con la media
        imputer = SimpleImputer(strategy='mean')
        df_imputado = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Separar las características (X) y el objetivo (y)
        X = df_imputado.drop('objetivo', axis=1)
        y = df_imputado['objetivo']

        # Normalizar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Entrenar el modelo de regresión
        regresion_model = RegresionModel()
        regresion_model.entrenar(X_scaled, y)

        # Entrenar la red neuronal
        red_neuronal_model = RedNeuronalModel(input_dim=X_scaled.shape[1])
        red_neuronal_model.entrenar(X_scaled, y)

        return {"message": "Modelos entrenados exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predecir/")
async def predecir():
    try:
        # Cargar los datos desde MongoDB
        df = ProcesarArchivoService.cargar_datos_mongo()

        # Crear la columna 'objetivo'
        df['objetivo'] = df['edad_']

        # Asegurarse de que la columna 'objetivo' está presente
        if 'objetivo' not in df.columns:
            raise HTTPException(status_code=400, detail="Columna 'objetivo' no encontrada en los datos")

        # Imputar valores faltantes con la media
        imputer = SimpleImputer(strategy='mean')
        df_imputado = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Normalizar los datos
        scaler = StandardScaler()
        X = df_imputado.drop('objetivo', axis=1)
        X_scaled = scaler.fit_transform(X)

        # Predicciones con regresión
        regresion_model = RegresionModel()
        predicciones_regresion = regresion_model.predecir(X_scaled)

        # Predicciones con red neuronal
        red_neuronal_model = RedNeuronalModel(input_dim=X_scaled.shape[1])
        predicciones_red_neuronal = red_neuronal_model.predecir(X_scaled)
        
        # Convertir las predicciones de la red neuronal a float
        predicciones_red_neuronal = [float(pred) for pred in predicciones_red_neuronal]

        print("Predicciones Regresión:", predicciones_regresion[:10])  # Imprime los primeros 10 valores
        print("Predicciones Red Neuronal:", predicciones_red_neuronal[:10])  # Imprime los primeros 10 valores

        # Crear gráfico interactivo con Plotly
        fig = go.Figure()

        # Agregar la primera línea de predicciones
        fig.add_trace(go.Scatter(
            y=predicciones_regresion,
            mode='markers+lines',
            name="Predicciones Regresión",
            marker=dict(size=6, color='blue'),
            line=dict(width=2, dash='solid')
        ))
        
        # Agregar la segunda línea de predicciones
        fig.add_trace(go.Scatter(
            y=predicciones_red_neuronal,
            mode='markers+lines',
            name="Predicciones Red Neuronal",
            marker=dict(size=6, color='orange'),
            line=dict(width=2, dash='dot')
        ))

        # Mejorar el diseño
        fig.update_layout(
            title="Comparación de Predicciones",
            xaxis_title="Índice",
            yaxis_title="Predicción",
            template="plotly_dark",
            autosize=True
        )

        # Guardar el gráfico como un archivo HTML para servirlo
        image_path = "static/images/predicciones_comparacion.html"
        fig.write_html(image_path)

        return {
            "predicciones_regresion": predicciones_regresion.tolist(),
            "predicciones_red_neuronal": predicciones_red_neuronal,
            "image_url": f"/static/images/predicciones_comparacion.html"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Servir el archivo HTML generado
@router.get("/static/images/{image_name}")
async def get_image(image_name: str):
    image_path = f"static/images/{image_name}"
    if os.path.exists(image_path):
        return HTMLResponse(content=open(image_path).read())
    raise HTTPException(status_code=404, detail="Image not found")
