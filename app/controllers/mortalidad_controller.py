from fastapi import APIRouter, HTTPException
from app.services.procesar_archivo_service import ProcesarArchivoService
from app.models.regresion_model import RegresionModel
from app.models.red_neuronal_model import RedNeuronalModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
from fastapi.responses import FileResponse

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

        # Generar gráfico de las predicciones con más espacio entre los puntos
        fig, ax = plt.subplots(figsize=(12, 6)) # Aumentamos el tamaño de la figura
        ax.plot(predicciones_regresion, label="Predicciones Regresión", marker='o', markersize=6, linestyle='-', linewidth=2)
        ax.plot(predicciones_red_neuronal, label="Predicciones Red Neuronal", marker='x', markersize=8, linestyle='--', linewidth=2)
        ax.set_xlabel("Índice", fontsize=14)
        ax.set_ylabel("Predicción", fontsize=14)
        ax.set_title("Comparación de Predicciones", fontsize=16)
        ax.legend()

        # Aumentar el espaciado entre los ticks en el eje X para mayor claridad
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        fig.tight_layout()  # Ajustar el espaciado de la figura automáticamente

        # Guardar el gráfico como imagen
        image_path = "static/images/predicciones_comparacion.png"
        fig.savefig(image_path)
        plt.close(fig)

        return {
            "predicciones_regresion": predicciones_regresion.tolist(),
            "predicciones_red_neuronal": predicciones_red_neuronal.tolist(),
            "image_url": f"/static/images/predicciones_comparacion.png"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Servir la imagen generada
@router.get("/static/images/{image_name}")
async def get_image(image_name: str):
    image_path = f"static/images/{image_name}"
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="Image not found")
