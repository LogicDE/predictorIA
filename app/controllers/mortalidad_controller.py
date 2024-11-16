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


@router.get("/dashboard/")
async def dashboard():
    try:
        # Cargar datos desde MongoDB
        df = ProcesarArchivoService.cargar_datos_mongo()

        df = df.fillna('No disponible')  # O usa df = df.applymap(lambda x: None if isinstance(x, float) and (x != x) else x)

        # Estadísticas descriptivas
        estadisticas = df.describe(include='all').fillna('No disponible').to_dict()

        # 1. Distribución de edades
        fig_edad = go.Figure()
        fig_edad.add_trace(go.Histogram(x=df['edad_'], nbinsx=20, marker=dict(color='blue')))
        fig_edad.update_layout(
            title="Distribución de Edades",
            xaxis_title="Edad",
            yaxis_title="Frecuencia",
            template="plotly_white"
        )
        edad_path = "static/images/distribucion_edades.html"
        fig_edad.write_html(edad_path)

        # 2. Distribución por sexo
        if 'sexo_' in df.columns:
            sexo_counts = df['sexo_'].value_counts()
            fig_sexo = go.Figure(data=[go.Pie(labels=sexo_counts.index, values=sexo_counts.values)])
            fig_sexo.update_layout(
                title="Distribución por Sexo"
            )
            sexo_path = "static/images/distribucion_sexo.html"
            fig_sexo.write_html(sexo_path)
        else:
            sexo_path = None

        # 3. Comparativa por año y mes
        if {'año', 'mes'}.issubset(df.columns):
            df_grouped = df.groupby(['año', 'mes']).size().reset_index(name='frecuencia')
            df_grouped = df_grouped.dropna(subset=['año', 'mes'])
            df_grouped['fecha'] = pd.to_datetime(
                df_grouped[['año', 'mes']].astype(str).agg('-'.join, axis=1) + '-01',  # Agregamos '-01' para el día
                errors='coerce'
            )
            df_grouped = df_grouped.dropna(subset=['fecha'])
            fig_tiempo = go.Figure()
            fig_tiempo.add_trace(go.Scatter(
                x=df_grouped['fecha'],
                y=df_grouped['frecuencia'],
                mode='lines+markers',
                name='Tendencia'
            ))
            fig_tiempo.update_layout(
                title="Frecuencia de Casos por Año y Mes",
                xaxis_title="Fecha",
                yaxis_title="Frecuencia",
                template="plotly_white"
            )
            tiempo_path = "static/images/frecuencia_tiempo.html"
            fig_tiempo.write_html(tiempo_path)
        else:
            tiempo_path = None

        # 4. Distribución por tipo de evento (nom_eve)
        if 'nom_eve' in df.columns:
            evento_counts = df['nom_eve'].value_counts()
            fig_evento = go.Figure(data=[go.Bar(
                x=evento_counts.index,
                y=evento_counts.values,
                marker=dict(color='orange')
            )])
            fig_evento.update_layout(
                title="Distribución por Tipo de Evento",
                xaxis_title="Evento",
                yaxis_title="Frecuencia",
                template="plotly_white"
            )
            evento_path = "static/images/distribucion_eventos.html"
            fig_evento.write_html(evento_path)
        else:
            evento_path = None

        # 5. Distribución de síntomas más comunes
        sintomas_comunes = ['fiebre', 'cefalea', 'vomito', 'diarrea']
        fig_sintomas = go.Figure()
        for sintoma in sintomas_comunes:
            if sintoma in df.columns:
                sintoma_counts = df[sintoma].value_counts()
                fig_sintomas.add_trace(go.Bar(
                    x=sintoma_counts.index,
                    y=sintoma_counts.values,
                    name=sintoma
                ))
        fig_sintomas.update_layout(
            title="Distribución de Síntomas Comunes",
            xaxis_title="Síntomas",
            yaxis_title="Frecuencia",
            template="plotly_white"
        )
        sintomas_path = "static/images/distribucion_sintomas.html"
        fig_sintomas.write_html(sintomas_path)

        # 6. Distribución geográfica de casos
        if 'cod_mun_o' in df.columns and 'localidad_' in df.columns and not df['localidad_'].isnull().all():
            geografia_counts = df.groupby('localidad_').size()
            fig_geografia = go.Figure(data=[go.Bar(
                x=geografia_counts.index,
                y=geografia_counts.values,
                marker=dict(color='green')
            )])
            fig_geografia.update_layout(
                title="Distribución Geográfica de Casos",
                xaxis_title="Localidad",
                yaxis_title="Frecuencia",
                template="plotly_white"
            )
            geografia_path = "static/images/distribucion_geografica.html"
            fig_geografia.write_html(geografia_path)
        else:
            geografia_path = None

        # 7. Relación entre tipo de régimen de salud y número de casos
        if 'tip_ss_' in df.columns and not df['tip_ss_'].isnull().all():
            regimen_counts = df.groupby('tip_ss_').size()
            fig_regimen = go.Figure(data=[go.Bar(
                x=regimen_counts.index,
                y=regimen_counts.values,
                marker=dict(color='purple')
            )])
            fig_regimen.update_layout(
                title="Relación entre Tipo de Régimen de Salud y Casos",
                xaxis_title="Régimen de Salud",
                yaxis_title="Frecuencia",
                template="plotly_white"
            )
            regimen_path = "static/images/relacion_regimen_salud.html"
            fig_regimen.write_html(regimen_path)
        else:
            regimen_path = None

        return {
            "estadisticas": estadisticas,
            "graficos": {
                "distribucion_edades": f"/{edad_path}" if edad_path else None,
                "distribucion_sexo": f"/{sexo_path}" if sexo_path else None,
                "frecuencia_tiempo": f"/{tiempo_path}" if tiempo_path else None,
                "distribucion_eventos": f"/{evento_path}" if evento_path else None,
                "distribucion_sintomas": f"/{sintomas_path}" if sintomas_path else None,
                "distribucion_geografica": f"/{geografia_path}" if geografia_path else None,
                "relacion_regimen_salud": f"/{regimen_path}" if regimen_path else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando el dashboard: {str(e)}")




# Servir el archivo HTML generado
@router.get("/static/images/{image_name}")
async def get_image(image_name: str):
    image_path = f"static/images/{image_name}"
    if os.path.exists(image_path):
        return HTMLResponse(content=open(image_path).read())
    raise HTTPException(status_code=404, detail="Image not found")
