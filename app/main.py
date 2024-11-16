from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from app.services.procesar_archivo_service import ProcesarArchivoService  # Asegúrate de que el import sea correcto
from app.controllers import  mortalidad_controller

app = FastAPI()

# Ruta para cargar archivos .xlsx
@app.post("/cargar-datos-xlsx/", response_description="Carga de datos desde archivos Excel (.xlsx)")
async def cargar_datos_xlsx(archivo: UploadFile = File(...), limite: int = 100):
    try:
        contenido = await archivo.read()
        archivo_bytes = BytesIO(contenido)
        
        # Procesamos el archivo usando la clase ProcesarArchivoService
        df = ProcesarArchivoService.cargar_datos_xlsx(archivo_bytes)

        # Convertimos el DataFrame a una lista de diccionarios
        datos_json = df.head(limite).to_dict(orient="records")

        # Insertamos los datos en MongoDB
        inserted_ids = ProcesarArchivoService.insertar_datos_mongo(datos_json)
        
        return JSONResponse(content={"datos_insertados": inserted_ids})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")


# Ruta para cargar archivos .xls
@app.post("/cargar-datos-xls/", response_description="Carga de datos desde archivos Excel antiguos (.xls)")
async def cargar_datos_xls(archivo: UploadFile = File(...), limite: int = 100):
    try:
        contenido = await archivo.read()
        archivo_bytes = BytesIO(contenido)
        
        # Procesamos el archivo usando la clase ProcesarArchivoService
        df = ProcesarArchivoService.cargar_datos_xls(archivo_bytes)

        # Convertimos el DataFrame a una lista de diccionarios
        datos_json = df.head(limite).to_dict(orient="records")

        # Insertamos los datos en MongoDB
        inserted_ids = ProcesarArchivoService.insertar_datos_mongo(datos_json)
        
        return JSONResponse(content={"datos_insertados": inserted_ids})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

app.include_router(mortalidad_controller.router)