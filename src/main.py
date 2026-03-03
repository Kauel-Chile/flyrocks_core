import os
import json
import uuid
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, status, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from core.config import Config
from service import run_tracking_pipeline

app = FastAPI(title="Flyrocks Tracker API - Concurrent")

# Habilitamos CORS para que cualquier Frontend (HTML local o React/Angular) pueda conectarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionario en memoria para rastrear múltiples trabajos simultáneos
# Estructura: { "job_id_1": { estado... }, "job_id_2": { estado... } }
jobs = {}

def update_progress(job_id: str, current: int, total: int, current_status: str, result_path: str = None):
    """Callback para actualizar el estado de un trabajo específico."""
    if job_id in jobs:
        jobs[job_id]["current_frame"] = current
        jobs[job_id]["total_frames"] = total
        jobs[job_id]["status"] = current_status
        
        if result_path:
            jobs[job_id]["result_file_path"] = result_path
            
        if current_status in ["Completado", "Error"]:
            jobs[job_id]["is_running"] = False

def background_tracking_task(config: Config, job_id: str):
    """Envuelve el pipeline, inyecta el callback y limpia la basura al terminar."""
    def callback(curr, tot, stat, res=None):
        update_progress(job_id, curr, tot, stat, res)
        
    try:
        # Ejecutamos el servicio pasándole el ID único
        run_tracking_pipeline(config=config, job_id=job_id, progress_callback=callback)
    except Exception as e:
        update_progress(job_id, 0, 0, f"Error: {str(e)}", None)
        print(f"Error en job {job_id}: {e}")
    finally:
        # IMPORTANTE: Eliminamos el video temporal para no llenar el disco del servidor
        if os.path.exists(config.VIDEO_PATH):
            try:
                os.remove(config.VIDEO_PATH)
                print(f"Video temporal {config.VIDEO_PATH} eliminado.")
            except Exception as cleanup_error:
                print(f"No se pudo eliminar el video temporal: {cleanup_error}")

# ==========================================
# 1. ENDPOINT REST: SUBIDA E INICIO (202 Accepted)
# ==========================================
@app.post("/api/v1/analyze", status_code=status.HTTP_202_ACCEPTED)
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    origin_zone: str = Form(...),
    expected_projection_zone: str = Form(...),
    h_matrix: str = Form(...)
):
    # Generamos un ID único para este análisis
    job_id = str(uuid.uuid4())
    
    # Inicializamos el estado en el diccionario global
    jobs[job_id] = {
        "is_running": True,
        "status": "Preparando...",
        "current_frame": 0,
        "total_frames": 0,
        "result_file_path": None
    }

    # Guardamos el video temporalmente con el job_id para evitar colisiones
    video_path = f"temp_{job_id}_{video.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    # Parseamos los datos JSON enviados desde el Frontend
    try:
        origin_list = json.loads(origin_zone)
        projection_list = json.loads(expected_projection_zone)
        h_matrix_list = json.loads(h_matrix)
    except json.JSONDecodeError:
        # Si envían un JSON mal formado, limpiamos y lanzamos error
        os.remove(video_path)
        del jobs[job_id]
        raise HTTPException(status_code=400, detail="Formato JSON inválido en las zonas o matriz.")

    # Construimos la configuración para este trabajo específico
    config = Config(
        video_path=video_path,
        origin_zone=origin_list,
        projection_zone=projection_list,
        h_matrix=h_matrix_list
    )

    # Enviamos a segundo plano (Threadpool)
    background_tasks.add_task(background_tracking_task, config, job_id)

    # Devolvemos inmediatamente el ID para que el frontend pueda escuchar
    return {
        "message": "Video e información recibidos. Procesamiento iniciado.",
        "job_id": job_id
    }

# ==========================================
# 2. ENDPOINT REST: DESCARGA DE RESULTADOS
# ==========================================
@app.get("/api/v1/results/download/{job_id}")
async def download_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="ID de trabajo no encontrado.")
        
    job = jobs[job_id]
    if job["is_running"]:
        raise HTTPException(status_code=400, detail="El análisis aún está en ejecución.")
        
    result_path = job.get("result_file_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Archivo de resultados no encontrado. Quizás falló.")
        
    return FileResponse(
        path=result_path, 
        media_type="application/json", 
        filename=f"flyrocks_resultados_{job_id}.json"
    )

# ==========================================
# 3. WEBSOCKET: ESTADO EN TIEMPO REAL
# ==========================================
@app.websocket("/ws/v1/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    
    if job_id not in jobs:
        await websocket.send_json({"error": "ID de trabajo no encontrado", "status": "Error"})
        await websocket.close()
        return

    try:
        while True:
            job = jobs[job_id]
            total = job["total_frames"]
            current = job["current_frame"]
            percentage = round((current / total * 100), 2) if total > 0 else 0

            await websocket.send_json({
                "status": job["status"],
                "current_frame": current,
                "total_frames": total,
                "percentage": percentage,
                "is_running": job["is_running"]
            })
            
            if not job["is_running"]:
                await asyncio.sleep(1) # Pequeño delay para asegurar que el frontend leyó el 100%
                break
                
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print(f"Cliente con Job ID {job_id} desconectado del WebSocket.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)