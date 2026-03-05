import os
import json
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, status, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from core.config import Config
from core.database import Job, engine, Session, SQLModel
from service import run_tracking_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lógica de inicio (Startup)
    SQLModel.metadata.create_all(engine)
    yield
    
app = FastAPI(title="Flyrocks Tracker API - Refactored", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- TAREAS DE FONDO ---
def background_tracking_task(config: Config, job_id: str):
    """Ejecuta el pipeline y actualiza la base de datos mediante el modelo."""
    def progress_callback(curr, tot, stat, res=None):
        # Determinamos si debe seguir marcado como corriendo
        running = stat not in ["Completado", "Error"]
        Job.update_status(
            job_id, 
            engine, 
            current_frame=curr, 
            total_frames=tot, 
            status=stat, 
            result_file_path=res,
            is_running=running
        )
        
    try:
        run_tracking_pipeline(config=config, job_id=job_id, progress_callback=progress_callback)
    except Exception as e:
        Job.update_status(job_id, engine, status=f"Error: {str(e)}", is_running=False)
    finally:
        # Limpieza de archivos temporales
        if os.path.exists(config.VIDEO_PATH):
            try:
                os.remove(config.VIDEO_PATH)
            except Exception as ex:
                print(f"Error limpiando video: {ex}")

# --- ENDPOINTS ---

@app.post("/api/analyze", status_code=status.HTTP_202_ACCEPTED)
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    origin_zone: str = Form(...),
    expected_projection_zone: str = Form(...),
    h_matrix: str = Form(...)
):
    # Crear registro inicial
    new_job = Job()
    with Session(engine) as session:
        session.add(new_job)
        session.commit()
        session.refresh(new_job)
    
    job_id = new_job.id
    video_path = f"temp_{job_id}_{video.filename}"

    # Guardar video físicamente
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    try:
        # Configuración del pipeline
        config = Config(
            video_path=video_path,
            origin_zone=json.loads(origin_zone),
            projection_zone=json.loads(expected_projection_zone),
            h_matrix=json.loads(h_matrix)
        )
        # Lanzar tarea asíncrona
        background_tasks.add_task(background_tracking_task, config, job_id)
        
        return {"job_id": job_id, "message": "Procesamiento iniciado"}
        
    except json.JSONDecodeError:
        os.remove(video_path)
        Job.update_status(job_id, engine, status="Error: JSON Inválido", is_running=False)
        raise HTTPException(status_code=400, detail="Formato JSON incorrecto en parámetros.")

@app.get("/api/results/{job_id}")
async def download_results(job_id: str):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="ID de trabajo no encontrado.")
        
        if job.is_running:
            raise HTTPException(status_code=400, detail="El análisis sigue en curso.")
            
        if not job.result_file_path or not os.path.exists(job.result_file_path):
            raise HTTPException(status_code=404, detail="Archivo de resultados no disponible.")
            
        return FileResponse(
            path=job.result_file_path, 
            filename=f"flyrocks_{job_id}.json"
        )

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    
    try:
        while True:
            with Session(engine) as session:
                job = session.get(Job, job_id)
                
                if not job:
                    await websocket.send_json({"error": "No encontrado"})
                    break

                percentage = round((job.current_frame / job.total_frames * 100), 2) if job.total_frames > 0 else 0
                
                await websocket.send_json({
                    "status": job.status,
                    "current": job.current_frame,
                    "total": job.total_frames,
                    "percentage": percentage,
                    "is_running": job.is_running
                })
                
                if not job.is_running:
                    break
                
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)