import asyncio
import asyncio

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from service import run_tracking_pipeline

app = FastAPI(title="Flyrocks Tracker API")

# Estado global simple para la aplicación
# (En producción con múltiples usuarios simultáneos, se usaría Redis o una BD)
job_state = {
    "is_running": False,
    "current_frame": 0,
    "total_frames": 0,
    "status": "Inactivo"
}

def update_progress(current: int, total: int, status: str):
    """Callback que el procesador de video llamará para actualizar el estado."""
    job_state["current_frame"] = current
    job_state["total_frames"] = total
    job_state["status"] = status
    
    if status in ["Completado", "Error: No se pudo leer el video"]:
        job_state["is_running"] = False

def background_tracking_task():
    """Wrapper para correr el pipeline e inyectar el callback."""
    job_state["is_running"] = True
    # Llamamos a nuestro servicio pasándole la función de actualización
    run_tracking_pipeline(progress_callback=update_progress)

# ==========================================
# ENDPOINTS REST
# ==========================================

@app.post("/api/start-analysis")
async def start_analysis(background_tasks: BackgroundTasks):
    """Endpoint para iniciar el procesamiento del video."""
    if job_state["is_running"]:
        return {"message": "El análisis ya se encuentra en ejecución."}
    
    job_state["status"] = "Preparando..."
    job_state["current_frame"] = 0
    job_state["total_frames"] = 0
    
    # FastAPI envía automáticamente las funciones síncronas a un ThreadPool
    background_tasks.add_task(background_tracking_task)
    
    return {"message": "Análisis iniciado en segundo plano."}

@app.get("/api/status")
async def get_status():
    """Endpoint REST clásico por si no quieres usar WebSockets."""
    return job_state

# ==========================================
# WEBSOCKETS
# ==========================================

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """
    El frontend se conecta aquí para escuchar los cambios en tiempo real.
    """
    await websocket.accept()
    try:
        while True:
            # Calculamos el porcentaje
            total = job_state["total_frames"]
            current = job_state["current_frame"]
            percentage = round((current / total * 100), 2) if total > 0 else 0

            # Preparamos la carga útil (payload)
            payload = {
                "status": job_state["status"],
                "current_frame": current,
                "total_frames": total,
                "percentage": percentage,
                "is_running": job_state["is_running"]
            }
            
            # Enviamos la info al front
            await websocket.send_json(payload)
            
            # Si el trabajo terminó y ya lo notificamos, podemos cerrar (opcional)
            if not job_state["is_running"] and job_state["status"] == "Completado":
                await asyncio.sleep(1) # Damos margen para que el front lo lea
                break
                
            # No saturamos el loop, revisamos cada 0.5 segundos
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("Cliente de WebSocket desconectado.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)